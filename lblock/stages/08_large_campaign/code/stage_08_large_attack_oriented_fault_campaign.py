# ============================================================
# Stage 06 — Parametric glitch/fault engine for LBlock-64/80
#
# This stage consumes the public 8-bit target contract from Stage 05
# and the public S-box timing map from Stage 04. It creates a controlled,
# reproducible simulation of timing-dependent clock glitches and final-round
# faults. The primary paper-faithful model is a 4-bit random-AND fault on the
# input of a final-round S-box. Four control models are also implemented.
#
# Public outputs contain only attacker-observable values and configured glitch
# parameters. Private ground truth contains keys, internal states, actual hit
# locations and labels. The public campaign is frozen before private validation.
# ============================================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import csv
import hashlib
import json
import math
import platform
import sys
import time

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# 1. Exact LBlock-64/80 reference core
# ============================================================

BLOCK_SIZE_BITS = 64
KEY_SIZE_BITS = 80
NUM_ROUNDS = 32
MASK32 = 0xFFFFFFFF
MASK80 = (1 << 80) - 1

# S0..S7 used in the round function.
SBOX: List[List[int]] = [
    [0xE, 0x9, 0xF, 0x0, 0xD, 0x4, 0xA, 0xB,
     0x1, 0x2, 0x8, 0x3, 0x7, 0x6, 0xC, 0x5],
    [0x4, 0xB, 0xE, 0x9, 0xF, 0xD, 0x0, 0xA,
     0x7, 0xC, 0x5, 0x6, 0x2, 0x8, 0x1, 0x3],
    [0x1, 0xE, 0x7, 0xC, 0xF, 0xD, 0x0, 0x6,
     0xB, 0x5, 0x9, 0x3, 0x2, 0x4, 0x8, 0xA],
    [0x7, 0x6, 0x8, 0xB, 0x0, 0xF, 0x3, 0xE,
     0x9, 0xA, 0xC, 0xD, 0x5, 0x2, 0x4, 0x1],
    [0xE, 0x5, 0xF, 0x0, 0x7, 0x2, 0xC, 0xD,
     0x1, 0x8, 0x4, 0x9, 0xB, 0xA, 0x6, 0x3],
    [0x2, 0xD, 0xB, 0xC, 0xF, 0xE, 0x0, 0x9,
     0x7, 0xA, 0x6, 0x3, 0x1, 0x8, 0x4, 0x5],
    [0xB, 0x9, 0x4, 0xE, 0x0, 0xF, 0xA, 0xD,
     0x6, 0xC, 0x5, 0x7, 0x3, 0x8, 0x1, 0x2],
    [0xD, 0xA, 0xF, 0x0, 0xE, 0x4, 0x9, 0xB,
     0x2, 0x1, 0x8, 0x3, 0x7, 0x5, 0xC, 0x6],
]

# S8 and S9 used by the 80-bit key schedule.
S8 = [0x8, 0x7, 0xE, 0x5, 0xF, 0xD, 0x0, 0x6,
      0xB, 0xC, 0x9, 0xA, 0x2, 0x4, 0x1, 0x3]
S9 = [0xB, 0x5, 0xF, 0x0, 0x7, 0x2, 0x9, 0xD,
      0x4, 0x8, 0x1, 0xC, 0xE, 0xA, 0x3, 0x6]

# Output nibble i receives the S-box output indexed by this table.
P_SOURCE_FOR_OUTPUT = [1, 3, 0, 2, 5, 7, 4, 6]

OFFICIAL_TEST_VECTORS = [
    {
        "plaintext_hex": "0000000000000000",
        "key_hex": "00000000000000000000",
        "ciphertext_hex": "c218185308e75bcd",
    },
    {
        "plaintext_hex": "0123456789abcdef",
        "key_hex": "0123456789abcdeffedc",
        "ciphertext_hex": "4b7179d8ebee0c26",
    },
]


def rol(value: int, shift: int, width: int) -> int:
    shift %= width
    mask = (1 << width) - 1
    value &= mask
    return value if shift == 0 else (
        ((value << shift) | (value >> (width - shift))) & mask
    )


def ror(value: int, shift: int, width: int) -> int:
    shift %= width
    mask = (1 << width) - 1
    value &= mask
    return value if shift == 0 else (
        ((value >> shift) | (value << (width - shift))) & mask
    )


def get_nibble(word: int, index: int) -> int:
    if not 0 <= index < 8:
        raise ValueError("nibble index must be in 0..7")
    return (word >> (4 * index)) & 0xF


def pack_nibbles(values: Sequence[int]) -> int:
    if len(values) != 8:
        raise ValueError("exactly eight nibbles are required")
    output = 0
    for index, value in enumerate(values):
        if not 0 <= int(value) <= 0xF:
            raise ValueError("nibble values must be in 0..15")
        output |= int(value) << (4 * index)
    return output & MASK32


def int_to_bits(value: int, width: int) -> str:
    return format(value & ((1 << width) - 1), f"0{width}b")


def key_schedule_lblock(master_key: int) -> List[int]:
    if not 0 <= master_key < (1 << 80):
        raise ValueError("master_key must be an 80-bit integer")

    key_register = master_key
    round_keys = [(key_register >> 48) & MASK32]

    for round_index in range(1, NUM_ROUNDS):
        bits = int_to_bits(key_register, 80)
        bits = bits[29:] + bits[:29]

        top = S9[int(bits[0:4], 2)]
        next_nibble = S8[int(bits[4:8], 2)]
        bits = f"{top:04b}{next_nibble:04b}" + bits[8:]

        counter_bits = f"{round_index:05b}"
        bit_list = list(bits)
        for offset, bit in enumerate(counter_bits):
            position = 29 + offset
            bit_list[position] = "1" if bit_list[position] != bit else "0"

        key_register = int("".join(bit_list), 2) & MASK80
        round_keys.append((key_register >> 48) & MASK32)

    return round_keys


def lblock_f_from_inputs(sbox_inputs: Sequence[int]) -> int:
    outputs = [SBOX[index][int(sbox_inputs[index])] for index in range(8)]
    permuted = [outputs[source] for source in P_SOURCE_FOR_OUTPUT]
    return pack_nibbles(permuted)


def lblock_f(x_word: int, round_key: int) -> int:
    xor_word = (x_word ^ round_key) & MASK32
    inputs = [get_nibble(xor_word, index) for index in range(8)]
    return lblock_f_from_inputs(inputs)


def encrypt_block_lblock(plaintext: int, master_key: int) -> int:
    round_keys = key_schedule_lblock(master_key)
    x0 = plaintext & MASK32
    x1 = (plaintext >> 32) & MASK32
    x_prev2, x_prev1 = x0, x1

    for round_index in range(NUM_ROUNDS):
        x_new = (
            lblock_f(x_prev1, round_keys[round_index])
            ^ rol(x_prev2, 8, 32)
        ) & MASK32
        x_prev2, x_prev1 = x_prev1, x_new

    return ((x_prev2 << 32) | x_prev1) & ((1 << 64) - 1)


def decrypt_block_lblock(ciphertext: int, master_key: int) -> int:
    round_keys = key_schedule_lblock(master_key)
    x32 = (ciphertext >> 32) & MASK32
    x33 = ciphertext & MASK32
    x_next2, x_next1 = x33, x32

    for round_index in range(NUM_ROUNDS - 1, -1, -1):
        x_previous = ror(
            x_next2 ^ lblock_f(x_next1, round_keys[round_index]),
            8,
            32,
        )
        x_next2, x_next1 = x_next1, x_previous

    return ((x_next2 << 32) | x_next1) & ((1 << 64) - 1)


def final_round_context(plaintext: int, master_key: int) -> Dict[str, Any]:
    """Return X31, X32, K32, final S-box inputs and the healthy ciphertext."""
    round_keys = key_schedule_lblock(master_key)
    x0 = plaintext & MASK32
    x1 = (plaintext >> 32) & MASK32
    states = [x0, x1]

    for round_index in range(31):
        x_new = (
            lblock_f(states[-1], round_keys[round_index])
            ^ rol(states[-2], 8, 32)
        ) & MASK32
        states.append(x_new)

    x31 = states[31]
    x32 = states[32]
    round_key_32 = round_keys[31]
    xor_input = (x32 ^ round_key_32) & MASK32
    sbox_inputs = [get_nibble(xor_input, index) for index in range(8)]
    f_output = lblock_f_from_inputs(sbox_inputs)
    x33 = (f_output ^ rol(x31, 8, 32)) & MASK32
    ciphertext = ((x32 << 32) | x33) & ((1 << 64) - 1)

    return {
        "x31": x31,
        "x32": x32,
        "round_key_32": round_key_32,
        "sbox_inputs": sbox_inputs,
        "f_output": f_output,
        "x33": x33,
        "ciphertext": ciphertext,
    }


def final_round_with_faulted_inputs(
    x31: int,
    x32: int,
    faulted_inputs: Sequence[int],
) -> int:
    f_faulted = lblock_f_from_inputs(faulted_inputs)
    x33_faulted = (f_faulted ^ rol(x31, 8, 32)) & MASK32
    return ((x32 << 32) | x33_faulted) & ((1 << 64) - 1)


# ============================================================
# 2. Configuration and data classes
# ============================================================

FAULT_MODELS = (
    "random_and_4",
    "random_and_2",
    "single_bit_flip",
    "stuck_at_bit",
    "random_nibble",
)

FAULT_CATEGORIES = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)
CATEGORY_TO_ID = {name: index for index, name in enumerate(FAULT_CATEGORIES)}


@dataclass(frozen=True)
class Stage06Config:
    input_stage5_run_directory: str
    output_root: str = "runs/stage_06"
    random_seed: int = 20260718

    number_of_experiments: int = 12000
    number_of_keys: int = 8
    number_of_sessions: int = 4

    # Target timing variability after Stage 03 alignment.
    global_timing_jitter_sigma_samples: float = 0.35
    local_sbox_jitter_sigma_samples: float = 0.18
    injection_timing_jitter_sigma_samples: float = 0.20
    session_timing_shift_sigma_samples: float = 0.25

    # Fault-model mixture. The primary 4-bit random-AND model dominates.
    random_and_4_weight: float = 0.65
    random_and_2_weight: float = 0.10
    single_bit_flip_weight: float = 0.10
    stuck_at_bit_weight: float = 0.08
    random_nibble_weight: float = 0.07

    # Synthetic response-trace generation.
    response_trace_noise_sigma: float = 0.055
    response_trace_baseline_sigma: float = 0.035
    response_trace_gain_sigma: float = 0.06
    save_response_traces: bool = True
    save_plots: bool = True

    # Public validation thresholds.
    minimum_category_fraction: float = 0.003
    minimum_primary_model_fraction: float = 0.55
    maximum_determinism_mismatches: int = 0
    deterministic_recheck_count: int = 64

    enable_private_validation: bool = True


@dataclass(frozen=True)
class GlitchParameters:
    target_sbox_index: int
    nominal_target_center_sample: float
    offset_samples: float
    width_samples: float
    strength: float
    repeat: int
    repeat_spacing_samples: float
    sampling_regime: str
    fault_model: str


@dataclass
class ExperimentRecord:
    public_row: Dict[str, Any]
    private_row: Dict[str, Any]
    response_trace: np.ndarray
    private_arrays: Dict[str, np.ndarray]


# ============================================================
# 3. File helpers
# ============================================================


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def hex_fixed(value: int, bits: int) -> str:
    return format(value & ((1 << bits) - 1), f"0{bits // 4}x")


def hamming_distance(value_a: int, value_b: int) -> int:
    return int(value_a ^ value_b).bit_count()


def sigmoid(value: float) -> float:
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def stable_json_hash(data: Any) -> str:
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ============================================================
# 4. Contract and source-data loading
# ============================================================


def load_stage_contracts(
    stage5_run_directory: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path]:
    stage5_summary_path = stage5_run_directory / "stage_05_summary.json"
    target_contract_path = (
        stage5_run_directory / "public" / "lblock_8bit_target_contract.json"
    )
    if not stage5_summary_path.is_file():
        raise FileNotFoundError(stage5_summary_path)
    if not target_contract_path.is_file():
        raise FileNotFoundError(target_contract_path)

    stage5_summary = read_json(stage5_summary_path)
    if not stage5_summary.get("all_checks_passed", False):
        raise RuntimeError("Stage 05 did not pass all checks")

    target_contract = read_json(target_contract_path)
    if int(target_contract.get("total_target_bits", 0)) != 8:
        raise RuntimeError("Stage 05 target contract does not define eight bits")
    if len(target_contract.get("selected_sbox_indices", [])) != 2:
        raise RuntimeError("Stage 05 must select exactly two S-boxes")

    stage4_run_directory = Path(
        stage5_summary["input_stage_04_run_directory"]
    ).expanduser().resolve()
    timing_map_path = (
        stage4_run_directory / "public" / "lblock_final_round_timing_map.json"
    )
    if not timing_map_path.is_file():
        raise FileNotFoundError(timing_map_path)
    timing_map = read_json(timing_map_path)

    stage3_run_directory = Path(
        stage5_summary["input_stage_03_run_directory"]
    ).expanduser().resolve()
    roi_path = stage3_run_directory / "public" / "final_round_roi_traces.npz"
    if not roi_path.is_file():
        raise FileNotFoundError(roi_path)

    return stage5_summary, target_contract, timing_map, target_contract_path, roi_path


def load_healthy_roi_source(roi_path: Path) -> Dict[str, np.ndarray]:
    with np.load(roi_path, allow_pickle=False) as data:
        traces = np.asarray(data["traces"], dtype=np.float32)
        trace_ids = np.asarray(data["trace_ids"], dtype=np.int32)
        absolute_samples = np.asarray(
            data["absolute_sample_indices"], dtype=np.int32
        )
        sample_axis_seconds = np.asarray(
            data["sample_axis_seconds"], dtype=np.float64
        )

    if traces.ndim != 2 or traces.shape[1] != absolute_samples.size:
        raise ValueError("Invalid Stage 03 ROI trace shape")
    if not np.all(np.isfinite(traces)):
        raise ValueError("Healthy ROI traces contain non-finite values")

    return {
        "traces": traces,
        "trace_ids": trace_ids,
        "absolute_samples": absolute_samples,
        "sample_axis_seconds": sample_axis_seconds,
    }


# ============================================================
# 5. Random key and parameter generation
# ============================================================


def random_80bit_integer(rng: np.random.Generator) -> int:
    high_16 = int(rng.integers(0, 1 << 16, dtype=np.uint32))
    low_64 = int(rng.integers(0, 1 << 64, dtype=np.uint64))
    return ((high_16 << 64) | low_64) & MASK80


def random_64bit_integer(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 1 << 64, dtype=np.uint64))


def build_key_pool(config: Stage06Config) -> List[int]:
    rng = np.random.default_rng(config.random_seed + 6001)
    keys: List[int] = []
    while len(keys) < config.number_of_keys:
        candidate = random_80bit_integer(rng)
        if candidate not in keys:
            keys.append(candidate)
    return keys


def model_weights(config: Stage06Config) -> np.ndarray:
    weights = np.asarray([
        config.random_and_4_weight,
        config.random_and_2_weight,
        config.single_bit_flip_weight,
        config.stuck_at_bit_weight,
        config.random_nibble_weight,
    ], dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("fault-model weights must be nonnegative and sum to one")
    return weights


def choose_sampling_regime(experiment_id: int) -> str:
    # A fixed 20-slot schedule guarantees coverage while preserving randomness
    # inside each regime: 20% miss, 45% target, 15% neighbour, 10% multi,
    # 10% invalid-oriented probes.
    slot = experiment_id % 20
    if slot < 4:
        return "miss_probe"
    if slot < 13:
        return "target_probe"
    if slot < 16:
        return "neighbor_probe"
    if slot < 18:
        return "multi_probe"
    return "invalid_probe"


def nearest_neighbor_index(target_index: int, rng: np.random.Generator) -> int:
    choices = []
    if target_index > 0:
        choices.append(target_index - 1)
    if target_index < 7:
        choices.append(target_index + 1)
    return int(choices[int(rng.integers(0, len(choices)))])


def sample_glitch_parameters(
    experiment_id: int,
    target_index: int,
    centers: np.ndarray,
    config: Stage06Config,
    rng: np.random.Generator,
) -> GlitchParameters:
    regime = choose_sampling_regime(experiment_id)
    target_center = float(centers[target_index])
    fault_model = str(rng.choice(FAULT_MODELS, p=model_weights(config)))

    if regime == "miss_probe":
        direction = -1.0 if rng.random() < 0.5 else 1.0
        offset = direction * float(rng.uniform(17.0, 30.0))
        width = float(rng.uniform(0.55, 2.0))
        strength = float(rng.uniform(0.08, 0.42))
        repeat = 1
        repeat_spacing = float(rng.uniform(1.0, 3.0))

    elif regime == "target_probe":
        offset = float(rng.uniform(-6.5, 6.5))
        width = float(rng.uniform(0.75, 4.75))
        strength = float(rng.uniform(0.30, 1.08))
        repeat = int(rng.choice([1, 1, 1, 2]))
        repeat_spacing = float(rng.uniform(1.5, 5.0))

    elif regime == "neighbor_probe":
        neighbor = nearest_neighbor_index(target_index, rng)
        neighbor_offset = float(centers[neighbor] - centers[target_index])
        offset = neighbor_offset + float(rng.normal(0.0, 2.0))
        width = float(rng.uniform(0.75, 4.5))
        strength = float(rng.uniform(0.38, 1.12))
        repeat = int(rng.choice([1, 1, 2]))
        repeat_spacing = float(rng.uniform(1.5, 5.5))

    elif regime == "multi_probe":
        neighbor = nearest_neighbor_index(target_index, rng)
        midpoint_offset = 0.5 * float(centers[neighbor] - centers[target_index])
        offset = midpoint_offset + float(rng.normal(0.0, 1.5))
        width = float(rng.uniform(6.5, 13.5))
        strength = float(rng.uniform(0.52, 1.18))
        repeat = int(rng.choice([1, 2, 2, 3]))
        repeat_spacing = float(rng.uniform(3.0, 9.5))

    else:  # invalid_probe
        offset = float(rng.uniform(-8.0, 8.0))
        width = float(rng.uniform(9.0, 18.0))
        strength = float(rng.uniform(1.05, 1.80))
        repeat = int(rng.integers(2, 5))
        repeat_spacing = float(rng.uniform(2.0, 9.5))

    return GlitchParameters(
        target_sbox_index=target_index,
        nominal_target_center_sample=target_center,
        offset_samples=offset,
        width_samples=width,
        strength=strength,
        repeat=repeat,
        repeat_spacing_samples=repeat_spacing,
        sampling_regime=regime,
        fault_model=fault_model,
    )


# ============================================================
# 6. Timing-overlap and event-probability engine
# ============================================================


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def pulse_centers(
    parameters: GlitchParameters,
    injection_jitter: float,
) -> np.ndarray:
    first_center = (
        parameters.nominal_target_center_sample
        + parameters.offset_samples
        + injection_jitter
    )
    return np.asarray([
        first_center + pulse_index * parameters.repeat_spacing_samples
        for pulse_index in range(parameters.repeat)
    ], dtype=np.float64)


def compute_hit_scores(
    parameters: GlitchParameters,
    pulse_center_values: np.ndarray,
    actual_centers: np.ndarray,
    core_widths: np.ndarray,
) -> np.ndarray:
    """Continuous exposure score for each of the eight final-round S-boxes."""
    scores = np.zeros(8, dtype=np.float64)
    pulse_width = max(parameters.width_samples, 0.25)
    half_width = pulse_width / 2.0

    for sbox_index in range(8):
        center = float(actual_centers[sbox_index])
        operation_width = max(float(core_widths[sbox_index]), 1.0)
        operation_start = center - operation_width / 2.0
        operation_end = center + operation_width / 2.0
        sigma = max(0.35 * operation_width, 0.9)

        survival = 1.0
        for center_of_pulse in pulse_center_values:
            pulse_start = float(center_of_pulse) - half_width
            pulse_end = float(center_of_pulse) + half_width
            overlap = interval_overlap(
                pulse_start,
                pulse_end,
                operation_start,
                operation_end,
            )
            overlap_ratio = overlap / pulse_width
            distance_kernel = math.exp(
                -0.5 * ((float(center_of_pulse) - center) / sigma) ** 2
            )
            exposure = float(np.clip(
                0.60 * overlap_ratio + 0.40 * distance_kernel,
                0.0,
                0.995,
            ))
            survival *= 1.0 - exposure

        scores[sbox_index] = 1.0 - survival

    return scores


def activation_probabilities(
    parameters: GlitchParameters,
    hit_scores: np.ndarray,
) -> np.ndarray:
    probabilities = np.empty(8, dtype=np.float64)
    width_bonus = 0.18 * max(parameters.width_samples - 2.0, 0.0)
    repeat_bonus = 0.40 * max(parameters.repeat - 1, 0)
    strength_bonus = 1.40 * (parameters.strength - 0.50)

    for index, exposure in enumerate(hit_scores):
        logit = (
            -5.0
            + 6.0 * float(exposure)
            + strength_bonus
            + width_bonus
            + repeat_bonus
        )
        probabilities[index] = float(np.clip(sigmoid(logit), 0.0005, 0.997))

    return probabilities


def invalid_probability(
    parameters: GlitchParameters,
    hit_scores: np.ndarray,
) -> float:
    energy = parameters.strength * parameters.width_samples * parameters.repeat
    exposed_operations = int(np.sum(hit_scores > 0.15))
    logit = (
        -8.0
        + 0.25 * energy
        + 0.18 * max(parameters.width_samples - 6.0, 0.0)
        + 0.35 * max(parameters.repeat - 1, 0)
        + 0.45 * max(exposed_operations - 2, 0)
    )
    return float(np.clip(sigmoid(logit), 0.0005, 0.985))


# ============================================================
# 7. Fault semantics
# ============================================================


def apply_fault_model(
    original: int,
    model: str,
    rng: np.random.Generator,
) -> Tuple[int, Dict[str, Any]]:
    if model == "random_and_4":
        mask = int(rng.integers(0, 16))
        faulted = original & mask
        return faulted, {"and_mask": mask}

    if model == "random_and_2":
        positions = sorted(
            int(value)
            for value in rng.choice(4, size=2, replace=False)
        )
        random_bits = int(rng.integers(0, 4))
        mask = 0xF
        for local_index, bit_position in enumerate(positions):
            bit_value = (random_bits >> local_index) & 1
            if bit_value == 0:
                mask &= ~(1 << bit_position)
            else:
                mask |= 1 << bit_position
        faulted = original & mask
        return faulted, {
            "and_mask": mask,
            "randomized_positions": positions,
        }

    if model == "single_bit_flip":
        bit_position = int(rng.integers(0, 4))
        faulted = original ^ (1 << bit_position)
        return faulted, {"bit_position": bit_position}

    if model == "stuck_at_bit":
        bit_position = int(rng.integers(0, 4))
        stuck_value = int(rng.integers(0, 2))
        faulted = (
            (original & ~(1 << bit_position))
            | (stuck_value << bit_position)
        )
        return faulted, {
            "bit_position": bit_position,
            "stuck_value": stuck_value,
        }

    if model == "random_nibble":
        replacement = int(rng.integers(0, 16))
        return replacement, {"replacement": replacement}

    raise ValueError(f"Unknown fault model: {model}")


def classify_fault_event(
    target_index: int,
    impacted_mask: np.ndarray,
    invalid: bool,
    ciphertext_equal: bool,
) -> str:
    if invalid:
        return "invalid_reset"

    impacted_indices = np.where(impacted_mask > 0)[0]
    impacted_count = int(impacted_indices.size)

    if impacted_count == 0:
        return "missed"
    if impacted_count >= 2:
        return "multi_hit"
    if int(impacted_indices[0]) != target_index:
        return "off_target"
    return "clean_target_ineffective" if ciphertext_equal else "clean_target_effective"


# ============================================================
# 8. Synthetic observable response traces
# ============================================================


def robust_standardize_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    median = float(np.median(trace))
    mad = float(1.4826 * np.median(np.abs(trace - median)))
    scale = mad if mad > 1e-10 else float(np.std(trace))
    if scale <= 1e-10:
        scale = 1.0
    return (trace - median) / scale


def add_gaussian_pulse(
    trace: np.ndarray,
    axis_samples: np.ndarray,
    center: float,
    sigma: float,
    amplitude: float,
) -> None:
    trace += amplitude * np.exp(
        -0.5 * ((axis_samples.astype(np.float64) - center) / max(sigma, 0.25)) ** 2
    )


def synthesize_response_trace(
    base_trace: np.ndarray,
    axis_samples: np.ndarray,
    pulse_center_values: np.ndarray,
    parameters: GlitchParameters,
    actual_centers: np.ndarray,
    original_inputs: np.ndarray,
    faulted_inputs: np.ndarray,
    impacted_mask: np.ndarray,
    invalid: bool,
    rng: np.random.Generator,
    config: Stage06Config,
) -> np.ndarray:
    trace = robust_standardize_trace(base_trace)
    gain = float(rng.normal(1.0, config.response_trace_gain_sigma))
    baseline = float(rng.normal(0.0, config.response_trace_baseline_sigma))
    trace = gain * trace + baseline

    # Clock-glitch signature: a bipolar pulse train visible to the attacker.
    glitch_sigma = max(0.32 * parameters.width_samples, 0.35)
    for pulse_center in pulse_center_values:
        amplitude = 0.55 + 0.95 * parameters.strength
        add_gaussian_pulse(trace, axis_samples, float(pulse_center), glitch_sigma, amplitude)
        add_gaussian_pulse(
            trace,
            axis_samples,
            float(pulse_center) + 0.65 * glitch_sigma,
            0.70 * glitch_sigma,
            -0.52 * amplitude,
        )

    # Data-dependent local response for each actually impacted operation.
    for sbox_index in np.where(impacted_mask > 0)[0]:
        input_hd = int(original_inputs[sbox_index] ^ faulted_inputs[sbox_index]).bit_count()
        output_hd = int(
            SBOX[int(sbox_index)][int(original_inputs[sbox_index])]
            ^ SBOX[int(sbox_index)][int(faulted_inputs[sbox_index])]
        ).bit_count()
        amplitude = 0.12 * input_hd + 0.10 * output_hd
        if amplitude > 0:
            add_gaussian_pulse(
                trace,
                axis_samples,
                float(actual_centers[sbox_index]) + 0.55,
                1.10,
                amplitude,
            )

    if invalid:
        first_pulse = float(np.min(pulse_center_values))
        drop_start = int(np.searchsorted(axis_samples, first_pulse + 1.5))
        drop_start = int(np.clip(drop_start, 0, trace.size - 1))
        if rng.random() < 0.55:
            # Reset-like decay.
            decay = np.exp(-np.linspace(0.0, 5.0, trace.size - drop_start))
            trace[drop_start:] *= decay
        else:
            # Timeout/saturation-like plateau.
            plateau = float(np.sign(rng.normal()) * (3.5 + parameters.strength))
            trace[drop_start:] = plateau + rng.normal(
                0.0, 0.08, size=trace.size - drop_start
            )

    trace += rng.normal(
        0.0,
        config.response_trace_noise_sigma,
        size=trace.size,
    )
    return trace.astype(np.float32)


def trace_features(
    trace: np.ndarray,
    axis_samples: np.ndarray,
    target_center: float,
    pulse_center_values: np.ndarray,
) -> Dict[str, float]:
    values = np.asarray(trace, dtype=np.float64)
    derivative = np.diff(values, prepend=values[0])
    target_mask = np.abs(axis_samples.astype(np.float64) - target_center) <= 5.0
    pulse_mask = np.zeros(values.size, dtype=bool)
    for center in pulse_center_values:
        pulse_mask |= np.abs(axis_samples.astype(np.float64) - float(center)) <= 4.0

    target_values = values[target_mask] if np.any(target_mask) else values
    pulse_values = values[pulse_mask] if np.any(pulse_mask) else values

    return {
        "trace_mean": float(np.mean(values)),
        "trace_std": float(np.std(values)),
        "trace_peak_to_peak": float(np.ptp(values)),
        "trace_max_absolute": float(np.max(np.abs(values))),
        "trace_high_frequency_energy": float(np.mean(derivative ** 2)),
        "target_window_energy": float(np.mean(target_values ** 2)),
        "pulse_window_energy": float(np.mean(pulse_values ** 2)),
        "saturation_fraction": float(np.mean(np.abs(values) > 4.0)),
    }


# ============================================================
# 9. One deterministic experiment
# ============================================================


def run_one_experiment(
    experiment_id: int,
    selected_targets: Sequence[int],
    timing_map: Mapping[str, Any],
    healthy_source: Mapping[str, np.ndarray],
    key_pool: Sequence[int],
    session_shifts: np.ndarray,
    config: Stage06Config,
) -> ExperimentRecord:
    seed_sequence = np.random.SeedSequence([config.random_seed, experiment_id, 6006])
    rng = np.random.default_rng(seed_sequence)

    target_index = int(selected_targets[experiment_id % len(selected_targets)])
    key_id = int((experiment_id // len(selected_targets)) % config.number_of_keys)
    session_id = int(
        (experiment_id // (len(selected_targets) * config.number_of_keys))
        % config.number_of_sessions
    )

    centers = np.asarray(
        [int(entry["center_sample"]) for entry in timing_map["sboxes"]],
        dtype=np.float64,
    )
    core_widths = np.asarray(
        [
            int(entry["core_window_end_sample_exclusive"])
            - int(entry["core_window_start_sample_inclusive"])
            for entry in timing_map["sboxes"]
        ],
        dtype=np.float64,
    )

    parameters = sample_glitch_parameters(
        experiment_id,
        target_index,
        centers,
        config,
        rng,
    )

    global_jitter = float(rng.normal(0.0, config.global_timing_jitter_sigma_samples))
    local_jitter = rng.normal(
        0.0,
        config.local_sbox_jitter_sigma_samples,
        size=8,
    )
    actual_centers = (
        centers
        + float(session_shifts[session_id])
        + global_jitter
        + local_jitter
    )
    injection_jitter = float(
        rng.normal(0.0, config.injection_timing_jitter_sigma_samples)
    )
    pulse_center_values = pulse_centers(parameters, injection_jitter)
    hit_scores = compute_hit_scores(
        parameters,
        pulse_center_values,
        actual_centers,
        core_widths,
    )
    probabilities = activation_probabilities(parameters, hit_scores)
    p_invalid = invalid_probability(parameters, hit_scores)
    invalid = bool(rng.random() < p_invalid)

    master_key = int(key_pool[key_id])
    plaintext = random_64bit_integer(rng)
    context = final_round_context(plaintext, master_key)
    healthy_ciphertext = int(context["ciphertext"])

    # Cross-check the final-round context against the full cipher on every row.
    full_ciphertext = encrypt_block_lblock(plaintext, master_key)
    if healthy_ciphertext != full_ciphertext:
        raise AssertionError("final-round reconstruction mismatch")

    original_inputs = np.asarray(context["sbox_inputs"], dtype=np.uint8)
    faulted_inputs = original_inputs.copy()
    impacted_mask = np.zeros(8, dtype=np.uint8)
    model_details: Dict[str, Any] = {}

    if not invalid:
        activated = rng.random(8) < probabilities
        impacted_mask[:] = activated.astype(np.uint8)
        for sbox_index in np.where(activated)[0]:
            faulted_value, details = apply_fault_model(
                int(original_inputs[sbox_index]),
                parameters.fault_model,
                rng,
            )
            faulted_inputs[sbox_index] = int(faulted_value)
            model_details[f"S{int(sbox_index)}"] = details

        faulty_ciphertext = final_round_with_faulted_inputs(
            int(context["x31"]),
            int(context["x32"]),
            faulted_inputs,
        )
        response_received = True
        ciphertext_equal = bool(faulty_ciphertext == healthy_ciphertext)
        ciphertext_hd = hamming_distance(faulty_ciphertext, healthy_ciphertext)
        invalid_subtype = ""
    else:
        faulty_ciphertext = None
        response_received = False
        ciphertext_equal = False
        ciphertext_hd = -1
        invalid_subtype = "reset" if rng.random() < 0.55 else "timeout"

    category = classify_fault_event(
        target_index,
        impacted_mask,
        invalid,
        ciphertext_equal,
    )

    base_index = int(rng.integers(0, healthy_source["traces"].shape[0]))
    base_trace = healthy_source["traces"][base_index]
    response_trace = synthesize_response_trace(
        base_trace,
        healthy_source["absolute_samples"],
        pulse_center_values,
        parameters,
        actual_centers,
        original_inputs,
        faulted_inputs,
        impacted_mask,
        invalid,
        rng,
        config,
    )
    features = trace_features(
        response_trace,
        healthy_source["absolute_samples"],
        float(centers[target_index]),
        pulse_center_values,
    )

    impacted_indices = [int(value) for value in np.where(impacted_mask > 0)[0]]
    target_impacted = bool(impacted_mask[target_index])
    off_target_impacted = any(value != target_index for value in impacted_indices)
    changed_input_count = int(np.sum(original_inputs != faulted_inputs))

    public_row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "target_sbox": f"S{target_index}",
        "target_sbox_index": target_index,
        "key_id": key_id,
        "session_id": session_id,
        "source_healthy_trace_id": int(healthy_source["trace_ids"][base_index]),
        "fault_model": parameters.fault_model,
        "nominal_target_center_sample": parameters.nominal_target_center_sample,
        "timing_offset_samples": parameters.offset_samples,
        "first_pulse_nominal_sample": (
            parameters.nominal_target_center_sample + parameters.offset_samples
        ),
        "width_samples": parameters.width_samples,
        "strength": parameters.strength,
        "repeat": parameters.repeat,
        "repeat_spacing_samples": parameters.repeat_spacing_samples,
        "plaintext_hex": hex_fixed(plaintext, 64),
        "healthy_ciphertext_hex": hex_fixed(healthy_ciphertext, 64),
        "response_received": response_received,
        "faulty_ciphertext_hex": (
            hex_fixed(int(faulty_ciphertext), 64) if faulty_ciphertext is not None else ""
        ),
        "ciphertext_equal": (ciphertext_equal if response_received else ""),
        "ciphertext_hamming_distance": ciphertext_hd,
        **features,
    }

    private_row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "category": category,
        "category_id": CATEGORY_TO_ID[category],
        "sampling_regime": parameters.sampling_regime,
        "target_sbox": f"S{target_index}",
        "target_sbox_index": target_index,
        "key_id": key_id,
        "session_id": session_id,
        "master_key_hex": hex_fixed(master_key, 80),
        "round_key_32_hex": hex_fixed(int(context["round_key_32"]), 32),
        "x31_hex": hex_fixed(int(context["x31"]), 32),
        "x32_hex": hex_fixed(int(context["x32"]), 32),
        "target_original_input": int(original_inputs[target_index]),
        "target_faulted_input": int(faulted_inputs[target_index]),
        "impacted_sboxes": ";".join(f"S{value}" for value in impacted_indices),
        "impacted_sbox_count": len(impacted_indices),
        "target_impacted": target_impacted,
        "off_target_impacted": off_target_impacted,
        "changed_sbox_input_count": changed_input_count,
        "fault_effective": bool(response_received and not ciphertext_equal),
        "invalid_subtype": invalid_subtype,
        "invalid_probability": p_invalid,
        "global_jitter_samples": global_jitter,
        "injection_jitter_samples": injection_jitter,
        "model_details_json": json.dumps(model_details, sort_keys=True),
    }

    private_arrays = {
        "master_key_bytes": np.frombuffer(master_key.to_bytes(10, "big"), dtype=np.uint8),
        "round_key_32": np.asarray(context["round_key_32"], dtype=np.uint32),
        "x31": np.asarray(context["x31"], dtype=np.uint32),
        "x32": np.asarray(context["x32"], dtype=np.uint32),
        "original_inputs": original_inputs.astype(np.uint8),
        "faulted_inputs": faulted_inputs.astype(np.uint8),
        "hit_scores": hit_scores.astype(np.float32),
        "activation_probabilities": probabilities.astype(np.float32),
        "actual_centers": actual_centers.astype(np.float32),
        "pulse_centers": np.pad(
            pulse_center_values.astype(np.float32),
            (0, 4 - pulse_center_values.size),
            constant_values=np.nan,
        )[:4],
        "impacted_mask": impacted_mask.astype(np.uint8),
        "category_id": np.asarray(CATEGORY_TO_ID[category], dtype=np.int8),
    }

    return ExperimentRecord(
        public_row=public_row,
        private_row=private_row,
        response_trace=response_trace,
        private_arrays=private_arrays,
    )


# ============================================================
# 10. Analytical random-AND validation
# ============================================================


def random_and_4_analytical_tables() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    transition_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    maximum_formula_error = 0.0
    subset_property_holds = True

    for original in range(16):
        ineffective_count = 0
        counts = np.zeros(16, dtype=np.int32)
        for mask in range(16):
            faulted = original & mask
            counts[faulted] += 1
            ineffective = faulted == original
            ineffective_count += int(ineffective)
            subset_property_holds &= (faulted & ~original) == 0
            transition_rows.append({
                "original_input": original,
                "original_input_hex": f"{original:x}",
                "mask": mask,
                "mask_hex": f"{mask:x}",
                "faulted_input": faulted,
                "faulted_input_hex": f"{faulted:x}",
                "ineffective": ineffective,
                "transition_probability": 1.0 / 16.0,
            })

        exact_probability = ineffective_count / 16.0
        analytic_probability = 2.0 ** (-int(original).bit_count())
        formula_error = abs(exact_probability - analytic_probability)
        maximum_formula_error = max(maximum_formula_error, formula_error)
        summary_rows.append({
            "original_input": original,
            "original_input_hex": f"{original:x}",
            "hamming_weight": int(original).bit_count(),
            "exact_ineffective_probability": exact_probability,
            "analytic_ineffective_probability": analytic_probability,
            "absolute_error": formula_error,
            "reachable_faulted_values": int(np.sum(counts > 0)),
        })

    validation = {
        "model": "random_and_4",
        "definition": "X' = X AND R, where R is uniform over all 4-bit values",
        "mask_count": 16,
        "input_count": 16,
        "subset_property_holds": bool(subset_property_holds),
        "maximum_ineffective_probability_formula_error": maximum_formula_error,
        "formula": "P[X'=X | X=x] = 2^{-HW(x)}",
        "all_checks_passed": bool(subset_property_holds and maximum_formula_error < 1e-12),
    }
    return transition_rows, summary_rows, validation


# ============================================================
# 11. Validation
# ============================================================


def reference_core_validation(config: Stage06Config) -> Dict[str, Any]:
    vector_results = []
    for vector in OFFICIAL_TEST_VECTORS:
        plaintext = int(vector["plaintext_hex"], 16)
        key = int(vector["key_hex"], 16)
        expected = int(vector["ciphertext_hex"], 16)
        obtained = encrypt_block_lblock(plaintext, key)
        recovered = decrypt_block_lblock(obtained, key)
        vector_results.append({
            **vector,
            "obtained_ciphertext_hex": hex_fixed(obtained, 64),
            "encryption_passed": obtained == expected,
            "decryption_passed": recovered == plaintext,
        })

    rng = np.random.default_rng(config.random_seed + 6061)
    random_roundtrips = 256
    random_passed = 0
    final_round_passed = 0
    for _ in range(random_roundtrips):
        key = random_80bit_integer(rng)
        plaintext = random_64bit_integer(rng)
        ciphertext = encrypt_block_lblock(plaintext, key)
        random_passed += int(decrypt_block_lblock(ciphertext, key) == plaintext)
        context = final_round_context(plaintext, key)
        final_round_passed += int(context["ciphertext"] == ciphertext)

    all_passed = (
        all(item["encryption_passed"] and item["decryption_passed"] for item in vector_results)
        and random_passed == random_roundtrips
        and final_round_passed == random_roundtrips
    )
    return {
        "all_passed": all_passed,
        "official_vectors": vector_results,
        "random_roundtrip_tests": random_roundtrips,
        "random_roundtrip_passed": random_passed,
        "final_round_reconstruction_passed": final_round_passed,
    }


def campaign_digest(records: Sequence[ExperimentRecord], count: int) -> str:
    compact = []
    for record in records[:count]:
        compact.append({
            "public": record.public_row,
            "private": {
                key: value
                for key, value in record.private_row.items()
                if key not in {"model_details_json"}
            },
            "model_details_json": record.private_row["model_details_json"],
            "trace_sha256": hashlib.sha256(record.response_trace.tobytes()).hexdigest(),
        })
    return stable_json_hash(compact)


def public_metadata_leakage_audit(public_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    columns = set(public_rows[0].keys())
    forbidden_exact = {
        "master_key_hex",
        "round_key_32_hex",
        "x31_hex",
        "x32_hex",
        "category",
        "category_id",
        "impacted_sboxes",
        "impacted_sbox_count",
        "target_impacted",
        "actual_centers",
        "hit_scores",
        "activation_probabilities",
        "sampling_regime",
        "model_details_json",
    }
    forbidden_present = sorted(columns & forbidden_exact)
    suspicious_tokens = ("master_key", "round_key", "internal_state", "actual_center")
    suspicious_columns = sorted(
        column for column in columns if any(token in column.lower() for token in suspicious_tokens)
    )
    passed = not forbidden_present and not suspicious_columns
    return {
        "passed": passed,
        "public_column_count": len(columns),
        "forbidden_columns_present": forbidden_present,
        "suspicious_columns_present": suspicious_columns,
        "allowed_observable_fields_include": [
            "glitch parameters",
            "plaintext",
            "healthy and returned ciphertexts",
            "ciphertext equality/difference",
            "response status",
            "trace-derived features",
        ],
    }


def validate_campaign(
    records: Sequence[ExperimentRecord],
    target_contract: Mapping[str, Any],
    analytical_validation: Mapping[str, Any],
    core_validation: Mapping[str, Any],
    determinism_digest_first: str,
    determinism_digest_second: str,
    config: Stage06Config,
) -> Dict[str, Any]:
    categories = [record.private_row["category"] for record in records]
    models = [record.public_row["fault_model"] for record in records]
    category_counts = {name: categories.count(name) for name in FAULT_CATEGORIES}
    model_counts = {name: models.count(name) for name in FAULT_MODELS}
    total = len(records)

    public_rows = [record.public_row for record in records]
    leakage_audit = public_metadata_leakage_audit(public_rows)

    consistency_failures = 0
    locality_failures = 0
    for record in records:
        category = record.private_row["category"]
        impacted_count = int(record.private_row["impacted_sbox_count"])
        target_impacted = bool(record.private_row["target_impacted"])
        response_received = bool(record.public_row["response_received"])
        equal = record.public_row["ciphertext_equal"]

        if category == "missed":
            ok = response_received and impacted_count == 0 and equal is True
        elif category == "clean_target_ineffective":
            ok = response_received and impacted_count == 1 and target_impacted and equal is True
        elif category == "clean_target_effective":
            ok = response_received and impacted_count == 1 and target_impacted and equal is False
        elif category == "off_target":
            ok = response_received and impacted_count == 1 and not target_impacted
        elif category == "multi_hit":
            ok = response_received and impacted_count >= 2
        else:
            ok = (not response_received) and impacted_count == 0
        consistency_failures += int(not ok)

        if category.startswith("clean_target_"):
            locality_failures += int(not (impacted_count == 1 and target_impacted))

    primary_fraction = model_counts["random_and_4"] / total
    minimum_count = max(1, int(math.floor(config.minimum_category_fraction * total)))
    category_coverage_passed = all(value >= minimum_count for value in category_counts.values())

    target_counts: Dict[str, int] = {}
    for record in records:
        target = str(record.public_row["target_sbox"])
        target_counts[target] = target_counts.get(target, 0) + 1
    target_balance = max(target_counts.values()) - min(target_counts.values())

    checks = {
        "reference_core": {
            "passed": bool(core_validation["all_passed"]),
        },
        "random_and_4_analytical_semantics": {
            "passed": bool(analytical_validation["all_checks_passed"]),
        },
        "experiment_count": {
            "passed": total == config.number_of_experiments,
            "expected": config.number_of_experiments,
            "observed": total,
        },
        "all_six_categories_present": {
            "passed": category_coverage_passed,
            "minimum_count_per_category": minimum_count,
            "counts": category_counts,
        },
        "primary_random_and_model_dominates": {
            "passed": primary_fraction >= config.minimum_primary_model_fraction,
            "observed_fraction": primary_fraction,
            "minimum_fraction": config.minimum_primary_model_fraction,
        },
        "category_semantics_consistent": {
            "passed": consistency_failures == 0,
            "failure_count": consistency_failures,
        },
        "clean_target_locality": {
            "passed": locality_failures == 0,
            "failure_count": locality_failures,
        },
        "public_metadata_leakage_audit": leakage_audit,
        "two_target_balance": {
            "passed": len(target_counts) == 2 and target_balance <= 1,
            "counts": target_counts,
        },
        "eight_bit_contract_preserved": {
            "passed": int(target_contract["total_target_bits"]) == 8,
            "bit_indices": target_contract["selected_last_round_key_bit_indices"],
        },
        "deterministic_generation": {
            "passed": determinism_digest_first == determinism_digest_second,
            "first_digest": determinism_digest_first,
            "second_digest": determinism_digest_second,
        },
        "finite_response_traces": {
            "passed": all(np.all(np.isfinite(record.response_trace)) for record in records),
        },
    }
    all_passed = all(bool(item["passed"]) for item in checks.values())
    return {
        "all_public_and_private_checks_passed": all_passed,
        "checks": checks,
    }


# ============================================================
# 12. Aggregate reports and plots
# ============================================================


def aggregate_campaign(records: Sequence[ExperimentRecord]) -> Dict[str, Any]:
    total = len(records)
    category_counts = {name: 0 for name in FAULT_CATEGORIES}
    model_counts = {name: 0 for name in FAULT_MODELS}
    target_category_counts: Dict[str, Dict[str, int]] = {}
    model_category_counts: Dict[str, Dict[str, int]] = {
        model: {category: 0 for category in FAULT_CATEGORIES}
        for model in FAULT_MODELS
    }

    impacted_counts = []
    invalid_probabilities = []
    for record in records:
        category = str(record.private_row["category"])
        model = str(record.public_row["fault_model"])
        target = str(record.public_row["target_sbox"])
        category_counts[category] += 1
        model_counts[model] += 1
        model_category_counts[model][category] += 1
        target_category_counts.setdefault(
            target, {name: 0 for name in FAULT_CATEGORIES}
        )[category] += 1
        impacted_counts.append(int(record.private_row["impacted_sbox_count"]))
        invalid_probabilities.append(float(record.private_row["invalid_probability"]))

    category_rates = {name: value / total for name, value in category_counts.items()}
    model_rates = {name: value / total for name, value in model_counts.items()}

    primary_clean = [
        record for record in records
        if record.public_row["fault_model"] == "random_and_4"
        and record.private_row["category"] in {
            "clean_target_ineffective", "clean_target_effective"
        }
    ]
    ineffective_primary = sum(
        record.private_row["category"] == "clean_target_ineffective"
        for record in primary_clean
    )

    return {
        "number_of_experiments": total,
        "category_counts": category_counts,
        "category_rates": category_rates,
        "fault_model_counts": model_counts,
        "fault_model_rates": model_rates,
        "target_category_counts": target_category_counts,
        "model_category_counts": model_category_counts,
        "mean_impacted_sbox_count": float(np.mean(impacted_counts)),
        "maximum_impacted_sbox_count": int(np.max(impacted_counts)),
        "mean_invalid_probability": float(np.mean(invalid_probabilities)),
        "primary_random_and_clean_target_count": len(primary_clean),
        "primary_random_and_clean_target_ineffective_count": int(ineffective_primary),
        "primary_random_and_clean_target_ineffective_rate": (
            float(ineffective_primary / len(primary_clean)) if primary_clean else None
        ),
    }


def save_public_plots(
    public_directory: Path,
    records: Sequence[ExperimentRecord],
    analytical_summary_rows: Sequence[Mapping[str, Any]],
    config: Stage06Config,
) -> List[str]:
    """Plots derived only from public parameters or analytical model semantics."""
    if plt is None or not config.save_plots:
        return []
    generated: List[str] = []

    x_values = [int(row["original_input"]) for row in analytical_summary_rows]
    analytic = [float(row["analytic_ineffective_probability"]) for row in analytical_summary_rows]
    exact = [float(row["exact_ineffective_probability"]) for row in analytical_summary_rows]
    fig = plt.figure(figsize=(10, 5.5))
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x_values, analytic, marker="o", label="Analytic 2^{-HW(x)}")
    axis.plot(x_values, exact, marker="x", linestyle="--", label="Exact enumeration")
    axis.set_xticks(x_values)
    axis.set_xlabel("Original 4-bit S-box input x")
    axis.set_ylabel("Ineffective probability")
    axis.set_title("4-bit random-AND ineffective-fault bias")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    path = public_directory / "random_and_4_ineffective_bias.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # Parameter-coverage plot uses configured values only and contains no label.
    offsets = np.asarray(
        [float(record.public_row["timing_offset_samples"]) for record in records],
        dtype=np.float64,
    )
    widths = np.asarray(
        [float(record.public_row["width_samples"]) for record in records],
        dtype=np.float64,
    )
    strengths = np.asarray(
        [float(record.public_row["strength"]) for record in records],
        dtype=np.float64,
    )
    fig = plt.figure(figsize=(10.5, 6))
    axis = fig.add_subplot(1, 1, 1)
    scatter = axis.scatter(offsets, widths, c=strengths, s=7, alpha=0.28)
    axis.set_xlabel("Configured offset from target center [samples]")
    axis.set_ylabel("Pulse width [samples]")
    axis.set_title("Public glitch-parameter coverage")
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis, label="Configured strength")
    fig.tight_layout()
    path = public_directory / "glitch_parameter_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    return generated


def save_validation_plots(
    validation_directory: Path,
    records: Sequence[ExperimentRecord],
    absolute_samples: np.ndarray,
    config: Stage06Config,
) -> List[str]:
    """Label-dependent plots stored outside the public attack-facing directory."""
    if plt is None or not config.save_plots:
        return []
    generated: List[str] = []

    categories = [record.private_row["category"] for record in records]
    counts = [categories.count(name) for name in FAULT_CATEGORIES]
    fig = plt.figure(figsize=(11, 5.5))
    axis = fig.add_subplot(1, 1, 1)
    positions = np.arange(len(FAULT_CATEGORIES))
    axis.bar(positions, counts)
    axis.set_xticks(positions)
    axis.set_xticklabels(FAULT_CATEGORIES, rotation=35, ha="right")
    axis.set_ylabel("Experiment count")
    axis.set_title("Stage 06 ground-truth event distribution")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = validation_directory / "fault_event_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    fig = plt.figure(figsize=(12, 6))
    axis = fig.add_subplot(1, 1, 1)
    category_y = {name: index for index, name in enumerate(FAULT_CATEGORIES)}
    offsets = np.asarray(
        [float(record.public_row["timing_offset_samples"]) for record in records],
        dtype=np.float64,
    )
    y_values = np.asarray([category_y[name] for name in categories], dtype=np.float64)
    axis.scatter(offsets, y_values, s=6, alpha=0.18)
    axis.set_yticks(range(len(FAULT_CATEGORIES)))
    axis.set_yticklabels(FAULT_CATEGORIES)
    axis.set_xlabel("Configured offset from target center [samples]")
    axis.set_title("Timing offset versus hidden event class")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = validation_directory / "timing_offset_vs_event_class.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    fig = plt.figure(figsize=(12, 7))
    axis = fig.add_subplot(1, 1, 1)
    vertical_offset = 0.0
    for category in FAULT_CATEGORIES:
        selected = next(record for record in records if record.private_row["category"] == category)
        standardized = robust_standardize_trace(selected.response_trace.astype(np.float64))
        axis.plot(absolute_samples, standardized + vertical_offset, label=category)
        vertical_offset += 5.0
    axis.set_xlabel("Absolute sample")
    axis.set_ylabel("Standardized response + vertical offset")
    axis.set_title("Representative traces selected using private labels")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.15)
    fig.tight_layout()
    path = validation_directory / "representative_fault_response_traces.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    return generated



# ============================================================
# Stage 08 — Large attack-oriented fault campaign
#
# This stage consumes only:
#   - the successful Stage 07 public theoretical prior,
#   - the successful Stage 06/05/04/03 run contracts needed to reconstruct
#     the public timing map and healthy final-round ROI source.
#
# Campaign generation does NOT read Stage 07 oracle parameter cells.
# The large campaign is divided by key into train, validation, test, and
# final-attack partitions. Public data are frozen before private labels and
# key material are written.
# ============================================================


@dataclass(frozen=True)
class Stage08Config:
    input_stage7_run_directory: str
    output_root: str = "runs/stage_08"
    random_seed: int = 20260718

    number_of_experiments: int = 64000
    number_of_keys: int = 24
    number_of_sessions: int = 8

    train_experiments: int = 32000
    validation_experiments: int = 8000
    test_experiments: int = 8000
    attack_experiments: int = 16000

    train_key_ids: Tuple[int, ...] = tuple(range(0, 16))
    validation_key_ids: Tuple[int, ...] = tuple(range(16, 20))
    test_key_ids: Tuple[int, ...] = (20, 21)
    attack_key_ids: Tuple[int, ...] = (22, 23)

    global_timing_jitter_sigma_samples: float = 0.35
    local_sbox_jitter_sigma_samples: float = 0.18
    injection_timing_jitter_sigma_samples: float = 0.20
    session_timing_shift_sigma_samples: float = 0.25

    # Must agree with the Stage 07 public theoretical prior.
    random_and_4_weight: float = 0.80
    random_and_2_weight: float = 0.06
    single_bit_flip_weight: float = 0.05
    stuck_at_bit_weight: float = 0.05
    random_nibble_weight: float = 0.04

    # Exact 100-slot public design. A large majority remains attack-oriented,
    # while explicit controls preserve the six-class ML problem.
    attack_core_fraction: float = 0.45
    attack_explore_fraction: float = 0.25
    boundary_scan_fraction: float = 0.08
    miss_control_fraction: float = 0.07
    neighbor_control_fraction: float = 0.06
    multi_control_fraction: float = 0.05
    invalid_control_fraction: float = 0.04

    response_trace_noise_sigma: float = 0.055
    response_trace_baseline_sigma: float = 0.035
    response_trace_gain_sigma: float = 0.06
    save_response_traces: bool = True
    save_plots: bool = True

    minimum_category_fraction: float = 0.002
    minimum_attack_primary_clean_ineffective_per_target_key: int = 120
    minimum_attack_primary_clean_effective_per_target_key: int = 250
    maximum_primary_ineffective_rate_error: float = 0.055
    deterministic_recheck_count: int = 96
    enable_private_validation: bool = True


@dataclass(frozen=True)
class CampaignPlanEntry:
    experiment_id: int
    campaign_partition: str
    key_id: int
    session_id: int
    target_sbox_index: int
    fault_model: str
    design_regime: str


PARTITION_NAMES = ("train", "validation", "test", "attack")
DESIGN_REGIMES = (
    "attack_core",
    "attack_explore",
    "boundary_scan",
    "miss_control",
    "neighbor_control",
    "multi_control",
    "invalid_control",
)


def stage08_model_weights(config: Stage08Config) -> Dict[str, float]:
    result = {
        "random_and_4": float(config.random_and_4_weight),
        "random_and_2": float(config.random_and_2_weight),
        "single_bit_flip": float(config.single_bit_flip_weight),
        "stuck_at_bit": float(config.stuck_at_bit_weight),
        "random_nibble": float(config.random_nibble_weight),
    }
    if any(value < 0.0 for value in result.values()):
        raise ValueError("Fault-model weights must be nonnegative")
    if not np.isclose(sum(result.values()), 1.0):
        raise ValueError("Fault-model weights must sum to one")
    return result


def stage08_regime_weights(config: Stage08Config) -> Dict[str, float]:
    result = {
        "attack_core": float(config.attack_core_fraction),
        "attack_explore": float(config.attack_explore_fraction),
        "boundary_scan": float(config.boundary_scan_fraction),
        "miss_control": float(config.miss_control_fraction),
        "neighbor_control": float(config.neighbor_control_fraction),
        "multi_control": float(config.multi_control_fraction),
        "invalid_control": float(config.invalid_control_fraction),
    }
    if any(value < 0.0 for value in result.values()):
        raise ValueError("Design-regime fractions must be nonnegative")
    if not np.isclose(sum(result.values()), 1.0):
        raise ValueError("Design-regime fractions must sum to one")
    return result


def counts_from_fractions(
    total: int,
    fractions: Mapping[str, float],
) -> Dict[str, int]:
    """Convert exact fractions to integer counts using largest remainder."""
    names = list(fractions.keys())
    raw = np.asarray(
        [total * float(fractions[name]) for name in names],
        dtype=np.float64,
    )
    base = np.floor(raw).astype(np.int64)
    remainder = int(total - int(np.sum(base)))
    order = np.argsort(-(raw - base))
    for index in order[:remainder]:
        base[index] += 1
    return {
        name: int(base[position])
        for position, name in enumerate(names)
    }


def shuffled_multiset(
    counts: Mapping[str, int],
    rng: np.random.Generator,
) -> List[str]:
    values: List[str] = []
    for name, count in counts.items():
        values.extend([str(name)] * int(count))
    rng.shuffle(values)
    return values


def validate_stage08_config(config: Stage08Config) -> Dict[str, Any]:
    partition_counts = {
        "train": config.train_experiments,
        "validation": config.validation_experiments,
        "test": config.test_experiments,
        "attack": config.attack_experiments,
    }
    partition_keys = {
        "train": config.train_key_ids,
        "validation": config.validation_key_ids,
        "test": config.test_key_ids,
        "attack": config.attack_key_ids,
    }

    all_key_ids = [
        int(value)
        for values in partition_keys.values()
        for value in values
    ]
    errors: List[str] = []

    if sum(partition_counts.values()) != config.number_of_experiments:
        errors.append("Partition experiment counts do not sum to total")
    if len(all_key_ids) != config.number_of_keys:
        errors.append("Partition key IDs do not cover number_of_keys")
    if sorted(all_key_ids) != list(range(config.number_of_keys)):
        errors.append("Key IDs must be a disjoint cover of 0..number_of_keys-1")
    if config.number_of_sessions < 2:
        errors.append("At least two sessions are required")

    for partition, total in partition_counts.items():
        key_count = len(partition_keys[partition])
        denominator = key_count * config.number_of_sessions
        if denominator <= 0 or total % denominator != 0:
            errors.append(
                f"{partition} count must be divisible by keys*sessions"
            )
        elif (total // denominator) % 2 != 0:
            errors.append(
                f"{partition} experiments per key/session must be even "
                "for exact target balance"
            )

    stage08_model_weights(config)
    stage08_regime_weights(config)

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "passed": True,
        "partition_counts": partition_counts,
        "partition_key_ids": {
            key: list(map(int, values))
            for key, values in partition_keys.items()
        },
    }


def load_stage08_contracts(
    stage7_run_directory: Path,
) -> Dict[str, Any]:
    stage7_summary_path = stage7_run_directory / "stage_07_summary.json"
    prior_path = (
        stage7_run_directory
        / "public_theory"
        / "theoretical_attack_prior.json"
    )

    if not stage7_summary_path.is_file():
        raise FileNotFoundError(stage7_summary_path)
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)

    stage7_summary = read_json(stage7_summary_path)
    prior = read_json(prior_path)

    if not stage7_summary.get("all_checks_passed", False):
        raise RuntimeError("Stage 07 did not pass all checks")
    if prior.get("ground_truth_used", True):
        raise RuntimeError("Stage 07 prior is not public-only")
    if prior.get("primary_fault_model") != "random_and_4":
        raise RuntimeError("Unexpected Stage 07 primary model")

    stage6_run_directory = Path(
        stage7_summary["input_stage_06_run_directory"]
    ).expanduser().resolve()
    stage6_summary_path = stage6_run_directory / "stage_06_summary.json"
    if not stage6_summary_path.is_file():
        raise FileNotFoundError(stage6_summary_path)

    stage6_summary = read_json(stage6_summary_path)
    if not stage6_summary.get("all_checks_passed", False):
        raise RuntimeError("Stage 06 did not pass all checks")

    stage5_run_directory = Path(
        stage6_summary["input_stage_05_run_directory"]
    ).expanduser().resolve()
    stage5_summary_path = stage5_run_directory / "stage_05_summary.json"
    target_contract_path = (
        stage5_run_directory
        / "public"
        / "lblock_8bit_target_contract.json"
    )
    if not stage5_summary_path.is_file():
        raise FileNotFoundError(stage5_summary_path)
    if not target_contract_path.is_file():
        raise FileNotFoundError(target_contract_path)

    stage5_summary = read_json(stage5_summary_path)
    target_contract = read_json(target_contract_path)
    if not stage5_summary.get("all_checks_passed", False):
        raise RuntimeError("Stage 05 did not pass all checks")
    if int(target_contract.get("total_target_bits", 0)) != 8:
        raise RuntimeError("Stage 05 does not define an 8-bit target")

    stage4_run_directory = Path(
        stage5_summary["input_stage_04_run_directory"]
    ).expanduser().resolve()
    timing_map_path = (
        stage4_run_directory
        / "public"
        / "lblock_final_round_timing_map.json"
    )
    if not timing_map_path.is_file():
        raise FileNotFoundError(timing_map_path)
    timing_map = read_json(timing_map_path)

    stage3_run_directory = Path(
        stage5_summary["input_stage_03_run_directory"]
    ).expanduser().resolve()
    roi_path = (
        stage3_run_directory
        / "public"
        / "final_round_roi_traces.npz"
    )
    if not roi_path.is_file():
        raise FileNotFoundError(roi_path)

    return {
        "stage7_summary": stage7_summary,
        "stage7_summary_path": stage7_summary_path,
        "prior": prior,
        "prior_path": prior_path,
        "stage6_summary": stage6_summary,
        "stage6_summary_path": stage6_summary_path,
        "stage6_run_directory": stage6_run_directory,
        "stage5_summary": stage5_summary,
        "stage5_summary_path": stage5_summary_path,
        "stage5_run_directory": stage5_run_directory,
        "target_contract": target_contract,
        "target_contract_path": target_contract_path,
        "stage4_run_directory": stage4_run_directory,
        "timing_map": timing_map,
        "timing_map_path": timing_map_path,
        "stage3_run_directory": stage3_run_directory,
        "roi_path": roi_path,
    }


def build_campaign_plan(config: Stage08Config) -> List[CampaignPlanEntry]:
    validate_stage08_config(config)

    partition_counts = {
        "train": config.train_experiments,
        "validation": config.validation_experiments,
        "test": config.test_experiments,
        "attack": config.attack_experiments,
    }
    partition_keys = {
        "train": config.train_key_ids,
        "validation": config.validation_key_ids,
        "test": config.test_key_ids,
        "attack": config.attack_key_ids,
    }

    model_fractions = stage08_model_weights(config)
    regime_fractions = stage08_regime_weights(config)
    selected_targets = (0, 5)

    temporary: List[Tuple[str, int, int, int, str, str]] = []
    seed_root = np.random.SeedSequence([config.random_seed, 8008])

    for partition_index, partition in enumerate(PARTITION_NAMES):
        keys = tuple(int(value) for value in partition_keys[partition])
        total = int(partition_counts[partition])
        per_key_session = total // (len(keys) * config.number_of_sessions)

        for key_id in keys:
            for session_id in range(config.number_of_sessions):
                local_seed = np.random.SeedSequence([
                    config.random_seed,
                    8100 + partition_index,
                    key_id,
                    session_id,
                ])
                rng = np.random.default_rng(local_seed)

                model_values = shuffled_multiset(
                    counts_from_fractions(per_key_session, model_fractions),
                    rng,
                )
                regime_values = shuffled_multiset(
                    counts_from_fractions(per_key_session, regime_fractions),
                    rng,
                )
                target_values = [
                    selected_targets[index % 2]
                    for index in range(per_key_session)
                ]
                rng.shuffle(target_values)

                for local_index in range(per_key_session):
                    temporary.append((
                        partition,
                        key_id,
                        session_id,
                        int(target_values[local_index]),
                        str(model_values[local_index]),
                        str(regime_values[local_index]),
                    ))

    if len(temporary) != config.number_of_experiments:
        raise AssertionError("Campaign plan length mismatch")

    # Shuffle global order so partitions and keys are not stored in large blocks.
    global_rng = np.random.default_rng(seed_root)
    order = global_rng.permutation(len(temporary))

    plan: List[CampaignPlanEntry] = []
    for experiment_id, source_index in enumerate(order):
        (
            partition,
            key_id,
            session_id,
            target_sbox_index,
            fault_model,
            design_regime,
        ) = temporary[int(source_index)]

        plan.append(CampaignPlanEntry(
            experiment_id=int(experiment_id),
            campaign_partition=str(partition),
            key_id=int(key_id),
            session_id=int(session_id),
            target_sbox_index=int(target_sbox_index),
            fault_model=str(fault_model),
            design_regime=str(design_regime),
        ))

    return plan


def target_contract_entry(
    target_contract: Mapping[str, Any],
    target_index: int,
) -> Mapping[str, Any]:
    for entry in target_contract["targets"]:
        if int(entry["sbox_index"]) == int(target_index):
            return entry
    raise KeyError(f"S{target_index} not found in target contract")


def sample_stage08_glitch_parameters(
    entry: CampaignPlanEntry,
    target_contract: Mapping[str, Any],
    centers: np.ndarray,
    rng: np.random.Generator,
) -> GlitchParameters:
    target_index = int(entry.target_sbox_index)
    center = float(centers[target_index])
    target_entry = target_contract_entry(target_contract, target_index)

    exploration_start_offset = (
        float(target_entry["exploration_window_start_sample_inclusive"])
        - center
    )
    exploration_end_offset = (
        float(target_entry["exploration_window_end_sample_exclusive"])
        - center
    )
    safe_start_offset = (
        float(target_entry["public_safe_interval_start_sample_inclusive"])
        - center
    )
    safe_end_offset = (
        float(target_entry["public_safe_interval_end_sample_exclusive"])
        - center
    )

    regime = entry.design_regime

    if regime == "attack_core":
        offset = float(rng.uniform(
            safe_start_offset - 0.35,
            safe_end_offset + 0.35,
        ))
        width = float(rng.uniform(0.85, 4.25))
        strength = float(rng.uniform(0.38, 1.08))
        repeat = int(rng.choice([1, 1, 1, 1, 2]))
        repeat_spacing = float(rng.uniform(1.5, 4.75))

    elif regime == "attack_explore":
        offset = float(rng.uniform(
            exploration_start_offset - 0.75,
            exploration_end_offset + 0.75,
        ))
        width = float(rng.uniform(0.60, 6.50))
        strength = float(rng.uniform(0.22, 1.25))
        repeat = int(rng.choice([1, 1, 1, 2, 2]))
        repeat_spacing = float(rng.uniform(1.25, 6.25))

    elif regime == "boundary_scan":
        boundary_offsets = np.asarray([
            exploration_start_offset,
            safe_start_offset,
            safe_end_offset,
            exploration_end_offset,
        ], dtype=np.float64)
        offset = float(
            rng.choice(boundary_offsets)
            + rng.normal(0.0, 0.85)
        )
        width = float(rng.uniform(0.55, 5.25))
        strength = float(rng.uniform(0.28, 1.15))
        repeat = int(rng.choice([1, 1, 2]))
        repeat_spacing = float(rng.uniform(1.5, 5.5))

    elif regime == "miss_control":
        direction = -1.0 if rng.random() < 0.5 else 1.0
        boundary = (
            exploration_start_offset
            if direction < 0
            else exploration_end_offset
        )
        offset = float(
            boundary
            + direction * rng.uniform(8.0, 24.0)
        )
        width = float(rng.uniform(0.50, 2.25))
        strength = float(rng.uniform(0.08, 0.55))
        repeat = 1
        repeat_spacing = float(rng.uniform(1.0, 3.0))

    elif regime == "neighbor_control":
        neighbor = nearest_neighbor_index(target_index, rng)
        offset = float(
            centers[neighbor] - center
            + rng.normal(0.0, 2.25)
        )
        width = float(rng.uniform(0.70, 4.75))
        strength = float(rng.uniform(0.38, 1.18))
        repeat = int(rng.choice([1, 1, 2]))
        repeat_spacing = float(rng.uniform(1.5, 5.75))

    elif regime == "multi_control":
        neighbor = nearest_neighbor_index(target_index, rng)
        midpoint = 0.5 * float(centers[neighbor] - center)
        offset = float(midpoint + rng.normal(0.0, 1.75))
        width = float(rng.uniform(6.75, 14.5))
        strength = float(rng.uniform(0.52, 1.30))
        repeat = int(rng.choice([1, 2, 2, 3]))
        repeat_spacing = float(rng.uniform(2.75, 9.5))

    elif regime == "invalid_control":
        offset = float(rng.uniform(
            exploration_start_offset - 2.0,
            exploration_end_offset + 2.0,
        ))
        width = float(rng.uniform(10.0, 20.0))
        strength = float(rng.uniform(1.08, 1.95))
        repeat = int(rng.integers(2, 5))
        repeat_spacing = float(rng.uniform(2.0, 9.5))

    else:
        raise ValueError(f"Unknown Stage 08 design regime: {regime}")

    return GlitchParameters(
        target_sbox_index=target_index,
        nominal_target_center_sample=center,
        offset_samples=offset,
        width_samples=width,
        strength=strength,
        repeat=repeat,
        repeat_spacing_samples=repeat_spacing,
        sampling_regime=regime,
        fault_model=entry.fault_model,
    )


def build_stage08_key_pool(config: Stage08Config) -> List[int]:
    rng = np.random.default_rng(config.random_seed + 8001)
    keys: List[int] = []
    while len(keys) < config.number_of_keys:
        candidate = random_80bit_integer(rng)
        if candidate not in keys:
            keys.append(candidate)
    return keys


def run_stage08_experiment(
    entry: CampaignPlanEntry,
    target_contract: Mapping[str, Any],
    timing_map: Mapping[str, Any],
    healthy_source: Mapping[str, np.ndarray],
    key_pool: Sequence[int],
    session_shifts: np.ndarray,
    config: Stage08Config,
) -> ExperimentRecord:
    experiment_id = int(entry.experiment_id)
    seed_sequence = np.random.SeedSequence([
        config.random_seed,
        experiment_id,
        8006,
    ])
    rng = np.random.default_rng(seed_sequence)

    target_index = int(entry.target_sbox_index)
    key_id = int(entry.key_id)
    session_id = int(entry.session_id)

    centers = np.asarray(
        [int(item["center_sample"]) for item in timing_map["sboxes"]],
        dtype=np.float64,
    )
    core_widths = np.asarray(
        [
            int(item["core_window_end_sample_exclusive"])
            - int(item["core_window_start_sample_inclusive"])
            for item in timing_map["sboxes"]
        ],
        dtype=np.float64,
    )

    parameters = sample_stage08_glitch_parameters(
        entry,
        target_contract,
        centers,
        rng,
    )

    global_jitter = float(
        rng.normal(0.0, config.global_timing_jitter_sigma_samples)
    )
    local_jitter = rng.normal(
        0.0,
        config.local_sbox_jitter_sigma_samples,
        size=8,
    )
    actual_centers = (
        centers
        + float(session_shifts[session_id])
        + global_jitter
        + local_jitter
    )
    injection_jitter = float(
        rng.normal(0.0, config.injection_timing_jitter_sigma_samples)
    )
    pulse_center_values = pulse_centers(parameters, injection_jitter)
    hit_scores = compute_hit_scores(
        parameters,
        pulse_center_values,
        actual_centers,
        core_widths,
    )
    probabilities = activation_probabilities(parameters, hit_scores)
    p_invalid = invalid_probability(parameters, hit_scores)
    invalid = bool(rng.random() < p_invalid)

    master_key = int(key_pool[key_id])
    plaintext = random_64bit_integer(rng)
    context = final_round_context(plaintext, master_key)
    healthy_ciphertext = int(context["ciphertext"])
    if healthy_ciphertext != encrypt_block_lblock(plaintext, master_key):
        raise AssertionError("Healthy final-round reconstruction mismatch")

    original_inputs = np.asarray(context["sbox_inputs"], dtype=np.uint8)
    faulted_inputs = original_inputs.copy()
    impacted_mask = np.zeros(8, dtype=np.uint8)
    model_details: Dict[str, Any] = {}

    if not invalid:
        activated = rng.random(8) < probabilities
        impacted_mask[:] = activated.astype(np.uint8)

        for sbox_index in np.where(activated)[0]:
            faulted_value, details = apply_fault_model(
                int(original_inputs[sbox_index]),
                parameters.fault_model,
                rng,
            )
            faulted_inputs[sbox_index] = int(faulted_value)
            model_details[f"S{int(sbox_index)}"] = details

        faulty_ciphertext = final_round_with_faulted_inputs(
            int(context["x31"]),
            int(context["x32"]),
            faulted_inputs,
        )
        response_received = True
        ciphertext_equal = bool(
            faulty_ciphertext == healthy_ciphertext
        )
        ciphertext_hd = hamming_distance(
            faulty_ciphertext,
            healthy_ciphertext,
        )
        invalid_subtype = ""
    else:
        faulty_ciphertext = None
        response_received = False
        ciphertext_equal = False
        ciphertext_hd = -1
        invalid_subtype = (
            "reset" if rng.random() < 0.55 else "timeout"
        )

    category = classify_fault_event(
        target_index,
        impacted_mask,
        invalid,
        ciphertext_equal,
    )

    base_index = int(
        rng.integers(0, healthy_source["traces"].shape[0])
    )
    response_trace = synthesize_response_trace(
        healthy_source["traces"][base_index],
        healthy_source["absolute_samples"],
        pulse_center_values,
        parameters,
        actual_centers,
        original_inputs,
        faulted_inputs,
        impacted_mask,
        invalid,
        rng,
        config,
    )
    features = trace_features(
        response_trace,
        healthy_source["absolute_samples"],
        float(centers[target_index]),
        pulse_center_values,
    )

    impacted_indices = [
        int(value)
        for value in np.where(impacted_mask > 0)[0]
    ]
    target_impacted = bool(impacted_mask[target_index])
    off_target_impacted = any(
        value != target_index
        for value in impacted_indices
    )
    changed_input_count = int(
        np.sum(original_inputs != faulted_inputs)
    )

    public_row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "campaign_partition": entry.campaign_partition,
        "target_sbox": f"S{target_index}",
        "target_sbox_index": target_index,
        "key_id": key_id,
        "session_id": session_id,
        "source_healthy_trace_id": int(
            healthy_source["trace_ids"][base_index]
        ),
        "fault_model": parameters.fault_model,
        "nominal_target_center_sample": (
            parameters.nominal_target_center_sample
        ),
        "timing_offset_samples": parameters.offset_samples,
        "first_pulse_nominal_sample": (
            parameters.nominal_target_center_sample
            + parameters.offset_samples
        ),
        "width_samples": parameters.width_samples,
        "strength": parameters.strength,
        "repeat": parameters.repeat,
        "repeat_spacing_samples": (
            parameters.repeat_spacing_samples
        ),
        "plaintext_hex": hex_fixed(plaintext, 64),
        "healthy_ciphertext_hex": hex_fixed(
            healthy_ciphertext,
            64,
        ),
        "response_received": response_received,
        "faulty_ciphertext_hex": (
            hex_fixed(int(faulty_ciphertext), 64)
            if faulty_ciphertext is not None
            else ""
        ),
        "ciphertext_equal": (
            ciphertext_equal if response_received else ""
        ),
        "ciphertext_hamming_distance": ciphertext_hd,
        **features,
    }

    private_row: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "campaign_partition": entry.campaign_partition,
        "category": category,
        "category_id": CATEGORY_TO_ID[category],
        "design_regime": entry.design_regime,
        "target_sbox": f"S{target_index}",
        "target_sbox_index": target_index,
        "key_id": key_id,
        "session_id": session_id,
        "master_key_hex": hex_fixed(master_key, 80),
        "round_key_32_hex": hex_fixed(
            int(context["round_key_32"]),
            32,
        ),
        "x31_hex": hex_fixed(int(context["x31"]), 32),
        "x32_hex": hex_fixed(int(context["x32"]), 32),
        "target_original_input": int(
            original_inputs[target_index]
        ),
        "target_faulted_input": int(
            faulted_inputs[target_index]
        ),
        "impacted_sboxes": ";".join(
            f"S{value}" for value in impacted_indices
        ),
        "impacted_sbox_count": len(impacted_indices),
        "target_impacted": target_impacted,
        "off_target_impacted": off_target_impacted,
        "changed_sbox_input_count": changed_input_count,
        "fault_effective": bool(
            response_received and not ciphertext_equal
        ),
        "invalid_subtype": invalid_subtype,
        "invalid_probability": p_invalid,
        "global_jitter_samples": global_jitter,
        "injection_jitter_samples": injection_jitter,
        "model_details_json": json.dumps(
            model_details,
            sort_keys=True,
        ),
    }

    private_arrays = {
        "master_key_bytes": np.frombuffer(
            master_key.to_bytes(10, "big"),
            dtype=np.uint8,
        ),
        "round_key_32": np.asarray(
            context["round_key_32"],
            dtype=np.uint32,
        ),
        "x31": np.asarray(context["x31"], dtype=np.uint32),
        "x32": np.asarray(context["x32"], dtype=np.uint32),
        "original_inputs": original_inputs.astype(np.uint8),
        "faulted_inputs": faulted_inputs.astype(np.uint8),
        "hit_scores": hit_scores.astype(np.float32),
        "activation_probabilities": probabilities.astype(
            np.float32
        ),
        "actual_centers": actual_centers.astype(np.float32),
        "pulse_centers": np.pad(
            pulse_center_values.astype(np.float32),
            (0, 4 - pulse_center_values.size),
            constant_values=np.nan,
        )[:4],
        "impacted_mask": impacted_mask.astype(np.uint8),
        "category_id": np.asarray(
            CATEGORY_TO_ID[category],
            dtype=np.int8,
        ),
    }

    return ExperimentRecord(
        public_row=public_row,
        private_row=private_row,
        response_trace=response_trace,
        private_arrays=private_arrays,
    )


def stage08_public_leakage_audit(
    public_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    generic = public_metadata_leakage_audit(public_rows)
    columns = set(public_rows[0].keys())
    extra_forbidden = {
        "design_regime",
        "key_role",
        "true_key_nibble",
        "target_original_input",
        "target_faulted_input",
    }
    present = sorted(columns & extra_forbidden)
    generic["stage08_extra_forbidden_columns_present"] = present
    generic["passed"] = bool(generic["passed"] and not present)
    return generic


def summarize_stage08_campaign(
    records: Sequence[ExperimentRecord],
) -> Dict[str, Any]:
    result = aggregate_campaign(records)
    result["partition_counts"] = {
        partition: sum(
            record.public_row["campaign_partition"] == partition
            for record in records
        )
        for partition in PARTITION_NAMES
    }
    result["partition_category_counts"] = {
        partition: {
            category: sum(
                record.public_row["campaign_partition"] == partition
                and record.private_row["category"] == category
                for record in records
            )
            for category in FAULT_CATEGORIES
        }
        for partition in PARTITION_NAMES
    }
    result["partition_model_counts"] = {
        partition: {
            model: sum(
                record.public_row["campaign_partition"] == partition
                and record.public_row["fault_model"] == model
                for record in records
            )
            for model in FAULT_MODELS
        }
        for partition in PARTITION_NAMES
    }
    result["design_regime_counts"] = {
        regime: sum(
            record.private_row["design_regime"] == regime
            for record in records
        )
        for regime in DESIGN_REGIMES
    }
    return result


def attack_readiness_statistics(
    records: Sequence[ExperimentRecord],
    config: Stage08Config,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    for key_id in config.attack_key_ids:
        for target_index in (0, 5):
            selected = [
                record
                for record in records
                if record.public_row["campaign_partition"] == "attack"
                and int(record.public_row["key_id"]) == int(key_id)
                and int(record.public_row["target_sbox_index"]) == target_index
                and record.public_row["fault_model"] == "random_and_4"
            ]
            category_counts = {
                category: sum(
                    record.private_row["category"] == category
                    for record in selected
                )
                for category in FAULT_CATEGORIES
            }
            clean_count = (
                category_counts["clean_target_ineffective"]
                + category_counts["clean_target_effective"]
            )
            ineffective_rate = (
                category_counts["clean_target_ineffective"] / clean_count
                if clean_count
                else float("nan")
            )
            input_counts = np.bincount(
                np.asarray([
                    int(record.private_row["target_original_input"])
                    for record in selected
                ], dtype=np.int32),
                minlength=16,
            )

            rows.append({
                "campaign_partition": "attack",
                "key_id": int(key_id),
                "target_sbox": f"S{target_index}",
                "target_sbox_index": target_index,
                "primary_model_attempt_count": len(selected),
                "clean_target_count": clean_count,
                "clean_ineffective_count": (
                    category_counts["clean_target_ineffective"]
                ),
                "clean_effective_count": (
                    category_counts["clean_target_effective"]
                ),
                "observed_ineffective_rate_given_clean": (
                    float(ineffective_rate)
                ),
                "minimum_original_input_count": int(
                    np.min(input_counts)
                ),
                "maximum_original_input_count": int(
                    np.max(input_counts)
                ),
                "original_input_imbalance_ratio": float(
                    np.max(input_counts)
                    / max(np.min(input_counts), 1)
                ),
                **{
                    f"input_{value:01x}_count": int(input_counts[value])
                    for value in range(16)
                },
            })

    all_ready = all(
        int(row["clean_ineffective_count"])
        >= config.minimum_attack_primary_clean_ineffective_per_target_key
        and int(row["clean_effective_count"])
        >= config.minimum_attack_primary_clean_effective_per_target_key
        and abs(
            float(row["observed_ineffective_rate_given_clean"])
            - 0.31640625
        ) <= config.maximum_primary_ineffective_rate_error
        for row in rows
    )

    return {
        "all_attack_key_target_groups_ready": bool(all_ready),
        "rows": rows,
    }


def validate_stage08_campaign(
    records: Sequence[ExperimentRecord],
    plan: Sequence[CampaignPlanEntry],
    config: Stage08Config,
    target_contract: Mapping[str, Any],
    prior: Mapping[str, Any],
    deterministic_first: str,
    deterministic_second: str,
) -> Dict[str, Any]:
    config_validation = validate_stage08_config(config)
    public_rows = [record.public_row for record in records]
    leakage = stage08_public_leakage_audit(public_rows)
    aggregate = summarize_stage08_campaign(records)
    attack_readiness = attack_readiness_statistics(records, config)

    expected_partitions = {
        "train": config.train_experiments,
        "validation": config.validation_experiments,
        "test": config.test_experiments,
        "attack": config.attack_experiments,
    }
    observed_partitions = aggregate["partition_counts"]

    category_counts = aggregate["category_counts"]
    minimum_category_count = max(
        1,
        int(config.minimum_category_fraction * len(records)),
    )

    target_counts = {
        target: sum(
            record.public_row["target_sbox"] == target
            for record in records
        )
        for target in ("S0", "S5")
    }

    plan_matches = all(
        int(record.public_row["experiment_id"]) == int(entry.experiment_id)
        and int(record.public_row["key_id"]) == int(entry.key_id)
        and int(record.public_row["session_id"]) == int(entry.session_id)
        and int(record.public_row["target_sbox_index"])
            == int(entry.target_sbox_index)
        and str(record.public_row["fault_model"]) == str(entry.fault_model)
        and str(record.public_row["campaign_partition"])
            == str(entry.campaign_partition)
        for record, entry in zip(records, plan)
    )

    primary_fraction = (
        aggregate["fault_model_counts"]["random_and_4"]
        / len(records)
    )
    prior_fraction = float(
        prior["recommended_stage_08_policy"]["primary_model_fraction"]
    )

    checks = {
        "configuration": config_validation,
        "experiment_count": {
            "passed": len(records) == config.number_of_experiments,
            "observed": len(records),
            "expected": config.number_of_experiments,
        },
        "campaign_plan_matches_records": {
            "passed": bool(plan_matches),
        },
        "partition_counts_exact": {
            "passed": observed_partitions == expected_partitions,
            "observed": observed_partitions,
            "expected": expected_partitions,
        },
        "two_target_balance": {
            "passed": abs(target_counts["S0"] - target_counts["S5"]) <= 1,
            "counts": target_counts,
        },
        "primary_model_matches_public_prior": {
            "passed": abs(primary_fraction - prior_fraction) <= 0.002,
            "observed_fraction": primary_fraction,
            "public_prior_fraction": prior_fraction,
        },
        "all_six_categories_present": {
            "passed": all(
                value >= minimum_category_count
                for value in category_counts.values()
            ),
            "minimum_count": minimum_category_count,
            "counts": category_counts,
        },
        "public_metadata_leakage_audit": leakage,
        "deterministic_generation": {
            "passed": deterministic_first == deterministic_second,
            "first_digest": deterministic_first,
            "second_digest": deterministic_second,
        },
        "eight_bit_contract_preserved": {
            "passed": int(target_contract["total_target_bits"]) == 8,
            "selected_bits": (
                target_contract["selected_last_round_key_bit_indices"]
            ),
        },
        "attack_partition_readiness": {
            "passed": bool(
                attack_readiness[
                    "all_attack_key_target_groups_ready"
                ]
            ),
            "minimum_clean_ineffective": (
                config.minimum_attack_primary_clean_ineffective_per_target_key
            ),
            "minimum_clean_effective": (
                config.minimum_attack_primary_clean_effective_per_target_key
            ),
        },
        "finite_response_traces": {
            "passed": all(
                np.all(np.isfinite(record.response_trace))
                for record in records
            ),
        },
    }

    return {
        "all_checks_passed": all(
            bool(check["passed"])
            for check in checks.values()
        ),
        "checks": checks,
        "attack_readiness": attack_readiness,
        "aggregate": aggregate,
    }


def save_stage08_public_plots(
    public_directory: Path,
    records: Sequence[ExperimentRecord],
    config: Stage08Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []

    offsets = np.asarray([
        float(record.public_row["timing_offset_samples"])
        for record in records
    ])
    widths = np.asarray([
        float(record.public_row["width_samples"])
        for record in records
    ])
    strengths = np.asarray([
        float(record.public_row["strength"])
        for record in records
    ])
    partitions = [
        str(record.public_row["campaign_partition"])
        for record in records
    ]

    fig = plt.figure(figsize=(11, 6))
    axis = fig.add_subplot(1, 1, 1)
    sample_count = min(12000, len(records))
    selected = np.linspace(
        0,
        len(records) - 1,
        sample_count,
        dtype=np.int32,
    )
    scatter = axis.scatter(
        offsets[selected],
        widths[selected],
        c=strengths[selected],
        s=5,
        alpha=0.25,
    )
    axis.set_xlabel("Timing offset from target center [samples]")
    axis.set_ylabel("Glitch width [samples]")
    axis.set_title("Stage 08 public parameter coverage")
    axis.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axis, label="Configured strength")
    fig.tight_layout()
    path = public_directory / "large_campaign_parameter_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    partition_counts = [
        partitions.count(partition)
        for partition in PARTITION_NAMES
    ]
    fig = plt.figure(figsize=(9, 5))
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(PARTITION_NAMES, partition_counts)
    axis.set_ylabel("Experiment count")
    axis.set_title("Public key-disjoint campaign partitions")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = public_directory / "campaign_partition_sizes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    model_counts = [
        sum(
            record.public_row["fault_model"] == model
            for record in records
        )
        for model in FAULT_MODELS
    ]
    fig = plt.figure(figsize=(10.5, 5.5))
    axis = fig.add_subplot(1, 1, 1)
    positions = np.arange(len(FAULT_MODELS))
    axis.bar(positions, model_counts)
    axis.set_xticks(positions)
    axis.set_xticklabels(FAULT_MODELS, rotation=30, ha="right")
    axis.set_ylabel("Experiment count")
    axis.set_title("Public fault-model allocation")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = public_directory / "fault_model_allocation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    return generated


def save_stage08_validation_plots(
    validation_directory: Path,
    records: Sequence[ExperimentRecord],
    config: Stage08Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []

    matrix = np.zeros(
        (len(PARTITION_NAMES), len(FAULT_CATEGORIES)),
        dtype=np.int64,
    )
    for partition_index, partition in enumerate(PARTITION_NAMES):
        for category_index, category in enumerate(FAULT_CATEGORIES):
            matrix[partition_index, category_index] = sum(
                record.public_row["campaign_partition"] == partition
                and record.private_row["category"] == category
                for record in records
            )

    fig = plt.figure(figsize=(12, 5.5))
    axis = fig.add_subplot(1, 1, 1)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_yticks(range(len(PARTITION_NAMES)))
    axis.set_yticklabels(PARTITION_NAMES)
    axis.set_xticks(range(len(FAULT_CATEGORIES)))
    axis.set_xticklabels(
        FAULT_CATEGORIES,
        rotation=35,
        ha="right",
    )
    axis.set_title("Hidden event classes by campaign partition")
    fig.colorbar(image, ax=axis, label="Count")
    fig.tight_layout()
    path = validation_directory / "partition_category_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    attack_records = [
        record
        for record in records
        if record.public_row["campaign_partition"] == "attack"
        and record.public_row["fault_model"] == "random_and_4"
    ]
    labels = [
        (
            int(record.public_row["key_id"]),
            str(record.public_row["target_sbox"]),
        )
        for record in attack_records
    ]
    unique_labels = sorted(set(labels))
    ineffective_counts = []
    effective_counts = []
    display_names = []

    for key_id, target in unique_labels:
        display_names.append(f"KID{key_id}-{target}")
        ineffective_counts.append(sum(
            int(record.public_row["key_id"]) == key_id
            and record.public_row["target_sbox"] == target
            and record.private_row["category"]
                == "clean_target_ineffective"
            for record in attack_records
        ))
        effective_counts.append(sum(
            int(record.public_row["key_id"]) == key_id
            and record.public_row["target_sbox"] == target
            and record.private_row["category"]
                == "clean_target_effective"
            for record in attack_records
        ))

    fig = plt.figure(figsize=(10.5, 5.5))
    axis = fig.add_subplot(1, 1, 1)
    positions = np.arange(len(display_names))
    axis.bar(
        positions - 0.18,
        ineffective_counts,
        width=0.36,
        label="Clean ineffective",
    )
    axis.bar(
        positions + 0.18,
        effective_counts,
        width=0.36,
        label="Clean effective",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(display_names)
    axis.set_ylabel("Ground-truth sample count")
    axis.set_title("Attack-partition primary-model readiness")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = validation_directory / "attack_partition_clean_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    return generated


def run_stage_08(config: Stage08Config) -> Dict[str, Any]:
    start_time = time.perf_counter()
    validate_stage08_config(config)

    stage7_run_directory = Path(
        config.input_stage7_run_directory
    ).expanduser().resolve()
    contracts = load_stage08_contracts(stage7_run_directory)

    prior = contracts["prior"]
    target_contract = contracts["target_contract"]
    timing_map = contracts["timing_map"]

    required_primary_fraction = float(
        prior["recommended_stage_08_policy"]["primary_model_fraction"]
    )
    if not np.isclose(
        required_primary_fraction,
        config.random_and_4_weight,
    ):
        raise ValueError(
            "Config primary-model fraction does not match Stage 07 public prior"
        )

    if not bool(
        prior["recommended_stage_08_policy"]["retain_both_selected_targets"]
    ):
        raise RuntimeError("Stage 07 prior does not retain both targets")

    healthy_source = load_healthy_roi_source(
        contracts["roi_path"]
    )
    plan = build_campaign_plan(config)
    key_pool = build_stage08_key_pool(config)

    session_rng = np.random.default_rng(
        config.random_seed + 8002
    )
    session_shifts = session_rng.normal(
        0.0,
        config.session_timing_shift_sigma_samples,
        size=config.number_of_sessions,
    ).astype(np.float64)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = (
        f"stage08_{timestamp}_seed{config.random_seed}"
    )
    run_directory = (
        Path(config.output_root)
        .expanduser()
        .resolve()
        / run_id
    )
    public_directory = run_directory / "public"
    private_directory = run_directory / "private_ground_truth"
    validation_directory = run_directory / "validation_only"
    public_directory.mkdir(parents=True, exist_ok=False)

    records: List[ExperimentRecord] = []
    progress_step = max(1, config.number_of_experiments // 20)

    for index, entry in enumerate(plan):
        records.append(run_stage08_experiment(
            entry,
            target_contract,
            timing_map,
            healthy_source,
            key_pool,
            session_shifts,
            config,
        ))
        if (
            (index + 1) % progress_step == 0
            or index + 1 == config.number_of_experiments
        ):
            print(
                f"Generated {index + 1:,}/"
                f"{config.number_of_experiments:,} experiments"
            )

    # Deterministic recheck on a fixed prefix.
    recheck_count = min(
        config.deterministic_recheck_count,
        len(plan),
    )
    first_digest = campaign_digest(records, recheck_count)
    rerun_records = [
        run_stage08_experiment(
            plan[index],
            target_contract,
            timing_map,
            healthy_source,
            key_pool,
            session_shifts,
            config,
        )
        for index in range(recheck_count)
    ]
    second_digest = campaign_digest(
        rerun_records,
        recheck_count,
    )

    public_rows = [record.public_row for record in records]
    response_traces = np.stack(
        [record.response_trace for record in records],
        axis=0,
    ).astype(np.float32)
    experiment_ids = np.asarray(
        [record.public_row["experiment_id"] for record in records],
        dtype=np.int32,
    )

    # ------------------------- public outputs -------------------------
    write_csv_rows(
        public_directory / "large_fault_campaign_public.csv",
        public_rows,
    )

    if config.save_response_traces:
        np.savez_compressed(
            public_directory / "large_fault_response_traces.npz",
            traces=response_traces,
            experiment_ids=experiment_ids,
            absolute_sample_indices=healthy_source["absolute_samples"],
            sample_axis_seconds=healthy_source["sample_axis_seconds"],
        )

    partition_manifest = {
        "stage": 8,
        "key_values_included": False,
        "split_unit": "key_id",
        "session_policy": (
            "Every key is represented in all configured sessions; "
            "key partitions remain disjoint."
        ),
        "partitions": {
            "train": {
                "key_ids": list(config.train_key_ids),
                "experiment_count": config.train_experiments,
            },
            "validation": {
                "key_ids": list(config.validation_key_ids),
                "experiment_count": config.validation_experiments,
            },
            "test": {
                "key_ids": list(config.test_key_ids),
                "experiment_count": config.test_experiments,
            },
            "attack": {
                "key_ids": list(config.attack_key_ids),
                "experiment_count": config.attack_experiments,
                "purpose": (
                    "Held-out final SIFA/SEFA/SHFA key-recovery experiments"
                ),
            },
        },
    }
    write_json(
        public_directory / "campaign_partition_manifest.json",
        partition_manifest,
    )

    design_policy = {
        "stage": 8,
        "source_prior": str(contracts["prior_path"]),
        "source_prior_sha256": sha256_file(contracts["prior_path"]),
        "ground_truth_or_oracle_used_for_generation": False,
        "oracle_parameter_cells_opened": False,
        "primary_fault_model": "random_and_4",
        "fault_model_fractions": stage08_model_weights(config),
        "design_regime_fractions": stage08_regime_weights(config),
        "selected_targets": target_contract["selected_sboxes"],
        "selected_key_nibbles": (
            target_contract["selected_last_round_key_nibbles"]
        ),
        "total_target_bits": int(target_contract["total_target_bits"]),
        "policy": (
            "Use public safe/exploration windows for attack-oriented probes; "
            "retain broad exploration and explicit control regimes."
        ),
    }
    write_json(
        public_directory / "campaign_design_policy.json",
        design_policy,
    )

    write_json(
        public_directory / "stage_08_config.json",
        {
            **asdict(config),
            "train_key_ids": list(config.train_key_ids),
            "validation_key_ids": list(config.validation_key_ids),
            "test_key_ids": list(config.test_key_ids),
            "attack_key_ids": list(config.attack_key_ids),
        },
    )

    write_json(
        public_directory / "trace_feature_definitions.json",
        {
            "source": "Stage 06 trace feature definitions",
            "features": [
                "trace_mean",
                "trace_std",
                "trace_peak_to_peak",
                "trace_max_abs",
                "trace_l1_energy",
                "trace_l2_energy",
                "trace_high_frequency_energy",
                "target_window_energy",
                "pulse_window_energy",
                "saturation_fraction",
            ],
            "private_labels_used": False,
        },
    )

    public_plots = save_stage08_public_plots(
        public_directory,
        records,
        config,
    )

    public_access_manifest = {
        "files_opened_before_public_freeze": [
            str(contracts["stage7_summary_path"]),
            str(contracts["prior_path"]),
            str(contracts["stage6_summary_path"]),
            str(contracts["stage5_summary_path"]),
            str(contracts["target_contract_path"]),
            str(contracts["timing_map_path"]),
            str(contracts["roi_path"]),
        ],
        "stage_07_oracle_files_opened": [],
        "forbidden_generation_inputs": {
            "oracle_parameter_cells": False,
            "oracle_campaign_recommendations": False,
            "stage_06_private_ground_truth": False,
            "stage_06_key_manifest": False,
        },
    }
    write_json(
        public_directory / "public_data_access_manifest.json",
        public_access_manifest,
    )

    freeze_files = sorted(
        path for path in public_directory.iterdir()
        if path.is_file()
    )
    freeze_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": (
            "Stage 08 public campaign and partitions were frozen before "
            "private labels, internal states, and key values were written."
        ),
        "files": {
            path.name: sha256_file(path)
            for path in freeze_files
        },
    }
    freeze_payload = json.dumps(
        freeze_manifest,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    freeze_sha256 = hashlib.sha256(freeze_payload).hexdigest()
    freeze_manifest["freeze_sha256"] = freeze_sha256
    write_json(
        run_directory / "public_freeze_manifest.json",
        freeze_manifest,
    )

    # ------------------------- private outputs ------------------------
    private_directory.mkdir(parents=True, exist_ok=True)
    private_rows = [record.private_row for record in records]
    write_csv_rows(
        private_directory / "large_fault_ground_truth.csv",
        private_rows,
    )

    private_array_names = list(records[0].private_arrays.keys())
    private_arrays = {
        name: np.stack(
            [record.private_arrays[name] for record in records],
            axis=0,
        )
        for name in private_array_names
    }
    np.savez_compressed(
        private_directory / "large_fault_ground_truth_arrays.npz",
        experiment_ids=experiment_ids,
        **private_arrays,
    )

    key_manifest = {
        "warning": (
            "Private validation material. Never use key values as ML features."
        ),
        "partition_key_ids": {
            "train": list(config.train_key_ids),
            "validation": list(config.validation_key_ids),
            "test": list(config.test_key_ids),
            "attack": list(config.attack_key_ids),
        },
        "keys": [
            {
                "key_id": key_id,
                "campaign_partition": next(
                    partition
                    for partition, ids in {
                        "train": config.train_key_ids,
                        "validation": config.validation_key_ids,
                        "test": config.test_key_ids,
                        "attack": config.attack_key_ids,
                    }.items()
                    if key_id in ids
                ),
                "master_key_hex": hex_fixed(
                    int(key_pool[key_id]),
                    80,
                ),
                "round_key_32_hex": hex_fixed(
                    key_schedule_lblock(
                        int(key_pool[key_id])
                    )[31],
                    32,
                ),
            }
            for key_id in range(config.number_of_keys)
        ],
    }
    write_json(
        private_directory / "key_manifest.json",
        key_manifest,
    )

    # ------------------------- validation -----------------------------
    validation_directory.mkdir(parents=True, exist_ok=True)
    validation = validate_stage08_campaign(
        records,
        plan,
        config,
        target_contract,
        prior,
        first_digest,
        second_digest,
    )
    write_json(
        validation_directory / "stage_08_campaign_validation.json",
        validation,
    )
    write_json(
        validation_directory / "campaign_aggregate_statistics.json",
        validation["aggregate"],
    )
    write_csv_rows(
        validation_directory / "attack_partition_readiness.csv",
        validation["attack_readiness"]["rows"],
    )
    write_json(
        validation_directory / "private_data_use_manifest.json",
        {
            "private_data_used_only_after_public_freeze": True,
            "public_freeze_sha256": freeze_sha256,
            "uses": [
                "event-class validation",
                "attack-partition clean-sample readiness",
                "input-balance checks",
                "final key-recovery truth for later stages",
            ],
            "private_data_not_permitted_as_ml_features": [
                "master_key_hex",
                "round_key_32_hex",
                "x31_hex",
                "x32_hex",
                "target_original_input",
                "target_faulted_input",
                "impacted_sboxes",
                "category",
            ],
        },
    )

    validation_plots = save_stage08_validation_plots(
        validation_directory,
        records,
        config,
    )

    aggregate = validation["aggregate"]
    attack_rows = validation["attack_readiness"]["rows"]
    elapsed_seconds = time.perf_counter() - start_time

    summary = {
        "stage": 8,
        "run_id": run_id,
        "run_directory": str(run_directory.resolve()),
        "input_stage_07_run_directory": str(stage7_run_directory),
        "input_stage_06_run_directory": str(
            contracts["stage6_run_directory"]
        ),
        "input_stage_05_run_directory": str(
            contracts["stage5_run_directory"]
        ),
        "input_stage_04_run_directory": str(
            contracts["stage4_run_directory"]
        ),
        "input_stage_03_run_directory": str(
            contracts["stage3_run_directory"]
        ),
        "public_directory": str(public_directory.resolve()),
        "private_ground_truth_directory": str(
            private_directory.resolve()
        ),
        "validation_only_directory": str(
            validation_directory.resolve()
        ),
        "all_checks_passed": bool(validation["all_checks_passed"]),
        "public_metadata_leakage_audit_passed": bool(
            validation["checks"][
                "public_metadata_leakage_audit"
            ]["passed"]
        ),
        "deterministic_generation_passed": bool(
            validation["checks"][
                "deterministic_generation"
            ]["passed"]
        ),
        "attack_partition_ready": bool(
            validation["checks"][
                "attack_partition_readiness"
            ]["passed"]
        ),
        "number_of_experiments": int(config.number_of_experiments),
        "number_of_response_trace_samples": int(
            response_traces.shape[1]
        ),
        "number_of_keys": int(config.number_of_keys),
        "number_of_sessions": int(config.number_of_sessions),
        "partition_counts": aggregate["partition_counts"],
        "partition_key_ids": {
            "train": list(config.train_key_ids),
            "validation": list(config.validation_key_ids),
            "test": list(config.test_key_ids),
            "attack": list(config.attack_key_ids),
        },
        "selected_sboxes": target_contract["selected_sboxes"],
        "selected_last_round_key_nibbles": (
            target_contract["selected_last_round_key_nibbles"]
        ),
        "selected_last_round_key_bit_indices": (
            target_contract["selected_last_round_key_bit_indices"]
        ),
        "total_target_bits": int(target_contract["total_target_bits"]),
        "primary_fault_model": "random_and_4",
        "fault_model_counts": aggregate["fault_model_counts"],
        "fault_model_rates": aggregate["fault_model_rates"],
        "category_counts": aggregate["category_counts"],
        "category_rates": aggregate["category_rates"],
        "design_regime_counts": aggregate["design_regime_counts"],
        "primary_random_and_clean_target_count": (
            aggregate["primary_random_and_clean_target_count"]
        ),
        "primary_random_and_clean_target_ineffective_count": (
            aggregate[
                "primary_random_and_clean_target_ineffective_count"
            ]
        ),
        "primary_random_and_clean_target_ineffective_rate": (
            aggregate[
                "primary_random_and_clean_target_ineffective_rate"
            ]
        ),
        "attack_key_target_readiness": attack_rows,
        "public_freeze_sha256": freeze_sha256,
        "stage_07_prior_sha256": sha256_file(contracts["prior_path"]),
        "oracle_files_used_for_generation": False,
        "elapsed_seconds": float(elapsed_seconds),
        "public_files": sorted(
            path.name
            for path in public_directory.iterdir()
            if path.is_file()
        ),
        "private_files": sorted(
            path.name
            for path in private_directory.iterdir()
            if path.is_file()
        ),
        "validation_files": sorted(
            path.name
            for path in validation_directory.iterdir()
            if path.is_file()
        ),
        "generated_plots": {
            "public": public_plots,
            "validation_only": validation_plots,
        },
    }

    write_json(
        run_directory / "stage_08_summary.json",
        summary,
    )

    write_json(
        run_directory / "run_manifest.json",
        {
            "stage": 8,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": {
                **asdict(config),
                "train_key_ids": list(config.train_key_ids),
                "validation_key_ids": list(config.validation_key_ids),
                "test_key_ids": list(config.test_key_ids),
                "attack_key_ids": list(config.attack_key_ids),
            },
            "input_sha256": {
                "stage_07_summary.json": sha256_file(
                    contracts["stage7_summary_path"]
                ),
                "theoretical_attack_prior.json": sha256_file(
                    contracts["prior_path"]
                ),
                "stage_06_summary.json": sha256_file(
                    contracts["stage6_summary_path"]
                ),
                "lblock_8bit_target_contract.json": sha256_file(
                    contracts["target_contract_path"]
                ),
                "lblock_final_round_timing_map.json": sha256_file(
                    contracts["timing_map_path"]
                ),
                "final_round_roi_traces.npz": sha256_file(
                    contracts["roi_path"]
                ),
            },
        },
    )

    print("\n" + "=" * 82)
    print("Stage 08 complete: large attack-oriented LBlock fault campaign")
    print("=" * 82)
    print("Run directory                  :", summary["run_directory"])
    print("All checks passed              :", summary["all_checks_passed"])
    print("Experiments                    :", summary["number_of_experiments"])
    print("Keys / sessions                :", summary["number_of_keys"], "/", summary["number_of_sessions"])
    print("Partitions                     :", summary["partition_counts"])
    print("Selected targets               :", summary["selected_sboxes"])
    print("Primary model fraction         :", summary["fault_model_rates"]["random_and_4"])
    print("Primary clean target count     :", summary["primary_random_and_clean_target_count"])
    print("Primary ineffective count      :", summary["primary_random_and_clean_target_ineffective_count"])
    print("Attack partition ready         :", summary["attack_partition_ready"])
    print("Public freeze SHA-256          :", summary["public_freeze_sha256"])
    print("Elapsed seconds                :", f"{summary['elapsed_seconds']:.3f}")
    print("=" * 82)

    if not summary["all_checks_passed"]:
        raise AssertionError(
            "Stage 08 failed validation. Inspect "
            "validation_only/stage_08_campaign_validation.json"
        )

    return summary


def load_stage_08_config(path: str | Path) -> Stage08Config:
    raw = read_json(Path(path))
    for field in (
        "train_key_ids",
        "validation_key_ids",
        "test_key_ids",
        "attack_key_ids",
    ):
        if field in raw:
            raw[field] = tuple(int(value) for value in raw[field])
    return Stage08Config(**raw)


if __name__ == "__main__":
    default_config = Stage08Config(
        input_stage7_run_directory=(
            './runs/stage_07'
            '/stage07_20260718_180834_112047_seed20260718'
        ),
        output_root=(
            './runs/stage_08'
        ),
    )
    run_stage_08(default_config)

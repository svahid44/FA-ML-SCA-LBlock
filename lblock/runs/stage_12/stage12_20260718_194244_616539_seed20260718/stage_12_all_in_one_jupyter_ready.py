# -*- coding: utf-8 -*-
"""
Stage 12 — Self-contained Closed-loop Fault Campaign for LBlock-64/80

این فایل شامل هر دو بخش زیر است:
1) هسته شبیه‌ساز Stage 08
2) Pipeline کامل Stage 12

روش استفاده:
- کل فایل را در یک سلول Jupyter کپی و اجرا کنید؛
- یا آن را مستقیم با Python اجرا کنید.

هیچ فایل Python جانبی برای Import لازم نیست.
"""

from __future__ import annotations

import sys
import types

# ============================================================
# Embedded Stage-08 engine
# ============================================================

_STAGE08_MODULE_NAME = "stage_08_large_attack_oriented_fault_campaign"
_stage08_module = types.ModuleType(_STAGE08_MODULE_NAME)
_stage08_module.__file__ = "<embedded_stage08_engine>"
sys.modules[_STAGE08_MODULE_NAME] = _stage08_module

_STAGE08_SOURCE = '# ============================================================\n# Stage 06 — Parametric glitch/fault engine for LBlock-64/80\n#\n# This stage consumes the public 8-bit target contract from Stage 05\n# and the public S-box timing map from Stage 04. It creates a controlled,\n# reproducible simulation of timing-dependent clock glitches and final-round\n# faults. The primary paper-faithful model is a 4-bit random-AND fault on the\n# input of a final-round S-box. Four control models are also implemented.\n#\n# Public outputs contain only attacker-observable values and configured glitch\n# parameters. Private ground truth contains keys, internal states, actual hit\n# locations and labels. The public campaign is frozen before private validation.\n# ============================================================\n\nfrom __future__ import annotations\n\nfrom dataclasses import asdict, dataclass\nfrom datetime import datetime\nfrom pathlib import Path\nfrom typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple\nimport csv\nimport hashlib\nimport json\nimport math\nimport platform\nimport sys\nimport time\n\nimport numpy as np\n\ntry:\n    import matplotlib.pyplot as plt\nexcept Exception:\n    plt = None\n\n\n# ============================================================\n# 1. Exact LBlock-64/80 reference core\n# ============================================================\n\nBLOCK_SIZE_BITS = 64\nKEY_SIZE_BITS = 80\nNUM_ROUNDS = 32\nMASK32 = 0xFFFFFFFF\nMASK80 = (1 << 80) - 1\n\n# S0..S7 used in the round function.\nSBOX: List[List[int]] = [\n    [0xE, 0x9, 0xF, 0x0, 0xD, 0x4, 0xA, 0xB,\n     0x1, 0x2, 0x8, 0x3, 0x7, 0x6, 0xC, 0x5],\n    [0x4, 0xB, 0xE, 0x9, 0xF, 0xD, 0x0, 0xA,\n     0x7, 0xC, 0x5, 0x6, 0x2, 0x8, 0x1, 0x3],\n    [0x1, 0xE, 0x7, 0xC, 0xF, 0xD, 0x0, 0x6,\n     0xB, 0x5, 0x9, 0x3, 0x2, 0x4, 0x8, 0xA],\n    [0x7, 0x6, 0x8, 0xB, 0x0, 0xF, 0x3, 0xE,\n     0x9, 0xA, 0xC, 0xD, 0x5, 0x2, 0x4, 0x1],\n    [0xE, 0x5, 0xF, 0x0, 0x7, 0x2, 0xC, 0xD,\n     0x1, 0x8, 0x4, 0x9, 0xB, 0xA, 0x6, 0x3],\n    [0x2, 0xD, 0xB, 0xC, 0xF, 0xE, 0x0, 0x9,\n     0x7, 0xA, 0x6, 0x3, 0x1, 0x8, 0x4, 0x5],\n    [0xB, 0x9, 0x4, 0xE, 0x0, 0xF, 0xA, 0xD,\n     0x6, 0xC, 0x5, 0x7, 0x3, 0x8, 0x1, 0x2],\n    [0xD, 0xA, 0xF, 0x0, 0xE, 0x4, 0x9, 0xB,\n     0x2, 0x1, 0x8, 0x3, 0x7, 0x5, 0xC, 0x6],\n]\n\n# S8 and S9 used by the 80-bit key schedule.\nS8 = [0x8, 0x7, 0xE, 0x5, 0xF, 0xD, 0x0, 0x6,\n      0xB, 0xC, 0x9, 0xA, 0x2, 0x4, 0x1, 0x3]\nS9 = [0xB, 0x5, 0xF, 0x0, 0x7, 0x2, 0x9, 0xD,\n      0x4, 0x8, 0x1, 0xC, 0xE, 0xA, 0x3, 0x6]\n\n# Output nibble i receives the S-box output indexed by this table.\nP_SOURCE_FOR_OUTPUT = [1, 3, 0, 2, 5, 7, 4, 6]\n\nOFFICIAL_TEST_VECTORS = [\n    {\n        "plaintext_hex": "0000000000000000",\n        "key_hex": "00000000000000000000",\n        "ciphertext_hex": "c218185308e75bcd",\n    },\n    {\n        "plaintext_hex": "0123456789abcdef",\n        "key_hex": "0123456789abcdeffedc",\n        "ciphertext_hex": "4b7179d8ebee0c26",\n    },\n]\n\n\ndef rol(value: int, shift: int, width: int) -> int:\n    shift %= width\n    mask = (1 << width) - 1\n    value &= mask\n    return value if shift == 0 else (\n        ((value << shift) | (value >> (width - shift))) & mask\n    )\n\n\ndef ror(value: int, shift: int, width: int) -> int:\n    shift %= width\n    mask = (1 << width) - 1\n    value &= mask\n    return value if shift == 0 else (\n        ((value >> shift) | (value << (width - shift))) & mask\n    )\n\n\ndef get_nibble(word: int, index: int) -> int:\n    if not 0 <= index < 8:\n        raise ValueError("nibble index must be in 0..7")\n    return (word >> (4 * index)) & 0xF\n\n\ndef pack_nibbles(values: Sequence[int]) -> int:\n    if len(values) != 8:\n        raise ValueError("exactly eight nibbles are required")\n    output = 0\n    for index, value in enumerate(values):\n        if not 0 <= int(value) <= 0xF:\n            raise ValueError("nibble values must be in 0..15")\n        output |= int(value) << (4 * index)\n    return output & MASK32\n\n\ndef int_to_bits(value: int, width: int) -> str:\n    return format(value & ((1 << width) - 1), f"0{width}b")\n\n\ndef key_schedule_lblock(master_key: int) -> List[int]:\n    if not 0 <= master_key < (1 << 80):\n        raise ValueError("master_key must be an 80-bit integer")\n\n    key_register = master_key\n    round_keys = [(key_register >> 48) & MASK32]\n\n    for round_index in range(1, NUM_ROUNDS):\n        bits = int_to_bits(key_register, 80)\n        bits = bits[29:] + bits[:29]\n\n        top = S9[int(bits[0:4], 2)]\n        next_nibble = S8[int(bits[4:8], 2)]\n        bits = f"{top:04b}{next_nibble:04b}" + bits[8:]\n\n        counter_bits = f"{round_index:05b}"\n        bit_list = list(bits)\n        for offset, bit in enumerate(counter_bits):\n            position = 29 + offset\n            bit_list[position] = "1" if bit_list[position] != bit else "0"\n\n        key_register = int("".join(bit_list), 2) & MASK80\n        round_keys.append((key_register >> 48) & MASK32)\n\n    return round_keys\n\n\ndef lblock_f_from_inputs(sbox_inputs: Sequence[int]) -> int:\n    outputs = [SBOX[index][int(sbox_inputs[index])] for index in range(8)]\n    permuted = [outputs[source] for source in P_SOURCE_FOR_OUTPUT]\n    return pack_nibbles(permuted)\n\n\ndef lblock_f(x_word: int, round_key: int) -> int:\n    xor_word = (x_word ^ round_key) & MASK32\n    inputs = [get_nibble(xor_word, index) for index in range(8)]\n    return lblock_f_from_inputs(inputs)\n\n\ndef encrypt_block_lblock(plaintext: int, master_key: int) -> int:\n    round_keys = key_schedule_lblock(master_key)\n    x0 = plaintext & MASK32\n    x1 = (plaintext >> 32) & MASK32\n    x_prev2, x_prev1 = x0, x1\n\n    for round_index in range(NUM_ROUNDS):\n        x_new = (\n            lblock_f(x_prev1, round_keys[round_index])\n            ^ rol(x_prev2, 8, 32)\n        ) & MASK32\n        x_prev2, x_prev1 = x_prev1, x_new\n\n    return ((x_prev2 << 32) | x_prev1) & ((1 << 64) - 1)\n\n\ndef decrypt_block_lblock(ciphertext: int, master_key: int) -> int:\n    round_keys = key_schedule_lblock(master_key)\n    x32 = (ciphertext >> 32) & MASK32\n    x33 = ciphertext & MASK32\n    x_next2, x_next1 = x33, x32\n\n    for round_index in range(NUM_ROUNDS - 1, -1, -1):\n        x_previous = ror(\n            x_next2 ^ lblock_f(x_next1, round_keys[round_index]),\n            8,\n            32,\n        )\n        x_next2, x_next1 = x_next1, x_previous\n\n    return ((x_next2 << 32) | x_next1) & ((1 << 64) - 1)\n\n\ndef final_round_context(plaintext: int, master_key: int) -> Dict[str, Any]:\n    """Return X31, X32, K32, final S-box inputs and the healthy ciphertext."""\n    round_keys = key_schedule_lblock(master_key)\n    x0 = plaintext & MASK32\n    x1 = (plaintext >> 32) & MASK32\n    states = [x0, x1]\n\n    for round_index in range(31):\n        x_new = (\n            lblock_f(states[-1], round_keys[round_index])\n            ^ rol(states[-2], 8, 32)\n        ) & MASK32\n        states.append(x_new)\n\n    x31 = states[31]\n    x32 = states[32]\n    round_key_32 = round_keys[31]\n    xor_input = (x32 ^ round_key_32) & MASK32\n    sbox_inputs = [get_nibble(xor_input, index) for index in range(8)]\n    f_output = lblock_f_from_inputs(sbox_inputs)\n    x33 = (f_output ^ rol(x31, 8, 32)) & MASK32\n    ciphertext = ((x32 << 32) | x33) & ((1 << 64) - 1)\n\n    return {\n        "x31": x31,\n        "x32": x32,\n        "round_key_32": round_key_32,\n        "sbox_inputs": sbox_inputs,\n        "f_output": f_output,\n        "x33": x33,\n        "ciphertext": ciphertext,\n    }\n\n\ndef final_round_with_faulted_inputs(\n    x31: int,\n    x32: int,\n    faulted_inputs: Sequence[int],\n) -> int:\n    f_faulted = lblock_f_from_inputs(faulted_inputs)\n    x33_faulted = (f_faulted ^ rol(x31, 8, 32)) & MASK32\n    return ((x32 << 32) | x33_faulted) & ((1 << 64) - 1)\n\n\n# ============================================================\n# 2. Configuration and data classes\n# ============================================================\n\nFAULT_MODELS = (\n    "random_and_4",\n    "random_and_2",\n    "single_bit_flip",\n    "stuck_at_bit",\n    "random_nibble",\n)\n\nFAULT_CATEGORIES = (\n    "missed",\n    "clean_target_ineffective",\n    "clean_target_effective",\n    "off_target",\n    "multi_hit",\n    "invalid_reset",\n)\nCATEGORY_TO_ID = {name: index for index, name in enumerate(FAULT_CATEGORIES)}\n\n\n@dataclass(frozen=True)\nclass Stage06Config:\n    input_stage5_run_directory: str\n    output_root: str = "runs/stage_06"\n    random_seed: int = 20260718\n\n    number_of_experiments: int = 12000\n    number_of_keys: int = 8\n    number_of_sessions: int = 4\n\n    # Target timing variability after Stage 03 alignment.\n    global_timing_jitter_sigma_samples: float = 0.35\n    local_sbox_jitter_sigma_samples: float = 0.18\n    injection_timing_jitter_sigma_samples: float = 0.20\n    session_timing_shift_sigma_samples: float = 0.25\n\n    # Fault-model mixture. The primary 4-bit random-AND model dominates.\n    random_and_4_weight: float = 0.65\n    random_and_2_weight: float = 0.10\n    single_bit_flip_weight: float = 0.10\n    stuck_at_bit_weight: float = 0.08\n    random_nibble_weight: float = 0.07\n\n    # Synthetic response-trace generation.\n    response_trace_noise_sigma: float = 0.055\n    response_trace_baseline_sigma: float = 0.035\n    response_trace_gain_sigma: float = 0.06\n    save_response_traces: bool = True\n    save_plots: bool = True\n\n    # Public validation thresholds.\n    minimum_category_fraction: float = 0.003\n    minimum_primary_model_fraction: float = 0.55\n    maximum_determinism_mismatches: int = 0\n    deterministic_recheck_count: int = 64\n\n    enable_private_validation: bool = True\n\n\n@dataclass(frozen=True)\nclass GlitchParameters:\n    target_sbox_index: int\n    nominal_target_center_sample: float\n    offset_samples: float\n    width_samples: float\n    strength: float\n    repeat: int\n    repeat_spacing_samples: float\n    sampling_regime: str\n    fault_model: str\n\n\n@dataclass\nclass ExperimentRecord:\n    public_row: Dict[str, Any]\n    private_row: Dict[str, Any]\n    response_trace: np.ndarray\n    private_arrays: Dict[str, np.ndarray]\n\n\n# ============================================================\n# 3. File helpers\n# ============================================================\n\n\ndef read_json(path: Path) -> Any:\n    with path.open("r", encoding="utf-8") as file:\n        return json.load(file)\n\n\ndef write_json(path: Path, data: Any) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    with path.open("w", encoding="utf-8") as file:\n        json.dump(data, file, ensure_ascii=False, indent=2)\n\n\ndef write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:\n    if not rows:\n        raise ValueError("CSV rows cannot be empty")\n    path.parent.mkdir(parents=True, exist_ok=True)\n    fieldnames: List[str] = []\n    seen = set()\n    for row in rows:\n        for key in row.keys():\n            if key not in seen:\n                seen.add(key)\n                fieldnames.append(key)\n    with path.open("w", encoding="utf-8", newline="") as file:\n        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")\n        writer.writeheader()\n        writer.writerows(rows)\n\n\ndef sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:\n    digest = hashlib.sha256()\n    with path.open("rb") as file:\n        while True:\n            block = file.read(chunk_size)\n            if not block:\n                break\n            digest.update(block)\n    return digest.hexdigest()\n\n\ndef hex_fixed(value: int, bits: int) -> str:\n    return format(value & ((1 << bits) - 1), f"0{bits // 4}x")\n\n\ndef hamming_distance(value_a: int, value_b: int) -> int:\n    return int(value_a ^ value_b).bit_count()\n\n\ndef sigmoid(value: float) -> float:\n    if value >= 0:\n        exponential = math.exp(-value)\n        return 1.0 / (1.0 + exponential)\n    exponential = math.exp(value)\n    return exponential / (1.0 + exponential)\n\n\ndef stable_json_hash(data: Any) -> str:\n    payload = json.dumps(\n        data,\n        ensure_ascii=False,\n        sort_keys=True,\n        separators=(",", ":"),\n    ).encode("utf-8")\n    return hashlib.sha256(payload).hexdigest()\n\n\n# ============================================================\n# 4. Contract and source-data loading\n# ============================================================\n\n\ndef load_stage_contracts(\n    stage5_run_directory: Path,\n) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Path, Path]:\n    stage5_summary_path = stage5_run_directory / "stage_05_summary.json"\n    target_contract_path = (\n        stage5_run_directory / "public" / "lblock_8bit_target_contract.json"\n    )\n    if not stage5_summary_path.is_file():\n        raise FileNotFoundError(stage5_summary_path)\n    if not target_contract_path.is_file():\n        raise FileNotFoundError(target_contract_path)\n\n    stage5_summary = read_json(stage5_summary_path)\n    if not stage5_summary.get("all_checks_passed", False):\n        raise RuntimeError("Stage 05 did not pass all checks")\n\n    target_contract = read_json(target_contract_path)\n    if int(target_contract.get("total_target_bits", 0)) != 8:\n        raise RuntimeError("Stage 05 target contract does not define eight bits")\n    if len(target_contract.get("selected_sbox_indices", [])) != 2:\n        raise RuntimeError("Stage 05 must select exactly two S-boxes")\n\n    stage4_run_directory = Path(\n        stage5_summary["input_stage_04_run_directory"]\n    ).expanduser().resolve()\n    timing_map_path = (\n        stage4_run_directory / "public" / "lblock_final_round_timing_map.json"\n    )\n    if not timing_map_path.is_file():\n        raise FileNotFoundError(timing_map_path)\n    timing_map = read_json(timing_map_path)\n\n    stage3_run_directory = Path(\n        stage5_summary["input_stage_03_run_directory"]\n    ).expanduser().resolve()\n    roi_path = stage3_run_directory / "public" / "final_round_roi_traces.npz"\n    if not roi_path.is_file():\n        raise FileNotFoundError(roi_path)\n\n    return stage5_summary, target_contract, timing_map, target_contract_path, roi_path\n\n\ndef load_healthy_roi_source(roi_path: Path) -> Dict[str, np.ndarray]:\n    with np.load(roi_path, allow_pickle=False) as data:\n        traces = np.asarray(data["traces"], dtype=np.float32)\n        trace_ids = np.asarray(data["trace_ids"], dtype=np.int32)\n        absolute_samples = np.asarray(\n            data["absolute_sample_indices"], dtype=np.int32\n        )\n        sample_axis_seconds = np.asarray(\n            data["sample_axis_seconds"], dtype=np.float64\n        )\n\n    if traces.ndim != 2 or traces.shape[1] != absolute_samples.size:\n        raise ValueError("Invalid Stage 03 ROI trace shape")\n    if not np.all(np.isfinite(traces)):\n        raise ValueError("Healthy ROI traces contain non-finite values")\n\n    return {\n        "traces": traces,\n        "trace_ids": trace_ids,\n        "absolute_samples": absolute_samples,\n        "sample_axis_seconds": sample_axis_seconds,\n    }\n\n\n# ============================================================\n# 5. Random key and parameter generation\n# ============================================================\n\n\ndef random_80bit_integer(rng: np.random.Generator) -> int:\n    high_16 = int(rng.integers(0, 1 << 16, dtype=np.uint32))\n    low_64 = int(rng.integers(0, 1 << 64, dtype=np.uint64))\n    return ((high_16 << 64) | low_64) & MASK80\n\n\ndef random_64bit_integer(rng: np.random.Generator) -> int:\n    return int(rng.integers(0, 1 << 64, dtype=np.uint64))\n\n\ndef build_key_pool(config: Stage06Config) -> List[int]:\n    rng = np.random.default_rng(config.random_seed + 6001)\n    keys: List[int] = []\n    while len(keys) < config.number_of_keys:\n        candidate = random_80bit_integer(rng)\n        if candidate not in keys:\n            keys.append(candidate)\n    return keys\n\n\ndef model_weights(config: Stage06Config) -> np.ndarray:\n    weights = np.asarray([\n        config.random_and_4_weight,\n        config.random_and_2_weight,\n        config.single_bit_flip_weight,\n        config.stuck_at_bit_weight,\n        config.random_nibble_weight,\n    ], dtype=np.float64)\n    if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):\n        raise ValueError("fault-model weights must be nonnegative and sum to one")\n    return weights\n\n\ndef choose_sampling_regime(experiment_id: int) -> str:\n    # A fixed 20-slot schedule guarantees coverage while preserving randomness\n    # inside each regime: 20% miss, 45% target, 15% neighbour, 10% multi,\n    # 10% invalid-oriented probes.\n    slot = experiment_id % 20\n    if slot < 4:\n        return "miss_probe"\n    if slot < 13:\n        return "target_probe"\n    if slot < 16:\n        return "neighbor_probe"\n    if slot < 18:\n        return "multi_probe"\n    return "invalid_probe"\n\n\ndef nearest_neighbor_index(target_index: int, rng: np.random.Generator) -> int:\n    choices = []\n    if target_index > 0:\n        choices.append(target_index - 1)\n    if target_index < 7:\n        choices.append(target_index + 1)\n    return int(choices[int(rng.integers(0, len(choices)))])\n\n\ndef sample_glitch_parameters(\n    experiment_id: int,\n    target_index: int,\n    centers: np.ndarray,\n    config: Stage06Config,\n    rng: np.random.Generator,\n) -> GlitchParameters:\n    regime = choose_sampling_regime(experiment_id)\n    target_center = float(centers[target_index])\n    fault_model = str(rng.choice(FAULT_MODELS, p=model_weights(config)))\n\n    if regime == "miss_probe":\n        direction = -1.0 if rng.random() < 0.5 else 1.0\n        offset = direction * float(rng.uniform(17.0, 30.0))\n        width = float(rng.uniform(0.55, 2.0))\n        strength = float(rng.uniform(0.08, 0.42))\n        repeat = 1\n        repeat_spacing = float(rng.uniform(1.0, 3.0))\n\n    elif regime == "target_probe":\n        offset = float(rng.uniform(-6.5, 6.5))\n        width = float(rng.uniform(0.75, 4.75))\n        strength = float(rng.uniform(0.30, 1.08))\n        repeat = int(rng.choice([1, 1, 1, 2]))\n        repeat_spacing = float(rng.uniform(1.5, 5.0))\n\n    elif regime == "neighbor_probe":\n        neighbor = nearest_neighbor_index(target_index, rng)\n        neighbor_offset = float(centers[neighbor] - centers[target_index])\n        offset = neighbor_offset + float(rng.normal(0.0, 2.0))\n        width = float(rng.uniform(0.75, 4.5))\n        strength = float(rng.uniform(0.38, 1.12))\n        repeat = int(rng.choice([1, 1, 2]))\n        repeat_spacing = float(rng.uniform(1.5, 5.5))\n\n    elif regime == "multi_probe":\n        neighbor = nearest_neighbor_index(target_index, rng)\n        midpoint_offset = 0.5 * float(centers[neighbor] - centers[target_index])\n        offset = midpoint_offset + float(rng.normal(0.0, 1.5))\n        width = float(rng.uniform(6.5, 13.5))\n        strength = float(rng.uniform(0.52, 1.18))\n        repeat = int(rng.choice([1, 2, 2, 3]))\n        repeat_spacing = float(rng.uniform(3.0, 9.5))\n\n    else:  # invalid_probe\n        offset = float(rng.uniform(-8.0, 8.0))\n        width = float(rng.uniform(9.0, 18.0))\n        strength = float(rng.uniform(1.05, 1.80))\n        repeat = int(rng.integers(2, 5))\n        repeat_spacing = float(rng.uniform(2.0, 9.5))\n\n    return GlitchParameters(\n        target_sbox_index=target_index,\n        nominal_target_center_sample=target_center,\n        offset_samples=offset,\n        width_samples=width,\n        strength=strength,\n        repeat=repeat,\n        repeat_spacing_samples=repeat_spacing,\n        sampling_regime=regime,\n        fault_model=fault_model,\n    )\n\n\n# ============================================================\n# 6. Timing-overlap and event-probability engine\n# ============================================================\n\n\ndef interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:\n    return max(0.0, min(end_a, end_b) - max(start_a, start_b))\n\n\ndef pulse_centers(\n    parameters: GlitchParameters,\n    injection_jitter: float,\n) -> np.ndarray:\n    first_center = (\n        parameters.nominal_target_center_sample\n        + parameters.offset_samples\n        + injection_jitter\n    )\n    return np.asarray([\n        first_center + pulse_index * parameters.repeat_spacing_samples\n        for pulse_index in range(parameters.repeat)\n    ], dtype=np.float64)\n\n\ndef compute_hit_scores(\n    parameters: GlitchParameters,\n    pulse_center_values: np.ndarray,\n    actual_centers: np.ndarray,\n    core_widths: np.ndarray,\n) -> np.ndarray:\n    """Continuous exposure score for each of the eight final-round S-boxes."""\n    scores = np.zeros(8, dtype=np.float64)\n    pulse_width = max(parameters.width_samples, 0.25)\n    half_width = pulse_width / 2.0\n\n    for sbox_index in range(8):\n        center = float(actual_centers[sbox_index])\n        operation_width = max(float(core_widths[sbox_index]), 1.0)\n        operation_start = center - operation_width / 2.0\n        operation_end = center + operation_width / 2.0\n        sigma = max(0.35 * operation_width, 0.9)\n\n        survival = 1.0\n        for center_of_pulse in pulse_center_values:\n            pulse_start = float(center_of_pulse) - half_width\n            pulse_end = float(center_of_pulse) + half_width\n            overlap = interval_overlap(\n                pulse_start,\n                pulse_end,\n                operation_start,\n                operation_end,\n            )\n            overlap_ratio = overlap / pulse_width\n            distance_kernel = math.exp(\n                -0.5 * ((float(center_of_pulse) - center) / sigma) ** 2\n            )\n            exposure = float(np.clip(\n                0.60 * overlap_ratio + 0.40 * distance_kernel,\n                0.0,\n                0.995,\n            ))\n            survival *= 1.0 - exposure\n\n        scores[sbox_index] = 1.0 - survival\n\n    return scores\n\n\ndef activation_probabilities(\n    parameters: GlitchParameters,\n    hit_scores: np.ndarray,\n) -> np.ndarray:\n    probabilities = np.empty(8, dtype=np.float64)\n    width_bonus = 0.18 * max(parameters.width_samples - 2.0, 0.0)\n    repeat_bonus = 0.40 * max(parameters.repeat - 1, 0)\n    strength_bonus = 1.40 * (parameters.strength - 0.50)\n\n    for index, exposure in enumerate(hit_scores):\n        logit = (\n            -5.0\n            + 6.0 * float(exposure)\n            + strength_bonus\n            + width_bonus\n            + repeat_bonus\n        )\n        probabilities[index] = float(np.clip(sigmoid(logit), 0.0005, 0.997))\n\n    return probabilities\n\n\ndef invalid_probability(\n    parameters: GlitchParameters,\n    hit_scores: np.ndarray,\n) -> float:\n    energy = parameters.strength * parameters.width_samples * parameters.repeat\n    exposed_operations = int(np.sum(hit_scores > 0.15))\n    logit = (\n        -8.0\n        + 0.25 * energy\n        + 0.18 * max(parameters.width_samples - 6.0, 0.0)\n        + 0.35 * max(parameters.repeat - 1, 0)\n        + 0.45 * max(exposed_operations - 2, 0)\n    )\n    return float(np.clip(sigmoid(logit), 0.0005, 0.985))\n\n\n# ============================================================\n# 7. Fault semantics\n# ============================================================\n\n\ndef apply_fault_model(\n    original: int,\n    model: str,\n    rng: np.random.Generator,\n) -> Tuple[int, Dict[str, Any]]:\n    if model == "random_and_4":\n        mask = int(rng.integers(0, 16))\n        faulted = original & mask\n        return faulted, {"and_mask": mask}\n\n    if model == "random_and_2":\n        positions = sorted(\n            int(value)\n            for value in rng.choice(4, size=2, replace=False)\n        )\n        random_bits = int(rng.integers(0, 4))\n        mask = 0xF\n        for local_index, bit_position in enumerate(positions):\n            bit_value = (random_bits >> local_index) & 1\n            if bit_value == 0:\n                mask &= ~(1 << bit_position)\n            else:\n                mask |= 1 << bit_position\n        faulted = original & mask\n        return faulted, {\n            "and_mask": mask,\n            "randomized_positions": positions,\n        }\n\n    if model == "single_bit_flip":\n        bit_position = int(rng.integers(0, 4))\n        faulted = original ^ (1 << bit_position)\n        return faulted, {"bit_position": bit_position}\n\n    if model == "stuck_at_bit":\n        bit_position = int(rng.integers(0, 4))\n        stuck_value = int(rng.integers(0, 2))\n        faulted = (\n            (original & ~(1 << bit_position))\n            | (stuck_value << bit_position)\n        )\n        return faulted, {\n            "bit_position": bit_position,\n            "stuck_value": stuck_value,\n        }\n\n    if model == "random_nibble":\n        replacement = int(rng.integers(0, 16))\n        return replacement, {"replacement": replacement}\n\n    raise ValueError(f"Unknown fault model: {model}")\n\n\ndef classify_fault_event(\n    target_index: int,\n    impacted_mask: np.ndarray,\n    invalid: bool,\n    ciphertext_equal: bool,\n) -> str:\n    if invalid:\n        return "invalid_reset"\n\n    impacted_indices = np.where(impacted_mask > 0)[0]\n    impacted_count = int(impacted_indices.size)\n\n    if impacted_count == 0:\n        return "missed"\n    if impacted_count >= 2:\n        return "multi_hit"\n    if int(impacted_indices[0]) != target_index:\n        return "off_target"\n    return "clean_target_ineffective" if ciphertext_equal else "clean_target_effective"\n\n\n# ============================================================\n# 8. Synthetic observable response traces\n# ============================================================\n\n\ndef robust_standardize_trace(trace: np.ndarray) -> np.ndarray:\n    trace = np.asarray(trace, dtype=np.float64)\n    median = float(np.median(trace))\n    mad = float(1.4826 * np.median(np.abs(trace - median)))\n    scale = mad if mad > 1e-10 else float(np.std(trace))\n    if scale <= 1e-10:\n        scale = 1.0\n    return (trace - median) / scale\n\n\ndef add_gaussian_pulse(\n    trace: np.ndarray,\n    axis_samples: np.ndarray,\n    center: float,\n    sigma: float,\n    amplitude: float,\n) -> None:\n    trace += amplitude * np.exp(\n        -0.5 * ((axis_samples.astype(np.float64) - center) / max(sigma, 0.25)) ** 2\n    )\n\n\ndef synthesize_response_trace(\n    base_trace: np.ndarray,\n    axis_samples: np.ndarray,\n    pulse_center_values: np.ndarray,\n    parameters: GlitchParameters,\n    actual_centers: np.ndarray,\n    original_inputs: np.ndarray,\n    faulted_inputs: np.ndarray,\n    impacted_mask: np.ndarray,\n    invalid: bool,\n    rng: np.random.Generator,\n    config: Stage06Config,\n) -> np.ndarray:\n    trace = robust_standardize_trace(base_trace)\n    gain = float(rng.normal(1.0, config.response_trace_gain_sigma))\n    baseline = float(rng.normal(0.0, config.response_trace_baseline_sigma))\n    trace = gain * trace + baseline\n\n    # Clock-glitch signature: a bipolar pulse train visible to the attacker.\n    glitch_sigma = max(0.32 * parameters.width_samples, 0.35)\n    for pulse_center in pulse_center_values:\n        amplitude = 0.55 + 0.95 * parameters.strength\n        add_gaussian_pulse(trace, axis_samples, float(pulse_center), glitch_sigma, amplitude)\n        add_gaussian_pulse(\n            trace,\n            axis_samples,\n            float(pulse_center) + 0.65 * glitch_sigma,\n            0.70 * glitch_sigma,\n            -0.52 * amplitude,\n        )\n\n    # Data-dependent local response for each actually impacted operation.\n    for sbox_index in np.where(impacted_mask > 0)[0]:\n        input_hd = int(original_inputs[sbox_index] ^ faulted_inputs[sbox_index]).bit_count()\n        output_hd = int(\n            SBOX[int(sbox_index)][int(original_inputs[sbox_index])]\n            ^ SBOX[int(sbox_index)][int(faulted_inputs[sbox_index])]\n        ).bit_count()\n        amplitude = 0.12 * input_hd + 0.10 * output_hd\n        if amplitude > 0:\n            add_gaussian_pulse(\n                trace,\n                axis_samples,\n                float(actual_centers[sbox_index]) + 0.55,\n                1.10,\n                amplitude,\n            )\n\n    if invalid:\n        first_pulse = float(np.min(pulse_center_values))\n        drop_start = int(np.searchsorted(axis_samples, first_pulse + 1.5))\n        drop_start = int(np.clip(drop_start, 0, trace.size - 1))\n        if rng.random() < 0.55:\n            # Reset-like decay.\n            decay = np.exp(-np.linspace(0.0, 5.0, trace.size - drop_start))\n            trace[drop_start:] *= decay\n        else:\n            # Timeout/saturation-like plateau.\n            plateau = float(np.sign(rng.normal()) * (3.5 + parameters.strength))\n            trace[drop_start:] = plateau + rng.normal(\n                0.0, 0.08, size=trace.size - drop_start\n            )\n\n    trace += rng.normal(\n        0.0,\n        config.response_trace_noise_sigma,\n        size=trace.size,\n    )\n    return trace.astype(np.float32)\n\n\ndef trace_features(\n    trace: np.ndarray,\n    axis_samples: np.ndarray,\n    target_center: float,\n    pulse_center_values: np.ndarray,\n) -> Dict[str, float]:\n    values = np.asarray(trace, dtype=np.float64)\n    derivative = np.diff(values, prepend=values[0])\n    target_mask = np.abs(axis_samples.astype(np.float64) - target_center) <= 5.0\n    pulse_mask = np.zeros(values.size, dtype=bool)\n    for center in pulse_center_values:\n        pulse_mask |= np.abs(axis_samples.astype(np.float64) - float(center)) <= 4.0\n\n    target_values = values[target_mask] if np.any(target_mask) else values\n    pulse_values = values[pulse_mask] if np.any(pulse_mask) else values\n\n    return {\n        "trace_mean": float(np.mean(values)),\n        "trace_std": float(np.std(values)),\n        "trace_peak_to_peak": float(np.ptp(values)),\n        "trace_max_absolute": float(np.max(np.abs(values))),\n        "trace_high_frequency_energy": float(np.mean(derivative ** 2)),\n        "target_window_energy": float(np.mean(target_values ** 2)),\n        "pulse_window_energy": float(np.mean(pulse_values ** 2)),\n        "saturation_fraction": float(np.mean(np.abs(values) > 4.0)),\n    }\n\n\n# ============================================================\n# 9. One deterministic experiment\n# ============================================================\n\n\ndef run_one_experiment(\n    experiment_id: int,\n    selected_targets: Sequence[int],\n    timing_map: Mapping[str, Any],\n    healthy_source: Mapping[str, np.ndarray],\n    key_pool: Sequence[int],\n    session_shifts: np.ndarray,\n    config: Stage06Config,\n) -> ExperimentRecord:\n    seed_sequence = np.random.SeedSequence([config.random_seed, experiment_id, 6006])\n    rng = np.random.default_rng(seed_sequence)\n\n    target_index = int(selected_targets[experiment_id % len(selected_targets)])\n    key_id = int((experiment_id // len(selected_targets)) % config.number_of_keys)\n    session_id = int(\n        (experiment_id // (len(selected_targets) * config.number_of_keys))\n        % config.number_of_sessions\n    )\n\n    centers = np.asarray(\n        [int(entry["center_sample"]) for entry in timing_map["sboxes"]],\n        dtype=np.float64,\n    )\n    core_widths = np.asarray(\n        [\n            int(entry["core_window_end_sample_exclusive"])\n            - int(entry["core_window_start_sample_inclusive"])\n            for entry in timing_map["sboxes"]\n        ],\n        dtype=np.float64,\n    )\n\n    parameters = sample_glitch_parameters(\n        experiment_id,\n        target_index,\n        centers,\n        config,\n        rng,\n    )\n\n    global_jitter = float(rng.normal(0.0, config.global_timing_jitter_sigma_samples))\n    local_jitter = rng.normal(\n        0.0,\n        config.local_sbox_jitter_sigma_samples,\n        size=8,\n    )\n    actual_centers = (\n        centers\n        + float(session_shifts[session_id])\n        + global_jitter\n        + local_jitter\n    )\n    injection_jitter = float(\n        rng.normal(0.0, config.injection_timing_jitter_sigma_samples)\n    )\n    pulse_center_values = pulse_centers(parameters, injection_jitter)\n    hit_scores = compute_hit_scores(\n        parameters,\n        pulse_center_values,\n        actual_centers,\n        core_widths,\n    )\n    probabilities = activation_probabilities(parameters, hit_scores)\n    p_invalid = invalid_probability(parameters, hit_scores)\n    invalid = bool(rng.random() < p_invalid)\n\n    master_key = int(key_pool[key_id])\n    plaintext = random_64bit_integer(rng)\n    context = final_round_context(plaintext, master_key)\n    healthy_ciphertext = int(context["ciphertext"])\n\n    # Cross-check the final-round context against the full cipher on every row.\n    full_ciphertext = encrypt_block_lblock(plaintext, master_key)\n    if healthy_ciphertext != full_ciphertext:\n        raise AssertionError("final-round reconstruction mismatch")\n\n    original_inputs = np.asarray(context["sbox_inputs"], dtype=np.uint8)\n    faulted_inputs = original_inputs.copy()\n    impacted_mask = np.zeros(8, dtype=np.uint8)\n    model_details: Dict[str, Any] = {}\n\n    if not invalid:\n        activated = rng.random(8) < probabilities\n        impacted_mask[:] = activated.astype(np.uint8)\n        for sbox_index in np.where(activated)[0]:\n            faulted_value, details = apply_fault_model(\n                int(original_inputs[sbox_index]),\n                parameters.fault_model,\n                rng,\n            )\n            faulted_inputs[sbox_index] = int(faulted_value)\n            model_details[f"S{int(sbox_index)}"] = details\n\n        faulty_ciphertext = final_round_with_faulted_inputs(\n            int(context["x31"]),\n            int(context["x32"]),\n            faulted_inputs,\n        )\n        response_received = True\n        ciphertext_equal = bool(faulty_ciphertext == healthy_ciphertext)\n        ciphertext_hd = hamming_distance(faulty_ciphertext, healthy_ciphertext)\n        invalid_subtype = ""\n    else:\n        faulty_ciphertext = None\n        response_received = False\n        ciphertext_equal = False\n        ciphertext_hd = -1\n        invalid_subtype = "reset" if rng.random() < 0.55 else "timeout"\n\n    category = classify_fault_event(\n        target_index,\n        impacted_mask,\n        invalid,\n        ciphertext_equal,\n    )\n\n    base_index = int(rng.integers(0, healthy_source["traces"].shape[0]))\n    base_trace = healthy_source["traces"][base_index]\n    response_trace = synthesize_response_trace(\n        base_trace,\n        healthy_source["absolute_samples"],\n        pulse_center_values,\n        parameters,\n        actual_centers,\n        original_inputs,\n        faulted_inputs,\n        impacted_mask,\n        invalid,\n        rng,\n        config,\n    )\n    features = trace_features(\n        response_trace,\n        healthy_source["absolute_samples"],\n        float(centers[target_index]),\n        pulse_center_values,\n    )\n\n    impacted_indices = [int(value) for value in np.where(impacted_mask > 0)[0]]\n    target_impacted = bool(impacted_mask[target_index])\n    off_target_impacted = any(value != target_index for value in impacted_indices)\n    changed_input_count = int(np.sum(original_inputs != faulted_inputs))\n\n    public_row: Dict[str, Any] = {\n        "experiment_id": experiment_id,\n        "target_sbox": f"S{target_index}",\n        "target_sbox_index": target_index,\n        "key_id": key_id,\n        "session_id": session_id,\n        "source_healthy_trace_id": int(healthy_source["trace_ids"][base_index]),\n        "fault_model": parameters.fault_model,\n        "nominal_target_center_sample": parameters.nominal_target_center_sample,\n        "timing_offset_samples": parameters.offset_samples,\n        "first_pulse_nominal_sample": (\n            parameters.nominal_target_center_sample + parameters.offset_samples\n        ),\n        "width_samples": parameters.width_samples,\n        "strength": parameters.strength,\n        "repeat": parameters.repeat,\n        "repeat_spacing_samples": parameters.repeat_spacing_samples,\n        "plaintext_hex": hex_fixed(plaintext, 64),\n        "healthy_ciphertext_hex": hex_fixed(healthy_ciphertext, 64),\n        "response_received": response_received,\n        "faulty_ciphertext_hex": (\n            hex_fixed(int(faulty_ciphertext), 64) if faulty_ciphertext is not None else ""\n        ),\n        "ciphertext_equal": (ciphertext_equal if response_received else ""),\n        "ciphertext_hamming_distance": ciphertext_hd,\n        **features,\n    }\n\n    private_row: Dict[str, Any] = {\n        "experiment_id": experiment_id,\n        "category": category,\n        "category_id": CATEGORY_TO_ID[category],\n        "sampling_regime": parameters.sampling_regime,\n        "target_sbox": f"S{target_index}",\n        "target_sbox_index": target_index,\n        "key_id": key_id,\n        "session_id": session_id,\n        "master_key_hex": hex_fixed(master_key, 80),\n        "round_key_32_hex": hex_fixed(int(context["round_key_32"]), 32),\n        "x31_hex": hex_fixed(int(context["x31"]), 32),\n        "x32_hex": hex_fixed(int(context["x32"]), 32),\n        "target_original_input": int(original_inputs[target_index]),\n        "target_faulted_input": int(faulted_inputs[target_index]),\n        "impacted_sboxes": ";".join(f"S{value}" for value in impacted_indices),\n        "impacted_sbox_count": len(impacted_indices),\n        "target_impacted": target_impacted,\n        "off_target_impacted": off_target_impacted,\n        "changed_sbox_input_count": changed_input_count,\n        "fault_effective": bool(response_received and not ciphertext_equal),\n        "invalid_subtype": invalid_subtype,\n        "invalid_probability": p_invalid,\n        "global_jitter_samples": global_jitter,\n        "injection_jitter_samples": injection_jitter,\n        "model_details_json": json.dumps(model_details, sort_keys=True),\n    }\n\n    private_arrays = {\n        "master_key_bytes": np.frombuffer(master_key.to_bytes(10, "big"), dtype=np.uint8),\n        "round_key_32": np.asarray(context["round_key_32"], dtype=np.uint32),\n        "x31": np.asarray(context["x31"], dtype=np.uint32),\n        "x32": np.asarray(context["x32"], dtype=np.uint32),\n        "original_inputs": original_inputs.astype(np.uint8),\n        "faulted_inputs": faulted_inputs.astype(np.uint8),\n        "hit_scores": hit_scores.astype(np.float32),\n        "activation_probabilities": probabilities.astype(np.float32),\n        "actual_centers": actual_centers.astype(np.float32),\n        "pulse_centers": np.pad(\n            pulse_center_values.astype(np.float32),\n            (0, 4 - pulse_center_values.size),\n            constant_values=np.nan,\n        )[:4],\n        "impacted_mask": impacted_mask.astype(np.uint8),\n        "category_id": np.asarray(CATEGORY_TO_ID[category], dtype=np.int8),\n    }\n\n    return ExperimentRecord(\n        public_row=public_row,\n        private_row=private_row,\n        response_trace=response_trace,\n        private_arrays=private_arrays,\n    )\n\n\n# ============================================================\n# 10. Analytical random-AND validation\n# ============================================================\n\n\ndef random_and_4_analytical_tables() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:\n    transition_rows: List[Dict[str, Any]] = []\n    summary_rows: List[Dict[str, Any]] = []\n    maximum_formula_error = 0.0\n    subset_property_holds = True\n\n    for original in range(16):\n        ineffective_count = 0\n        counts = np.zeros(16, dtype=np.int32)\n        for mask in range(16):\n            faulted = original & mask\n            counts[faulted] += 1\n            ineffective = faulted == original\n            ineffective_count += int(ineffective)\n            subset_property_holds &= (faulted & ~original) == 0\n            transition_rows.append({\n                "original_input": original,\n                "original_input_hex": f"{original:x}",\n                "mask": mask,\n                "mask_hex": f"{mask:x}",\n                "faulted_input": faulted,\n                "faulted_input_hex": f"{faulted:x}",\n                "ineffective": ineffective,\n                "transition_probability": 1.0 / 16.0,\n            })\n\n        exact_probability = ineffective_count / 16.0\n        analytic_probability = 2.0 ** (-int(original).bit_count())\n        formula_error = abs(exact_probability - analytic_probability)\n        maximum_formula_error = max(maximum_formula_error, formula_error)\n        summary_rows.append({\n            "original_input": original,\n            "original_input_hex": f"{original:x}",\n            "hamming_weight": int(original).bit_count(),\n            "exact_ineffective_probability": exact_probability,\n            "analytic_ineffective_probability": analytic_probability,\n            "absolute_error": formula_error,\n            "reachable_faulted_values": int(np.sum(counts > 0)),\n        })\n\n    validation = {\n        "model": "random_and_4",\n        "definition": "X\' = X AND R, where R is uniform over all 4-bit values",\n        "mask_count": 16,\n        "input_count": 16,\n        "subset_property_holds": bool(subset_property_holds),\n        "maximum_ineffective_probability_formula_error": maximum_formula_error,\n        "formula": "P[X\'=X | X=x] = 2^{-HW(x)}",\n        "all_checks_passed": bool(subset_property_holds and maximum_formula_error < 1e-12),\n    }\n    return transition_rows, summary_rows, validation\n\n\n# ============================================================\n# 11. Validation\n# ============================================================\n\n\ndef reference_core_validation(config: Stage06Config) -> Dict[str, Any]:\n    vector_results = []\n    for vector in OFFICIAL_TEST_VECTORS:\n        plaintext = int(vector["plaintext_hex"], 16)\n        key = int(vector["key_hex"], 16)\n        expected = int(vector["ciphertext_hex"], 16)\n        obtained = encrypt_block_lblock(plaintext, key)\n        recovered = decrypt_block_lblock(obtained, key)\n        vector_results.append({\n            **vector,\n            "obtained_ciphertext_hex": hex_fixed(obtained, 64),\n            "encryption_passed": obtained == expected,\n            "decryption_passed": recovered == plaintext,\n        })\n\n    rng = np.random.default_rng(config.random_seed + 6061)\n    random_roundtrips = 256\n    random_passed = 0\n    final_round_passed = 0\n    for _ in range(random_roundtrips):\n        key = random_80bit_integer(rng)\n        plaintext = random_64bit_integer(rng)\n        ciphertext = encrypt_block_lblock(plaintext, key)\n        random_passed += int(decrypt_block_lblock(ciphertext, key) == plaintext)\n        context = final_round_context(plaintext, key)\n        final_round_passed += int(context["ciphertext"] == ciphertext)\n\n    all_passed = (\n        all(item["encryption_passed"] and item["decryption_passed"] for item in vector_results)\n        and random_passed == random_roundtrips\n        and final_round_passed == random_roundtrips\n    )\n    return {\n        "all_passed": all_passed,\n        "official_vectors": vector_results,\n        "random_roundtrip_tests": random_roundtrips,\n        "random_roundtrip_passed": random_passed,\n        "final_round_reconstruction_passed": final_round_passed,\n    }\n\n\ndef campaign_digest(records: Sequence[ExperimentRecord], count: int) -> str:\n    compact = []\n    for record in records[:count]:\n        compact.append({\n            "public": record.public_row,\n            "private": {\n                key: value\n                for key, value in record.private_row.items()\n                if key not in {"model_details_json"}\n            },\n            "model_details_json": record.private_row["model_details_json"],\n            "trace_sha256": hashlib.sha256(record.response_trace.tobytes()).hexdigest(),\n        })\n    return stable_json_hash(compact)\n\n\ndef public_metadata_leakage_audit(public_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:\n    columns = set(public_rows[0].keys())\n    forbidden_exact = {\n        "master_key_hex",\n        "round_key_32_hex",\n        "x31_hex",\n        "x32_hex",\n        "category",\n        "category_id",\n        "impacted_sboxes",\n        "impacted_sbox_count",\n        "target_impacted",\n        "actual_centers",\n        "hit_scores",\n        "activation_probabilities",\n        "sampling_regime",\n        "model_details_json",\n    }\n    forbidden_present = sorted(columns & forbidden_exact)\n    suspicious_tokens = ("master_key", "round_key", "internal_state", "actual_center")\n    suspicious_columns = sorted(\n        column for column in columns if any(token in column.lower() for token in suspicious_tokens)\n    )\n    passed = not forbidden_present and not suspicious_columns\n    return {\n        "passed": passed,\n        "public_column_count": len(columns),\n        "forbidden_columns_present": forbidden_present,\n        "suspicious_columns_present": suspicious_columns,\n        "allowed_observable_fields_include": [\n            "glitch parameters",\n            "plaintext",\n            "healthy and returned ciphertexts",\n            "ciphertext equality/difference",\n            "response status",\n            "trace-derived features",\n        ],\n    }\n\n\ndef validate_campaign(\n    records: Sequence[ExperimentRecord],\n    target_contract: Mapping[str, Any],\n    analytical_validation: Mapping[str, Any],\n    core_validation: Mapping[str, Any],\n    determinism_digest_first: str,\n    determinism_digest_second: str,\n    config: Stage06Config,\n) -> Dict[str, Any]:\n    categories = [record.private_row["category"] for record in records]\n    models = [record.public_row["fault_model"] for record in records]\n    category_counts = {name: categories.count(name) for name in FAULT_CATEGORIES}\n    model_counts = {name: models.count(name) for name in FAULT_MODELS}\n    total = len(records)\n\n    public_rows = [record.public_row for record in records]\n    leakage_audit = public_metadata_leakage_audit(public_rows)\n\n    consistency_failures = 0\n    locality_failures = 0\n    for record in records:\n        category = record.private_row["category"]\n        impacted_count = int(record.private_row["impacted_sbox_count"])\n        target_impacted = bool(record.private_row["target_impacted"])\n        response_received = bool(record.public_row["response_received"])\n        equal = record.public_row["ciphertext_equal"]\n\n        if category == "missed":\n            ok = response_received and impacted_count == 0 and equal is True\n        elif category == "clean_target_ineffective":\n            ok = response_received and impacted_count == 1 and target_impacted and equal is True\n        elif category == "clean_target_effective":\n            ok = response_received and impacted_count == 1 and target_impacted and equal is False\n        elif category == "off_target":\n            ok = response_received and impacted_count == 1 and not target_impacted\n        elif category == "multi_hit":\n            ok = response_received and impacted_count >= 2\n        else:\n            ok = (not response_received) and impacted_count == 0\n        consistency_failures += int(not ok)\n\n        if category.startswith("clean_target_"):\n            locality_failures += int(not (impacted_count == 1 and target_impacted))\n\n    primary_fraction = model_counts["random_and_4"] / total\n    minimum_count = max(1, int(math.floor(config.minimum_category_fraction * total)))\n    category_coverage_passed = all(value >= minimum_count for value in category_counts.values())\n\n    target_counts: Dict[str, int] = {}\n    for record in records:\n        target = str(record.public_row["target_sbox"])\n        target_counts[target] = target_counts.get(target, 0) + 1\n    target_balance = max(target_counts.values()) - min(target_counts.values())\n\n    checks = {\n        "reference_core": {\n            "passed": bool(core_validation["all_passed"]),\n        },\n        "random_and_4_analytical_semantics": {\n            "passed": bool(analytical_validation["all_checks_passed"]),\n        },\n        "experiment_count": {\n            "passed": total == config.number_of_experiments,\n            "expected": config.number_of_experiments,\n            "observed": total,\n        },\n        "all_six_categories_present": {\n            "passed": category_coverage_passed,\n            "minimum_count_per_category": minimum_count,\n            "counts": category_counts,\n        },\n        "primary_random_and_model_dominates": {\n            "passed": primary_fraction >= config.minimum_primary_model_fraction,\n            "observed_fraction": primary_fraction,\n            "minimum_fraction": config.minimum_primary_model_fraction,\n        },\n        "category_semantics_consistent": {\n            "passed": consistency_failures == 0,\n            "failure_count": consistency_failures,\n        },\n        "clean_target_locality": {\n            "passed": locality_failures == 0,\n            "failure_count": locality_failures,\n        },\n        "public_metadata_leakage_audit": leakage_audit,\n        "two_target_balance": {\n            "passed": len(target_counts) == 2 and target_balance <= 1,\n            "counts": target_counts,\n        },\n        "eight_bit_contract_preserved": {\n            "passed": int(target_contract["total_target_bits"]) == 8,\n            "bit_indices": target_contract["selected_last_round_key_bit_indices"],\n        },\n        "deterministic_generation": {\n            "passed": determinism_digest_first == determinism_digest_second,\n            "first_digest": determinism_digest_first,\n            "second_digest": determinism_digest_second,\n        },\n        "finite_response_traces": {\n            "passed": all(np.all(np.isfinite(record.response_trace)) for record in records),\n        },\n    }\n    all_passed = all(bool(item["passed"]) for item in checks.values())\n    return {\n        "all_public_and_private_checks_passed": all_passed,\n        "checks": checks,\n    }\n\n\n# ============================================================\n# 12. Aggregate reports and plots\n# ============================================================\n\n\ndef aggregate_campaign(records: Sequence[ExperimentRecord]) -> Dict[str, Any]:\n    total = len(records)\n    category_counts = {name: 0 for name in FAULT_CATEGORIES}\n    model_counts = {name: 0 for name in FAULT_MODELS}\n    target_category_counts: Dict[str, Dict[str, int]] = {}\n    model_category_counts: Dict[str, Dict[str, int]] = {\n        model: {category: 0 for category in FAULT_CATEGORIES}\n        for model in FAULT_MODELS\n    }\n\n    impacted_counts = []\n    invalid_probabilities = []\n    for record in records:\n        category = str(record.private_row["category"])\n        model = str(record.public_row["fault_model"])\n        target = str(record.public_row["target_sbox"])\n        category_counts[category] += 1\n        model_counts[model] += 1\n        model_category_counts[model][category] += 1\n        target_category_counts.setdefault(\n            target, {name: 0 for name in FAULT_CATEGORIES}\n        )[category] += 1\n        impacted_counts.append(int(record.private_row["impacted_sbox_count"]))\n        invalid_probabilities.append(float(record.private_row["invalid_probability"]))\n\n    category_rates = {name: value / total for name, value in category_counts.items()}\n    model_rates = {name: value / total for name, value in model_counts.items()}\n\n    primary_clean = [\n        record for record in records\n        if record.public_row["fault_model"] == "random_and_4"\n        and record.private_row["category"] in {\n            "clean_target_ineffective", "clean_target_effective"\n        }\n    ]\n    ineffective_primary = sum(\n        record.private_row["category"] == "clean_target_ineffective"\n        for record in primary_clean\n    )\n\n    return {\n        "number_of_experiments": total,\n        "category_counts": category_counts,\n        "category_rates": category_rates,\n        "fault_model_counts": model_counts,\n        "fault_model_rates": model_rates,\n        "target_category_counts": target_category_counts,\n        "model_category_counts": model_category_counts,\n        "mean_impacted_sbox_count": float(np.mean(impacted_counts)),\n        "maximum_impacted_sbox_count": int(np.max(impacted_counts)),\n        "mean_invalid_probability": float(np.mean(invalid_probabilities)),\n        "primary_random_and_clean_target_count": len(primary_clean),\n        "primary_random_and_clean_target_ineffective_count": int(ineffective_primary),\n        "primary_random_and_clean_target_ineffective_rate": (\n            float(ineffective_primary / len(primary_clean)) if primary_clean else None\n        ),\n    }\n\n\ndef save_public_plots(\n    public_directory: Path,\n    records: Sequence[ExperimentRecord],\n    analytical_summary_rows: Sequence[Mapping[str, Any]],\n    config: Stage06Config,\n) -> List[str]:\n    """Plots derived only from public parameters or analytical model semantics."""\n    if plt is None or not config.save_plots:\n        return []\n    generated: List[str] = []\n\n    x_values = [int(row["original_input"]) for row in analytical_summary_rows]\n    analytic = [float(row["analytic_ineffective_probability"]) for row in analytical_summary_rows]\n    exact = [float(row["exact_ineffective_probability"]) for row in analytical_summary_rows]\n    fig = plt.figure(figsize=(10, 5.5))\n    axis = fig.add_subplot(1, 1, 1)\n    axis.plot(x_values, analytic, marker="o", label="Analytic 2^{-HW(x)}")\n    axis.plot(x_values, exact, marker="x", linestyle="--", label="Exact enumeration")\n    axis.set_xticks(x_values)\n    axis.set_xlabel("Original 4-bit S-box input x")\n    axis.set_ylabel("Ineffective probability")\n    axis.set_title("4-bit random-AND ineffective-fault bias")\n    axis.grid(alpha=0.2)\n    axis.legend()\n    fig.tight_layout()\n    path = public_directory / "random_and_4_ineffective_bias.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    # Parameter-coverage plot uses configured values only and contains no label.\n    offsets = np.asarray(\n        [float(record.public_row["timing_offset_samples"]) for record in records],\n        dtype=np.float64,\n    )\n    widths = np.asarray(\n        [float(record.public_row["width_samples"]) for record in records],\n        dtype=np.float64,\n    )\n    strengths = np.asarray(\n        [float(record.public_row["strength"]) for record in records],\n        dtype=np.float64,\n    )\n    fig = plt.figure(figsize=(10.5, 6))\n    axis = fig.add_subplot(1, 1, 1)\n    scatter = axis.scatter(offsets, widths, c=strengths, s=7, alpha=0.28)\n    axis.set_xlabel("Configured offset from target center [samples]")\n    axis.set_ylabel("Pulse width [samples]")\n    axis.set_title("Public glitch-parameter coverage")\n    axis.grid(alpha=0.2)\n    fig.colorbar(scatter, ax=axis, label="Configured strength")\n    fig.tight_layout()\n    path = public_directory / "glitch_parameter_coverage.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    return generated\n\n\ndef save_validation_plots(\n    validation_directory: Path,\n    records: Sequence[ExperimentRecord],\n    absolute_samples: np.ndarray,\n    config: Stage06Config,\n) -> List[str]:\n    """Label-dependent plots stored outside the public attack-facing directory."""\n    if plt is None or not config.save_plots:\n        return []\n    generated: List[str] = []\n\n    categories = [record.private_row["category"] for record in records]\n    counts = [categories.count(name) for name in FAULT_CATEGORIES]\n    fig = plt.figure(figsize=(11, 5.5))\n    axis = fig.add_subplot(1, 1, 1)\n    positions = np.arange(len(FAULT_CATEGORIES))\n    axis.bar(positions, counts)\n    axis.set_xticks(positions)\n    axis.set_xticklabels(FAULT_CATEGORIES, rotation=35, ha="right")\n    axis.set_ylabel("Experiment count")\n    axis.set_title("Stage 06 ground-truth event distribution")\n    axis.grid(axis="y", alpha=0.2)\n    fig.tight_layout()\n    path = validation_directory / "fault_event_distribution.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    fig = plt.figure(figsize=(12, 6))\n    axis = fig.add_subplot(1, 1, 1)\n    category_y = {name: index for index, name in enumerate(FAULT_CATEGORIES)}\n    offsets = np.asarray(\n        [float(record.public_row["timing_offset_samples"]) for record in records],\n        dtype=np.float64,\n    )\n    y_values = np.asarray([category_y[name] for name in categories], dtype=np.float64)\n    axis.scatter(offsets, y_values, s=6, alpha=0.18)\n    axis.set_yticks(range(len(FAULT_CATEGORIES)))\n    axis.set_yticklabels(FAULT_CATEGORIES)\n    axis.set_xlabel("Configured offset from target center [samples]")\n    axis.set_title("Timing offset versus hidden event class")\n    axis.grid(alpha=0.2)\n    fig.tight_layout()\n    path = validation_directory / "timing_offset_vs_event_class.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    fig = plt.figure(figsize=(12, 7))\n    axis = fig.add_subplot(1, 1, 1)\n    vertical_offset = 0.0\n    for category in FAULT_CATEGORIES:\n        selected = next(record for record in records if record.private_row["category"] == category)\n        standardized = robust_standardize_trace(selected.response_trace.astype(np.float64))\n        axis.plot(absolute_samples, standardized + vertical_offset, label=category)\n        vertical_offset += 5.0\n    axis.set_xlabel("Absolute sample")\n    axis.set_ylabel("Standardized response + vertical offset")\n    axis.set_title("Representative traces selected using private labels")\n    axis.legend(loc="upper right", fontsize=8)\n    axis.grid(alpha=0.15)\n    fig.tight_layout()\n    path = validation_directory / "representative_fault_response_traces.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    return generated\n\n\n\n# ============================================================\n# Stage 08 — Large attack-oriented fault campaign\n#\n# This stage consumes only:\n#   - the successful Stage 07 public theoretical prior,\n#   - the successful Stage 06/05/04/03 run contracts needed to reconstruct\n#     the public timing map and healthy final-round ROI source.\n#\n# Campaign generation does NOT read Stage 07 oracle parameter cells.\n# The large campaign is divided by key into train, validation, test, and\n# final-attack partitions. Public data are frozen before private labels and\n# key material are written.\n# ============================================================\n\n\n@dataclass(frozen=True)\nclass Stage08Config:\n    input_stage7_run_directory: str\n    output_root: str = "runs/stage_08"\n    random_seed: int = 20260718\n\n    number_of_experiments: int = 64000\n    number_of_keys: int = 24\n    number_of_sessions: int = 8\n\n    train_experiments: int = 32000\n    validation_experiments: int = 8000\n    test_experiments: int = 8000\n    attack_experiments: int = 16000\n\n    train_key_ids: Tuple[int, ...] = tuple(range(0, 16))\n    validation_key_ids: Tuple[int, ...] = tuple(range(16, 20))\n    test_key_ids: Tuple[int, ...] = (20, 21)\n    attack_key_ids: Tuple[int, ...] = (22, 23)\n\n    global_timing_jitter_sigma_samples: float = 0.35\n    local_sbox_jitter_sigma_samples: float = 0.18\n    injection_timing_jitter_sigma_samples: float = 0.20\n    session_timing_shift_sigma_samples: float = 0.25\n\n    # Must agree with the Stage 07 public theoretical prior.\n    random_and_4_weight: float = 0.80\n    random_and_2_weight: float = 0.06\n    single_bit_flip_weight: float = 0.05\n    stuck_at_bit_weight: float = 0.05\n    random_nibble_weight: float = 0.04\n\n    # Exact 100-slot public design. A large majority remains attack-oriented,\n    # while explicit controls preserve the six-class ML problem.\n    attack_core_fraction: float = 0.45\n    attack_explore_fraction: float = 0.25\n    boundary_scan_fraction: float = 0.08\n    miss_control_fraction: float = 0.07\n    neighbor_control_fraction: float = 0.06\n    multi_control_fraction: float = 0.05\n    invalid_control_fraction: float = 0.04\n\n    response_trace_noise_sigma: float = 0.055\n    response_trace_baseline_sigma: float = 0.035\n    response_trace_gain_sigma: float = 0.06\n    save_response_traces: bool = True\n    save_plots: bool = True\n\n    minimum_category_fraction: float = 0.002\n    minimum_attack_primary_clean_ineffective_per_target_key: int = 120\n    minimum_attack_primary_clean_effective_per_target_key: int = 250\n    maximum_primary_ineffective_rate_error: float = 0.055\n    deterministic_recheck_count: int = 96\n    enable_private_validation: bool = True\n\n\n@dataclass(frozen=True)\nclass CampaignPlanEntry:\n    experiment_id: int\n    campaign_partition: str\n    key_id: int\n    session_id: int\n    target_sbox_index: int\n    fault_model: str\n    design_regime: str\n\n\nPARTITION_NAMES = ("train", "validation", "test", "attack")\nDESIGN_REGIMES = (\n    "attack_core",\n    "attack_explore",\n    "boundary_scan",\n    "miss_control",\n    "neighbor_control",\n    "multi_control",\n    "invalid_control",\n)\n\n\ndef stage08_model_weights(config: Stage08Config) -> Dict[str, float]:\n    result = {\n        "random_and_4": float(config.random_and_4_weight),\n        "random_and_2": float(config.random_and_2_weight),\n        "single_bit_flip": float(config.single_bit_flip_weight),\n        "stuck_at_bit": float(config.stuck_at_bit_weight),\n        "random_nibble": float(config.random_nibble_weight),\n    }\n    if any(value < 0.0 for value in result.values()):\n        raise ValueError("Fault-model weights must be nonnegative")\n    if not np.isclose(sum(result.values()), 1.0):\n        raise ValueError("Fault-model weights must sum to one")\n    return result\n\n\ndef stage08_regime_weights(config: Stage08Config) -> Dict[str, float]:\n    result = {\n        "attack_core": float(config.attack_core_fraction),\n        "attack_explore": float(config.attack_explore_fraction),\n        "boundary_scan": float(config.boundary_scan_fraction),\n        "miss_control": float(config.miss_control_fraction),\n        "neighbor_control": float(config.neighbor_control_fraction),\n        "multi_control": float(config.multi_control_fraction),\n        "invalid_control": float(config.invalid_control_fraction),\n    }\n    if any(value < 0.0 for value in result.values()):\n        raise ValueError("Design-regime fractions must be nonnegative")\n    if not np.isclose(sum(result.values()), 1.0):\n        raise ValueError("Design-regime fractions must sum to one")\n    return result\n\n\ndef counts_from_fractions(\n    total: int,\n    fractions: Mapping[str, float],\n) -> Dict[str, int]:\n    """Convert exact fractions to integer counts using largest remainder."""\n    names = list(fractions.keys())\n    raw = np.asarray(\n        [total * float(fractions[name]) for name in names],\n        dtype=np.float64,\n    )\n    base = np.floor(raw).astype(np.int64)\n    remainder = int(total - int(np.sum(base)))\n    order = np.argsort(-(raw - base))\n    for index in order[:remainder]:\n        base[index] += 1\n    return {\n        name: int(base[position])\n        for position, name in enumerate(names)\n    }\n\n\ndef shuffled_multiset(\n    counts: Mapping[str, int],\n    rng: np.random.Generator,\n) -> List[str]:\n    values: List[str] = []\n    for name, count in counts.items():\n        values.extend([str(name)] * int(count))\n    rng.shuffle(values)\n    return values\n\n\ndef validate_stage08_config(config: Stage08Config) -> Dict[str, Any]:\n    partition_counts = {\n        "train": config.train_experiments,\n        "validation": config.validation_experiments,\n        "test": config.test_experiments,\n        "attack": config.attack_experiments,\n    }\n    partition_keys = {\n        "train": config.train_key_ids,\n        "validation": config.validation_key_ids,\n        "test": config.test_key_ids,\n        "attack": config.attack_key_ids,\n    }\n\n    all_key_ids = [\n        int(value)\n        for values in partition_keys.values()\n        for value in values\n    ]\n    errors: List[str] = []\n\n    if sum(partition_counts.values()) != config.number_of_experiments:\n        errors.append("Partition experiment counts do not sum to total")\n    if len(all_key_ids) != config.number_of_keys:\n        errors.append("Partition key IDs do not cover number_of_keys")\n    if sorted(all_key_ids) != list(range(config.number_of_keys)):\n        errors.append("Key IDs must be a disjoint cover of 0..number_of_keys-1")\n    if config.number_of_sessions < 2:\n        errors.append("At least two sessions are required")\n\n    for partition, total in partition_counts.items():\n        key_count = len(partition_keys[partition])\n        denominator = key_count * config.number_of_sessions\n        if denominator <= 0 or total % denominator != 0:\n            errors.append(\n                f"{partition} count must be divisible by keys*sessions"\n            )\n        elif (total // denominator) % 2 != 0:\n            errors.append(\n                f"{partition} experiments per key/session must be even "\n                "for exact target balance"\n            )\n\n    stage08_model_weights(config)\n    stage08_regime_weights(config)\n\n    if errors:\n        raise ValueError("; ".join(errors))\n\n    return {\n        "passed": True,\n        "partition_counts": partition_counts,\n        "partition_key_ids": {\n            key: list(map(int, values))\n            for key, values in partition_keys.items()\n        },\n    }\n\n\ndef load_stage08_contracts(\n    stage7_run_directory: Path,\n) -> Dict[str, Any]:\n    stage7_summary_path = stage7_run_directory / "stage_07_summary.json"\n    prior_path = (\n        stage7_run_directory\n        / "public_theory"\n        / "theoretical_attack_prior.json"\n    )\n\n    if not stage7_summary_path.is_file():\n        raise FileNotFoundError(stage7_summary_path)\n    if not prior_path.is_file():\n        raise FileNotFoundError(prior_path)\n\n    stage7_summary = read_json(stage7_summary_path)\n    prior = read_json(prior_path)\n\n    if not stage7_summary.get("all_checks_passed", False):\n        raise RuntimeError("Stage 07 did not pass all checks")\n    if prior.get("ground_truth_used", True):\n        raise RuntimeError("Stage 07 prior is not public-only")\n    if prior.get("primary_fault_model") != "random_and_4":\n        raise RuntimeError("Unexpected Stage 07 primary model")\n\n    stage6_run_directory = Path(\n        stage7_summary["input_stage_06_run_directory"]\n    ).expanduser().resolve()\n    stage6_summary_path = stage6_run_directory / "stage_06_summary.json"\n    if not stage6_summary_path.is_file():\n        raise FileNotFoundError(stage6_summary_path)\n\n    stage6_summary = read_json(stage6_summary_path)\n    if not stage6_summary.get("all_checks_passed", False):\n        raise RuntimeError("Stage 06 did not pass all checks")\n\n    stage5_run_directory = Path(\n        stage6_summary["input_stage_05_run_directory"]\n    ).expanduser().resolve()\n    stage5_summary_path = stage5_run_directory / "stage_05_summary.json"\n    target_contract_path = (\n        stage5_run_directory\n        / "public"\n        / "lblock_8bit_target_contract.json"\n    )\n    if not stage5_summary_path.is_file():\n        raise FileNotFoundError(stage5_summary_path)\n    if not target_contract_path.is_file():\n        raise FileNotFoundError(target_contract_path)\n\n    stage5_summary = read_json(stage5_summary_path)\n    target_contract = read_json(target_contract_path)\n    if not stage5_summary.get("all_checks_passed", False):\n        raise RuntimeError("Stage 05 did not pass all checks")\n    if int(target_contract.get("total_target_bits", 0)) != 8:\n        raise RuntimeError("Stage 05 does not define an 8-bit target")\n\n    stage4_run_directory = Path(\n        stage5_summary["input_stage_04_run_directory"]\n    ).expanduser().resolve()\n    timing_map_path = (\n        stage4_run_directory\n        / "public"\n        / "lblock_final_round_timing_map.json"\n    )\n    if not timing_map_path.is_file():\n        raise FileNotFoundError(timing_map_path)\n    timing_map = read_json(timing_map_path)\n\n    stage3_run_directory = Path(\n        stage5_summary["input_stage_03_run_directory"]\n    ).expanduser().resolve()\n    roi_path = (\n        stage3_run_directory\n        / "public"\n        / "final_round_roi_traces.npz"\n    )\n    if not roi_path.is_file():\n        raise FileNotFoundError(roi_path)\n\n    return {\n        "stage7_summary": stage7_summary,\n        "stage7_summary_path": stage7_summary_path,\n        "prior": prior,\n        "prior_path": prior_path,\n        "stage6_summary": stage6_summary,\n        "stage6_summary_path": stage6_summary_path,\n        "stage6_run_directory": stage6_run_directory,\n        "stage5_summary": stage5_summary,\n        "stage5_summary_path": stage5_summary_path,\n        "stage5_run_directory": stage5_run_directory,\n        "target_contract": target_contract,\n        "target_contract_path": target_contract_path,\n        "stage4_run_directory": stage4_run_directory,\n        "timing_map": timing_map,\n        "timing_map_path": timing_map_path,\n        "stage3_run_directory": stage3_run_directory,\n        "roi_path": roi_path,\n    }\n\n\ndef build_campaign_plan(config: Stage08Config) -> List[CampaignPlanEntry]:\n    validate_stage08_config(config)\n\n    partition_counts = {\n        "train": config.train_experiments,\n        "validation": config.validation_experiments,\n        "test": config.test_experiments,\n        "attack": config.attack_experiments,\n    }\n    partition_keys = {\n        "train": config.train_key_ids,\n        "validation": config.validation_key_ids,\n        "test": config.test_key_ids,\n        "attack": config.attack_key_ids,\n    }\n\n    model_fractions = stage08_model_weights(config)\n    regime_fractions = stage08_regime_weights(config)\n    selected_targets = (0, 5)\n\n    temporary: List[Tuple[str, int, int, int, str, str]] = []\n    seed_root = np.random.SeedSequence([config.random_seed, 8008])\n\n    for partition_index, partition in enumerate(PARTITION_NAMES):\n        keys = tuple(int(value) for value in partition_keys[partition])\n        total = int(partition_counts[partition])\n        per_key_session = total // (len(keys) * config.number_of_sessions)\n\n        for key_id in keys:\n            for session_id in range(config.number_of_sessions):\n                local_seed = np.random.SeedSequence([\n                    config.random_seed,\n                    8100 + partition_index,\n                    key_id,\n                    session_id,\n                ])\n                rng = np.random.default_rng(local_seed)\n\n                model_values = shuffled_multiset(\n                    counts_from_fractions(per_key_session, model_fractions),\n                    rng,\n                )\n                regime_values = shuffled_multiset(\n                    counts_from_fractions(per_key_session, regime_fractions),\n                    rng,\n                )\n                target_values = [\n                    selected_targets[index % 2]\n                    for index in range(per_key_session)\n                ]\n                rng.shuffle(target_values)\n\n                for local_index in range(per_key_session):\n                    temporary.append((\n                        partition,\n                        key_id,\n                        session_id,\n                        int(target_values[local_index]),\n                        str(model_values[local_index]),\n                        str(regime_values[local_index]),\n                    ))\n\n    if len(temporary) != config.number_of_experiments:\n        raise AssertionError("Campaign plan length mismatch")\n\n    # Shuffle global order so partitions and keys are not stored in large blocks.\n    global_rng = np.random.default_rng(seed_root)\n    order = global_rng.permutation(len(temporary))\n\n    plan: List[CampaignPlanEntry] = []\n    for experiment_id, source_index in enumerate(order):\n        (\n            partition,\n            key_id,\n            session_id,\n            target_sbox_index,\n            fault_model,\n            design_regime,\n        ) = temporary[int(source_index)]\n\n        plan.append(CampaignPlanEntry(\n            experiment_id=int(experiment_id),\n            campaign_partition=str(partition),\n            key_id=int(key_id),\n            session_id=int(session_id),\n            target_sbox_index=int(target_sbox_index),\n            fault_model=str(fault_model),\n            design_regime=str(design_regime),\n        ))\n\n    return plan\n\n\ndef target_contract_entry(\n    target_contract: Mapping[str, Any],\n    target_index: int,\n) -> Mapping[str, Any]:\n    for entry in target_contract["targets"]:\n        if int(entry["sbox_index"]) == int(target_index):\n            return entry\n    raise KeyError(f"S{target_index} not found in target contract")\n\n\ndef sample_stage08_glitch_parameters(\n    entry: CampaignPlanEntry,\n    target_contract: Mapping[str, Any],\n    centers: np.ndarray,\n    rng: np.random.Generator,\n) -> GlitchParameters:\n    target_index = int(entry.target_sbox_index)\n    center = float(centers[target_index])\n    target_entry = target_contract_entry(target_contract, target_index)\n\n    exploration_start_offset = (\n        float(target_entry["exploration_window_start_sample_inclusive"])\n        - center\n    )\n    exploration_end_offset = (\n        float(target_entry["exploration_window_end_sample_exclusive"])\n        - center\n    )\n    safe_start_offset = (\n        float(target_entry["public_safe_interval_start_sample_inclusive"])\n        - center\n    )\n    safe_end_offset = (\n        float(target_entry["public_safe_interval_end_sample_exclusive"])\n        - center\n    )\n\n    regime = entry.design_regime\n\n    if regime == "attack_core":\n        offset = float(rng.uniform(\n            safe_start_offset - 0.35,\n            safe_end_offset + 0.35,\n        ))\n        width = float(rng.uniform(0.85, 4.25))\n        strength = float(rng.uniform(0.38, 1.08))\n        repeat = int(rng.choice([1, 1, 1, 1, 2]))\n        repeat_spacing = float(rng.uniform(1.5, 4.75))\n\n    elif regime == "attack_explore":\n        offset = float(rng.uniform(\n            exploration_start_offset - 0.75,\n            exploration_end_offset + 0.75,\n        ))\n        width = float(rng.uniform(0.60, 6.50))\n        strength = float(rng.uniform(0.22, 1.25))\n        repeat = int(rng.choice([1, 1, 1, 2, 2]))\n        repeat_spacing = float(rng.uniform(1.25, 6.25))\n\n    elif regime == "boundary_scan":\n        boundary_offsets = np.asarray([\n            exploration_start_offset,\n            safe_start_offset,\n            safe_end_offset,\n            exploration_end_offset,\n        ], dtype=np.float64)\n        offset = float(\n            rng.choice(boundary_offsets)\n            + rng.normal(0.0, 0.85)\n        )\n        width = float(rng.uniform(0.55, 5.25))\n        strength = float(rng.uniform(0.28, 1.15))\n        repeat = int(rng.choice([1, 1, 2]))\n        repeat_spacing = float(rng.uniform(1.5, 5.5))\n\n    elif regime == "miss_control":\n        direction = -1.0 if rng.random() < 0.5 else 1.0\n        boundary = (\n            exploration_start_offset\n            if direction < 0\n            else exploration_end_offset\n        )\n        offset = float(\n            boundary\n            + direction * rng.uniform(8.0, 24.0)\n        )\n        width = float(rng.uniform(0.50, 2.25))\n        strength = float(rng.uniform(0.08, 0.55))\n        repeat = 1\n        repeat_spacing = float(rng.uniform(1.0, 3.0))\n\n    elif regime == "neighbor_control":\n        neighbor = nearest_neighbor_index(target_index, rng)\n        offset = float(\n            centers[neighbor] - center\n            + rng.normal(0.0, 2.25)\n        )\n        width = float(rng.uniform(0.70, 4.75))\n        strength = float(rng.uniform(0.38, 1.18))\n        repeat = int(rng.choice([1, 1, 2]))\n        repeat_spacing = float(rng.uniform(1.5, 5.75))\n\n    elif regime == "multi_control":\n        neighbor = nearest_neighbor_index(target_index, rng)\n        midpoint = 0.5 * float(centers[neighbor] - center)\n        offset = float(midpoint + rng.normal(0.0, 1.75))\n        width = float(rng.uniform(6.75, 14.5))\n        strength = float(rng.uniform(0.52, 1.30))\n        repeat = int(rng.choice([1, 2, 2, 3]))\n        repeat_spacing = float(rng.uniform(2.75, 9.5))\n\n    elif regime == "invalid_control":\n        offset = float(rng.uniform(\n            exploration_start_offset - 2.0,\n            exploration_end_offset + 2.0,\n        ))\n        width = float(rng.uniform(10.0, 20.0))\n        strength = float(rng.uniform(1.08, 1.95))\n        repeat = int(rng.integers(2, 5))\n        repeat_spacing = float(rng.uniform(2.0, 9.5))\n\n    else:\n        raise ValueError(f"Unknown Stage 08 design regime: {regime}")\n\n    return GlitchParameters(\n        target_sbox_index=target_index,\n        nominal_target_center_sample=center,\n        offset_samples=offset,\n        width_samples=width,\n        strength=strength,\n        repeat=repeat,\n        repeat_spacing_samples=repeat_spacing,\n        sampling_regime=regime,\n        fault_model=entry.fault_model,\n    )\n\n\ndef build_stage08_key_pool(config: Stage08Config) -> List[int]:\n    rng = np.random.default_rng(config.random_seed + 8001)\n    keys: List[int] = []\n    while len(keys) < config.number_of_keys:\n        candidate = random_80bit_integer(rng)\n        if candidate not in keys:\n            keys.append(candidate)\n    return keys\n\n\ndef run_stage08_experiment(\n    entry: CampaignPlanEntry,\n    target_contract: Mapping[str, Any],\n    timing_map: Mapping[str, Any],\n    healthy_source: Mapping[str, np.ndarray],\n    key_pool: Sequence[int],\n    session_shifts: np.ndarray,\n    config: Stage08Config,\n) -> ExperimentRecord:\n    experiment_id = int(entry.experiment_id)\n    seed_sequence = np.random.SeedSequence([\n        config.random_seed,\n        experiment_id,\n        8006,\n    ])\n    rng = np.random.default_rng(seed_sequence)\n\n    target_index = int(entry.target_sbox_index)\n    key_id = int(entry.key_id)\n    session_id = int(entry.session_id)\n\n    centers = np.asarray(\n        [int(item["center_sample"]) for item in timing_map["sboxes"]],\n        dtype=np.float64,\n    )\n    core_widths = np.asarray(\n        [\n            int(item["core_window_end_sample_exclusive"])\n            - int(item["core_window_start_sample_inclusive"])\n            for item in timing_map["sboxes"]\n        ],\n        dtype=np.float64,\n    )\n\n    parameters = sample_stage08_glitch_parameters(\n        entry,\n        target_contract,\n        centers,\n        rng,\n    )\n\n    global_jitter = float(\n        rng.normal(0.0, config.global_timing_jitter_sigma_samples)\n    )\n    local_jitter = rng.normal(\n        0.0,\n        config.local_sbox_jitter_sigma_samples,\n        size=8,\n    )\n    actual_centers = (\n        centers\n        + float(session_shifts[session_id])\n        + global_jitter\n        + local_jitter\n    )\n    injection_jitter = float(\n        rng.normal(0.0, config.injection_timing_jitter_sigma_samples)\n    )\n    pulse_center_values = pulse_centers(parameters, injection_jitter)\n    hit_scores = compute_hit_scores(\n        parameters,\n        pulse_center_values,\n        actual_centers,\n        core_widths,\n    )\n    probabilities = activation_probabilities(parameters, hit_scores)\n    p_invalid = invalid_probability(parameters, hit_scores)\n    invalid = bool(rng.random() < p_invalid)\n\n    master_key = int(key_pool[key_id])\n    plaintext = random_64bit_integer(rng)\n    context = final_round_context(plaintext, master_key)\n    healthy_ciphertext = int(context["ciphertext"])\n    if healthy_ciphertext != encrypt_block_lblock(plaintext, master_key):\n        raise AssertionError("Healthy final-round reconstruction mismatch")\n\n    original_inputs = np.asarray(context["sbox_inputs"], dtype=np.uint8)\n    faulted_inputs = original_inputs.copy()\n    impacted_mask = np.zeros(8, dtype=np.uint8)\n    model_details: Dict[str, Any] = {}\n\n    if not invalid:\n        activated = rng.random(8) < probabilities\n        impacted_mask[:] = activated.astype(np.uint8)\n\n        for sbox_index in np.where(activated)[0]:\n            faulted_value, details = apply_fault_model(\n                int(original_inputs[sbox_index]),\n                parameters.fault_model,\n                rng,\n            )\n            faulted_inputs[sbox_index] = int(faulted_value)\n            model_details[f"S{int(sbox_index)}"] = details\n\n        faulty_ciphertext = final_round_with_faulted_inputs(\n            int(context["x31"]),\n            int(context["x32"]),\n            faulted_inputs,\n        )\n        response_received = True\n        ciphertext_equal = bool(\n            faulty_ciphertext == healthy_ciphertext\n        )\n        ciphertext_hd = hamming_distance(\n            faulty_ciphertext,\n            healthy_ciphertext,\n        )\n        invalid_subtype = ""\n    else:\n        faulty_ciphertext = None\n        response_received = False\n        ciphertext_equal = False\n        ciphertext_hd = -1\n        invalid_subtype = (\n            "reset" if rng.random() < 0.55 else "timeout"\n        )\n\n    category = classify_fault_event(\n        target_index,\n        impacted_mask,\n        invalid,\n        ciphertext_equal,\n    )\n\n    base_index = int(\n        rng.integers(0, healthy_source["traces"].shape[0])\n    )\n    response_trace = synthesize_response_trace(\n        healthy_source["traces"][base_index],\n        healthy_source["absolute_samples"],\n        pulse_center_values,\n        parameters,\n        actual_centers,\n        original_inputs,\n        faulted_inputs,\n        impacted_mask,\n        invalid,\n        rng,\n        config,\n    )\n    features = trace_features(\n        response_trace,\n        healthy_source["absolute_samples"],\n        float(centers[target_index]),\n        pulse_center_values,\n    )\n\n    impacted_indices = [\n        int(value)\n        for value in np.where(impacted_mask > 0)[0]\n    ]\n    target_impacted = bool(impacted_mask[target_index])\n    off_target_impacted = any(\n        value != target_index\n        for value in impacted_indices\n    )\n    changed_input_count = int(\n        np.sum(original_inputs != faulted_inputs)\n    )\n\n    public_row: Dict[str, Any] = {\n        "experiment_id": experiment_id,\n        "campaign_partition": entry.campaign_partition,\n        "target_sbox": f"S{target_index}",\n        "target_sbox_index": target_index,\n        "key_id": key_id,\n        "session_id": session_id,\n        "source_healthy_trace_id": int(\n            healthy_source["trace_ids"][base_index]\n        ),\n        "fault_model": parameters.fault_model,\n        "nominal_target_center_sample": (\n            parameters.nominal_target_center_sample\n        ),\n        "timing_offset_samples": parameters.offset_samples,\n        "first_pulse_nominal_sample": (\n            parameters.nominal_target_center_sample\n            + parameters.offset_samples\n        ),\n        "width_samples": parameters.width_samples,\n        "strength": parameters.strength,\n        "repeat": parameters.repeat,\n        "repeat_spacing_samples": (\n            parameters.repeat_spacing_samples\n        ),\n        "plaintext_hex": hex_fixed(plaintext, 64),\n        "healthy_ciphertext_hex": hex_fixed(\n            healthy_ciphertext,\n            64,\n        ),\n        "response_received": response_received,\n        "faulty_ciphertext_hex": (\n            hex_fixed(int(faulty_ciphertext), 64)\n            if faulty_ciphertext is not None\n            else ""\n        ),\n        "ciphertext_equal": (\n            ciphertext_equal if response_received else ""\n        ),\n        "ciphertext_hamming_distance": ciphertext_hd,\n        **features,\n    }\n\n    private_row: Dict[str, Any] = {\n        "experiment_id": experiment_id,\n        "campaign_partition": entry.campaign_partition,\n        "category": category,\n        "category_id": CATEGORY_TO_ID[category],\n        "design_regime": entry.design_regime,\n        "target_sbox": f"S{target_index}",\n        "target_sbox_index": target_index,\n        "key_id": key_id,\n        "session_id": session_id,\n        "master_key_hex": hex_fixed(master_key, 80),\n        "round_key_32_hex": hex_fixed(\n            int(context["round_key_32"]),\n            32,\n        ),\n        "x31_hex": hex_fixed(int(context["x31"]), 32),\n        "x32_hex": hex_fixed(int(context["x32"]), 32),\n        "target_original_input": int(\n            original_inputs[target_index]\n        ),\n        "target_faulted_input": int(\n            faulted_inputs[target_index]\n        ),\n        "impacted_sboxes": ";".join(\n            f"S{value}" for value in impacted_indices\n        ),\n        "impacted_sbox_count": len(impacted_indices),\n        "target_impacted": target_impacted,\n        "off_target_impacted": off_target_impacted,\n        "changed_sbox_input_count": changed_input_count,\n        "fault_effective": bool(\n            response_received and not ciphertext_equal\n        ),\n        "invalid_subtype": invalid_subtype,\n        "invalid_probability": p_invalid,\n        "global_jitter_samples": global_jitter,\n        "injection_jitter_samples": injection_jitter,\n        "model_details_json": json.dumps(\n            model_details,\n            sort_keys=True,\n        ),\n    }\n\n    private_arrays = {\n        "master_key_bytes": np.frombuffer(\n            master_key.to_bytes(10, "big"),\n            dtype=np.uint8,\n        ),\n        "round_key_32": np.asarray(\n            context["round_key_32"],\n            dtype=np.uint32,\n        ),\n        "x31": np.asarray(context["x31"], dtype=np.uint32),\n        "x32": np.asarray(context["x32"], dtype=np.uint32),\n        "original_inputs": original_inputs.astype(np.uint8),\n        "faulted_inputs": faulted_inputs.astype(np.uint8),\n        "hit_scores": hit_scores.astype(np.float32),\n        "activation_probabilities": probabilities.astype(\n            np.float32\n        ),\n        "actual_centers": actual_centers.astype(np.float32),\n        "pulse_centers": np.pad(\n            pulse_center_values.astype(np.float32),\n            (0, 4 - pulse_center_values.size),\n            constant_values=np.nan,\n        )[:4],\n        "impacted_mask": impacted_mask.astype(np.uint8),\n        "category_id": np.asarray(\n            CATEGORY_TO_ID[category],\n            dtype=np.int8,\n        ),\n    }\n\n    return ExperimentRecord(\n        public_row=public_row,\n        private_row=private_row,\n        response_trace=response_trace,\n        private_arrays=private_arrays,\n    )\n\n\ndef stage08_public_leakage_audit(\n    public_rows: Sequence[Mapping[str, Any]],\n) -> Dict[str, Any]:\n    generic = public_metadata_leakage_audit(public_rows)\n    columns = set(public_rows[0].keys())\n    extra_forbidden = {\n        "design_regime",\n        "key_role",\n        "true_key_nibble",\n        "target_original_input",\n        "target_faulted_input",\n    }\n    present = sorted(columns & extra_forbidden)\n    generic["stage08_extra_forbidden_columns_present"] = present\n    generic["passed"] = bool(generic["passed"] and not present)\n    return generic\n\n\ndef summarize_stage08_campaign(\n    records: Sequence[ExperimentRecord],\n) -> Dict[str, Any]:\n    result = aggregate_campaign(records)\n    result["partition_counts"] = {\n        partition: sum(\n            record.public_row["campaign_partition"] == partition\n            for record in records\n        )\n        for partition in PARTITION_NAMES\n    }\n    result["partition_category_counts"] = {\n        partition: {\n            category: sum(\n                record.public_row["campaign_partition"] == partition\n                and record.private_row["category"] == category\n                for record in records\n            )\n            for category in FAULT_CATEGORIES\n        }\n        for partition in PARTITION_NAMES\n    }\n    result["partition_model_counts"] = {\n        partition: {\n            model: sum(\n                record.public_row["campaign_partition"] == partition\n                and record.public_row["fault_model"] == model\n                for record in records\n            )\n            for model in FAULT_MODELS\n        }\n        for partition in PARTITION_NAMES\n    }\n    result["design_regime_counts"] = {\n        regime: sum(\n            record.private_row["design_regime"] == regime\n            for record in records\n        )\n        for regime in DESIGN_REGIMES\n    }\n    return result\n\n\ndef attack_readiness_statistics(\n    records: Sequence[ExperimentRecord],\n    config: Stage08Config,\n) -> Dict[str, Any]:\n    rows: List[Dict[str, Any]] = []\n\n    for key_id in config.attack_key_ids:\n        for target_index in (0, 5):\n            selected = [\n                record\n                for record in records\n                if record.public_row["campaign_partition"] == "attack"\n                and int(record.public_row["key_id"]) == int(key_id)\n                and int(record.public_row["target_sbox_index"]) == target_index\n                and record.public_row["fault_model"] == "random_and_4"\n            ]\n            category_counts = {\n                category: sum(\n                    record.private_row["category"] == category\n                    for record in selected\n                )\n                for category in FAULT_CATEGORIES\n            }\n            clean_count = (\n                category_counts["clean_target_ineffective"]\n                + category_counts["clean_target_effective"]\n            )\n            ineffective_rate = (\n                category_counts["clean_target_ineffective"] / clean_count\n                if clean_count\n                else float("nan")\n            )\n            input_counts = np.bincount(\n                np.asarray([\n                    int(record.private_row["target_original_input"])\n                    for record in selected\n                ], dtype=np.int32),\n                minlength=16,\n            )\n\n            rows.append({\n                "campaign_partition": "attack",\n                "key_id": int(key_id),\n                "target_sbox": f"S{target_index}",\n                "target_sbox_index": target_index,\n                "primary_model_attempt_count": len(selected),\n                "clean_target_count": clean_count,\n                "clean_ineffective_count": (\n                    category_counts["clean_target_ineffective"]\n                ),\n                "clean_effective_count": (\n                    category_counts["clean_target_effective"]\n                ),\n                "observed_ineffective_rate_given_clean": (\n                    float(ineffective_rate)\n                ),\n                "minimum_original_input_count": int(\n                    np.min(input_counts)\n                ),\n                "maximum_original_input_count": int(\n                    np.max(input_counts)\n                ),\n                "original_input_imbalance_ratio": float(\n                    np.max(input_counts)\n                    / max(np.min(input_counts), 1)\n                ),\n                **{\n                    f"input_{value:01x}_count": int(input_counts[value])\n                    for value in range(16)\n                },\n            })\n\n    all_ready = all(\n        int(row["clean_ineffective_count"])\n        >= config.minimum_attack_primary_clean_ineffective_per_target_key\n        and int(row["clean_effective_count"])\n        >= config.minimum_attack_primary_clean_effective_per_target_key\n        and abs(\n            float(row["observed_ineffective_rate_given_clean"])\n            - 0.31640625\n        ) <= config.maximum_primary_ineffective_rate_error\n        for row in rows\n    )\n\n    return {\n        "all_attack_key_target_groups_ready": bool(all_ready),\n        "rows": rows,\n    }\n\n\ndef validate_stage08_campaign(\n    records: Sequence[ExperimentRecord],\n    plan: Sequence[CampaignPlanEntry],\n    config: Stage08Config,\n    target_contract: Mapping[str, Any],\n    prior: Mapping[str, Any],\n    deterministic_first: str,\n    deterministic_second: str,\n) -> Dict[str, Any]:\n    config_validation = validate_stage08_config(config)\n    public_rows = [record.public_row for record in records]\n    leakage = stage08_public_leakage_audit(public_rows)\n    aggregate = summarize_stage08_campaign(records)\n    attack_readiness = attack_readiness_statistics(records, config)\n\n    expected_partitions = {\n        "train": config.train_experiments,\n        "validation": config.validation_experiments,\n        "test": config.test_experiments,\n        "attack": config.attack_experiments,\n    }\n    observed_partitions = aggregate["partition_counts"]\n\n    category_counts = aggregate["category_counts"]\n    minimum_category_count = max(\n        1,\n        int(config.minimum_category_fraction * len(records)),\n    )\n\n    target_counts = {\n        target: sum(\n            record.public_row["target_sbox"] == target\n            for record in records\n        )\n        for target in ("S0", "S5")\n    }\n\n    plan_matches = all(\n        int(record.public_row["experiment_id"]) == int(entry.experiment_id)\n        and int(record.public_row["key_id"]) == int(entry.key_id)\n        and int(record.public_row["session_id"]) == int(entry.session_id)\n        and int(record.public_row["target_sbox_index"])\n            == int(entry.target_sbox_index)\n        and str(record.public_row["fault_model"]) == str(entry.fault_model)\n        and str(record.public_row["campaign_partition"])\n            == str(entry.campaign_partition)\n        for record, entry in zip(records, plan)\n    )\n\n    primary_fraction = (\n        aggregate["fault_model_counts"]["random_and_4"]\n        / len(records)\n    )\n    prior_fraction = float(\n        prior["recommended_stage_08_policy"]["primary_model_fraction"]\n    )\n\n    checks = {\n        "configuration": config_validation,\n        "experiment_count": {\n            "passed": len(records) == config.number_of_experiments,\n            "observed": len(records),\n            "expected": config.number_of_experiments,\n        },\n        "campaign_plan_matches_records": {\n            "passed": bool(plan_matches),\n        },\n        "partition_counts_exact": {\n            "passed": observed_partitions == expected_partitions,\n            "observed": observed_partitions,\n            "expected": expected_partitions,\n        },\n        "two_target_balance": {\n            "passed": abs(target_counts["S0"] - target_counts["S5"]) <= 1,\n            "counts": target_counts,\n        },\n        "primary_model_matches_public_prior": {\n            "passed": abs(primary_fraction - prior_fraction) <= 0.002,\n            "observed_fraction": primary_fraction,\n            "public_prior_fraction": prior_fraction,\n        },\n        "all_six_categories_present": {\n            "passed": all(\n                value >= minimum_category_count\n                for value in category_counts.values()\n            ),\n            "minimum_count": minimum_category_count,\n            "counts": category_counts,\n        },\n        "public_metadata_leakage_audit": leakage,\n        "deterministic_generation": {\n            "passed": deterministic_first == deterministic_second,\n            "first_digest": deterministic_first,\n            "second_digest": deterministic_second,\n        },\n        "eight_bit_contract_preserved": {\n            "passed": int(target_contract["total_target_bits"]) == 8,\n            "selected_bits": (\n                target_contract["selected_last_round_key_bit_indices"]\n            ),\n        },\n        "attack_partition_readiness": {\n            "passed": bool(\n                attack_readiness[\n                    "all_attack_key_target_groups_ready"\n                ]\n            ),\n            "minimum_clean_ineffective": (\n                config.minimum_attack_primary_clean_ineffective_per_target_key\n            ),\n            "minimum_clean_effective": (\n                config.minimum_attack_primary_clean_effective_per_target_key\n            ),\n        },\n        "finite_response_traces": {\n            "passed": all(\n                np.all(np.isfinite(record.response_trace))\n                for record in records\n            ),\n        },\n    }\n\n    return {\n        "all_checks_passed": all(\n            bool(check["passed"])\n            for check in checks.values()\n        ),\n        "checks": checks,\n        "attack_readiness": attack_readiness,\n        "aggregate": aggregate,\n    }\n\n\ndef save_stage08_public_plots(\n    public_directory: Path,\n    records: Sequence[ExperimentRecord],\n    config: Stage08Config,\n) -> List[str]:\n    if plt is None or not config.save_plots:\n        return []\n\n    generated: List[str] = []\n\n    offsets = np.asarray([\n        float(record.public_row["timing_offset_samples"])\n        for record in records\n    ])\n    widths = np.asarray([\n        float(record.public_row["width_samples"])\n        for record in records\n    ])\n    strengths = np.asarray([\n        float(record.public_row["strength"])\n        for record in records\n    ])\n    partitions = [\n        str(record.public_row["campaign_partition"])\n        for record in records\n    ]\n\n    fig = plt.figure(figsize=(11, 6))\n    axis = fig.add_subplot(1, 1, 1)\n    sample_count = min(12000, len(records))\n    selected = np.linspace(\n        0,\n        len(records) - 1,\n        sample_count,\n        dtype=np.int32,\n    )\n    scatter = axis.scatter(\n        offsets[selected],\n        widths[selected],\n        c=strengths[selected],\n        s=5,\n        alpha=0.25,\n    )\n    axis.set_xlabel("Timing offset from target center [samples]")\n    axis.set_ylabel("Glitch width [samples]")\n    axis.set_title("Stage 08 public parameter coverage")\n    axis.grid(alpha=0.2)\n    fig.colorbar(scatter, ax=axis, label="Configured strength")\n    fig.tight_layout()\n    path = public_directory / "large_campaign_parameter_coverage.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    partition_counts = [\n        partitions.count(partition)\n        for partition in PARTITION_NAMES\n    ]\n    fig = plt.figure(figsize=(9, 5))\n    axis = fig.add_subplot(1, 1, 1)\n    axis.bar(PARTITION_NAMES, partition_counts)\n    axis.set_ylabel("Experiment count")\n    axis.set_title("Public key-disjoint campaign partitions")\n    axis.grid(axis="y", alpha=0.2)\n    fig.tight_layout()\n    path = public_directory / "campaign_partition_sizes.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    model_counts = [\n        sum(\n            record.public_row["fault_model"] == model\n            for record in records\n        )\n        for model in FAULT_MODELS\n    ]\n    fig = plt.figure(figsize=(10.5, 5.5))\n    axis = fig.add_subplot(1, 1, 1)\n    positions = np.arange(len(FAULT_MODELS))\n    axis.bar(positions, model_counts)\n    axis.set_xticks(positions)\n    axis.set_xticklabels(FAULT_MODELS, rotation=30, ha="right")\n    axis.set_ylabel("Experiment count")\n    axis.set_title("Public fault-model allocation")\n    axis.grid(axis="y", alpha=0.2)\n    fig.tight_layout()\n    path = public_directory / "fault_model_allocation.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    return generated\n\n\ndef save_stage08_validation_plots(\n    validation_directory: Path,\n    records: Sequence[ExperimentRecord],\n    config: Stage08Config,\n) -> List[str]:\n    if plt is None or not config.save_plots:\n        return []\n\n    generated: List[str] = []\n\n    matrix = np.zeros(\n        (len(PARTITION_NAMES), len(FAULT_CATEGORIES)),\n        dtype=np.int64,\n    )\n    for partition_index, partition in enumerate(PARTITION_NAMES):\n        for category_index, category in enumerate(FAULT_CATEGORIES):\n            matrix[partition_index, category_index] = sum(\n                record.public_row["campaign_partition"] == partition\n                and record.private_row["category"] == category\n                for record in records\n            )\n\n    fig = plt.figure(figsize=(12, 5.5))\n    axis = fig.add_subplot(1, 1, 1)\n    image = axis.imshow(matrix, aspect="auto")\n    axis.set_yticks(range(len(PARTITION_NAMES)))\n    axis.set_yticklabels(PARTITION_NAMES)\n    axis.set_xticks(range(len(FAULT_CATEGORIES)))\n    axis.set_xticklabels(\n        FAULT_CATEGORIES,\n        rotation=35,\n        ha="right",\n    )\n    axis.set_title("Hidden event classes by campaign partition")\n    fig.colorbar(image, ax=axis, label="Count")\n    fig.tight_layout()\n    path = validation_directory / "partition_category_distribution.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    attack_records = [\n        record\n        for record in records\n        if record.public_row["campaign_partition"] == "attack"\n        and record.public_row["fault_model"] == "random_and_4"\n    ]\n    labels = [\n        (\n            int(record.public_row["key_id"]),\n            str(record.public_row["target_sbox"]),\n        )\n        for record in attack_records\n    ]\n    unique_labels = sorted(set(labels))\n    ineffective_counts = []\n    effective_counts = []\n    display_names = []\n\n    for key_id, target in unique_labels:\n        display_names.append(f"KID{key_id}-{target}")\n        ineffective_counts.append(sum(\n            int(record.public_row["key_id"]) == key_id\n            and record.public_row["target_sbox"] == target\n            and record.private_row["category"]\n                == "clean_target_ineffective"\n            for record in attack_records\n        ))\n        effective_counts.append(sum(\n            int(record.public_row["key_id"]) == key_id\n            and record.public_row["target_sbox"] == target\n            and record.private_row["category"]\n                == "clean_target_effective"\n            for record in attack_records\n        ))\n\n    fig = plt.figure(figsize=(10.5, 5.5))\n    axis = fig.add_subplot(1, 1, 1)\n    positions = np.arange(len(display_names))\n    axis.bar(\n        positions - 0.18,\n        ineffective_counts,\n        width=0.36,\n        label="Clean ineffective",\n    )\n    axis.bar(\n        positions + 0.18,\n        effective_counts,\n        width=0.36,\n        label="Clean effective",\n    )\n    axis.set_xticks(positions)\n    axis.set_xticklabels(display_names)\n    axis.set_ylabel("Ground-truth sample count")\n    axis.set_title("Attack-partition primary-model readiness")\n    axis.legend()\n    axis.grid(axis="y", alpha=0.2)\n    fig.tight_layout()\n    path = validation_directory / "attack_partition_clean_counts.png"\n    fig.savefig(path, dpi=180)\n    plt.close(fig)\n    generated.append(path.name)\n\n    return generated\n\n\ndef run_stage_08(config: Stage08Config) -> Dict[str, Any]:\n    start_time = time.perf_counter()\n    validate_stage08_config(config)\n\n    stage7_run_directory = Path(\n        config.input_stage7_run_directory\n    ).expanduser().resolve()\n    contracts = load_stage08_contracts(stage7_run_directory)\n\n    prior = contracts["prior"]\n    target_contract = contracts["target_contract"]\n    timing_map = contracts["timing_map"]\n\n    required_primary_fraction = float(\n        prior["recommended_stage_08_policy"]["primary_model_fraction"]\n    )\n    if not np.isclose(\n        required_primary_fraction,\n        config.random_and_4_weight,\n    ):\n        raise ValueError(\n            "Config primary-model fraction does not match Stage 07 public prior"\n        )\n\n    if not bool(\n        prior["recommended_stage_08_policy"]["retain_both_selected_targets"]\n    ):\n        raise RuntimeError("Stage 07 prior does not retain both targets")\n\n    healthy_source = load_healthy_roi_source(\n        contracts["roi_path"]\n    )\n    plan = build_campaign_plan(config)\n    key_pool = build_stage08_key_pool(config)\n\n    session_rng = np.random.default_rng(\n        config.random_seed + 8002\n    )\n    session_shifts = session_rng.normal(\n        0.0,\n        config.session_timing_shift_sigma_samples,\n        size=config.number_of_sessions,\n    ).astype(np.float64)\n\n    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")\n    run_id = (\n        f"stage08_{timestamp}_seed{config.random_seed}"\n    )\n    run_directory = (\n        Path(config.output_root)\n        .expanduser()\n        .resolve()\n        / run_id\n    )\n    public_directory = run_directory / "public"\n    private_directory = run_directory / "private_ground_truth"\n    validation_directory = run_directory / "validation_only"\n    public_directory.mkdir(parents=True, exist_ok=False)\n\n    records: List[ExperimentRecord] = []\n    progress_step = max(1, config.number_of_experiments // 20)\n\n    for index, entry in enumerate(plan):\n        records.append(run_stage08_experiment(\n            entry,\n            target_contract,\n            timing_map,\n            healthy_source,\n            key_pool,\n            session_shifts,\n            config,\n        ))\n        if (\n            (index + 1) % progress_step == 0\n            or index + 1 == config.number_of_experiments\n        ):\n            print(\n                f"Generated {index + 1:,}/"\n                f"{config.number_of_experiments:,} experiments"\n            )\n\n    # Deterministic recheck on a fixed prefix.\n    recheck_count = min(\n        config.deterministic_recheck_count,\n        len(plan),\n    )\n    first_digest = campaign_digest(records, recheck_count)\n    rerun_records = [\n        run_stage08_experiment(\n            plan[index],\n            target_contract,\n            timing_map,\n            healthy_source,\n            key_pool,\n            session_shifts,\n            config,\n        )\n        for index in range(recheck_count)\n    ]\n    second_digest = campaign_digest(\n        rerun_records,\n        recheck_count,\n    )\n\n    public_rows = [record.public_row for record in records]\n    response_traces = np.stack(\n        [record.response_trace for record in records],\n        axis=0,\n    ).astype(np.float32)\n    experiment_ids = np.asarray(\n        [record.public_row["experiment_id"] for record in records],\n        dtype=np.int32,\n    )\n\n    # ------------------------- public outputs -------------------------\n    write_csv_rows(\n        public_directory / "large_fault_campaign_public.csv",\n        public_rows,\n    )\n\n    if config.save_response_traces:\n        np.savez_compressed(\n            public_directory / "large_fault_response_traces.npz",\n            traces=response_traces,\n            experiment_ids=experiment_ids,\n            absolute_sample_indices=healthy_source["absolute_samples"],\n            sample_axis_seconds=healthy_source["sample_axis_seconds"],\n        )\n\n    partition_manifest = {\n        "stage": 8,\n        "key_values_included": False,\n        "split_unit": "key_id",\n        "session_policy": (\n            "Every key is represented in all configured sessions; "\n            "key partitions remain disjoint."\n        ),\n        "partitions": {\n            "train": {\n                "key_ids": list(config.train_key_ids),\n                "experiment_count": config.train_experiments,\n            },\n            "validation": {\n                "key_ids": list(config.validation_key_ids),\n                "experiment_count": config.validation_experiments,\n            },\n            "test": {\n                "key_ids": list(config.test_key_ids),\n                "experiment_count": config.test_experiments,\n            },\n            "attack": {\n                "key_ids": list(config.attack_key_ids),\n                "experiment_count": config.attack_experiments,\n                "purpose": (\n                    "Held-out final SIFA/SEFA/SHFA key-recovery experiments"\n                ),\n            },\n        },\n    }\n    write_json(\n        public_directory / "campaign_partition_manifest.json",\n        partition_manifest,\n    )\n\n    design_policy = {\n        "stage": 8,\n        "source_prior": str(contracts["prior_path"]),\n        "source_prior_sha256": sha256_file(contracts["prior_path"]),\n        "ground_truth_or_oracle_used_for_generation": False,\n        "oracle_parameter_cells_opened": False,\n        "primary_fault_model": "random_and_4",\n        "fault_model_fractions": stage08_model_weights(config),\n        "design_regime_fractions": stage08_regime_weights(config),\n        "selected_targets": target_contract["selected_sboxes"],\n        "selected_key_nibbles": (\n            target_contract["selected_last_round_key_nibbles"]\n        ),\n        "total_target_bits": int(target_contract["total_target_bits"]),\n        "policy": (\n            "Use public safe/exploration windows for attack-oriented probes; "\n            "retain broad exploration and explicit control regimes."\n        ),\n    }\n    write_json(\n        public_directory / "campaign_design_policy.json",\n        design_policy,\n    )\n\n    write_json(\n        public_directory / "stage_08_config.json",\n        {\n            **asdict(config),\n            "train_key_ids": list(config.train_key_ids),\n            "validation_key_ids": list(config.validation_key_ids),\n            "test_key_ids": list(config.test_key_ids),\n            "attack_key_ids": list(config.attack_key_ids),\n        },\n    )\n\n    write_json(\n        public_directory / "trace_feature_definitions.json",\n        {\n            "source": "Stage 06 trace feature definitions",\n            "features": [\n                "trace_mean",\n                "trace_std",\n                "trace_peak_to_peak",\n                "trace_max_abs",\n                "trace_l1_energy",\n                "trace_l2_energy",\n                "trace_high_frequency_energy",\n                "target_window_energy",\n                "pulse_window_energy",\n                "saturation_fraction",\n            ],\n            "private_labels_used": False,\n        },\n    )\n\n    public_plots = save_stage08_public_plots(\n        public_directory,\n        records,\n        config,\n    )\n\n    public_access_manifest = {\n        "files_opened_before_public_freeze": [\n            str(contracts["stage7_summary_path"]),\n            str(contracts["prior_path"]),\n            str(contracts["stage6_summary_path"]),\n            str(contracts["stage5_summary_path"]),\n            str(contracts["target_contract_path"]),\n            str(contracts["timing_map_path"]),\n            str(contracts["roi_path"]),\n        ],\n        "stage_07_oracle_files_opened": [],\n        "forbidden_generation_inputs": {\n            "oracle_parameter_cells": False,\n            "oracle_campaign_recommendations": False,\n            "stage_06_private_ground_truth": False,\n            "stage_06_key_manifest": False,\n        },\n    }\n    write_json(\n        public_directory / "public_data_access_manifest.json",\n        public_access_manifest,\n    )\n\n    freeze_files = sorted(\n        path for path in public_directory.iterdir()\n        if path.is_file()\n    )\n    freeze_manifest = {\n        "created_at": datetime.now().isoformat(timespec="seconds"),\n        "statement": (\n            "Stage 08 public campaign and partitions were frozen before "\n            "private labels, internal states, and key values were written."\n        ),\n        "files": {\n            path.name: sha256_file(path)\n            for path in freeze_files\n        },\n    }\n    freeze_payload = json.dumps(\n        freeze_manifest,\n        ensure_ascii=False,\n        sort_keys=True,\n    ).encode("utf-8")\n    freeze_sha256 = hashlib.sha256(freeze_payload).hexdigest()\n    freeze_manifest["freeze_sha256"] = freeze_sha256\n    write_json(\n        run_directory / "public_freeze_manifest.json",\n        freeze_manifest,\n    )\n\n    # ------------------------- private outputs ------------------------\n    private_directory.mkdir(parents=True, exist_ok=True)\n    private_rows = [record.private_row for record in records]\n    write_csv_rows(\n        private_directory / "large_fault_ground_truth.csv",\n        private_rows,\n    )\n\n    private_array_names = list(records[0].private_arrays.keys())\n    private_arrays = {\n        name: np.stack(\n            [record.private_arrays[name] for record in records],\n            axis=0,\n        )\n        for name in private_array_names\n    }\n    np.savez_compressed(\n        private_directory / "large_fault_ground_truth_arrays.npz",\n        experiment_ids=experiment_ids,\n        **private_arrays,\n    )\n\n    key_manifest = {\n        "warning": (\n            "Private validation material. Never use key values as ML features."\n        ),\n        "partition_key_ids": {\n            "train": list(config.train_key_ids),\n            "validation": list(config.validation_key_ids),\n            "test": list(config.test_key_ids),\n            "attack": list(config.attack_key_ids),\n        },\n        "keys": [\n            {\n                "key_id": key_id,\n                "campaign_partition": next(\n                    partition\n                    for partition, ids in {\n                        "train": config.train_key_ids,\n                        "validation": config.validation_key_ids,\n                        "test": config.test_key_ids,\n                        "attack": config.attack_key_ids,\n                    }.items()\n                    if key_id in ids\n                ),\n                "master_key_hex": hex_fixed(\n                    int(key_pool[key_id]),\n                    80,\n                ),\n                "round_key_32_hex": hex_fixed(\n                    key_schedule_lblock(\n                        int(key_pool[key_id])\n                    )[31],\n                    32,\n                ),\n            }\n            for key_id in range(config.number_of_keys)\n        ],\n    }\n    write_json(\n        private_directory / "key_manifest.json",\n        key_manifest,\n    )\n\n    # ------------------------- validation -----------------------------\n    validation_directory.mkdir(parents=True, exist_ok=True)\n    validation = validate_stage08_campaign(\n        records,\n        plan,\n        config,\n        target_contract,\n        prior,\n        first_digest,\n        second_digest,\n    )\n    write_json(\n        validation_directory / "stage_08_campaign_validation.json",\n        validation,\n    )\n    write_json(\n        validation_directory / "campaign_aggregate_statistics.json",\n        validation["aggregate"],\n    )\n    write_csv_rows(\n        validation_directory / "attack_partition_readiness.csv",\n        validation["attack_readiness"]["rows"],\n    )\n    write_json(\n        validation_directory / "private_data_use_manifest.json",\n        {\n            "private_data_used_only_after_public_freeze": True,\n            "public_freeze_sha256": freeze_sha256,\n            "uses": [\n                "event-class validation",\n                "attack-partition clean-sample readiness",\n                "input-balance checks",\n                "final key-recovery truth for later stages",\n            ],\n            "private_data_not_permitted_as_ml_features": [\n                "master_key_hex",\n                "round_key_32_hex",\n                "x31_hex",\n                "x32_hex",\n                "target_original_input",\n                "target_faulted_input",\n                "impacted_sboxes",\n                "category",\n            ],\n        },\n    )\n\n    validation_plots = save_stage08_validation_plots(\n        validation_directory,\n        records,\n        config,\n    )\n\n    aggregate = validation["aggregate"]\n    attack_rows = validation["attack_readiness"]["rows"]\n    elapsed_seconds = time.perf_counter() - start_time\n\n    summary = {\n        "stage": 8,\n        "run_id": run_id,\n        "run_directory": str(run_directory.resolve()),\n        "input_stage_07_run_directory": str(stage7_run_directory),\n        "input_stage_06_run_directory": str(\n            contracts["stage6_run_directory"]\n        ),\n        "input_stage_05_run_directory": str(\n            contracts["stage5_run_directory"]\n        ),\n        "input_stage_04_run_directory": str(\n            contracts["stage4_run_directory"]\n        ),\n        "input_stage_03_run_directory": str(\n            contracts["stage3_run_directory"]\n        ),\n        "public_directory": str(public_directory.resolve()),\n        "private_ground_truth_directory": str(\n            private_directory.resolve()\n        ),\n        "validation_only_directory": str(\n            validation_directory.resolve()\n        ),\n        "all_checks_passed": bool(validation["all_checks_passed"]),\n        "public_metadata_leakage_audit_passed": bool(\n            validation["checks"][\n                "public_metadata_leakage_audit"\n            ]["passed"]\n        ),\n        "deterministic_generation_passed": bool(\n            validation["checks"][\n                "deterministic_generation"\n            ]["passed"]\n        ),\n        "attack_partition_ready": bool(\n            validation["checks"][\n                "attack_partition_readiness"\n            ]["passed"]\n        ),\n        "number_of_experiments": int(config.number_of_experiments),\n        "number_of_response_trace_samples": int(\n            response_traces.shape[1]\n        ),\n        "number_of_keys": int(config.number_of_keys),\n        "number_of_sessions": int(config.number_of_sessions),\n        "partition_counts": aggregate["partition_counts"],\n        "partition_key_ids": {\n            "train": list(config.train_key_ids),\n            "validation": list(config.validation_key_ids),\n            "test": list(config.test_key_ids),\n            "attack": list(config.attack_key_ids),\n        },\n        "selected_sboxes": target_contract["selected_sboxes"],\n        "selected_last_round_key_nibbles": (\n            target_contract["selected_last_round_key_nibbles"]\n        ),\n        "selected_last_round_key_bit_indices": (\n            target_contract["selected_last_round_key_bit_indices"]\n        ),\n        "total_target_bits": int(target_contract["total_target_bits"]),\n        "primary_fault_model": "random_and_4",\n        "fault_model_counts": aggregate["fault_model_counts"],\n        "fault_model_rates": aggregate["fault_model_rates"],\n        "category_counts": aggregate["category_counts"],\n        "category_rates": aggregate["category_rates"],\n        "design_regime_counts": aggregate["design_regime_counts"],\n        "primary_random_and_clean_target_count": (\n            aggregate["primary_random_and_clean_target_count"]\n        ),\n        "primary_random_and_clean_target_ineffective_count": (\n            aggregate[\n                "primary_random_and_clean_target_ineffective_count"\n            ]\n        ),\n        "primary_random_and_clean_target_ineffective_rate": (\n            aggregate[\n                "primary_random_and_clean_target_ineffective_rate"\n            ]\n        ),\n        "attack_key_target_readiness": attack_rows,\n        "public_freeze_sha256": freeze_sha256,\n        "stage_07_prior_sha256": sha256_file(contracts["prior_path"]),\n        "oracle_files_used_for_generation": False,\n        "elapsed_seconds": float(elapsed_seconds),\n        "public_files": sorted(\n            path.name\n            for path in public_directory.iterdir()\n            if path.is_file()\n        ),\n        "private_files": sorted(\n            path.name\n            for path in private_directory.iterdir()\n            if path.is_file()\n        ),\n        "validation_files": sorted(\n            path.name\n            for path in validation_directory.iterdir()\n            if path.is_file()\n        ),\n        "generated_plots": {\n            "public": public_plots,\n            "validation_only": validation_plots,\n        },\n    }\n\n    write_json(\n        run_directory / "stage_08_summary.json",\n        summary,\n    )\n\n    write_json(\n        run_directory / "run_manifest.json",\n        {\n            "stage": 8,\n            "run_id": run_id,\n            "created_at": datetime.now().isoformat(timespec="seconds"),\n            "python_version": sys.version,\n            "platform": platform.platform(),\n            "config": {\n                **asdict(config),\n                "train_key_ids": list(config.train_key_ids),\n                "validation_key_ids": list(config.validation_key_ids),\n                "test_key_ids": list(config.test_key_ids),\n                "attack_key_ids": list(config.attack_key_ids),\n            },\n            "input_sha256": {\n                "stage_07_summary.json": sha256_file(\n                    contracts["stage7_summary_path"]\n                ),\n                "theoretical_attack_prior.json": sha256_file(\n                    contracts["prior_path"]\n                ),\n                "stage_06_summary.json": sha256_file(\n                    contracts["stage6_summary_path"]\n                ),\n                "lblock_8bit_target_contract.json": sha256_file(\n                    contracts["target_contract_path"]\n                ),\n                "lblock_final_round_timing_map.json": sha256_file(\n                    contracts["timing_map_path"]\n                ),\n                "final_round_roi_traces.npz": sha256_file(\n                    contracts["roi_path"]\n                ),\n            },\n        },\n    )\n\n    print("\\n" + "=" * 82)\n    print("Stage 08 complete: large attack-oriented LBlock fault campaign")\n    print("=" * 82)\n    print("Run directory                  :", summary["run_directory"])\n    print("All checks passed              :", summary["all_checks_passed"])\n    print("Experiments                    :", summary["number_of_experiments"])\n    print("Keys / sessions                :", summary["number_of_keys"], "/", summary["number_of_sessions"])\n    print("Partitions                     :", summary["partition_counts"])\n    print("Selected targets               :", summary["selected_sboxes"])\n    print("Primary model fraction         :", summary["fault_model_rates"]["random_and_4"])\n    print("Primary clean target count     :", summary["primary_random_and_clean_target_count"])\n    print("Primary ineffective count      :", summary["primary_random_and_clean_target_ineffective_count"])\n    print("Attack partition ready         :", summary["attack_partition_ready"])\n    print("Public freeze SHA-256          :", summary["public_freeze_sha256"])\n    print("Elapsed seconds                :", f"{summary[\'elapsed_seconds\']:.3f}")\n    print("=" * 82)\n\n    if not summary["all_checks_passed"]:\n        raise AssertionError(\n            "Stage 08 failed validation. Inspect "\n            "validation_only/stage_08_campaign_validation.json"\n        )\n\n    return summary\n\n\ndef load_stage_08_config(path: str | Path) -> Stage08Config:\n    raw = read_json(Path(path))\n    for field in (\n        "train_key_ids",\n        "validation_key_ids",\n        "test_key_ids",\n        "attack_key_ids",\n    ):\n        if field in raw:\n            raw[field] = tuple(int(value) for value in raw[field])\n    return Stage08Config(**raw)\n\n'

exec(
    compile(
        _STAGE08_SOURCE,
        _stage08_module.__file__,
        "exec",
    ),
    _stage08_module.__dict__,
)

print("Embedded Stage-08 engine loaded successfully.")

# ============================================================
# Stage-12 pipeline
# ============================================================

"""
Stage 12 — Leakage-safe closed-loop fault campaign for LBlock-64/80.

This stage validates the Stage-11 pre-injection optimizer with a fresh,
pre-registered simulation campaign.  Adaptation uses only public observations
and the frozen Stage-10 quality classifier.  Private simulator labels are not
opened until the complete public campaign, classifier probabilities, and
closed-loop selection history have been frozen.

The design contains four sequential batches.  The first three batches use six
fresh adaptation keys.  The fourth batch is a confirmation batch on two new
keys that were never used for policy updates.  Each batch contains:

  * optimizer-guided exploitation,
  * optimizer-guided exploration,
  * a randomized attack-oriented baseline,
  * explicit missed/neighbor/multi-hit/invalid safety controls.

The confirmation batch is the primary unbiased test of empirical uplift.
"""


from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import csv
import hashlib
import json
import math
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

# The package bundles the validated Stage-08 simulation core.  Keeping the
# physical simulator unchanged is important: Stage 12 changes only the public
# campaign policy, not the underlying fault-generation mechanism.
try:
    import stage_08_large_attack_oriented_fault_campaign as engine
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stage_08_large_attack_oriented_fault_campaign.py must be in the "
        "same directory as this script/notebook."
    ) from exc


# ============================================================
# 1. Configuration and fixed contracts
# ============================================================


@dataclass(frozen=True)
class Stage12Config:
    input_stage11_run_directory: str
    output_root: str = "runs/stage_12"
    random_seed: int = 20260718

    number_of_experiments: int = 24000
    number_of_batches: int = 4
    experiments_per_batch: int = 6000
    number_of_keys: int = 8
    number_of_sessions: int = 4
    confirmation_batch_index: int = 3

    # Per-batch pre-registered allocation.  Seventy percent remains optimizer
    # guided, twenty percent is a same-campaign randomized baseline, and ten
    # percent preserves safety/negative controls.
    guided_exploit_fraction: float = 0.50
    guided_explore_fraction: float = 0.20
    randomized_baseline_fraction: float = 0.20
    safety_control_fraction: float = 0.10

    sifa_objective_fraction: float = 0.35
    sefa_objective_fraction: float = 0.35
    shfa_objective_fraction: float = 0.30

    # Closed-loop recommendation-arm updating.  The outcome is a public
    # Stage-10 probability, never a private simulator category.
    prior_strength: float = 8.0
    exploit_ucb_coefficient: float = 0.22
    explore_ucb_coefficient: float = 0.38
    disagreement_weight: float = 0.25
    selection_softmax_temperature: float = 0.08

    # Small implementation jitter around Stage-11 recommendations.
    exploit_offset_jitter_sigma: float = 0.15
    explore_offset_jitter_sigma: float = 0.35
    exploit_relative_parameter_jitter: float = 0.03
    explore_relative_parameter_jitter: float = 0.07

    # The same timing/noise model used by Stage 08.
    global_timing_jitter_sigma_samples: float = 0.35
    local_sbox_jitter_sigma_samples: float = 0.18
    injection_timing_jitter_sigma_samples: float = 0.20
    session_timing_shift_sigma_samples: float = 0.25
    response_trace_noise_sigma: float = 0.055
    response_trace_baseline_sigma: float = 0.035
    response_trace_gain_sigma: float = 0.06

    # Stage-09 feature-view reconstruction constants.
    target_window_radius_samples: int = 24
    pulse_window_radius_samples: int = 24
    highpass_moving_average_width: int = 9
    trace_standard_deviation_floor: float = 1.0e-6

    bootstrap_repetitions: int = 1000
    ece_bins: int = 15
    save_plots: bool = True
    deterministic_recheck_count: int = 32


CLASS_NAMES: Tuple[str, ...] = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
OBJECTIVES: Tuple[str, ...] = ("SIFA", "SEFA", "SHFA")
TARGETS: Tuple[str, ...] = ("S0", "S5")
GUIDED_ARMS: Tuple[str, ...] = ("guided_exploit", "guided_explore")
ALL_ARMS: Tuple[str, ...] = (
    "guided_exploit",
    "guided_explore",
    "baseline_random",
    "safety_control",
)
SAFETY_REGIMES: Tuple[str, ...] = (
    "miss_control",
    "neighbor_control",
    "multi_control",
    "invalid_control",
)

PRIMARY_TABULAR_FEATURES: Tuple[str, ...] = (
    "target_is_s5",
    "timing_offset_samples",
    "absolute_timing_offset_samples",
    "width_samples",
    "strength",
    "repeat",
    "repeat_spacing_samples",
    "pulse_span_samples",
    "glitch_energy_proxy",
    "response_received_numeric",
    "ciphertext_equal_observed",
    "ciphertext_equal_missing",
    "ciphertext_hamming_distance_observed",
    "ciphertext_hamming_distance_missing",
    "trace_mean",
    "trace_std",
    "trace_peak_to_peak",
    "trace_max_absolute",
    "trace_high_frequency_energy",
    "target_window_energy",
    "pulse_window_energy",
    "saturation_fraction",
    "pulse_to_target_energy_ratio",
    "high_frequency_to_variance_ratio",
    "target_window_valid_fraction",
    "pulse_window_valid_fraction",
)


@dataclass(frozen=True)
class ClosedLoopPlanEntry:
    experiment_id: int
    batch_index: int
    batch_role: str
    key_id: int
    session_id: int
    target_sbox_index: int
    objective: str
    campaign_arm: str
    recommendation_mode: str
    recommendation_candidate_id: int
    recommendation_rank: int
    safety_regime: str
    fault_model: str = "random_and_4"


# ============================================================
# 2. Generic helpers and freeze verification
# ============================================================


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_manifest(root: Path, manifest_path: Path) -> Dict[str, Any]:
    manifest = read_json(manifest_path)
    mismatches: List[Dict[str, str]] = []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            mismatches.append({"file": relative, "reason": "missing"})
            continue
        observed = sha256_file(path)
        if observed != expected:
            mismatches.append({
                "file": relative,
                "reason": "sha256_mismatch",
                "expected": expected,
                "observed": observed,
            })
    return {
        "passed": len(mismatches) == 0,
        "freeze_sha256": manifest.get("freeze_sha256", ""),
        "mismatches": mismatches,
    }


def counts_from_fractions(total: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    names = list(fractions)
    raw = np.asarray([total * float(fractions[name]) for name in names])
    base = np.floor(raw).astype(int)
    remainder = total - int(base.sum())
    order = np.argsort(-(raw - base))
    for index in order[:remainder]:
        base[index] += 1
    return {name: int(base[i]) for i, name in enumerate(names)}


def validate_config(config: Stage12Config) -> Dict[str, Any]:
    errors: List[str] = []
    if config.number_of_batches * config.experiments_per_batch != config.number_of_experiments:
        errors.append("batches * experiments_per_batch must equal total")
    if config.number_of_keys != 2 * config.number_of_batches:
        errors.append("exactly two fresh keys per batch are required")
    if config.confirmation_batch_index != config.number_of_batches - 1:
        errors.append("confirmation batch must be the final batch")
    arm_sum = (
        config.guided_exploit_fraction
        + config.guided_explore_fraction
        + config.randomized_baseline_fraction
        + config.safety_control_fraction
    )
    if not np.isclose(arm_sum, 1.0):
        errors.append("campaign-arm fractions must sum to one")
    objective_sum = (
        config.sifa_objective_fraction
        + config.sefa_objective_fraction
        + config.shfa_objective_fraction
    )
    if not np.isclose(objective_sum, 1.0):
        errors.append("objective fractions must sum to one")
    if config.experiments_per_batch % (2 * config.number_of_sessions) != 0:
        errors.append("batch size must divide evenly over two keys and sessions")
    if errors:
        raise ValueError("; ".join(errors))
    return {"passed": True}


def resolve_stage_contracts(stage11_dir: Path) -> Dict[str, Any]:
    summary11_path = stage11_dir / "stage_11_summary.json"
    freeze11_path = stage11_dir / "optimizer_freeze_manifest.json"
    if not summary11_path.is_file() or not freeze11_path.is_file():
        raise FileNotFoundError("Stage 11 summary/freeze is missing")
    summary11 = read_json(summary11_path)
    if not summary11.get("all_checks_passed", False):
        raise RuntimeError("Stage 11 did not pass all checks")
    verify11 = verify_manifest(stage11_dir, freeze11_path)
    if not verify11["passed"]:
        raise RuntimeError("Stage 11 optimizer freeze verification failed")

    stage10_dir = Path(summary11["input_stage_10_run_directory"]).expanduser().resolve()
    summary10 = read_json(stage10_dir / "stage_10_summary.json")
    verify10 = verify_manifest(stage10_dir, stage10_dir / "model_freeze_manifest.json")
    if not summary10.get("all_checks_passed", False) or not verify10["passed"]:
        raise RuntimeError("Stage 10 model freeze verification failed")

    stage9_dir = Path(summary10["input_stage_09_run_directory"]).expanduser().resolve()
    summary9 = read_json(stage9_dir / "stage_09_summary.json")
    verify9 = verify_manifest(stage9_dir / "public_ml", stage9_dir / "public_ml_freeze_manifest.json")
    if not summary9.get("all_checks_passed", False) or not verify9["passed"]:
        raise RuntimeError("Stage 09 public ML freeze verification failed")

    stage8_dir = Path(summary9["input_stage_08_run_directory"]).expanduser().resolve()
    summary8 = read_json(stage8_dir / "stage_08_summary.json")
    if not summary8.get("all_checks_passed", False):
        raise RuntimeError("Stage 08 did not pass all checks")

    stage3_dir = Path(summary8["input_stage_03_run_directory"]).expanduser().resolve()
    stage4_dir = Path(summary8["input_stage_04_run_directory"]).expanduser().resolve()
    stage5_dir = Path(summary8["input_stage_05_run_directory"]).expanduser().resolve()

    timing_map = read_json(stage4_dir / "public" / "lblock_final_round_timing_map.json")
    target_contract = read_json(stage5_dir / "public" / "lblock_8bit_target_contract.json")
    healthy_source = engine.load_healthy_roi_source(
        stage3_dir / "public" / "final_round_roi_traces.npz"
    )

    exploit_path = stage11_dir / "recommendations" / "exploit_parameter_recommendations.csv"
    explore_path = stage11_dir / "recommendations" / "explore_parameter_recommendations.csv"
    policy_path = stage11_dir / "recommendations" / "stage_12_campaign_policy.json"
    for path in (exploit_path, explore_path, policy_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    model10_path = stage10_dir / "models" / "fault_quality_deployment_model.joblib"
    model10 = joblib.load(model10_path)
    if model10.get("attack_labels_used", True):
        raise RuntimeError("Stage 10 deployment model reports Attack-label use")

    return {
        "stage11_summary": summary11,
        "stage10_summary": summary10,
        "stage9_summary": summary9,
        "stage8_summary": summary8,
        "stage11_verify": verify11,
        "stage10_verify": verify10,
        "stage9_verify": verify9,
        "stage10_dir": stage10_dir,
        "stage9_dir": stage9_dir,
        "stage8_dir": stage8_dir,
        "stage3_dir": stage3_dir,
        "stage4_dir": stage4_dir,
        "stage5_dir": stage5_dir,
        "timing_map": timing_map,
        "target_contract": target_contract,
        "healthy_source": healthy_source,
        "exploit_recommendations": pd.read_csv(exploit_path),
        "explore_recommendations": pd.read_csv(explore_path),
        "stage11_policy": read_json(policy_path),
        "stage10_model": model10,
    }


# ============================================================
# 3. Stage-10 public feature reconstruction and scoring
# ============================================================


def moving_average_same(values: np.ndarray, width: int) -> np.ndarray:
    radius = width // 2
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    cumulative = np.cumsum(padded, axis=1, dtype=np.float64)
    cumulative = np.pad(cumulative, ((0, 0), (1, 0)), mode="constant")
    average = (cumulative[:, width:] - cumulative[:, :-width]) / float(width)
    return average.astype(np.float32)


def extract_aligned_windows(
    traces: np.ndarray,
    absolute_samples: np.ndarray,
    centers: np.ndarray,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    width = 2 * radius + 1
    output = np.zeros((traces.shape[0], width), dtype=np.float32)
    valid_fraction = np.zeros(traces.shape[0], dtype=np.float32)
    first_sample = int(absolute_samples[0])
    last_sample = int(absolute_samples[-1])
    for row_index in range(traces.shape[0]):
        center = int(round(float(centers[row_index])))
        requested_start = center - radius
        requested_end = center + radius
        valid_start = max(requested_start, first_sample)
        valid_end = min(requested_end, last_sample)
        if valid_end < valid_start:
            continue
        source_start = valid_start - first_sample
        source_end = valid_end - first_sample + 1
        destination_start = valid_start - requested_start
        destination_end = destination_start + source_end - source_start
        output[row_index, destination_start:destination_end] = traces[
            row_index, source_start:source_end
        ]
        valid_fraction[row_index] = (source_end - source_start) / float(width)
    return output, valid_fraction


def _bool_observed(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    missing = series.isna() | (series.astype(str).str.strip() == "")
    normalized = series.astype(str).str.lower().map({
        "true": 1.0,
        "false": 0.0,
        "1": 1.0,
        "0": 0.0,
    }).fillna(0.0)
    return normalized.to_numpy(dtype=np.float64), missing.to_numpy(dtype=np.float64)


def build_stage10_matrix(
    public_frame: pd.DataFrame,
    traces: np.ndarray,
    absolute_samples: np.ndarray,
    model_bundle: Mapping[str, Any],
    config: Stage12Config,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    trace_mean = np.mean(traces, axis=1, keepdims=True, dtype=np.float64)
    trace_std = np.std(traces, axis=1, keepdims=True, dtype=np.float64)
    trace_std = np.maximum(trace_std, config.trace_standard_deviation_floor)
    zscore = ((traces - trace_mean) / trace_std).astype(np.float32)
    highpass = zscore - moving_average_same(zscore, config.highpass_moving_average_width)

    target_window, target_valid = extract_aligned_windows(
        zscore,
        absolute_samples,
        public_frame["nominal_target_center_sample"].to_numpy(float),
        config.target_window_radius_samples,
    )
    pulse_window, pulse_valid = extract_aligned_windows(
        zscore,
        absolute_samples,
        public_frame["first_pulse_nominal_sample"].to_numpy(float),
        config.pulse_window_radius_samples,
    )

    response_observed, _ = _bool_observed(public_frame["response_received"])
    equal_observed, equal_missing = _bool_observed(public_frame["ciphertext_equal"])
    ciphertext_hd = pd.to_numeric(
        public_frame["ciphertext_hamming_distance"], errors="coerce"
    )
    hd_missing = ciphertext_hd.isna().to_numpy(float)
    hd_observed = ciphertext_hd.fillna(-1.0).to_numpy(float)

    width = public_frame["width_samples"].to_numpy(float)
    strength = public_frame["strength"].to_numpy(float)
    repeat = public_frame["repeat"].to_numpy(float)
    spacing = public_frame["repeat_spacing_samples"].to_numpy(float)
    offset = public_frame["timing_offset_samples"].to_numpy(float)
    pulse_span = width + np.maximum(repeat - 1.0, 0.0) * spacing
    glitch_energy = width * strength * repeat

    target_energy = public_frame["target_window_energy"].to_numpy(float)
    pulse_energy = public_frame["pulse_window_energy"].to_numpy(float)
    trace_std_feature = public_frame["trace_std"].to_numpy(float)
    hf_energy = public_frame["trace_high_frequency_energy"].to_numpy(float)

    tabular = pd.DataFrame({
        "target_is_s5": (public_frame["target_sbox_index"].to_numpy(int) == 5).astype(float),
        "timing_offset_samples": offset,
        "absolute_timing_offset_samples": np.abs(offset),
        "width_samples": width,
        "strength": strength,
        "repeat": repeat,
        "repeat_spacing_samples": spacing,
        "pulse_span_samples": pulse_span,
        "glitch_energy_proxy": glitch_energy,
        "response_received_numeric": response_observed,
        "ciphertext_equal_observed": equal_observed,
        "ciphertext_equal_missing": equal_missing,
        "ciphertext_hamming_distance_observed": hd_observed,
        "ciphertext_hamming_distance_missing": hd_missing,
        "trace_mean": public_frame["trace_mean"].to_numpy(float),
        "trace_std": trace_std_feature,
        "trace_peak_to_peak": public_frame["trace_peak_to_peak"].to_numpy(float),
        "trace_max_absolute": public_frame["trace_max_absolute"].to_numpy(float),
        "trace_high_frequency_energy": hf_energy,
        "target_window_energy": target_energy,
        "pulse_window_energy": pulse_energy,
        "saturation_fraction": public_frame["saturation_fraction"].to_numpy(float),
        "pulse_to_target_energy_ratio": pulse_energy / np.maximum(target_energy, 1.0e-9),
        "high_frequency_to_variance_ratio": hf_energy / np.maximum(trace_std_feature ** 2, 1.0e-9),
        "target_window_valid_fraction": target_valid,
        "pulse_window_valid_fraction": pulse_valid,
    })

    matrix = np.concatenate([
        tabular.loc[:, PRIMARY_TABULAR_FEATURES].to_numpy(np.float32),
        highpass.astype(np.float32),
        target_window.astype(np.float32),
        pulse_window.astype(np.float32),
    ], axis=1)

    expected_names = list(model_bundle["feature_names"])
    generated_names = list(PRIMARY_TABULAR_FEATURES)
    generated_names += [f"full_trace_highpass_{i:03d}" for i in range(highpass.shape[1])]
    generated_names += [f"target_aligned_zscore_{i:03d}" for i in range(target_window.shape[1])]
    generated_names += [f"pulse_aligned_zscore_{i:03d}" for i in range(pulse_window.shape[1])]
    if generated_names != expected_names:
        raise RuntimeError("Stage-10 feature-name/order contract mismatch")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite values in Stage-10 classifier matrix")

    return matrix, {
        "full_trace_highpass": highpass,
        "target_aligned_zscore": target_window,
        "pulse_aligned_zscore": pulse_window,
    }


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1.0e-9, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def score_public_batch(
    public_frame: pd.DataFrame,
    traces: np.ndarray,
    absolute_samples: np.ndarray,
    model_bundle: Mapping[str, Any],
    config: Stage12Config,
) -> pd.DataFrame:
    matrix, _ = build_stage10_matrix(
        public_frame,
        traces,
        absolute_samples,
        model_bundle,
        config,
    )
    raw = model_bundle["estimator"].predict_proba(matrix)
    calibrated = temperature_scale(raw, float(model_bundle["temperature"]))
    class_names = list(model_bundle["class_names"])
    output = pd.DataFrame({"experiment_id": public_frame["experiment_id"].to_numpy(int)})
    for index, name in enumerate(class_names):
        output[f"p_{name}"] = calibrated[:, index]
    output["predicted_category"] = [class_names[i] for i in np.argmax(calibrated, axis=1)]
    output["prediction_confidence"] = np.max(calibrated, axis=1)
    output["p_clean_target"] = (
        output["p_clean_target_ineffective"] + output["p_clean_target_effective"]
    )
    output["p_attack_unusable"] = (
        output["p_missed"] + output["p_off_target"]
        + output["p_multi_hit"] + output["p_invalid_reset"]
    )
    return output


def objective_public_score(frame: pd.DataFrame) -> np.ndarray:
    objective = frame["objective"].astype(str).to_numpy()
    p_i = frame["p_clean_target_ineffective"].to_numpy(float)
    p_e = frame["p_clean_target_effective"].to_numpy(float)
    prior_i = 81.0 / 256.0
    prior_e = 175.0 / 256.0
    a = p_i / prior_i
    b = p_e / prior_e
    shfa = 2.0 * a * b / np.maximum(a + b, 1.0e-12)
    score = np.where(objective == "SIFA", p_i, np.where(objective == "SEFA", p_e, shfa))
    return np.clip(score, 0.0, 1.0)


# ============================================================
# 4. Closed-loop recommendation selection and campaign planning
# ============================================================


def recommendation_utility_column(objective: str) -> str:
    return f"robust_utility_{objective}"


def initialize_arm_statistics(recommendations: pd.DataFrame, config: Stage12Config) -> Dict[Tuple[str, str, str, int], Dict[str, float]]:
    stats: Dict[Tuple[str, str, str, int], Dict[str, float]] = {}
    for _, row in recommendations.iterrows():
        mode = str(row["recommendation_mode"])
        target = str(row["target_sbox"])
        objective = str(row["objective"])
        candidate = int(row["candidate_id"])
        prior_mean = float(np.clip(row[recommendation_utility_column(objective)], 0.0, 1.0))
        stats[(mode, target, objective, candidate)] = {
            "prior_mean": prior_mean,
            "prior_strength": float(config.prior_strength),
            "count": 0.0,
            "score_sum": 0.0,
        }
    return stats


def softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = (scores - np.max(scores)) / max(float(temperature), 1.0e-6)
    weights = np.exp(np.clip(scaled, -60.0, 60.0))
    return weights / np.sum(weights)


def choose_recommendation(
    subset: pd.DataFrame,
    stats: Mapping[Tuple[str, str, str, int], Mapping[str, float]],
    batch_index: int,
    mode: str,
    rng: np.random.Generator,
    config: Stage12Config,
) -> pd.Series:
    values: List[float] = []
    total_observations = 1.0 + sum(
        item["count"] for key, item in stats.items() if key[0] == mode
    )
    for _, row in subset.iterrows():
        key = (mode, str(row["target_sbox"]), str(row["objective"]), int(row["candidate_id"]))
        item = stats[key]
        numerator = item["prior_strength"] * item["prior_mean"] + item["score_sum"]
        denominator = item["prior_strength"] + item["count"]
        posterior = numerator / max(denominator, 1.0e-12)
        coefficient = (
            config.exploit_ucb_coefficient
            if mode == "exploit"
            else config.explore_ucb_coefficient
        )
        bonus = coefficient * math.sqrt(
            math.log(total_observations + 1.0) / (item["count"] + 1.0)
        )
        disagreement = float(row.get("model_disagreement", 0.0))
        value = posterior + bonus
        if mode == "explore":
            value += config.disagreement_weight * disagreement
        # Batch zero still respects Stage-11 ordering but has no feedback.
        values.append(float(value))
    probabilities = softmax_weights(np.asarray(values), config.selection_softmax_temperature)
    index = int(rng.choice(len(subset), p=probabilities))
    return subset.iloc[index]


def perturb_recommendation(
    row: Mapping[str, Any],
    mode: str,
    bounds: Mapping[str, Any],
    center: float,
    rng: np.random.Generator,
    config: Stage12Config,
) -> engine.GlitchParameters:
    if mode == "exploit":
        offset_sigma = config.exploit_offset_jitter_sigma
        relative = config.exploit_relative_parameter_jitter
    else:
        offset_sigma = config.explore_offset_jitter_sigma
        relative = config.explore_relative_parameter_jitter

    offset = float(row["timing_offset_samples"]) + float(rng.normal(0.0, offset_sigma))
    width = float(row["width_samples"]) * float(rng.lognormal(0.0, relative))
    strength = float(row["strength"]) * float(rng.lognormal(0.0, relative))
    repeat = int(row["repeat"])
    spacing = float(row["repeat_spacing_samples"]) * float(rng.lognormal(0.0, relative))

    offset = float(np.clip(offset, bounds["timing_offset_samples"]["lower"], bounds["timing_offset_samples"]["upper"]))
    width = float(np.clip(width, bounds["width_samples"]["lower"], bounds["width_samples"]["upper"]))
    strength = float(np.clip(strength, bounds["strength"]["lower"], bounds["strength"]["upper"]))
    spacing = float(np.clip(spacing, bounds["repeat_spacing_samples"]["lower"], bounds["repeat_spacing_samples"]["upper"]))
    repeat = int(np.clip(repeat, min(bounds["repeat"]["allowed"]), max(bounds["repeat"]["allowed"])))

    return engine.GlitchParameters(
        target_sbox_index=int(row["target_sbox"][-1]),
        nominal_target_center_sample=float(center),
        offset_samples=offset,
        width_samples=width,
        strength=strength,
        repeat=repeat,
        repeat_spacing_samples=spacing,
        sampling_regime=f"guided_{mode}",
        fault_model="random_and_4",
    )


def objective_target_multiset(total: int, config: Stage12Config) -> List[Tuple[str, int]]:
    objective_counts = counts_from_fractions(total, {
        "SIFA": config.sifa_objective_fraction,
        "SEFA": config.sefa_objective_fraction,
        "SHFA": config.shfa_objective_fraction,
    })
    values: List[Tuple[str, int]] = []
    for objective, count in objective_counts.items():
        target_counts = counts_from_fractions(count, {"S0": 0.5, "S5": 0.5})
        values.extend([(objective, 0)] * target_counts["S0"])
        values.extend([(objective, 5)] * target_counts["S5"])
    return values


def build_batch_skeleton(batch_index: int, config: Stage12Config, rng: np.random.Generator) -> List[Dict[str, Any]]:
    arm_counts = counts_from_fractions(config.experiments_per_batch, {
        "guided_exploit": config.guided_exploit_fraction,
        "guided_explore": config.guided_explore_fraction,
        "baseline_random": config.randomized_baseline_fraction,
        "safety_control": config.safety_control_fraction,
    })
    rows: List[Dict[str, Any]] = []
    for arm in ("guided_exploit", "guided_explore", "baseline_random"):
        combinations = objective_target_multiset(arm_counts[arm], config)
        rng.shuffle(combinations)
        for objective, target_index in combinations:
            rows.append({
                "campaign_arm": arm,
                "objective": objective,
                "target_sbox_index": target_index,
                "safety_regime": "",
            })

    safety_total = arm_counts["safety_control"]
    safety_counts = counts_from_fractions(safety_total, {name: 0.25 for name in SAFETY_REGIMES})
    for regime, count in safety_counts.items():
        target_counts = counts_from_fractions(count, {"S0": 0.5, "S5": 0.5})
        for target_name, target_count in target_counts.items():
            target_index = 0 if target_name == "S0" else 5
            for _ in range(target_count):
                rows.append({
                    "campaign_arm": "safety_control",
                    "objective": "CONTROL",
                    "target_sbox_index": target_index,
                    "safety_regime": regime,
                })

    if len(rows) != config.experiments_per_batch:
        raise AssertionError("Batch skeleton count mismatch")
    rng.shuffle(rows)

    keys = [2 * batch_index, 2 * batch_index + 1]
    key_sessions = [(key, session) for key in keys for session in range(config.number_of_sessions)]
    per_key_session = config.experiments_per_batch // len(key_sessions)
    assignments = [pair for pair in key_sessions for _ in range(per_key_session)]
    rng.shuffle(assignments)
    for row, (key_id, session_id) in zip(rows, assignments):
        row["key_id"] = key_id
        row["session_id"] = session_id
    return rows


def update_arm_statistics(
    batch_frame: pd.DataFrame,
    statistics: MutableMapping[Tuple[str, str, str, int], MutableMapping[str, float]],
) -> None:
    guided = batch_frame[batch_frame["campaign_arm"].isin(GUIDED_ARMS)].copy()
    for _, row in guided.iterrows():
        mode = "exploit" if row["campaign_arm"] == "guided_exploit" else "explore"
        key = (
            mode,
            str(row["target_sbox"]),
            str(row["objective"]),
            int(row["recommendation_candidate_id"]),
        )
        statistics[key]["count"] += 1.0
        statistics[key]["score_sum"] += float(row["public_objective_score"])


# ============================================================
# 5. Custom experiment execution using the unchanged Stage-08 engine
# ============================================================


def build_new_key_pool(config: Stage12Config) -> List[int]:
    rng = np.random.default_rng(config.random_seed + 12001)
    keys: List[int] = []
    while len(keys) < config.number_of_keys:
        candidate = engine.random_80bit_integer(rng)
        if candidate not in keys:
            keys.append(candidate)
    return keys


def simulator_config_proxy(config: Stage12Config) -> Any:
    class Proxy:
        pass
    proxy = Proxy()
    for name in (
        "response_trace_noise_sigma",
        "response_trace_baseline_sigma",
        "response_trace_gain_sigma",
    ):
        setattr(proxy, name, getattr(config, name))
    return proxy


def run_custom_experiment(
    plan: ClosedLoopPlanEntry,
    parameters: engine.GlitchParameters,
    timing_map: Mapping[str, Any],
    healthy_source: Mapping[str, np.ndarray],
    key_pool: Sequence[int],
    session_shifts: np.ndarray,
    config: Stage12Config,
) -> engine.ExperimentRecord:
    rng = np.random.default_rng(np.random.SeedSequence([
        config.random_seed,
        int(plan.experiment_id),
        12006,
    ]))
    target_index = int(plan.target_sbox_index)
    centers = np.asarray([int(item["center_sample"]) for item in timing_map["sboxes"]], dtype=float)
    core_widths = np.asarray([
        int(item["core_window_end_sample_exclusive"])
        - int(item["core_window_start_sample_inclusive"])
        for item in timing_map["sboxes"]
    ], dtype=float)

    global_jitter = float(rng.normal(0.0, config.global_timing_jitter_sigma_samples))
    local_jitter = rng.normal(0.0, config.local_sbox_jitter_sigma_samples, size=8)
    actual_centers = centers + float(session_shifts[plan.session_id]) + global_jitter + local_jitter
    injection_jitter = float(rng.normal(0.0, config.injection_timing_jitter_sigma_samples))
    pulse_values = engine.pulse_centers(parameters, injection_jitter)
    hit_scores = engine.compute_hit_scores(parameters, pulse_values, actual_centers, core_widths)
    activation = engine.activation_probabilities(parameters, hit_scores)
    p_invalid = engine.invalid_probability(parameters, hit_scores)
    invalid = bool(rng.random() < p_invalid)

    master_key = int(key_pool[plan.key_id])
    plaintext = engine.random_64bit_integer(rng)
    context = engine.final_round_context(plaintext, master_key)
    healthy_ciphertext = int(context["ciphertext"])
    original_inputs = np.asarray(context["sbox_inputs"], dtype=np.uint8)
    faulted_inputs = original_inputs.copy()
    impacted_mask = np.zeros(8, dtype=np.uint8)
    model_details: Dict[str, Any] = {}

    if not invalid:
        activated = rng.random(8) < activation
        impacted_mask[:] = activated.astype(np.uint8)
        for sbox_index in np.where(activated)[0]:
            value, details = engine.apply_fault_model(
                int(original_inputs[sbox_index]), parameters.fault_model, rng
            )
            faulted_inputs[sbox_index] = int(value)
            model_details[f"S{int(sbox_index)}"] = details
        faulty_ciphertext = engine.final_round_with_faulted_inputs(
            int(context["x31"]), int(context["x32"]), faulted_inputs
        )
        response_received = True
        ciphertext_equal = bool(faulty_ciphertext == healthy_ciphertext)
        ciphertext_hd = engine.hamming_distance(faulty_ciphertext, healthy_ciphertext)
        invalid_subtype = ""
    else:
        faulty_ciphertext = None
        response_received = False
        ciphertext_equal = False
        ciphertext_hd = -1
        invalid_subtype = "reset" if rng.random() < 0.55 else "timeout"

    category = engine.classify_fault_event(target_index, impacted_mask, invalid, ciphertext_equal)
    base_index = int(rng.integers(0, healthy_source["traces"].shape[0]))
    response_trace = engine.synthesize_response_trace(
        healthy_source["traces"][base_index],
        healthy_source["absolute_samples"],
        pulse_values,
        parameters,
        actual_centers,
        original_inputs,
        faulted_inputs,
        impacted_mask,
        invalid,
        rng,
        simulator_config_proxy(config),
    )
    features = engine.trace_features(
        response_trace,
        healthy_source["absolute_samples"],
        float(centers[target_index]),
        pulse_values,
    )

    impacted_indices = [int(value) for value in np.where(impacted_mask > 0)[0]]
    public_row: Dict[str, Any] = {
        "experiment_id": int(plan.experiment_id),
        "campaign_partition": "closed_loop",
        "batch_index": int(plan.batch_index),
        "batch_role": plan.batch_role,
        "campaign_arm": plan.campaign_arm,
        "objective": plan.objective,
        "recommendation_mode": plan.recommendation_mode,
        "recommendation_candidate_id": int(plan.recommendation_candidate_id),
        "recommendation_rank": int(plan.recommendation_rank),
        "safety_regime": plan.safety_regime,
        "target_sbox": f"S{target_index}",
        "target_sbox_index": target_index,
        "key_id": int(plan.key_id),
        "session_id": int(plan.session_id),
        "source_healthy_trace_id": int(healthy_source["trace_ids"][base_index]),
        "fault_model": parameters.fault_model,
        "nominal_target_center_sample": parameters.nominal_target_center_sample,
        "timing_offset_samples": parameters.offset_samples,
        "first_pulse_nominal_sample": parameters.nominal_target_center_sample + parameters.offset_samples,
        "width_samples": parameters.width_samples,
        "strength": parameters.strength,
        "repeat": parameters.repeat,
        "repeat_spacing_samples": parameters.repeat_spacing_samples,
        "plaintext_hex": engine.hex_fixed(plaintext, 64),
        "healthy_ciphertext_hex": engine.hex_fixed(healthy_ciphertext, 64),
        "response_received": response_received,
        "faulty_ciphertext_hex": engine.hex_fixed(int(faulty_ciphertext), 64) if faulty_ciphertext is not None else "",
        "ciphertext_equal": ciphertext_equal if response_received else "",
        "ciphertext_hamming_distance": ciphertext_hd if response_received else np.nan,
        **features,
    }

    private_row: Dict[str, Any] = {
        "experiment_id": int(plan.experiment_id),
        "batch_index": int(plan.batch_index),
        "batch_role": plan.batch_role,
        "campaign_arm": plan.campaign_arm,
        "objective": plan.objective,
        "category": category,
        "category_id": CLASS_TO_ID[category],
        "target_sbox": f"S{target_index}",
        "target_sbox_index": target_index,
        "key_id": int(plan.key_id),
        "session_id": int(plan.session_id),
        "target_original_input": int(original_inputs[target_index]),
        "target_faulted_input": int(faulted_inputs[target_index]),
        "impacted_sboxes": ";".join(f"S{value}" for value in impacted_indices),
        "impacted_sbox_count": len(impacted_indices),
        "target_impacted": bool(impacted_mask[target_index]),
        "off_target_impacted": any(value != target_index for value in impacted_indices),
        "changed_sbox_input_count": int(np.sum(original_inputs != faulted_inputs)),
        "fault_effective": bool(response_received and not ciphertext_equal),
        "invalid_subtype": invalid_subtype,
        "invalid_probability": p_invalid,
        "global_jitter_samples": global_jitter,
        "injection_jitter_samples": injection_jitter,
        "model_details_json": json.dumps(model_details, sort_keys=True),
    }
    private_arrays = {
        "round_key_32": np.asarray(context["round_key_32"], dtype=np.uint32),
        "x31": np.asarray(context["x31"], dtype=np.uint32),
        "x32": np.asarray(context["x32"], dtype=np.uint32),
        "original_inputs": original_inputs.astype(np.uint8),
        "faulted_inputs": faulted_inputs.astype(np.uint8),
        "hit_scores": hit_scores.astype(np.float32),
        "activation_probabilities": activation.astype(np.float32),
        "actual_centers": actual_centers.astype(np.float32),
        "impacted_mask": impacted_mask.astype(np.uint8),
        "category_id": np.asarray(CLASS_TO_ID[category], dtype=np.int8),
    }
    return engine.ExperimentRecord(
        public_row=public_row,
        private_row=private_row,
        response_trace=response_trace.astype(np.float32),
        private_arrays=private_arrays,
    )


# ============================================================
# 6. Private-after-freeze evaluation
# ============================================================


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(mask):
            result += np.mean(mask) * abs(np.mean(confidence[mask]) - np.mean(correct[mask]))
    return float(result)


def classifier_metrics(labels: pd.DataFrame, probabilities: pd.DataFrame, config: Stage12Config) -> Dict[str, Any]:
    merged = labels[["experiment_id", "category_id", "category"]].merge(
        probabilities, on="experiment_id", validate="one_to_one"
    )
    y_true = merged["category_id"].to_numpy(int)
    matrix = merged[[f"p_{name}" for name in CLASS_NAMES]].to_numpy(float)
    y_pred = np.argmax(matrix, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    metrics = {
        "number_of_rows": int(len(merged)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_true, matrix, labels=np.arange(len(CLASS_NAMES)))),
        "multiclass_brier_score": float(np.mean(np.sum((matrix - np.eye(len(CLASS_NAMES))[y_true]) ** 2, axis=1))),
        "expected_calibration_error": expected_calibration_error(y_true, matrix, config.ece_bins),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
    }
    branch = {}
    for name in ("clean_target_ineffective", "clean_target_effective"):
        index = CLASS_TO_ID[name]
        binary = (y_true == index).astype(int)
        score = matrix[:, index]
        branch[name] = {
            "support": int(binary.sum()),
            "prevalence": float(binary.mean()),
            "average_precision": float(average_precision_score(binary, score)),
            "roc_auc": float(roc_auc_score(binary, score)),
        }
    metrics["branch_metrics"] = branch
    return metrics


def empirical_objective_value(frame: pd.DataFrame, objective: str) -> float:
    if len(frame) == 0:
        return float("nan")
    categories = frame["category"].astype(str)
    rate_i = float(np.mean(categories == "clean_target_ineffective"))
    rate_e = float(np.mean(categories == "clean_target_effective"))
    if objective == "SIFA":
        return rate_i
    if objective == "SEFA":
        return rate_e
    a = rate_i / (81.0 / 256.0)
    b = rate_e / (175.0 / 256.0)
    return float(2.0 * a * b / max(a + b, 1.0e-12))


def clustered_uplift_bootstrap(
    guided: pd.DataFrame,
    baseline: pd.DataFrame,
    objective: str,
    repetitions: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Cluster bootstrap using pre-aggregated key/session counts.

    The first implementation resampled pandas frames directly.  This
    equivalent count-based implementation is much faster for the full 24k
    campaign and preserves key/session clustering.
    """
    guided_value = empirical_objective_value(guided, objective)
    baseline_value = empirical_objective_value(baseline, objective)
    difference = guided_value - baseline_value
    ratio = guided_value / max(baseline_value, 1.0e-12)

    cluster_keys = sorted(set(
        list(zip(guided["key_id"].astype(int), guided["session_id"].astype(int)))
        + list(zip(baseline["key_id"].astype(int), baseline["session_id"].astype(int)))
    ))

    def aggregate(frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for key_id, session_id in cluster_keys:
            part = frame[
                (frame["key_id"].to_numpy(int) == key_id)
                & (frame["session_id"].to_numpy(int) == session_id)
            ]
            categories = part["category"].astype(str).to_numpy()
            rows.append([
                len(part),
                int(np.sum(categories == "clean_target_ineffective")),
                int(np.sum(categories == "clean_target_effective")),
            ])
        return np.asarray(rows, dtype=np.float64)

    g_counts = aggregate(guided)
    b_counts = aggregate(baseline)

    def value_from_counts(counts: np.ndarray) -> float:
        total = float(np.sum(counts[:, 0]))
        if total <= 0.0:
            return float("nan")
        rate_i = float(np.sum(counts[:, 1]) / total)
        rate_e = float(np.sum(counts[:, 2]) / total)
        if objective == "SIFA":
            return rate_i
        if objective == "SEFA":
            return rate_e
        a = rate_i / (81.0 / 256.0)
        b = rate_e / (175.0 / 256.0)
        return float(2.0 * a * b / max(a + b, 1.0e-12))

    differences = np.empty(repetitions, dtype=np.float64)
    ratios = np.empty(repetitions, dtype=np.float64)
    cluster_count = len(cluster_keys)
    if cluster_count:
        for repetition in range(repetitions):
            sampled = rng.integers(0, cluster_count, size=cluster_count)
            gv = value_from_counts(g_counts[sampled])
            bv = value_from_counts(b_counts[sampled])
            differences[repetition] = gv - bv
            ratios[repetition] = gv / max(bv, 1.0e-12)
    else:
        differences[:] = np.nan
        ratios[:] = np.nan

    finite_differences = differences[np.isfinite(differences)]
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "guided_count": int(len(guided)),
        "baseline_count": int(len(baseline)),
        "guided_value": float(guided_value),
        "baseline_value": float(baseline_value),
        "absolute_lift": float(difference),
        "uplift_ratio": float(ratio),
        "absolute_lift_ci95_low": float(np.quantile(finite_differences, 0.025)) if finite_differences.size else float("nan"),
        "absolute_lift_ci95_high": float(np.quantile(finite_differences, 0.975)) if finite_differences.size else float("nan"),
        "uplift_ratio_ci95_low": float(np.quantile(finite_ratios, 0.025)) if finite_ratios.size else float("nan"),
        "uplift_ratio_ci95_high": float(np.quantile(finite_ratios, 0.975)) if finite_ratios.size else float("nan"),
    }


def build_uplift_table(
    merged: pd.DataFrame,
    batch_filter: Iterable[int],
    config: Stage12Config,
    seed_offset: int,
) -> pd.DataFrame:
    subset = merged[merged["batch_index"].isin(list(batch_filter))].copy()
    rows: List[Dict[str, Any]] = []
    for target in TARGETS:
        for objective in OBJECTIVES:
            guided = subset[
                (subset["target_sbox"] == target)
                & (subset["objective"] == objective)
                & (subset["campaign_arm"] == "guided_exploit")
            ]
            baseline = subset[
                (subset["target_sbox"] == target)
                & (subset["objective"] == objective)
                & (subset["campaign_arm"] == "baseline_random")
            ]
            result = clustered_uplift_bootstrap(
                guided,
                baseline,
                objective,
                config.bootstrap_repetitions,
                np.random.default_rng(config.random_seed + seed_offset + len(rows)),
            )
            rows.append({"target_sbox": target, "objective": objective, **result})
    return pd.DataFrame(rows)


# ============================================================
# 7. Plots
# ============================================================


def save_public_plots(public_dir: Path, batch_summary: pd.DataFrame, enabled: bool) -> List[str]:
    if not enabled or plt is None:
        return []
    files: List[str] = []
    figure, axis = plt.subplots(figsize=(10, 5))
    for arm in ("guided_exploit", "guided_explore", "baseline_random"):
        group = batch_summary[batch_summary["campaign_arm"] == arm]
        if len(group):
            axis.plot(group["batch_index"], group["mean_public_objective_score"], marker="o", label=arm)
    axis.set_xlabel("Batch index")
    axis.set_ylabel("Mean frozen Stage-10 objective score")
    axis.set_title("Public closed-loop feedback by batch")
    axis.legend()
    axis.grid(alpha=0.25)
    path = public_dir / "public_closed_loop_feedback_by_batch.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    files.append(path.name)
    return files


def save_validation_plots(validation_dir: Path, confirmation: pd.DataFrame, metrics: Mapping[str, Any], enabled: bool) -> List[str]:
    if not enabled or plt is None:
        return []
    files: List[str] = []
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [f"{row.target_sbox}-{row.objective}" for row in confirmation.itertuples()]
    x = np.arange(len(labels))
    axis.bar(x, confirmation["uplift_ratio"].to_numpy(float))
    axis.axhline(1.0, linewidth=1.2)
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Guided / randomized baseline")
    axis.set_title("Confirmation-batch empirical uplift")
    path = validation_dir / "confirmation_empirical_uplift.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    files.append(path.name)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix)
    axis.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Stage-10 classifier on fresh Stage-12 campaign")
    figure.colorbar(image, ax=axis)
    path = validation_dir / "fresh_campaign_classifier_confusion_matrix.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    files.append(path.name)
    return files


# ============================================================
# 8. Main execution
# ============================================================


def run_stage_12(config: Stage12Config) -> Dict[str, Any]:
    started = time.perf_counter()
    validate_config(config)
    stage11_dir = Path(config.input_stage11_run_directory).expanduser().resolve()
    contracts = resolve_stage_contracts(stage11_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage12_{timestamp}_seed{config.random_seed}"
    run_dir = Path(config.output_root).expanduser().resolve() / run_id
    public_dir = run_dir / "public_closed_loop"
    private_dir = run_dir / "private_ground_truth"
    locked_dir = run_dir / "locked_attack_truth"
    validation_dir = run_dir / "validation_only"
    for directory in (public_dir, private_dir, locked_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exploit = contracts["exploit_recommendations"].copy()
    explore = contracts["explore_recommendations"].copy()
    recommendations = pd.concat([exploit, explore], ignore_index=True)
    required_recommendation_groups = {
        (mode, target, objective)
        for mode in ("exploit", "explore")
        for target in TARGETS
        for objective in OBJECTIVES
    }
    observed_groups = set(zip(
        recommendations["recommendation_mode"].astype(str),
        recommendations["target_sbox"].astype(str),
        recommendations["objective"].astype(str),
    ))
    if not required_recommendation_groups.issubset(observed_groups):
        raise RuntimeError("Stage 11 recommendation groups are incomplete")

    stats = initialize_arm_statistics(recommendations, config)
    key_pool = build_new_key_pool(config)
    session_rng = np.random.default_rng(config.random_seed + 12002)
    session_shifts = session_rng.normal(
        0.0, config.session_timing_shift_sigma_samples, size=config.number_of_sessions
    )
    centers = np.asarray([
        int(item["center_sample"]) for item in contracts["timing_map"]["sboxes"]
    ], dtype=float)
    bounds_by_target = contracts["stage11_summary"]["candidate_pool_summary"]

    public_rows: List[Dict[str, Any]] = []
    private_rows: List[Dict[str, Any]] = []
    traces: List[np.ndarray] = []
    private_array_lists: Dict[str, List[np.ndarray]] = {}
    probability_frames: List[pd.DataFrame] = []
    selection_log_rows: List[Dict[str, Any]] = []
    global_experiment_id = 0
    private_feedback_accessed = False
    confirmation_policy_frozen = False

    for batch_index in range(config.number_of_batches):
        batch_role = "confirmation" if batch_index == config.confirmation_batch_index else "adaptive"
        if batch_role == "confirmation":
            confirmation_policy_frozen = True

        batch_rng = np.random.default_rng(np.random.SeedSequence([
            config.random_seed, 12010, batch_index
        ]))
        skeleton = build_batch_skeleton(batch_index, config, batch_rng)
        batch_records: List[engine.ExperimentRecord] = []
        batch_plans: List[ClosedLoopPlanEntry] = []

        for item in skeleton:
            arm = str(item["campaign_arm"])
            target_index = int(item["target_sbox_index"])
            target_name = f"S{target_index}"
            objective = str(item["objective"])
            recommendation_mode = ""
            candidate_id = -1
            recommendation_rank = -1

            if arm in GUIDED_ARMS:
                recommendation_mode = "exploit" if arm == "guided_exploit" else "explore"
                subset = recommendations[
                    (recommendations["recommendation_mode"] == recommendation_mode)
                    & (recommendations["target_sbox"] == target_name)
                    & (recommendations["objective"] == objective)
                ].reset_index(drop=True)
                recommendation = choose_recommendation(
                    subset, stats, batch_index, recommendation_mode, batch_rng, config
                )
                candidate_id = int(recommendation["candidate_id"])
                recommendation_rank = int(recommendation["rank"])
                parameters = perturb_recommendation(
                    recommendation,
                    recommendation_mode,
                    bounds_by_target[target_name]["parameter_bounds"],
                    centers[target_index],
                    batch_rng,
                    config,
                )
            else:
                if arm == "baseline_random":
                    regime = "attack_core" if batch_rng.random() < 0.70 else "attack_explore"
                else:
                    regime = str(item["safety_regime"])
                temporary = engine.CampaignPlanEntry(
                    experiment_id=global_experiment_id,
                    campaign_partition="closed_loop",
                    key_id=int(item["key_id"]),
                    session_id=int(item["session_id"]),
                    target_sbox_index=target_index,
                    fault_model="random_and_4",
                    design_regime=regime,
                )
                parameters = engine.sample_stage08_glitch_parameters(
                    temporary,
                    contracts["target_contract"],
                    centers,
                    batch_rng,
                )

            plan = ClosedLoopPlanEntry(
                experiment_id=global_experiment_id,
                batch_index=batch_index,
                batch_role=batch_role,
                key_id=int(item["key_id"]),
                session_id=int(item["session_id"]),
                target_sbox_index=target_index,
                objective=objective,
                campaign_arm=arm,
                recommendation_mode=recommendation_mode,
                recommendation_candidate_id=candidate_id,
                recommendation_rank=recommendation_rank,
                safety_regime=str(item["safety_regime"]),
            )
            record = run_custom_experiment(
                plan,
                parameters,
                contracts["timing_map"],
                contracts["healthy_source"],
                key_pool,
                session_shifts,
                config,
            )
            batch_records.append(record)
            batch_plans.append(plan)
            global_experiment_id += 1

        batch_public = pd.DataFrame([record.public_row for record in batch_records])
        batch_traces = np.stack([record.response_trace for record in batch_records]).astype(np.float32)
        batch_probabilities = score_public_batch(
            batch_public,
            batch_traces,
            contracts["healthy_source"]["absolute_samples"],
            contracts["stage10_model"],
            config,
        )
        batch_feedback = batch_public.merge(batch_probabilities, on="experiment_id", validate="one_to_one")
        batch_feedback["public_objective_score"] = objective_public_score(batch_feedback)

        # Adapt only after an adaptive batch.  The confirmation policy is frozen
        # before the final keys are generated and is never updated from them.
        if batch_role == "adaptive":
            update_arm_statistics(batch_feedback, stats)

        guided_feedback = batch_feedback[batch_feedback["campaign_arm"].isin(GUIDED_ARMS)]
        for group, group_frame in guided_feedback.groupby([
            "batch_index", "campaign_arm", "target_sbox", "objective",
            "recommendation_candidate_id", "recommendation_rank"
        ]):
            selection_log_rows.append({
                "batch_index": int(group[0]),
                "campaign_arm": str(group[1]),
                "target_sbox": str(group[2]),
                "objective": str(group[3]),
                "recommendation_candidate_id": int(group[4]),
                "recommendation_rank": int(group[5]),
                "selection_count": int(len(group_frame)),
                "mean_public_objective_score": float(group_frame["public_objective_score"].mean()),
                "mean_p_clean_target_ineffective": float(group_frame["p_clean_target_ineffective"].mean()),
                "mean_p_clean_target_effective": float(group_frame["p_clean_target_effective"].mean()),
            })

        public_rows.extend(record.public_row for record in batch_records)
        private_rows.extend(record.private_row for record in batch_records)
        traces.extend(record.response_trace for record in batch_records)
        for record in batch_records:
            for name, value in record.private_arrays.items():
                private_array_lists.setdefault(name, []).append(np.asarray(value))
        probability_frames.append(batch_probabilities)

    public_frame = pd.DataFrame(public_rows).sort_values("experiment_id").reset_index(drop=True)
    private_frame = pd.DataFrame(private_rows).sort_values("experiment_id").reset_index(drop=True)
    trace_matrix = np.stack(traces).astype(np.float32)
    probabilities = pd.concat(probability_frames, ignore_index=True).sort_values("experiment_id").reset_index(drop=True)
    public_with_scores = public_frame.merge(probabilities, on="experiment_id", validate="one_to_one")
    public_with_scores["public_objective_score"] = objective_public_score(public_with_scores)

    # Determinism check: reconstruct a fixed prefix from only the stored public
    # parameters and verify both the public record and response trace exactly.
    deterministic_recheck_count = min(config.deterministic_recheck_count, len(public_frame))
    deterministic_mismatches = 0
    deterministic_fields = [
        "plaintext_hex", "healthy_ciphertext_hex", "response_received",
        "faulty_ciphertext_hex", "ciphertext_equal",
        "ciphertext_hamming_distance", "source_healthy_trace_id",
    ]
    for row_index in range(deterministic_recheck_count):
        row = public_frame.iloc[row_index]
        reconstructed_plan = ClosedLoopPlanEntry(
            experiment_id=int(row["experiment_id"]),
            batch_index=int(row["batch_index"]),
            batch_role=str(row["batch_role"]),
            key_id=int(row["key_id"]),
            session_id=int(row["session_id"]),
            target_sbox_index=int(row["target_sbox_index"]),
            objective=str(row["objective"]),
            campaign_arm=str(row["campaign_arm"]),
            recommendation_mode=str(row["recommendation_mode"]),
            recommendation_candidate_id=int(row["recommendation_candidate_id"]),
            recommendation_rank=int(row["recommendation_rank"]),
            safety_regime=str(row["safety_regime"]),
        )
        reconstructed_parameters = engine.GlitchParameters(
            target_sbox_index=int(row["target_sbox_index"]),
            nominal_target_center_sample=float(row["nominal_target_center_sample"]),
            offset_samples=float(row["timing_offset_samples"]),
            width_samples=float(row["width_samples"]),
            strength=float(row["strength"]),
            repeat=int(row["repeat"]),
            repeat_spacing_samples=float(row["repeat_spacing_samples"]),
            sampling_regime=str(row["campaign_arm"]),
            fault_model=str(row["fault_model"]),
        )
        repeated = run_custom_experiment(
            reconstructed_plan, reconstructed_parameters,
            contracts["timing_map"], contracts["healthy_source"],
            key_pool, session_shifts, config,
        )
        trace_equal = np.array_equal(repeated.response_trace, trace_matrix[row_index])
        fields_equal = True
        original_public = public_rows[row_index]
        for field in deterministic_fields:
            left = original_public[field]
            right = repeated.public_row[field]
            if isinstance(left, float) and np.isnan(left):
                if not (isinstance(right, float) and np.isnan(right)):
                    fields_equal = False
                    break
            elif left != right:
                fields_equal = False
                break
        if not (trace_equal and fields_equal):
            deterministic_mismatches += 1
    deterministic_recheck_passed = deterministic_mismatches == 0

    # Reload the frozen teacher and verify that serialization does not change
    # its probabilities on a public prefix of the new campaign.
    reloaded_teacher = joblib.load(
        contracts["stage10_dir"] / "models" / "fault_quality_deployment_model.joblib"
    )
    reload_count = min(256, len(public_frame))
    reload_probabilities = score_public_batch(
        public_frame.iloc[:reload_count].reset_index(drop=True),
        trace_matrix[:reload_count],
        contracts["healthy_source"]["absolute_samples"],
        reloaded_teacher, config,
    )
    probability_columns = [f"p_{name}" for name in CLASS_NAMES]
    original_prefix = probabilities.iloc[:reload_count][probability_columns].to_numpy(float)
    reloaded_prefix = reload_probabilities[probability_columns].to_numpy(float)
    model_reload_max_difference = float(np.max(np.abs(original_prefix - reloaded_prefix)))
    model_reload_check_passed = model_reload_max_difference <= 1.0e-12

    batch_summary = (
        public_with_scores.groupby(["batch_index", "batch_role", "campaign_arm"], as_index=False)
        .agg(
            number_of_rows=("experiment_id", "size"),
            mean_public_objective_score=("public_objective_score", "mean"),
            mean_p_clean_target_ineffective=("p_clean_target_ineffective", "mean"),
            mean_p_clean_target_effective=("p_clean_target_effective", "mean"),
            mean_p_invalid_reset=("p_invalid_reset", "mean"),
        )
    )

    campaign_contract = {
        "stage": 12,
        "design": "four-batch public-feedback closed loop with unseen-key confirmation",
        "number_of_experiments": config.number_of_experiments,
        "number_of_batches": config.number_of_batches,
        "experiments_per_batch": config.experiments_per_batch,
        "adaptation_key_ids": list(range(0, 6)),
        "confirmation_key_ids": [6, 7],
        "campaign_arm_fractions": {
            "guided_exploit": config.guided_exploit_fraction,
            "guided_explore": config.guided_explore_fraction,
            "baseline_random": config.randomized_baseline_fraction,
            "safety_control": config.safety_control_fraction,
        },
        "objective_fractions": {
            "SIFA": config.sifa_objective_fraction,
            "SEFA": config.sefa_objective_fraction,
            "SHFA": config.shfa_objective_fraction,
        },
        "feedback_source": "Frozen Stage-10 public probability outputs only",
        "private_feedback_used": False,
        "confirmation_policy_frozen_before_batch": bool(confirmation_policy_frozen),
        "primary_confirmation_comparison": "guided_exploit versus baseline_random",
        "fault_model_for_attack_arms": "random_and_4",
    }

    public_paths = {
        "campaign": public_dir / "closed_loop_campaign_public.csv",
        "traces": public_dir / "closed_loop_response_traces.npz",
        "probabilities": public_dir / "closed_loop_quality_probabilities_public.csv",
        "selection_log": public_dir / "closed_loop_selection_log_public.csv",
        "batch_summary": public_dir / "closed_loop_batch_summary_public.csv",
        "contract": public_dir / "closed_loop_campaign_contract.json",
        "attack_payload": public_dir / "confirmation_attack_payload_public.csv",
        "access_manifest": public_dir / "public_feedback_access_manifest.json",
    }
    write_csv(public_paths["campaign"], public_frame)
    np.savez_compressed(
        public_paths["traces"],
        experiment_ids=public_frame["experiment_id"].to_numpy(np.int64),
        traces=trace_matrix,
        absolute_samples=contracts["healthy_source"]["absolute_samples"].astype(np.int64),
    )
    write_csv(public_paths["probabilities"], probabilities)
    write_csv(public_paths["selection_log"], pd.DataFrame(selection_log_rows))
    write_csv(public_paths["batch_summary"], batch_summary)
    write_json(public_paths["contract"], campaign_contract)
    confirmation_payload = public_with_scores[
        public_with_scores["batch_role"] == "confirmation"
    ].copy()
    write_csv(public_paths["attack_payload"], confirmation_payload)
    write_json(public_paths["access_manifest"], {
        "private_labels_accessed_during_closed_loop": False,
        "private_key_truth_accessed_during_closed_loop": False,
        "stage09_locked_attack_labels_accessed": False,
        "stage10_attack_rows_accessed": False,
        "stage11_test_rows_accessed": False,
        "statement": "All policy updates used only public response observations and frozen Stage-10 probabilities.",
    })
    public_plot_files = save_public_plots(public_dir, batch_summary, config.save_plots)

    public_files = [path for path in public_dir.iterdir() if path.is_file()]
    public_manifest_files = {
        path.relative_to(public_dir).as_posix(): sha256_file(path)
        for path in sorted(public_files)
    }
    public_freeze = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": "Stage-12 public campaign, traces, classifier scores, and adaptive selection history were frozen before private simulator labels were opened.",
        "source_stage_11_optimizer_freeze_sha256": contracts["stage11_verify"]["freeze_sha256"],
        "source_stage_10_model_freeze_sha256": contracts["stage10_verify"]["freeze_sha256"],
        "files": public_manifest_files,
    }
    public_freeze["freeze_sha256"] = stable_json_hash(public_freeze)
    write_json(run_dir / "public_closed_loop_freeze_manifest.json", public_freeze)

    # ---------------- Private-after-freeze section ----------------
    private_feedback_accessed = True
    write_csv(private_dir / "closed_loop_ground_truth.csv", private_frame)
    stacked_private = {
        name: np.stack(values)
        for name, values in private_array_lists.items()
    }
    stacked_private["experiment_ids"] = private_frame["experiment_id"].to_numpy(np.int64)
    np.savez_compressed(private_dir / "closed_loop_ground_truth_arrays.npz", **stacked_private)

    # The confirmation keys are intentionally locked for Stages 13–15.  Stage
    # 12 evaluates categories but never uses key values in its validation.
    confirmation_truth = {
        "statement": "Do not open before the corresponding Stage-13/14/15 attack score has been frozen.",
        "confirmation_key_ids": [6, 7],
        "keys": [],
    }
    for key_id in (6, 7):
        master_key = int(key_pool[key_id])
        round_key_32 = int(engine.key_schedule_lblock(master_key)[31])
        confirmation_truth["keys"].append({
            "key_id": key_id,
            "master_key_hex": engine.hex_fixed(master_key, 80),
            "round_key_32_hex": engine.hex_fixed(round_key_32, 32),
            "K32_0": int(engine.get_nibble(round_key_32, 0)),
            "K32_5": int(engine.get_nibble(round_key_32, 5)),
        })
    locked_truth_path = locked_dir / "confirmation_key_truth_LOCKED.json"
    write_json(locked_truth_path, confirmation_truth)
    write_json(locked_dir / "DO_NOT_OPEN_BEFORE_ATTACK_FREEZE.json", {
        "locked_file": locked_truth_path.name,
        "sha256": sha256_file(locked_truth_path),
        "reason": "Prevents key-answer leakage into SIFA/SEFA/SHFA implementation and tuning.",
    })

    labels = private_frame[[
        "experiment_id", "category_id", "category", "batch_index", "batch_role",
        "campaign_arm", "objective", "target_sbox", "key_id", "session_id",
    ]].copy()
    merged = public_with_scores.merge(
        labels[["experiment_id", "category_id", "category"]],
        on="experiment_id",
        validate="one_to_one",
    )
    fresh_classifier_metrics = classifier_metrics(labels, probabilities, config)
    confirmation_uplift = build_uplift_table(
        merged, [config.confirmation_batch_index], config, 12100
    )
    cumulative_uplift = build_uplift_table(
        merged, range(config.number_of_batches), config, 12200
    )
    write_csv(validation_dir / "confirmation_empirical_uplift.csv", confirmation_uplift)
    write_csv(validation_dir / "cumulative_empirical_uplift.csv", cumulative_uplift)
    write_json(validation_dir / "fresh_campaign_classifier_metrics.json", fresh_classifier_metrics)

    category_rates = (
        merged.groupby(["batch_index", "batch_role", "campaign_arm", "target_sbox", "objective", "category"], as_index=False)
        .size()
    )
    totals = category_rates.groupby(
        ["batch_index", "batch_role", "campaign_arm", "target_sbox", "objective"]
    )["size"].transform("sum")
    category_rates["rate"] = category_rates["size"] / totals
    write_csv(validation_dir / "realized_category_rates.csv", category_rates)

    confirmation_mean_uplift = float(confirmation_uplift["uplift_ratio"].replace([np.inf, -np.inf], np.nan).mean())
    confirmation_sifa_sefa = confirmation_uplift[
        confirmation_uplift["objective"].isin(["SIFA", "SEFA"])
    ]
    confirmation_branch_mean_uplift = float(confirmation_sifa_sefa["uplift_ratio"].replace([np.inf, -np.inf], np.nan).mean())
    positive_groups = int(np.sum(confirmation_uplift["uplift_ratio"] > 1.0))
    closed_loop_success = bool(
        confirmation_branch_mean_uplift > 1.0
        and positive_groups >= 4
    )

    validation_checks = {
        "stage_09_public_freeze_verified": bool(contracts["stage9_verify"]["passed"]),
        "stage_10_model_freeze_verified": bool(contracts["stage10_verify"]["passed"]),
        "stage_11_optimizer_freeze_verified": bool(contracts["stage11_verify"]["passed"]),
        "public_feedback_only": True,
        "private_feedback_used_for_adaptation": False,
        "private_labels_opened_only_after_public_freeze": (run_dir / "public_closed_loop_freeze_manifest.json").is_file(),
        "row_count_exact": len(public_frame) == config.number_of_experiments,
        "trace_alignment_exact": trace_matrix.shape == (
            config.number_of_experiments,
            contracts["healthy_source"]["traces"].shape[1],
        ),
        "probability_row_count_exact": len(probabilities) == config.number_of_experiments,
        "maximum_probability_sum_error": float(np.max(np.abs(
            probabilities[[f"p_{name}" for name in CLASS_NAMES]].sum(axis=1).to_numpy(float) - 1.0
        ))),
        "confirmation_keys_disjoint_from_adaptation": set([6, 7]).isdisjoint(set(range(0, 6))),
        "confirmation_policy_frozen": confirmation_policy_frozen,
        "locked_confirmation_truth_created": locked_truth_path.is_file(),
        "deterministic_recheck_passed": deterministic_recheck_passed,
        "deterministic_recheck_count": deterministic_recheck_count,
        "deterministic_mismatch_count": deterministic_mismatches,
        "model_reload_check_passed": model_reload_check_passed,
        "model_reload_max_probability_difference": model_reload_max_difference,
        "closed_loop_empirical_success": closed_loop_success,
    }
    integrity_keys = [
        "stage_09_public_freeze_verified",
        "stage_10_model_freeze_verified",
        "stage_11_optimizer_freeze_verified",
        "private_labels_opened_only_after_public_freeze",
        "row_count_exact",
        "trace_alignment_exact",
        "probability_row_count_exact",
        "confirmation_keys_disjoint_from_adaptation",
        "confirmation_policy_frozen",
        "locked_confirmation_truth_created",
        "deterministic_recheck_passed",
        "model_reload_check_passed",
    ]
    all_checks_passed = all(bool(validation_checks[key]) for key in integrity_keys)
    write_json(validation_dir / "stage_12_validation_checks.json", validation_checks)
    validation_plot_files = save_validation_plots(
        validation_dir, confirmation_uplift, fresh_classifier_metrics, config.save_plots
    )

    run_manifest = {
        "stage": 12,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "config": asdict(config),
        "public_freeze_sha256": public_freeze["freeze_sha256"],
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    summary = {
        "stage": 12,
        "run_id": run_id,
        "run_directory": str(run_dir),
        "input_stage_11_run_directory": str(stage11_dir),
        "input_stage_10_run_directory": str(contracts["stage10_dir"]),
        "input_stage_09_run_directory": str(contracts["stage9_dir"]),
        "input_stage_08_run_directory": str(contracts["stage8_dir"]),
        "all_checks_passed": bool(all_checks_passed),
        "closed_loop_empirical_success": bool(closed_loop_success),
        "public_feedback_only": True,
        "private_labels_opened_after_public_freeze": True,
        "stage_09_public_freeze_verified": True,
        "stage_10_model_freeze_verified": True,
        "stage_11_optimizer_freeze_verified": True,
        "stage_10_model_freeze_sha256": contracts["stage10_verify"]["freeze_sha256"],
        "stage_11_optimizer_freeze_sha256": contracts["stage11_verify"]["freeze_sha256"],
        "number_of_experiments": int(len(public_frame)),
        "number_of_batches": config.number_of_batches,
        "experiments_per_batch": config.experiments_per_batch,
        "number_of_keys": config.number_of_keys,
        "number_of_sessions": config.number_of_sessions,
        "adaptation_key_ids": list(range(0, 6)),
        "confirmation_key_ids": [6, 7],
        "campaign_arm_counts": {
            str(key): int(value)
            for key, value in public_frame["campaign_arm"].value_counts().sort_index().items()
        },
        "objective_counts": {
            str(key): int(value)
            for key, value in public_frame[public_frame["objective"] != "CONTROL"]["objective"].value_counts().sort_index().items()
        },
        "target_counts": {
            str(key): int(value)
            for key, value in public_frame["target_sbox"].value_counts().sort_index().items()
        },
        "category_counts": {
            str(key): int(value)
            for key, value in private_frame["category"].value_counts().sort_index().items()
        },
        "fresh_campaign_classifier_metrics": {
            key: fresh_classifier_metrics[key]
            for key in (
                "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                "log_loss", "multiclass_brier_score", "expected_calibration_error",
            )
        },
        "fresh_campaign_branch_metrics": fresh_classifier_metrics["branch_metrics"],
        "confirmation_uplift_by_target_objective": confirmation_uplift.to_dict(orient="records"),
        "confirmation_mean_uplift": confirmation_mean_uplift,
        "confirmation_branch_mean_uplift": confirmation_branch_mean_uplift,
        "confirmation_positive_group_count": positive_groups,
        "public_closed_loop_freeze_sha256": public_freeze["freeze_sha256"],
        "locked_confirmation_truth_sha256": sha256_file(locked_truth_path),
        "attack_payload_row_count": int(len(confirmation_payload)),
        "deterministic_recheck_passed": deterministic_recheck_passed,
        "deterministic_recheck_count": deterministic_recheck_count,
        "model_reload_check_passed": model_reload_check_passed,
        "model_reload_max_probability_difference": model_reload_max_difference,
        "public_files": sorted(path.name for path in public_dir.iterdir() if path.is_file()),
        "private_files": sorted(path.name for path in private_dir.iterdir() if path.is_file()),
        "locked_files": sorted(path.name for path in locked_dir.iterdir() if path.is_file()),
        "validation_files": sorted(path.name for path in validation_dir.iterdir() if path.is_file()),
        "generated_plots": {
            "public": public_plot_files,
            "validation": validation_plot_files,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(run_dir / "stage_12_summary.json", summary)
    return summary


def load_stage_12_config(path: str | Path) -> Stage12Config:
    value = read_json(Path(path).expanduser().resolve())
    return Stage12Config(**value)


# ============================================================
# اجرای مستقیم در Jupyter یا Python
# ============================================================
#
# در Jupyter متغیر __file__ وجود ندارد؛ بنابراین تنظیمات را مستقیم
# در همین فایل می‌سازیم و هیچ stage_12_config.json جداگانه‌ای لازم نیست.

stage_12_config = Stage12Config(
    input_stage11_run_directory=(
        r"C:\Users\SADRA\Desktop\LBlock\runs\stage_11"
        r"\stage11_20260718_191029_290452_seed20260718"
    ),
    output_root=(
        r"C:\Users\SADRA\Desktop\LBlock\runs\stage_12"
    ),
    random_seed=20260718,
    number_of_experiments=24000,
    number_of_batches=4,
    experiments_per_batch=6000,
    number_of_keys=8,
    number_of_sessions=4,
    confirmation_batch_index=3,
    guided_exploit_fraction=0.50,
    guided_explore_fraction=0.20,
    randomized_baseline_fraction=0.20,
    safety_control_fraction=0.10,
    sifa_objective_fraction=0.35,
    sefa_objective_fraction=0.35,
    shfa_objective_fraction=0.30,
    prior_strength=8.0,
    exploit_ucb_coefficient=0.22,
    explore_ucb_coefficient=0.38,
    disagreement_weight=0.25,
    selection_softmax_temperature=0.08,
    exploit_offset_jitter_sigma=0.15,
    explore_offset_jitter_sigma=0.35,
    exploit_relative_parameter_jitter=0.03,
    explore_relative_parameter_jitter=0.07,
    global_timing_jitter_sigma_samples=0.35,
    local_sbox_jitter_sigma_samples=0.18,
    injection_timing_jitter_sigma_samples=0.20,
    session_timing_shift_sigma_samples=0.25,
    response_trace_noise_sigma=0.055,
    response_trace_baseline_sigma=0.035,
    response_trace_gain_sigma=0.06,
    target_window_radius_samples=24,
    pulse_window_radius_samples=24,
    highpass_moving_average_width=9,
    trace_standard_deviation_floor=1.0e-6,
    bootstrap_repetitions=1000,
    ece_bins=15,
    save_plots=True,
    deterministic_recheck_count=32,
)

print("Stage-12 configuration created.")
print("Input :", stage_12_config.input_stage11_run_directory)
print("Output:", stage_12_config.output_root)

stage_12_summary = run_stage_12(stage_12_config)

print("\nStage 12 completed.")
print(json.dumps(stage_12_summary, ensure_ascii=False, indent=2))


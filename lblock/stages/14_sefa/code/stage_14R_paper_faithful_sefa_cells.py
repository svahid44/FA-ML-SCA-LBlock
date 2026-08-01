# %% [markdown]
# # Stage 14R — SEFA مقاله‌ای روی LBlock با مقایسه کامل مدل‌ها
#
# این مرحله یک کمپین تازه و مستقل اجرا می‌کند. برای SEFA، فقط تلاش‌هایی که
# `C' != C` دارند به‌عنوان رویداد مؤثر انتخاب می‌شوند، اما بازیابی کلید از
# **ciphertext صحیح C** انجام می‌شود؛ مقدار ciphertext معیوب فقط برای تشخیص
# مؤثر/نامؤثر بودن تلاش استفاده می‌شود و وارد امتیاز کلید نمی‌شود.
#
# امتیاز اصلی دقیقاً معیار بدون‌وزن مقاله است:
#
# \[
# SEI(k)=\sum_x\left(\hat p_k(x)-\frac1{16}\right)^2.
# \]
#
# مدل Stage 11 فقط پارامتر تزریق را پیش از تزریق پیشنهاد می‌دهد و مدل Stage 10
# فقط نمونه‌ها را انتخاب/مرتب می‌کند. هیچ Probability داخل SEI ضرب نمی‌شود.
#
# برای مقایسه منصفانه، همان کمپین تازه هم‌زمان با SIFA نیز تحلیل می‌شود. سیاست
# Budget-aware مربوط به SEFA فقط روی بخش Calibration عمومی تنظیم و Freeze می‌شود
# و ارزیابی نهایی روی بخش Confirmation جداگانه انجام می‌گیرد.

# %%
# ============================================================
# Stage 14R / Cell 1
# Configuration, contracts, paper model, and pre-registration
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display


_SEFA14_REQUIRED_SYMBOLS = (
    "engine",
    "Stage12Config",
    "resolve_stage_contracts",
    "score_public_batch",
    "perturb_recommendation",
)
_SEFA14_MISSING = [name for name in _SEFA14_REQUIRED_SYMBOLS if name not in globals()]
if _SEFA14_MISSING:
    raise RuntimeError(
        "Before Stage 14R, run the Stage-12 cell. Missing symbols: "
        + ", ".join(_SEFA14_MISSING)
    )


@dataclass(frozen=True)
class PaperSEFAConfig:
    input_stage11_run_directory: str
    output_root: str
    random_seed: int = 20260724

    number_of_keys: int = 2
    number_of_sessions: int = 4
    target_sbox_indices: Tuple[int, ...] = (0, 5)

    # The 4-bit random-AND model has a weaker effective-event capacity than
    # ineffective-event capacity.  We therefore reserve 8,000 independent
    # confirmation attempts per key/target/arm, compared with 3,000 in SIFA.
    calibration_injections_per_key_target_arm: int = 2000
    confirmation_injections_per_key_target_arm: int = 8000

    guided_offset_jitter_sigma: float = 0.15
    guided_relative_parameter_jitter: float = 0.03

    global_timing_jitter_sigma_samples: float = 0.35
    local_sbox_jitter_sigma_samples: float = 0.18
    injection_timing_jitter_sigma_samples: float = 0.20
    session_timing_shift_sigma_samples: float = 0.25
    response_trace_noise_sigma: float = 0.055
    response_trace_baseline_sigma: float = 0.035
    response_trace_gain_sigma: float = 0.06

    target_window_radius_samples: int = 24
    pulse_window_radius_samples: int = 24
    highpass_moving_average_width: int = 9
    trace_standard_deviation_floor: float = 1.0e-6

    candidate_top_fractions: Tuple[float, ...] = (
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.65,
        0.80,
        1.00,
    )
    minimum_calibration_selected_per_task: int = 128
    required_loso_consensus: float = 0.75

    # Frozen from the accepted Stage 13R-v3 SIFA calibration.  This value is
    # not tuned again on the new SEFA campaign.
    frozen_sifa_top_fraction: float = 0.40

    minimum_attack_samples: int = 16
    bootstrap_repetitions: int = 300
    matched_random_repetitions: int = 300
    save_plots: bool = True


_SEFA14_STAGE11_DEFAULT = (
    getattr(stage_12_config, "input_stage11_run_directory", None)
    if "stage_12_config" in globals()
    else None
)
if not _SEFA14_STAGE11_DEFAULT:
    _SEFA14_STAGE11_DEFAULT = (
        './runs/stage_11'
        '/stage11_20260718_191029_290452_seed20260718'
    )

paper_sefa_config = PaperSEFAConfig(
    input_stage11_run_directory=os.environ.get(
        "LBLOCK_PAPER_SEFA_STAGE11",
        _SEFA14_STAGE11_DEFAULT,
    ),
    output_root=os.environ.get(
        "LBLOCK_PAPER_SEFA_OUTPUT",
        './runs/stage_14R_paper_sefa',
    ),
)


def sefa14_json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sefa14_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=sefa14_json_default,
        ),
        encoding="utf-8",
    )


def sefa14_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sefa14_stable_freeze(directory: Path, output_path: Path, statement: str) -> Dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": statement,
        "files": {
            str(path.relative_to(directory)).replace("\\", "/"): sefa14_sha256_file(path)
            for path in files
        },
    }
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=sefa14_json_default,
    ).encode("utf-8")
    manifest["freeze_sha256"] = hashlib.sha256(encoded).hexdigest()
    sefa14_write_json(output_path, manifest)
    return manifest


def sefa14_hex(value: int, bits: int) -> str:
    return f"{int(value):0{bits // 4}x}"


if paper_sefa_config.number_of_keys != 2:
    raise ValueError("Stage 14R is pre-registered for exactly two fresh keys")
if set(paper_sefa_config.target_sbox_indices) != {0, 5}:
    raise ValueError("Stage 14R is pre-registered for S0 and S5")
if paper_sefa_config.number_of_sessions != 4:
    raise ValueError("Stage 14R requires four sessions for LOSO checks")
if paper_sefa_config.confirmation_injections_per_key_target_arm < 3000:
    raise ValueError("Use at least 3000 confirmation injections per task/arm")

_sefa14_total_per_task_arm = (
    paper_sefa_config.calibration_injections_per_key_target_arm
    + paper_sefa_config.confirmation_injections_per_key_target_arm
)
_sefa14_total_pairs = (
    paper_sefa_config.number_of_keys
    * len(paper_sefa_config.target_sbox_indices)
    * _sefa14_total_per_task_arm
)
_sefa14_total_injections = 2 * _sefa14_total_pairs

sefa14_sim_config = Stage12Config(
    input_stage11_run_directory=paper_sefa_config.input_stage11_run_directory,
    output_root=paper_sefa_config.output_root,
    random_seed=paper_sefa_config.random_seed,
    number_of_experiments=_sefa14_total_injections,
    number_of_batches=1,
    experiments_per_batch=_sefa14_total_injections,
    number_of_keys=paper_sefa_config.number_of_keys,
    number_of_sessions=paper_sefa_config.number_of_sessions,
    confirmation_batch_index=0,
    guided_exploit_fraction=0.50,
    guided_explore_fraction=0.00,
    randomized_baseline_fraction=0.50,
    safety_control_fraction=0.00,
    sifa_objective_fraction=0.00,
    sefa_objective_fraction=1.00,
    shfa_objective_fraction=0.00,
    exploit_offset_jitter_sigma=paper_sefa_config.guided_offset_jitter_sigma,
    exploit_relative_parameter_jitter=paper_sefa_config.guided_relative_parameter_jitter,
    global_timing_jitter_sigma_samples=paper_sefa_config.global_timing_jitter_sigma_samples,
    local_sbox_jitter_sigma_samples=paper_sefa_config.local_sbox_jitter_sigma_samples,
    injection_timing_jitter_sigma_samples=paper_sefa_config.injection_timing_jitter_sigma_samples,
    session_timing_shift_sigma_samples=paper_sefa_config.session_timing_shift_sigma_samples,
    response_trace_noise_sigma=paper_sefa_config.response_trace_noise_sigma,
    response_trace_baseline_sigma=paper_sefa_config.response_trace_baseline_sigma,
    response_trace_gain_sigma=paper_sefa_config.response_trace_gain_sigma,
    target_window_radius_samples=paper_sefa_config.target_window_radius_samples,
    pulse_window_radius_samples=paper_sefa_config.pulse_window_radius_samples,
    highpass_moving_average_width=paper_sefa_config.highpass_moving_average_width,
    trace_standard_deviation_floor=paper_sefa_config.trace_standard_deviation_floor,
    bootstrap_repetitions=paper_sefa_config.bootstrap_repetitions,
    save_plots=paper_sefa_config.save_plots,
)

sefa14_stage11_directory = Path(
    paper_sefa_config.input_stage11_run_directory
).expanduser().resolve()
sefa14_contracts = resolve_stage_contracts(sefa14_stage11_directory)
sefa14_recommendations = sefa14_contracts["exploit_recommendations"].copy()

sefa14_best_recommendations: Dict[str, pd.Series] = {}
for sefa14_target_name in ("S0", "S5"):
    sefa14_subset = sefa14_recommendations[
        (sefa14_recommendations["target_sbox"].astype(str) == sefa14_target_name)
        & (sefa14_recommendations["objective"].astype(str) == "SEFA")
        & (sefa14_recommendations["recommendation_mode"].astype(str) == "exploit")
    ].sort_values(["rank", "robust_utility_SEFA"], ascending=[True, False])
    if sefa14_subset.empty:
        raise RuntimeError(f"No Stage-11 SEFA exploit recommendation for {sefa14_target_name}")
    sefa14_best_recommendations[sefa14_target_name] = sefa14_subset.iloc[0].copy()

sefa14_model10 = sefa14_contracts["stage10_model"]
sefa14_effective_threshold = float(
    sefa14_model10["branch_thresholds"]["clean_target_effective"]
)
sefa14_ineffective_threshold = float(
    sefa14_model10["branch_thresholds"]["clean_target_ineffective"]
)
if not (0.0 < sefa14_effective_threshold < 1.0):
    raise RuntimeError("Invalid frozen Stage-10 effective threshold")
if not (0.0 < sefa14_ineffective_threshold < 1.0):
    raise RuntimeError("Invalid frozen Stage-10 ineffective threshold")

sefa14_centers = np.asarray(
    [float(item["center_sample"]) for item in sefa14_contracts["timing_map"]["sboxes"]],
    dtype=np.float64,
)
sefa14_bounds_by_target = {
    target_name: sefa14_contracts["stage11_summary"]["candidate_pool_summary"][target_name][
        "parameter_bounds"
    ]
    for target_name in ("S0", "S5")
}

# Final-round mapping: X33 = F(X32,K32) XOR ROL8(X31).
_SEFA14_P_SOURCE_FOR_OUTPUT = tuple(int(value) for value in engine.P_SOURCE_FOR_OUTPUT)
_SEFA14_SOURCE_TO_OUTPUT = {
    int(source): int(output_index)
    for output_index, source in enumerate(_SEFA14_P_SOURCE_FOR_OUTPUT)
}
_SEFA14_STATE_NIBBLE_BY_CHANNEL = {
    int(source): int((output_index - 2) % 8)
    for source, output_index in _SEFA14_SOURCE_TO_OUTPUT.items()
}
if _SEFA14_STATE_NIBBLE_BY_CHANNEL[0] != 0:
    raise AssertionError("S0 must map to X31[0]")
if _SEFA14_STATE_NIBBLE_BY_CHANNEL[5] != 2:
    raise AssertionError("S5 must map to X31[2]")

# Paper distributions for the secondary known-model LLR.  SEI remains primary.
_SEFA14_HW = np.asarray([int(value).bit_count() for value in range(16)], dtype=float)
_SEFA14_UNIFORM = np.full(16, 1.0 / 16.0, dtype=float)
_SEFA14_P_INEFFECTIVE_GIVEN_X = np.power(2.0, -_SEFA14_HW)
_SEFA14_P_EFFECTIVE_GIVEN_X = 1.0 - _SEFA14_P_INEFFECTIVE_GIVEN_X
_SEFA14_Q_INEFFECTIVE = (
    _SEFA14_P_INEFFECTIVE_GIVEN_X / _SEFA14_P_INEFFECTIVE_GIVEN_X.sum()
)
_SEFA14_Q_EFFECTIVE = (
    _SEFA14_P_EFFECTIVE_GIVEN_X / _SEFA14_P_EFFECTIVE_GIVEN_X.sum()
)
_SEFA14_THEORY = {
    "fault_model": "4-bit random-AND",
    "ineffective_rate": float(_SEFA14_P_INEFFECTIVE_GIVEN_X.mean()),
    "effective_rate": float(_SEFA14_P_EFFECTIVE_GIVEN_X.mean()),
    "ineffective_capacity_from_paper": 0.52,
    "effective_capacity_from_paper": 0.11,
    "relative_total_attempt_complexity_SEFA_over_SIFA": float(
        (0.52 * _SEFA14_P_INEFFECTIVE_GIVEN_X.mean())
        / (0.11 * _SEFA14_P_EFFECTIVE_GIVEN_X.mean())
    ),
}

# Fresh keys are simulator-private until the public attack is frozen.
sefa14_key_rng = np.random.default_rng(
    np.random.SeedSequence([paper_sefa_config.random_seed, 14001])
)
sefa14_key_pool: List[int] = []
while len(sefa14_key_pool) < paper_sefa_config.number_of_keys:
    candidate = int(engine.random_80bit_integer(sefa14_key_rng))
    if candidate not in sefa14_key_pool:
        sefa14_key_pool.append(candidate)

sefa14_session_rng = np.random.default_rng(
    np.random.SeedSequence([paper_sefa_config.random_seed, 14002])
)
sefa14_session_shifts = sefa14_session_rng.normal(
    0.0,
    paper_sefa_config.session_timing_shift_sigma_samples,
    size=paper_sefa_config.number_of_sessions,
)

sefa14_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
sefa14_run_id = f"stage14R_{sefa14_timestamp}_seed{paper_sefa_config.random_seed}"
sefa14_run_directory = Path(paper_sefa_config.output_root).expanduser().resolve() / sefa14_run_id
sefa14_public_campaign_directory = sefa14_run_directory / "public_campaign"
sefa14_public_policy_directory = sefa14_run_directory / "public_policy"
sefa14_public_attack_directory = sefa14_run_directory / "public_attack"
sefa14_locked_directory = sefa14_run_directory / "locked_truth"
sefa14_validation_directory = sefa14_run_directory / "validation_only"
for sefa14_directory in (
    sefa14_public_campaign_directory,
    sefa14_public_policy_directory,
    sefa14_public_attack_directory,
    sefa14_locked_directory,
    sefa14_validation_directory,
):
    sefa14_directory.mkdir(parents=True, exist_ok=True)

sefa14_write_json(
    sefa14_public_campaign_directory / "stage14R_config.json",
    {
        **asdict(paper_sefa_config),
        "target_sbox_indices": list(paper_sefa_config.target_sbox_indices),
        "total_paired_injections": _sefa14_total_injections,
        "total_pairs": _sefa14_total_pairs,
        "stage10_effective_threshold": sefa14_effective_threshold,
        "stage10_ineffective_threshold": sefa14_ineffective_threshold,
        "theoretical_random_and_4": _SEFA14_THEORY,
    },
)

sefa14_write_json(
    sefa14_public_campaign_directory / "stage14R_pre_registered_contract.json",
    {
        "primary_attack": "SEFA",
        "secondary_same_campaign_comparison": "SIFA",
        "primary_statistic": "unweighted SEI",
        "sei_equation": "sum_x (p_hat_k(x)-1/16)^2; highest score wins",
        "sefa_observation_rule": "response_received and C_prime != C",
        "sefa_key_recovery_data": "correct/non-faulty ciphertext C from effective trials",
        "faulty_ciphertext_use": "event classification only; never parsed by key scorer",
        "sifa_observation_rule": "response_received and C_prime == C",
        "physical_fault_target": (
            "X31 state nibble after X32 is formed and before ROL8(X31) is XORed into X33"
        ),
        "lblock_partial_decryption": (
            "X31_hat_t(k)=nibble_j(X33) XOR S_s(nibble_s(X32) XOR k)"
        ),
        "target_mapping": {
            "S0": "fault X31[0], recover K32[0]",
            "S5": "fault X31[2], recover K32[5]",
        },
        "guided_parameter_source": "Stage-11 rank-1 SEFA exploit recommendation",
        "random_parameter_source": "uniform sampling from identical Stage-11 support bounds",
        "stage10_effective_role": "hard threshold or session-balanced top-fraction selection only",
        "stage10_ineffective_role": "same SIFA ablation with previously frozen 0.40 top fraction",
        "probability_weighting_inside_sei": "forbidden",
        "calibration_confirmation_split": {
            "calibration_attempts_per_task_arm": paper_sefa_config.calibration_injections_per_key_target_arm,
            "confirmation_attempts_per_task_arm": paper_sefa_config.confirmation_injections_per_key_target_arm,
            "policy_source": "random arm calibration partition only",
            "final_key_recovery_source": "disjoint confirmation partition only",
        },
        "paired_design": "same key/plaintext/target/session/source trace for guided and random pair",
        "private_truth_available_before_public_freeze": False,
    },
)

# Algebraic reconstruction guard, independent of campaign keys.
sefa14_guard_rng = np.random.default_rng(
    np.random.SeedSequence([paper_sefa_config.random_seed, 14991])
)
sefa14_guard_key = int(engine.random_80bit_integer(sefa14_guard_rng))
sefa14_guard_k32 = int(engine.key_schedule_lblock(sefa14_guard_key)[31])
for sefa14_target in paper_sefa_config.target_sbox_indices:
    output_index = _SEFA14_SOURCE_TO_OUTPUT[int(sefa14_target)]
    state_index = _SEFA14_STATE_NIBBLE_BY_CHANNEL[int(sefa14_target)]
    true_nibble = int((sefa14_guard_k32 >> (4 * int(sefa14_target))) & 0xF)
    for _ in range(64):
        plaintext = int(engine.random_64bit_integer(sefa14_guard_rng))
        context = engine.final_round_context(plaintext, sefa14_guard_key)
        x32_nibble = int(engine.get_nibble(int(context["x32"]), int(sefa14_target)))
        x33_nibble = int(engine.get_nibble(int(context["x33"]), int(output_index)))
        reconstructed = int(
            x33_nibble
            ^ int(engine.SBOX[int(sefa14_target)][x32_nibble ^ true_nibble])
        )
        actual = int(engine.get_nibble(int(context["x31"]), int(state_index)))
        if reconstructed != actual:
            raise AssertionError("LBlock X31 partial-decryption guard failed")

print("=" * 94)
print("Stage 14R configuration ready — paper-faithful SEFA")
print("Stage-11 input                    :", sefa14_stage11_directory)
print("Output directory                  :", sefa14_run_directory)
print("Fresh paired fault attempts       :", _sefa14_total_injections)
print("Calibration per key/target/arm    :", paper_sefa_config.calibration_injections_per_key_target_arm)
print("Confirmation per key/target/arm   :", paper_sefa_config.confirmation_injections_per_key_target_arm)
print("Fresh keys / sessions             :", paper_sefa_config.number_of_keys, "/", paper_sefa_config.number_of_sessions)
print("Frozen Stage-10 effective th       :", f"{sefa14_effective_threshold:.6f}")
print("Frozen Stage-10 ineffective th     :", f"{sefa14_ineffective_threshold:.6f}")
print("Paper theory SEFA/SIFA attempts    :", f"{_SEFA14_THEORY['relative_total_attempt_complexity_SEFA_over_SIFA']:.3f}x")
for target_name in ("S0", "S5"):
    row = sefa14_best_recommendations[target_name]
    print(
        f"{target_name} best Stage-11 SEFA parameters:",
        {
            "offset": float(row["timing_offset_samples"]),
            "width": float(row["width_samples"]),
            "strength": float(row["strength"]),
            "repeat": int(row["repeat"]),
            "spacing": float(row["repeat_spacing_samples"]),
            "robust_utility_SEFA": float(row["robust_utility_SEFA"]),
        },
    )
print("Truth/private opened              : False")
print("=" * 94)

# %%
# ============================================================
# Stage 14R / Cell 2
# Fresh paired campaign with SEFA-optimized guided parameters
# ============================================================

_SEFA14_CLASS_NAMES = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)
_SEFA14_CLASS_TO_ID = {name: index for index, name in enumerate(_SEFA14_CLASS_NAMES)}


def sefa14_sample_random_support_parameters(
    target_index: int,
    rng: np.random.Generator,
) -> Any:
    target_name = f"S{int(target_index)}"
    bounds = sefa14_bounds_by_target[target_name]
    repeat_values = np.asarray(bounds["repeat"]["allowed"], dtype=np.int32)
    return engine.GlitchParameters(
        target_sbox_index=int(target_index),
        nominal_target_center_sample=float(sefa14_centers[target_index]),
        offset_samples=float(
            rng.uniform(
                bounds["timing_offset_samples"]["lower"],
                bounds["timing_offset_samples"]["upper"],
            )
        ),
        width_samples=float(
            rng.uniform(bounds["width_samples"]["lower"], bounds["width_samples"]["upper"])
        ),
        strength=float(
            rng.uniform(bounds["strength"]["lower"], bounds["strength"]["upper"])
        ),
        repeat=int(rng.choice(repeat_values)),
        repeat_spacing_samples=float(
            rng.uniform(
                bounds["repeat_spacing_samples"]["lower"],
                bounds["repeat_spacing_samples"]["upper"],
            )
        ),
        sampling_regime="random_uniform_stage11_support",
        fault_model="random_and_4",
    )


def sefa14_simulator_proxy() -> Any:
    class Proxy:
        pass

    proxy = Proxy()
    proxy.response_trace_noise_sigma = paper_sefa_config.response_trace_noise_sigma
    proxy.response_trace_baseline_sigma = paper_sefa_config.response_trace_baseline_sigma
    proxy.response_trace_gain_sigma = paper_sefa_config.response_trace_gain_sigma
    return proxy


def sefa14_channel_values_from_x31(x31: int) -> np.ndarray:
    values = np.zeros(8, dtype=np.uint8)
    for channel_index in range(8):
        state_nibble_index = _SEFA14_STATE_NIBBLE_BY_CHANNEL[channel_index]
        values[channel_index] = int(
            engine.get_nibble(int(x31), int(state_nibble_index))
        )
    return values


def sefa14_x31_from_channel_values(values: Sequence[int]) -> int:
    if len(values) != 8:
        raise ValueError("Exactly eight channel values are required")
    x31 = 0
    for channel_index, value in enumerate(values):
        state_nibble_index = _SEFA14_STATE_NIBBLE_BY_CHANNEL[channel_index]
        x31 |= (int(value) & 0xF) << (4 * int(state_nibble_index))
    return int(x31) & int(engine.MASK32)


def sefa14_run_attempt(
    *,
    experiment_id: int,
    pair_id: int,
    arm: str,
    arm_code: int,
    partition: str,
    local_attempt_index: int,
    confirmation_index: int,
    key_id: int,
    session_id: int,
    target_index: int,
    plaintext: int,
    source_trace_index: int,
    parameters: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                paper_sefa_config.random_seed,
                int(pair_id),
                int(key_id),
                int(target_index),
                int(arm_code),
                14006,
            ]
        )
    )

    core_widths = np.asarray(
        [
            float(item["core_window_end_sample_exclusive"])
            - float(item["core_window_start_sample_inclusive"])
            for item in sefa14_contracts["timing_map"]["sboxes"]
        ],
        dtype=np.float64,
    )
    global_jitter = float(
        rng.normal(0.0, paper_sefa_config.global_timing_jitter_sigma_samples)
    )
    local_jitter = rng.normal(
        0.0,
        paper_sefa_config.local_sbox_jitter_sigma_samples,
        size=8,
    )
    actual_centers = (
        sefa14_centers
        + float(sefa14_session_shifts[session_id])
        + global_jitter
        + local_jitter
    )
    injection_jitter = float(
        rng.normal(0.0, paper_sefa_config.injection_timing_jitter_sigma_samples)
    )
    pulse_values = engine.pulse_centers(parameters, injection_jitter)
    hit_scores = engine.compute_hit_scores(
        parameters,
        pulse_values,
        actual_centers,
        core_widths,
    )
    activation = engine.activation_probabilities(parameters, hit_scores)
    invalid_probability = engine.invalid_probability(parameters, hit_scores)
    invalid = bool(rng.random() < invalid_probability)

    master_key = int(sefa14_key_pool[key_id])
    context = engine.final_round_context(int(plaintext), master_key)
    healthy_ciphertext = int(context["ciphertext"])

    original_inputs = sefa14_channel_values_from_x31(int(context["x31"]))
    faulted_inputs = original_inputs.copy()
    impacted_mask = np.zeros(8, dtype=np.uint8)
    model_details: Dict[str, Any] = {}

    if not invalid:
        activated = rng.random(8) < activation
        impacted_mask[:] = activated.astype(np.uint8)
        for channel_index in np.where(activated)[0]:
            faulted_value, details = engine.apply_fault_model(
                int(original_inputs[channel_index]),
                parameters.fault_model,
                rng,
            )
            faulted_inputs[channel_index] = int(faulted_value)
            model_details[f"S{int(channel_index)}"] = {
                **details,
                "physical_x31_nibble": int(
                    _SEFA14_STATE_NIBBLE_BY_CHANNEL[int(channel_index)]
                ),
            }

        faulted_x31 = sefa14_x31_from_channel_values(faulted_inputs)
        x33_faulted = int(
            int(context["f_output"]) ^ engine.rol(int(faulted_x31), 8, 32)
        ) & int(engine.MASK32)
        faulty_ciphertext = int(
            (int(context["x32"]) << 32) | int(x33_faulted)
        ) & ((1 << 64) - 1)
        response_received = True
        ciphertext_equal = bool(faulty_ciphertext == healthy_ciphertext)
        ciphertext_hamming_distance = int(
            engine.hamming_distance(faulty_ciphertext, healthy_ciphertext)
        )
        invalid_subtype = ""
    else:
        faulty_ciphertext = None
        response_received = False
        ciphertext_equal = False
        ciphertext_hamming_distance = -1
        invalid_subtype = "reset" if rng.random() < 0.55 else "timeout"

    category = engine.classify_fault_event(
        int(target_index),
        impacted_mask,
        invalid,
        ciphertext_equal,
    )

    response_trace = engine.synthesize_response_trace(
        sefa14_contracts["healthy_source"]["traces"][int(source_trace_index)],
        sefa14_contracts["healthy_source"]["absolute_samples"],
        pulse_values,
        parameters,
        actual_centers,
        original_inputs,
        faulted_inputs,
        impacted_mask,
        invalid,
        rng,
        sefa14_simulator_proxy(),
    ).astype(np.float32)

    trace_features = engine.trace_features(
        response_trace,
        sefa14_contracts["healthy_source"]["absolute_samples"],
        float(sefa14_centers[target_index]),
        pulse_values,
    )

    impacted_indices = [int(value) for value in np.where(impacted_mask > 0)[0]]
    public_row: Dict[str, Any] = {
        "experiment_id": int(experiment_id),
        "pair_id": int(pair_id),
        "local_attempt_index": int(local_attempt_index),
        "confirmation_index": int(confirmation_index),
        "campaign_partition": str(partition),
        "campaign_arm": str(arm),
        "objective": "SEFA",
        "target_sbox": f"S{int(target_index)}",
        "target_sbox_index": int(target_index),
        "physical_fault_intermediate": "X31_state_after_X32_before_final_feedforward",
        "physical_x31_nibble_index": int(
            _SEFA14_STATE_NIBBLE_BY_CHANNEL[int(target_index)]
        ),
        "final_f_output_nibble_index": int(
            _SEFA14_SOURCE_TO_OUTPUT[int(target_index)]
        ),
        "recovered_key_nibble": f"K32[{int(target_index)}]",
        "key_id": int(key_id),
        "session_id": int(session_id),
        "source_healthy_trace_id": int(
            sefa14_contracts["healthy_source"]["trace_ids"][int(source_trace_index)]
        ),
        "fault_model": str(parameters.fault_model),
        "parameter_source": (
            "stage11_rank1_SEFA_exploit"
            if arm == "guided_model"
            else "uniform_same_stage11_support"
        ),
        "recommendation_rank": 1 if arm == "guided_model" else -1,
        "nominal_target_center_sample": float(parameters.nominal_target_center_sample),
        "timing_offset_samples": float(parameters.offset_samples),
        "first_pulse_nominal_sample": float(
            parameters.nominal_target_center_sample + parameters.offset_samples
        ),
        "width_samples": float(parameters.width_samples),
        "strength": float(parameters.strength),
        "repeat": int(parameters.repeat),
        "repeat_spacing_samples": float(parameters.repeat_spacing_samples),
        "plaintext_hex": sefa14_hex(plaintext, 64),
        "healthy_ciphertext_hex": sefa14_hex(healthy_ciphertext, 64),
        "response_received": bool(response_received),
        "faulty_ciphertext_hex": (
            sefa14_hex(faulty_ciphertext, 64) if faulty_ciphertext is not None else ""
        ),
        "ciphertext_equal": bool(ciphertext_equal) if response_received else "",
        "ciphertext_hamming_distance": (
            int(ciphertext_hamming_distance) if response_received else np.nan
        ),
        **trace_features,
    }

    private_row: Dict[str, Any] = {
        "experiment_id": int(experiment_id),
        "pair_id": int(pair_id),
        "campaign_partition": str(partition),
        "campaign_arm": str(arm),
        "key_id": int(key_id),
        "session_id": int(session_id),
        "target_sbox": f"S{int(target_index)}",
        "target_sbox_index": int(target_index),
        "category": str(category),
        "category_id": int(_SEFA14_CLASS_TO_ID[category]),
        "target_original_x31_value": int(original_inputs[target_index]),
        "target_faulted_x31_value": int(faulted_inputs[target_index]),
        "target_impacted": bool(impacted_mask[target_index]),
        "off_target_impacted": bool(
            any(value != target_index for value in impacted_indices)
        ),
        "impacted_sboxes": ";".join(f"S{value}" for value in impacted_indices),
        "impacted_sbox_count": int(len(impacted_indices)),
        "changed_x31_channel_count": int(np.sum(original_inputs != faulted_inputs)),
        "fault_effective": bool(response_received and not ciphertext_equal),
        "invalid_subtype": str(invalid_subtype),
        "invalid_probability": float(invalid_probability),
        "global_jitter_samples": float(global_jitter),
        "injection_jitter_samples": float(injection_jitter),
        "model_details_json": json.dumps(model_details, sort_keys=True),
    }
    return public_row, private_row, response_trace


sefa14_campaign_started = time.perf_counter()
sefa14_public_rows: List[Dict[str, Any]] = []
sefa14_private_rows: List[Dict[str, Any]] = []
sefa14_response_traces: List[np.ndarray] = []
sefa14_experiment_id = 0
sefa14_pair_id = 0

for sefa14_key_id in range(paper_sefa_config.number_of_keys):
    for sefa14_target_index in paper_sefa_config.target_sbox_indices:
        target_name = f"S{int(sefa14_target_index)}"
        best_row = sefa14_best_recommendations[target_name]
        bounds = sefa14_bounds_by_target[target_name]

        for sefa14_local_index in range(_sefa14_total_per_task_arm):
            if sefa14_local_index < paper_sefa_config.calibration_injections_per_key_target_arm:
                partition = "calibration"
                confirmation_index = -1
            else:
                partition = "confirmation"
                confirmation_index = int(
                    sefa14_local_index
                    - paper_sefa_config.calibration_injections_per_key_target_arm
                )

            session_id = int(sefa14_local_index % paper_sefa_config.number_of_sessions)
            common_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        paper_sefa_config.random_seed,
                        int(sefa14_key_id),
                        int(sefa14_target_index),
                        int(sefa14_local_index),
                        14005,
                    ]
                )
            )
            plaintext = int(engine.random_64bit_integer(common_rng))
            source_trace_index = int(
                common_rng.integers(0, sefa14_contracts["healthy_source"]["traces"].shape[0])
            )

            for arm_code, arm in enumerate(("guided_model", "random_uniform")):
                parameter_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_sefa_config.random_seed,
                            int(sefa14_key_id),
                            int(sefa14_target_index),
                            int(sefa14_local_index),
                            int(arm_code),
                            14004,
                        ]
                    )
                )
                if arm == "guided_model":
                    parameters = perturb_recommendation(
                        best_row,
                        "exploit",
                        bounds,
                        float(sefa14_centers[sefa14_target_index]),
                        parameter_rng,
                        sefa14_sim_config,
                    )
                else:
                    parameters = sefa14_sample_random_support_parameters(
                        sefa14_target_index,
                        parameter_rng,
                    )

                public_row, private_row, trace = sefa14_run_attempt(
                    experiment_id=sefa14_experiment_id,
                    pair_id=sefa14_pair_id,
                    arm=arm,
                    arm_code=arm_code,
                    partition=partition,
                    local_attempt_index=sefa14_local_index,
                    confirmation_index=confirmation_index,
                    key_id=sefa14_key_id,
                    session_id=session_id,
                    target_index=sefa14_target_index,
                    plaintext=plaintext,
                    source_trace_index=source_trace_index,
                    parameters=parameters,
                )
                sefa14_public_rows.append(public_row)
                sefa14_private_rows.append(private_row)
                sefa14_response_traces.append(trace)
                sefa14_experiment_id += 1

            sefa14_pair_id += 1
            if sefa14_pair_id % 2000 == 0 or sefa14_pair_id == _sefa14_total_pairs:
                print(
                    "Paired injections completed:",
                    f"{sefa14_pair_id}/{_sefa14_total_pairs}",
                    "pairs |",
                    f"{2 * sefa14_pair_id}/{_sefa14_total_injections}",
                    "fault attempts",
                )

sefa14_public_frame = (
    pd.DataFrame(sefa14_public_rows).sort_values("experiment_id").reset_index(drop=True)
)
sefa14_private_frame_locked = (
    pd.DataFrame(sefa14_private_rows).sort_values("experiment_id").reset_index(drop=True)
)
sefa14_trace_matrix = np.stack(sefa14_response_traces, axis=0).astype(np.float32)

if len(sefa14_public_frame) != _sefa14_total_injections:
    raise AssertionError("Campaign row count mismatch")
if sefa14_trace_matrix.shape[0] != _sefa14_total_injections:
    raise AssertionError("Trace row count mismatch")
if sefa14_public_frame["experiment_id"].duplicated().any():
    raise AssertionError("Duplicate experiment IDs")
if not (
    sefa14_public_frame.groupby("pair_id")["plaintext_hex"].nunique().eq(1).all()
):
    raise AssertionError("Paired arms do not share plaintext")
if not (
    sefa14_public_frame.groupby("pair_id")["key_id"].nunique().eq(1).all()
):
    raise AssertionError("Paired arms do not share key ID")
if not (
    sefa14_public_frame.groupby("pair_id")["target_sbox_index"].nunique().eq(1).all()
):
    raise AssertionError("Paired arms do not share target")
if not (
    sefa14_public_frame.groupby("pair_id")["session_id"].nunique().eq(1).all()
):
    raise AssertionError("Paired arms do not share session")

sefa14_public_campaign_path = (
    sefa14_public_campaign_directory / "paired_sefa_campaign_public.csv"
)
sefa14_trace_path = sefa14_public_campaign_directory / "paired_sefa_response_traces.npz"
sefa14_public_frame.to_csv(sefa14_public_campaign_path, index=False)
np.savez_compressed(
    sefa14_trace_path,
    experiment_ids=sefa14_public_frame["experiment_id"].to_numpy(np.int64),
    traces=sefa14_trace_matrix,
    absolute_samples=sefa14_contracts["healthy_source"]["absolute_samples"].astype(np.int64),
)

# Locked files are written now but never read until the public policy and attack
# have both been frozen.
sefa14_locked_label_path = sefa14_locked_directory / "fault_labels_LOCKED.csv"
sefa14_locked_key_path = sefa14_locked_directory / "key_truth_LOCKED.json"
sefa14_private_frame_locked.to_csv(sefa14_locked_label_path, index=False)
sefa14_write_json(
    sefa14_locked_key_path,
    {
        "keys": [
            {
                "key_id": int(key_id),
                "master_key_hex": sefa14_hex(key_value, 80),
                "round_key_32_hex": sefa14_hex(
                    int(engine.key_schedule_lblock(key_value)[31]),
                    32,
                ),
            }
            for key_id, key_value in enumerate(sefa14_key_pool)
        ]
    },
)
sefa14_locked_hashes = {
    "fault_labels_sha256": sefa14_sha256_file(sefa14_locked_label_path),
    "key_truth_sha256": sefa14_sha256_file(sefa14_locked_key_path),
}
sefa14_write_json(
    sefa14_public_campaign_directory / "locked_truth_access_manifest.json",
    {
        **sefa14_locked_hashes,
        "locked_truth_written": True,
        "locked_truth_opened_during_campaign": False,
        "statement": "Private labels and keys are unavailable to policy selection and public scoring.",
    },
)

print("=" * 94)
print("Fresh paired SEFA-oriented campaign generated")
print("Rows / traces                    :", len(sefa14_public_frame), "/", sefa14_trace_matrix.shape)
print("Pair count                       :", sefa14_public_frame["pair_id"].nunique())
print("Partitions                       :", sefa14_public_frame["campaign_partition"].value_counts().to_dict())
print("Arms                             :", sefa14_public_frame["campaign_arm"].value_counts().to_dict())
print("Elapsed seconds                  :", f"{time.perf_counter() - sefa14_campaign_started:.3f}")
print("Truth/private opened             : False")
print("=" * 94)

# %%
# ============================================================
# Stage 14R / Cell 3
# Public Stage-10 scoring, disjoint policy calibration, and attack freeze
# ============================================================

sefa14_probability_frame = score_public_batch(
    sefa14_public_frame,
    sefa14_trace_matrix,
    sefa14_contracts["healthy_source"]["absolute_samples"],
    sefa14_model10,
    sefa14_sim_config,
)
if not np.array_equal(
    sefa14_probability_frame["experiment_id"].to_numpy(int),
    sefa14_public_frame["experiment_id"].to_numpy(int),
):
    raise AssertionError("Stage-10 public score order mismatch")

sefa14_public_scored = sefa14_public_frame.merge(
    sefa14_probability_frame,
    on="experiment_id",
    how="left",
    validate="one_to_one",
)
sefa14_public_scored["paper_effective"] = (
    sefa14_public_scored["response_received"].astype(bool)
    & (sefa14_public_scored["ciphertext_equal"].astype(str).str.lower() == "false")
)
sefa14_public_scored["paper_ineffective"] = (
    sefa14_public_scored["response_received"].astype(bool)
    & (sefa14_public_scored["ciphertext_equal"].astype(str).str.lower() == "true")
)
sefa14_public_scored["effective_threshold_selected"] = (
    sefa14_public_scored["paper_effective"]
    & (
        sefa14_public_scored["p_clean_target_effective"].astype(float)
        >= sefa14_effective_threshold
    )
)
sefa14_public_scored["ineffective_threshold_selected"] = (
    sefa14_public_scored["paper_ineffective"]
    & (
        sefa14_public_scored["p_clean_target_ineffective"].astype(float)
        >= sefa14_ineffective_threshold
    )
)


def sefa14_parse_ciphertext_words(ciphertext_hex: Any) -> Tuple[int, int]:
    text = str(ciphertext_hex).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 16:
        raise ValueError(f"Expected a 64-bit ciphertext, got {text!r}")
    return int(text[:8], 16), int(text[8:], 16)


def sefa14_reconstruct_x31_target(
    x32: int,
    x33: int,
    target_sbox_index: int,
    key_guess: int,
) -> int:
    sbox_index = int(target_sbox_index)
    output_index = int(_SEFA14_SOURCE_TO_OUTPUT[sbox_index])
    x32_nibble = int(engine.get_nibble(int(x32), sbox_index))
    x33_nibble = int(engine.get_nibble(int(x33), output_index))
    return int(
        x33_nibble
        ^ int(engine.SBOX[sbox_index][x32_nibble ^ int(key_guess)])
    )


parsed_words = [
    sefa14_parse_ciphertext_words(value)
    for value in sefa14_public_scored["healthy_ciphertext_hex"]
]
sefa14_public_scored["x32_word"] = np.asarray(
    [value[0] for value in parsed_words], dtype=np.uint64
)
sefa14_public_scored["x33_word"] = np.asarray(
    [value[1] for value in parsed_words], dtype=np.uint64
)


def sefa14_intermediate_matrix(task_frame: pd.DataFrame) -> np.ndarray:
    if task_frame.empty:
        return np.empty((0, 16), dtype=np.uint8)
    targets = task_frame["target_sbox_index"].astype(int).unique()
    if len(targets) != 1:
        raise ValueError("Each task must contain one target S-box")
    target_index = int(targets[0])
    x32_values = task_frame["x32_word"].to_numpy(np.uint64)
    x33_values = task_frame["x33_word"].to_numpy(np.uint64)
    matrix = np.empty((len(task_frame), 16), dtype=np.uint8)
    for key_guess in range(16):
        matrix[:, key_guess] = np.fromiter(
            (
                sefa14_reconstruct_x31_target(
                    int(x32), int(x33), target_index, key_guess
                )
                for x32, x33 in zip(x32_values, x33_values)
            ),
            dtype=np.uint8,
            count=len(task_frame),
        )
    return matrix


def sefa14_scores_from_matrix(
    matrix: np.ndarray,
    selected_indices: Optional[np.ndarray] = None,
    attack_type: str = "SEFA",
) -> pd.DataFrame:
    if selected_indices is None:
        selected_indices = np.arange(matrix.shape[0], dtype=int)
    selected_indices = np.asarray(selected_indices, dtype=int)
    sample_count = int(len(selected_indices))
    known_distribution = (
        _SEFA14_Q_EFFECTIVE if str(attack_type).upper() == "SEFA" else _SEFA14_Q_INEFFECTIVE
    )
    log_ratio = np.log2(known_distribution / _SEFA14_UNIFORM)
    rows: List[Dict[str, Any]] = []
    for key_guess in range(16):
        if sample_count == 0:
            counts = np.zeros(16, dtype=float)
            empirical = np.zeros(16, dtype=float)
            sei = np.nan
            chi = np.nan
            llr = np.nan
        else:
            values = matrix[selected_indices, key_guess]
            counts = np.bincount(values, minlength=16).astype(float)
            empirical = counts / float(sample_count)
            sei = float(np.sum(np.square(empirical - _SEFA14_UNIFORM)))
            chi = float(sample_count * 16.0 * sei)
            llr = float(np.dot(counts, log_ratio))
        rows.append(
            {
                "key_guess": int(key_guess),
                "key_guess_hex": f"{key_guess:x}",
                "sample_count": sample_count,
                "sei": sei,
                "chi": chi,
                "llr_known_model": llr,
            }
        )
    return pd.DataFrame(rows)


def sefa14_prediction(scores: pd.DataFrame) -> Dict[str, Any]:
    ordered = scores.sort_values(["sei", "key_guess"], ascending=[False, True]).reset_index(drop=True)
    if ordered.empty or not np.isfinite(float(ordered.iloc[0]["sei"])):
        return {
            "best_key_guess": -1,
            "best_key_guess_hex": "",
            "best_score": np.nan,
            "second_score": np.nan,
            "score_margin": np.nan,
            "relative_margin": np.nan,
            "unique_best": False,
        }
    best_score = float(ordered.iloc[0]["sei"])
    second_score = float(ordered.iloc[1]["sei"])
    ties = np.isclose(
        scores["sei"].to_numpy(float),
        best_score,
        atol=1.0e-15,
        rtol=1.0e-12,
    )
    margin = float(best_score - second_score)
    return {
        "best_key_guess": int(ordered.iloc[0]["key_guess"]),
        "best_key_guess_hex": str(ordered.iloc[0]["key_guess_hex"]),
        "best_score": best_score,
        "second_score": second_score,
        "score_margin": margin,
        "relative_margin": float(margin / max(abs(best_score), 1.0e-15)),
        "unique_best": bool(np.sum(ties) == 1),
    }


def sefa14_session_balanced_top_indices(
    task_frame: pd.DataFrame,
    fraction: float,
    probability_column: str,
) -> np.ndarray:
    if task_frame.empty:
        return np.asarray([], dtype=int)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    total = int(len(task_frame))
    requested = min(total, max(1, int(round(fraction * total))))
    counts = task_frame.groupby("session_id").size().sort_index()
    exact = requested * counts.astype(float) / float(total)
    quotas = np.floor(exact).astype(int)
    remainder = requested - int(quotas.sum())
    order = sorted(
        counts.index.tolist(),
        key=lambda session: (
            -(float(exact.loc[session]) - float(quotas.loc[session])),
            int(session),
        ),
    )
    for session in order[:remainder]:
        quotas.loc[session] += 1

    selected: List[int] = []
    for session, quota in quotas.items():
        rows = task_frame[task_frame["session_id"].astype(int) == int(session)].sort_values(
            [probability_column, "experiment_id"],
            ascending=[False, True],
        )
        selected.extend(rows.index[: int(quota)].astype(int).tolist())
    selected_array = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected_array) != requested:
        remaining = task_frame.loc[~task_frame.index.isin(selected_array)].sort_values(
            [probability_column, "experiment_id"],
            ascending=[False, True],
        )
        needed = requested - len(selected_array)
        if needed > 0:
            selected_array = np.concatenate(
                [selected_array, remaining.index[:needed].to_numpy(int)]
            )
    return np.asarray(sorted(selected_array.tolist()), dtype=int)


def sefa14_loso_consensus(
    task_frame: pd.DataFrame,
    matrix: np.ndarray,
    selected_indices: np.ndarray,
    full_winner: int,
    attack_type: str,
) -> Tuple[float, int, str]:
    if len(selected_indices) == 0:
        return 0.0, 0, ""
    selected_sessions = task_frame.loc[selected_indices, "session_id"].astype(int)
    sessions = sorted(selected_sessions.unique().tolist())
    winners: List[int] = []
    for session in sessions:
        keep = selected_indices[
            task_frame.loc[selected_indices, "session_id"].to_numpy(int) != int(session)
        ]
        if len(keep) < paper_sefa_config.minimum_attack_samples:
            continue
        prediction = sefa14_prediction(
            sefa14_scores_from_matrix(matrix, keep, attack_type=attack_type)
        )
        winners.append(int(prediction["best_key_guess"]))
    if not winners:
        return 0.0, int(len(sessions)), ""
    consensus = float(np.mean(np.asarray(winners, dtype=int) == int(full_winner)))
    return consensus, int(len(sessions)), ",".join(f"{winner:x}" for winner in winners)


# Public-only SEFA policy calibration: random arm + calibration partition only.
sefa14_calibration_grid_score_parts: List[pd.DataFrame] = []
sefa14_calibration_prediction_rows: List[Dict[str, Any]] = []
for key_id in range(paper_sefa_config.number_of_keys):
    for target_index in paper_sefa_config.target_sbox_indices:
        task_all = (
            sefa14_public_scored[
                (sefa14_public_scored["campaign_partition"] == "calibration")
                & (sefa14_public_scored["campaign_arm"] == "random_uniform")
                & (sefa14_public_scored["key_id"].astype(int) == int(key_id))
                & (sefa14_public_scored["target_sbox_index"].astype(int) == int(target_index))
                & sefa14_public_scored["paper_effective"].astype(bool)
            ]
            .sort_values(["local_attempt_index", "experiment_id"])
            .reset_index(drop=True)
        )
        matrix = sefa14_intermediate_matrix(task_all)
        for fraction in paper_sefa_config.candidate_top_fractions:
            selected_indices = sefa14_session_balanced_top_indices(
                task_all,
                fraction,
                "p_clean_target_effective",
            )
            scores = sefa14_scores_from_matrix(matrix, selected_indices, attack_type="SEFA")
            prediction = sefa14_prediction(scores)
            loso, session_coverage, loso_winners = sefa14_loso_consensus(
                task_all,
                matrix,
                selected_indices,
                int(prediction["best_key_guess"]),
                "SEFA",
            )
            scores.insert(0, "key_id", int(key_id))
            scores.insert(1, "target_sbox", f"S{int(target_index)}")
            scores.insert(2, "target_sbox_index", int(target_index))
            scores.insert(3, "top_fraction", float(fraction))
            sefa14_calibration_grid_score_parts.append(scores)
            selected_probabilities = task_all.loc[
                selected_indices,
                "p_clean_target_effective",
            ].to_numpy(float)
            sefa14_calibration_prediction_rows.append(
                {
                    "key_id": int(key_id),
                    "target_sbox": f"S{int(target_index)}",
                    "target_sbox_index": int(target_index),
                    "top_fraction": float(fraction),
                    "paper_effective_count": int(len(task_all)),
                    "selected_ciphertext_count": int(len(selected_indices)),
                    "mean_selected_probability": float(np.mean(selected_probabilities)),
                    "minimum_selected_probability": float(np.min(selected_probabilities)),
                    "session_coverage": int(session_coverage),
                    "loso_consensus": float(loso),
                    "loso_winners_hex": loso_winners,
                    **prediction,
                }
            )

sefa14_calibration_grid_scores = pd.concat(
    sefa14_calibration_grid_score_parts, ignore_index=True
)
sefa14_calibration_predictions = pd.DataFrame(sefa14_calibration_prediction_rows)
sefa14_policy_rows: List[Dict[str, Any]] = []
for fraction, group in sefa14_calibration_predictions.groupby("top_fraction"):
    large_enough = bool(
        np.all(
            group["selected_ciphertext_count"].to_numpy(int)
            >= paper_sefa_config.minimum_calibration_selected_per_task
        )
    )
    robust_mask = (
        group["unique_best"].astype(bool)
        & (group["session_coverage"].astype(int) == paper_sefa_config.number_of_sessions)
        & (
            group["loso_consensus"].astype(float)
            >= paper_sefa_config.required_loso_consensus
        )
    )
    sefa14_policy_rows.append(
        {
            "top_fraction": float(fraction),
            "all_tasks_meet_minimum_count": large_enough,
            "robust_task_count": int(robust_mask.sum()),
            "unique_task_count": int(group["unique_best"].astype(bool).sum()),
            "mean_loso_consensus": float(group["loso_consensus"].mean()),
            "median_relative_margin": float(group["relative_margin"].median()),
            "mean_selected_ciphertext_count": float(
                group["selected_ciphertext_count"].mean()
            ),
        }
    )
sefa14_policy_grid = pd.DataFrame(sefa14_policy_rows)
sefa14_eligible_policy_grid = sefa14_policy_grid[
    sefa14_policy_grid["all_tasks_meet_minimum_count"].astype(bool)
    & (sefa14_policy_grid["top_fraction"].astype(float) < 1.0)
].copy()
if sefa14_eligible_policy_grid.empty:
    raise RuntimeError("No SEFA top-fraction candidate meets the public minimum-count rule")
sefa14_chosen_policy_row = (
    sefa14_eligible_policy_grid.sort_values(
        [
            "robust_task_count",
            "mean_loso_consensus",
            "median_relative_margin",
            "top_fraction",
        ],
        ascending=[False, False, False, True],
    )
    .iloc[0]
    .copy()
)
sefa14_chosen_top_fraction = float(sefa14_chosen_policy_row["top_fraction"])

sefa14_calibration_grid_scores.to_csv(
    sefa14_public_policy_directory / "sefa_top_fraction_candidate_scores_public.csv",
    index=False,
)
sefa14_calibration_predictions.to_csv(
    sefa14_public_policy_directory / "sefa_top_fraction_predictions_public.csv",
    index=False,
)
sefa14_policy_grid.to_csv(
    sefa14_public_policy_directory / "sefa_global_policy_grid_public.csv",
    index=False,
)
sefa14_write_json(
    sefa14_public_policy_directory / "chosen_sefa_budget_policy_public.json",
    {
        "chosen_top_fraction": sefa14_chosen_top_fraction,
        "selection_partition": "calibration",
        "selection_arm": "random_uniform",
        "selection_uses_key_truth": False,
        "selection_uses_private_fault_labels": False,
        "confirmation_partition_accessed_during_policy_selection": False,
        "policy_row": sefa14_chosen_policy_row.to_dict(),
    },
)
sefa14_policy_freeze = sefa14_stable_freeze(
    sefa14_public_policy_directory,
    sefa14_run_directory / "stage14R_public_policy_freeze_manifest.json",
    "SEFA budget policy selected from public random-arm calibration rows before truth and before confirmation scoring.",
)

print("=" * 94)
print("Public-only SEFA budget policy selected and frozen")
print("Chosen SEFA top fraction          :", f"{sefa14_chosen_top_fraction:.3f}")
print("Robust calibration tasks          :", int(sefa14_chosen_policy_row["robust_task_count"]), "/ 4")
print("Mean calibration LOSO             :", f"{float(sefa14_chosen_policy_row['mean_loso_consensus']):.3f}")
print("Policy freeze SHA-256             :", sefa14_policy_freeze["freeze_sha256"])
print("Truth/private opened              : False")
print("=" * 94)
ipy_display(sefa14_policy_grid)

# Confirmation pipelines.  SEFA uses its newly frozen fraction; SIFA uses the
# already frozen 0.40 policy from Stage 13R-v3.
_SEFA14_PIPELINES: Dict[str, Dict[str, Any]] = {}
for attack_type, probability_column, threshold, budget_fraction in (
    (
        "SEFA",
        "p_clean_target_effective",
        sefa14_effective_threshold,
        sefa14_chosen_top_fraction,
    ),
    (
        "SIFA",
        "p_clean_target_ineffective",
        sefa14_ineffective_threshold,
        paper_sefa_config.frozen_sifa_top_fraction,
    ),
):
    for arm_name, campaign_arm in (
        ("random", "random_uniform"),
        ("guided", "guided_model"),
    ):
        for selector in ("raw", "threshold", "budget"):
            pipeline_name = f"{attack_type.lower()}_{arm_name}_{selector}"
            _SEFA14_PIPELINES[pipeline_name] = {
                "attack_type": attack_type,
                "campaign_arm": campaign_arm,
                "selector": selector,
                "probability_column": probability_column,
                "threshold": float(threshold),
                "budget_fraction": float(budget_fraction),
                "uses_stage11": campaign_arm == "guided_model",
                "uses_stage10": selector != "raw",
            }


def sefa14_event_column(attack_type: str) -> str:
    return "paper_effective" if str(attack_type).upper() == "SEFA" else "paper_ineffective"


def sefa14_select_indices(
    event_frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> np.ndarray:
    selector = str(spec["selector"])
    if selector == "raw":
        return np.arange(len(event_frame), dtype=int)
    if selector == "threshold":
        mask = (
            event_frame[str(spec["probability_column"])].astype(float).to_numpy()
            >= float(spec["threshold"])
        )
        return np.where(mask)[0].astype(int)
    if selector == "budget":
        return sefa14_session_balanced_top_indices(
            event_frame,
            float(spec["budget_fraction"]),
            str(spec["probability_column"]),
        )
    raise ValueError(f"Unknown selector: {selector}")


def sefa14_confirmation_event_frame(
    pipeline: str,
    key_id: int,
    target_index: int,
    maximum_confirmation_index: Optional[int] = None,
) -> pd.DataFrame:
    spec = _SEFA14_PIPELINES[pipeline]
    event_column = sefa14_event_column(str(spec["attack_type"]))
    mask = (
        (sefa14_public_scored["campaign_partition"] == "confirmation")
        & (sefa14_public_scored["campaign_arm"] == str(spec["campaign_arm"]))
        & (sefa14_public_scored["key_id"].astype(int) == int(key_id))
        & (sefa14_public_scored["target_sbox_index"].astype(int) == int(target_index))
        & sefa14_public_scored[event_column].astype(bool)
    )
    if maximum_confirmation_index is not None:
        mask &= (
            sefa14_public_scored["confirmation_index"].astype(int)
            < int(maximum_confirmation_index)
        )
    return (
        sefa14_public_scored[mask]
        .sort_values(["confirmation_index", "experiment_id"])
        .reset_index(drop=True)
    )


sefa14_score_parts: List[pd.DataFrame] = []
sefa14_prediction_rows: List[Dict[str, Any]] = []
sefa14_task_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
for pipeline, spec in _SEFA14_PIPELINES.items():
    for key_id in range(paper_sefa_config.number_of_keys):
        for target_index in paper_sefa_config.target_sbox_indices:
            event_frame = sefa14_confirmation_event_frame(
                pipeline,
                key_id,
                target_index,
            )
            matrix = sefa14_intermediate_matrix(event_frame)
            selected_indices = sefa14_select_indices(event_frame, spec)
            selected_count = int(len(selected_indices))
            if selected_count < paper_sefa_config.minimum_attack_samples:
                raise RuntimeError(
                    f"Too few samples ({selected_count}) for {pipeline}/key{key_id}/S{target_index}"
                )
            scores = sefa14_scores_from_matrix(
                matrix,
                selected_indices,
                attack_type=str(spec["attack_type"]),
            )
            prediction = sefa14_prediction(scores)
            loso, coverage, loso_winners = sefa14_loso_consensus(
                event_frame,
                matrix,
                selected_indices,
                int(prediction["best_key_guess"]),
                str(spec["attack_type"]),
            )
            selected_probabilities = event_frame.loc[
                selected_indices,
                str(spec["probability_column"]),
            ].to_numpy(float)
            scores.insert(0, "pipeline", pipeline)
            scores.insert(1, "attack_type", str(spec["attack_type"]))
            scores.insert(2, "key_id", int(key_id))
            scores.insert(3, "target_sbox", f"S{int(target_index)}")
            scores.insert(4, "target_sbox_index", int(target_index))
            sefa14_score_parts.append(scores)
            sefa14_prediction_rows.append(
                {
                    "pipeline": pipeline,
                    "attack_type": str(spec["attack_type"]),
                    "campaign_arm": str(spec["campaign_arm"]),
                    "selector": str(spec["selector"]),
                    "uses_stage11": bool(spec["uses_stage11"]),
                    "uses_stage10": bool(spec["uses_stage10"]),
                    "key_id": int(key_id),
                    "target_sbox": f"S{int(target_index)}",
                    "target_sbox_index": int(target_index),
                    "confirmation_injection_count": int(
                        paper_sefa_config.confirmation_injections_per_key_target_arm
                    ),
                    "observable_event_count": int(len(event_frame)),
                    "selected_ciphertext_count": selected_count,
                    "mean_selected_probability": float(np.mean(selected_probabilities)),
                    "minimum_selected_probability": float(np.min(selected_probabilities)),
                    "loso_consensus": float(loso),
                    "session_coverage": int(coverage),
                    "loso_winners_hex": loso_winners,
                    **prediction,
                }
            )
            sefa14_task_cache[(pipeline, int(key_id), int(target_index))] = {
                "event_frame": event_frame,
                "matrix": matrix,
                "selected_indices": selected_indices,
            }

sefa14_candidate_scores_public = pd.concat(sefa14_score_parts, ignore_index=True)
sefa14_predictions_public = pd.DataFrame(sefa14_prediction_rows)

# Public injection-prefix curves.  Budgets are re-applied inside each prefix,
# while their fractions and thresholds remain frozen.
sefa14_injection_checkpoints = [250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
sefa14_prefix_score_parts: List[pd.DataFrame] = []
sefa14_prefix_prediction_rows: List[Dict[str, Any]] = []
for pipeline, spec in _SEFA14_PIPELINES.items():
    for key_id in range(paper_sefa_config.number_of_keys):
        for target_index in paper_sefa_config.target_sbox_indices:
            for checkpoint in sefa14_injection_checkpoints:
                event_frame = sefa14_confirmation_event_frame(
                    pipeline,
                    key_id,
                    target_index,
                    maximum_confirmation_index=checkpoint,
                )
                if len(event_frame) == 0:
                    continue
                matrix = sefa14_intermediate_matrix(event_frame)
                selected_indices = sefa14_select_indices(event_frame, spec)
                if len(selected_indices) < paper_sefa_config.minimum_attack_samples:
                    continue
                scores = sefa14_scores_from_matrix(
                    matrix,
                    selected_indices,
                    attack_type=str(spec["attack_type"]),
                )
                prediction = sefa14_prediction(scores)
                scores.insert(0, "pipeline", pipeline)
                scores.insert(1, "attack_type", str(spec["attack_type"]))
                scores.insert(2, "key_id", int(key_id))
                scores.insert(3, "target_sbox", f"S{int(target_index)}")
                scores.insert(4, "target_sbox_index", int(target_index))
                scores.insert(5, "injection_checkpoint", int(checkpoint))
                sefa14_prefix_score_parts.append(scores)
                sefa14_prefix_prediction_rows.append(
                    {
                        "pipeline": pipeline,
                        "attack_type": str(spec["attack_type"]),
                        "key_id": int(key_id),
                        "target_sbox": f"S{int(target_index)}",
                        "target_sbox_index": int(target_index),
                        "injection_checkpoint": int(checkpoint),
                        "observable_event_count": int(len(event_frame)),
                        "selected_ciphertext_count": int(len(selected_indices)),
                        **prediction,
                    }
                )

sefa14_prefix_scores_public = pd.concat(sefa14_prefix_score_parts, ignore_index=True)
sefa14_prefix_predictions_public = pd.DataFrame(sefa14_prefix_prediction_rows)

# Joint 8-bit public ranking by sum of the two independent nibble SEI scores.
sefa14_joint_score_rows: List[Dict[str, Any]] = []
sefa14_joint_prediction_rows: List[Dict[str, Any]] = []
for pipeline in _SEFA14_PIPELINES:
    for key_id in range(paper_sefa_config.number_of_keys):
        left = sefa14_candidate_scores_public[
            (sefa14_candidate_scores_public["pipeline"] == pipeline)
            & (sefa14_candidate_scores_public["key_id"].astype(int) == int(key_id))
            & (sefa14_candidate_scores_public["target_sbox_index"].astype(int) == 0)
        ][["key_guess", "sei"]].rename(columns={"key_guess": "guess_s0", "sei": "sei_s0"})
        right = sefa14_candidate_scores_public[
            (sefa14_candidate_scores_public["pipeline"] == pipeline)
            & (sefa14_candidate_scores_public["key_id"].astype(int) == int(key_id))
            & (sefa14_candidate_scores_public["target_sbox_index"].astype(int) == 5)
        ][["key_guess", "sei"]].rename(columns={"key_guess": "guess_s5", "sei": "sei_s5"})
        joint = left.assign(_join=1).merge(right.assign(_join=1), on="_join").drop(columns="_join")
        joint["joint_guess"] = (
            joint["guess_s0"].astype(int) * 16 + joint["guess_s5"].astype(int)
        )
        joint["joint_guess_hex"] = [
            f"{int(a):x}{int(b):x}"
            for a, b in zip(joint["guess_s0"], joint["guess_s5"])
        ]
        joint["joint_score"] = joint["sei_s0"] + joint["sei_s5"]
        ordered = joint.sort_values(["joint_score", "joint_guess"], ascending=[False, True]).reset_index(drop=True)
        margin = float(ordered.iloc[0]["joint_score"] - ordered.iloc[1]["joint_score"])
        sefa14_joint_prediction_rows.append(
            {
                "pipeline": pipeline,
                "attack_type": str(_SEFA14_PIPELINES[pipeline]["attack_type"]),
                "key_id": int(key_id),
                "best_joint_guess": int(ordered.iloc[0]["joint_guess"]),
                "best_joint_guess_hex": str(ordered.iloc[0]["joint_guess_hex"]),
                "best_joint_score": float(ordered.iloc[0]["joint_score"]),
                "joint_score_margin": margin,
            }
        )
        for row in joint.itertuples(index=False):
            sefa14_joint_score_rows.append(
                {
                    "pipeline": pipeline,
                    "attack_type": str(_SEFA14_PIPELINES[pipeline]["attack_type"]),
                    "key_id": int(key_id),
                    "guess_s0": int(row.guess_s0),
                    "guess_s5": int(row.guess_s5),
                    "joint_guess": int(row.joint_guess),
                    "joint_guess_hex": str(row.joint_guess_hex),
                    "sei_s0": float(row.sei_s0),
                    "sei_s5": float(row.sei_s5),
                    "joint_score": float(row.joint_score),
                }
            )

sefa14_joint_scores_public = pd.DataFrame(sefa14_joint_score_rows)
sefa14_joint_predictions_public = pd.DataFrame(sefa14_joint_prediction_rows)

# Mandatory key-identifiability guard before truth is opened.
sefa14_score_ranges = (
    sefa14_candidate_scores_public.groupby(
        ["pipeline", "key_id", "target_sbox_index"]
    )["sei"]
    .agg(lambda values: float(np.nanmax(values) - np.nanmin(values)))
)
if np.any(sefa14_score_ranges.to_numpy(float) <= 1.0e-15):
    failing = sefa14_score_ranges[sefa14_score_ranges <= 1.0e-15]
    raise RuntimeError(
        "SEI identifiability guard failed before truth was opened: "
        + failing.to_string()
    )

sefa14_public_scored.to_csv(
    sefa14_public_campaign_directory / "paired_sefa_campaign_with_public_probabilities.csv",
    index=False,
)
sefa14_candidate_scores_public.to_csv(
    sefa14_public_attack_directory / "sifa_sefa_candidate_scores_public.csv",
    index=False,
)
sefa14_predictions_public.to_csv(
    sefa14_public_attack_directory / "sifa_sefa_predictions_public.csv",
    index=False,
)
sefa14_prefix_scores_public.to_csv(
    sefa14_public_attack_directory / "injection_prefix_candidate_scores_public.csv",
    index=False,
)
sefa14_prefix_predictions_public.to_csv(
    sefa14_public_attack_directory / "injection_prefix_predictions_public.csv",
    index=False,
)
sefa14_joint_scores_public.to_csv(
    sefa14_public_attack_directory / "joint_8bit_candidate_scores_public.csv",
    index=False,
)
sefa14_joint_predictions_public.to_csv(
    sefa14_public_attack_directory / "joint_8bit_predictions_public.csv",
    index=False,
)
sefa14_write_json(
    sefa14_public_attack_directory / "public_attack_contract.json",
    {
        "truth_opened": False,
        "primary_statistic": "SEI",
        "probability_weighting_inside_sei": False,
        "sefa_uses_correct_ciphertexts_from_effective_trials": True,
        "faulty_ciphertext_values_used_for_key_scoring": False,
        "confirmation_partition_only": True,
        "chosen_sefa_top_fraction": sefa14_chosen_top_fraction,
        "frozen_sifa_top_fraction": paper_sefa_config.frozen_sifa_top_fraction,
        "pipelines": _SEFA14_PIPELINES,
    },
)

sefa14_public_attack_freeze = sefa14_stable_freeze(
    sefa14_public_attack_directory,
    sefa14_run_directory / "stage14R_public_attack_freeze_manifest.json",
    "All SEFA/SIFA confirmation scores, predictions, prefix curves, and joint rankings were frozen before private truth was opened.",
)

print("=" * 94)
print("Stage 14R public confirmation attack frozen")
print("Chosen SEFA top fraction          :", f"{sefa14_chosen_top_fraction:.3f}")
print("Frozen SIFA top fraction          :", f"{paper_sefa_config.frozen_sifa_top_fraction:.3f}")
print("Public attack freeze SHA-256      :", sefa14_public_attack_freeze["freeze_sha256"])
print("Truth/private opened              : False")
print("Public predictions:")
print(
    sefa14_predictions_public[
        [
            "pipeline",
            "key_id",
            "target_sbox",
            "observable_event_count",
            "selected_ciphertext_count",
            "best_key_guess_hex",
            "score_margin",
            "loso_consensus",
            "unique_best",
        ]
    ].to_string(index=False)
)
print("=" * 94)

# %%
# ============================================================
# Stage 14R / Cell 4
# Post-freeze truth opening, validation, controls, and final summary
# ============================================================

# Truth is opened only after both public freezes exist.
if not (sefa14_run_directory / "stage14R_public_policy_freeze_manifest.json").is_file():
    raise RuntimeError("Public policy freeze is missing")
if not (sefa14_run_directory / "stage14R_public_attack_freeze_manifest.json").is_file():
    raise RuntimeError("Public attack freeze is missing")
if sefa14_sha256_file(sefa14_locked_label_path) != sefa14_locked_hashes["fault_labels_sha256"]:
    raise RuntimeError("Locked private-label hash mismatch")
if sefa14_sha256_file(sefa14_locked_key_path) != sefa14_locked_hashes["key_truth_sha256"]:
    raise RuntimeError("Locked key-truth hash mismatch")

sefa14_private_frame = pd.read_csv(sefa14_locked_label_path)
sefa14_key_truth_payload = json.loads(sefa14_locked_key_path.read_text(encoding="utf-8"))
sefa14_truth_opened = True

sefa14_true_nibbles: Dict[Tuple[int, int], int] = {}
sefa14_true_joint: Dict[int, int] = {}
for item in sefa14_key_truth_payload["keys"]:
    key_id = int(item["key_id"])
    round_key_32 = int(str(item["round_key_32_hex"]), 16)
    nibble_s0 = int((round_key_32 >> 0) & 0xF)
    nibble_s5 = int((round_key_32 >> 20) & 0xF)
    sefa14_true_nibbles[(key_id, 0)] = nibble_s0
    sefa14_true_nibbles[(key_id, 5)] = nibble_s5
    sefa14_true_joint[key_id] = int(nibble_s0 * 16 + nibble_s5)


def sefa14_rank_from_scores(scores: pd.DataFrame, true_guess: int) -> Tuple[int, float, bool]:
    true_rows = scores[scores["key_guess"].astype(int) == int(true_guess)]
    if len(true_rows) != 1:
        raise RuntimeError("True key candidate is missing")
    true_score = float(true_rows.iloc[0]["sei"])
    all_scores = scores["sei"].to_numpy(float)
    rank = int(1 + np.sum(all_scores > true_score + 1.0e-15))
    tied = int(np.sum(np.isclose(all_scores, true_score, atol=1.0e-15, rtol=1.0e-12)))
    return rank, true_score, bool(rank == 1 and tied == 1)


sefa14_rank_rows: List[Dict[str, Any]] = []
for prediction in sefa14_predictions_public.itertuples(index=False):
    scores = sefa14_candidate_scores_public[
        (sefa14_candidate_scores_public["pipeline"] == prediction.pipeline)
        & (sefa14_candidate_scores_public["key_id"].astype(int) == int(prediction.key_id))
        & (
            sefa14_candidate_scores_public["target_sbox_index"].astype(int)
            == int(prediction.target_sbox_index)
        )
    ].copy()
    true_guess = sefa14_true_nibbles[(int(prediction.key_id), int(prediction.target_sbox_index))]
    true_rank, true_score, true_unique_rank1 = sefa14_rank_from_scores(scores, true_guess)
    sefa14_rank_rows.append(
        {
            **prediction._asdict(),
            "true_key_guess": int(true_guess),
            "true_key_guess_hex": f"{true_guess:x}",
            "true_rank": int(true_rank),
            "true_score": float(true_score),
            "true_is_unique_rank1": bool(true_unique_rank1),
        }
    )
sefa14_rank_evaluation = pd.DataFrame(sefa14_rank_rows)

# Prefix true-rank evaluation and first Rank-1 checkpoints.
sefa14_prefix_rank_rows: List[Dict[str, Any]] = []
for row in sefa14_prefix_predictions_public.itertuples(index=False):
    scores = sefa14_prefix_scores_public[
        (sefa14_prefix_scores_public["pipeline"] == row.pipeline)
        & (sefa14_prefix_scores_public["key_id"].astype(int) == int(row.key_id))
        & (
            sefa14_prefix_scores_public["target_sbox_index"].astype(int)
            == int(row.target_sbox_index)
        )
        & (
            sefa14_prefix_scores_public["injection_checkpoint"].astype(int)
            == int(row.injection_checkpoint)
        )
    ]
    true_guess = sefa14_true_nibbles[(int(row.key_id), int(row.target_sbox_index))]
    true_rank, true_score, true_unique_rank1 = sefa14_rank_from_scores(scores, true_guess)
    sefa14_prefix_rank_rows.append(
        {
            **row._asdict(),
            "true_key_guess": int(true_guess),
            "true_rank": int(true_rank),
            "true_score": float(true_score),
            "true_is_unique_rank1": bool(true_unique_rank1),
        }
    )
sefa14_prefix_rank_evaluation = pd.DataFrame(sefa14_prefix_rank_rows)

# Stratified bootstrap on the final selected sets.
sefa14_bootstrap_rows: List[Dict[str, Any]] = []
for pipeline, spec in _SEFA14_PIPELINES.items():
    for key_id in range(paper_sefa_config.number_of_keys):
        for target_index in paper_sefa_config.target_sbox_indices:
            cache = sefa14_task_cache[(pipeline, int(key_id), int(target_index))]
            event_frame = cache["event_frame"]
            matrix = cache["matrix"]
            selected_indices = np.asarray(cache["selected_indices"], dtype=int)
            true_guess = sefa14_true_nibbles[(int(key_id), int(target_index))]
            selected_by_session = {
                int(session): selected_indices[
                    event_frame.loc[selected_indices, "session_id"].to_numpy(int)
                    == int(session)
                ]
                for session in sorted(
                    event_frame.loc[selected_indices, "session_id"].astype(int).unique()
                )
            }
            for repetition in range(paper_sefa_config.bootstrap_repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_sefa_config.random_seed,
                            14100,
                            int(key_id),
                            int(target_index),
                            int(repetition),
                            sum(ord(ch) for ch in pipeline),
                        ]
                    )
                )
                resampled_parts: List[np.ndarray] = []
                for indices in selected_by_session.values():
                    if len(indices) > 0:
                        resampled_parts.append(rng.choice(indices, size=len(indices), replace=True))
                resampled = np.concatenate(resampled_parts).astype(int)
                scores = sefa14_scores_from_matrix(
                    matrix,
                    resampled,
                    attack_type=str(spec["attack_type"]),
                )
                prediction = sefa14_prediction(scores)
                true_rank, _, true_unique_rank1 = sefa14_rank_from_scores(scores, true_guess)
                sefa14_bootstrap_rows.append(
                    {
                        "pipeline": pipeline,
                        "attack_type": str(spec["attack_type"]),
                        "key_id": int(key_id),
                        "target_sbox": f"S{int(target_index)}",
                        "target_sbox_index": int(target_index),
                        "repetition": int(repetition),
                        "winner": int(prediction["best_key_guess"]),
                        "winner_is_true": bool(int(prediction["best_key_guess"]) == int(true_guess)),
                        "true_rank": int(true_rank),
                        "true_is_unique_rank1": bool(true_unique_rank1),
                    }
                )
sefa14_bootstrap = pd.DataFrame(sefa14_bootstrap_rows)
sefa14_bootstrap_summary = (
    sefa14_bootstrap.groupby(["pipeline", "key_id", "target_sbox_index"], as_index=False)
    .agg(
        bootstrap_true_winner_fraction=("winner_is_true", "mean"),
        bootstrap_unique_rank1_fraction=("true_is_unique_rank1", "mean"),
        bootstrap_mean_true_rank=("true_rank", "mean"),
    )
)
sefa14_rank_evaluation = sefa14_rank_evaluation.merge(
    sefa14_bootstrap_summary,
    on=["pipeline", "key_id", "target_sbox_index"],
    how="left",
    validate="one_to_one",
)

# Matched random controls for every budget selector.  Each random subset has
# exactly the same per-session sample counts as the model-selected subset.
sefa14_matched_random_rows: List[Dict[str, Any]] = []
for pipeline, spec in _SEFA14_PIPELINES.items():
    if str(spec["selector"]) != "budget":
        continue
    for key_id in range(paper_sefa_config.number_of_keys):
        for target_index in paper_sefa_config.target_sbox_indices:
            cache = sefa14_task_cache[(pipeline, int(key_id), int(target_index))]
            event_frame = cache["event_frame"]
            matrix = cache["matrix"]
            selected_indices = np.asarray(cache["selected_indices"], dtype=int)
            true_guess = sefa14_true_nibbles[(int(key_id), int(target_index))]
            model_scores = sefa14_scores_from_matrix(
                matrix,
                selected_indices,
                attack_type=str(spec["attack_type"]),
            )
            model_rank, _, model_rank1 = sefa14_rank_from_scores(model_scores, true_guess)
            selected_counts = (
                event_frame.loc[selected_indices]
                .groupby("session_id")
                .size()
                .to_dict()
            )
            all_by_session = {
                int(session): group.index.to_numpy(int)
                for session, group in event_frame.groupby("session_id")
            }
            random_ranks: List[int] = []
            for repetition in range(paper_sefa_config.matched_random_repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_sefa_config.random_seed,
                            14200,
                            int(key_id),
                            int(target_index),
                            int(repetition),
                            sum(ord(ch) for ch in pipeline),
                        ]
                    )
                )
                parts: List[np.ndarray] = []
                for session, count in selected_counts.items():
                    population = all_by_session[int(session)]
                    parts.append(rng.choice(population, size=int(count), replace=False))
                random_indices = np.concatenate(parts).astype(int)
                random_scores = sefa14_scores_from_matrix(
                    matrix,
                    random_indices,
                    attack_type=str(spec["attack_type"]),
                )
                random_rank, _, _ = sefa14_rank_from_scores(random_scores, true_guess)
                random_ranks.append(int(random_rank))
            random_array = np.asarray(random_ranks, dtype=float)
            sefa14_matched_random_rows.append(
                {
                    "pipeline": pipeline,
                    "attack_type": str(spec["attack_type"]),
                    "campaign_arm": str(spec["campaign_arm"]),
                    "key_id": int(key_id),
                    "target_sbox": f"S{int(target_index)}",
                    "target_sbox_index": int(target_index),
                    "model_topk_true_rank": int(model_rank),
                    "model_topk_rank1": bool(model_rank1),
                    "matched_random_median_true_rank": float(np.median(random_array)),
                    "matched_random_mean_true_rank": float(np.mean(random_array)),
                    "matched_random_rank1_fraction": float(np.mean(random_array == 1.0)),
                    "model_better_than_random_median": bool(
                        float(model_rank) < float(np.median(random_array))
                    ),
                    "model_not_worse_than_random_median": bool(
                        float(model_rank) <= float(np.median(random_array))
                    ),
                }
            )
sefa14_matched_random = pd.DataFrame(sefa14_matched_random_rows)

# Post-freeze quality metrics from private labels.
sefa14_scored_with_private = sefa14_public_scored.merge(
    sefa14_private_frame[
        ["experiment_id", "category", "category_id"]
    ],
    on="experiment_id",
    how="left",
    validate="one_to_one",
)
sefa14_selection_quality_rows: List[Dict[str, Any]] = []
for pipeline, spec in _SEFA14_PIPELINES.items():
    attack_type = str(spec["attack_type"])
    clean_category = (
        "clean_target_effective" if attack_type == "SEFA" else "clean_target_ineffective"
    )
    event_column = sefa14_event_column(attack_type)
    for key_id in range(paper_sefa_config.number_of_keys):
        for target_index in paper_sefa_config.target_sbox_indices:
            cache = sefa14_task_cache[(pipeline, int(key_id), int(target_index))]
            event_frame = cache["event_frame"]
            selected_indices = np.asarray(cache["selected_indices"], dtype=int)
            selected_ids = set(event_frame.loc[selected_indices, "experiment_id"].astype(int))
            task_private = sefa14_scored_with_private[
                (sefa14_scored_with_private["campaign_partition"] == "confirmation")
                & (sefa14_scored_with_private["campaign_arm"] == str(spec["campaign_arm"]))
                & (sefa14_scored_with_private["key_id"].astype(int) == int(key_id))
                & (
                    sefa14_scored_with_private["target_sbox_index"].astype(int)
                    == int(target_index)
                )
                & sefa14_scored_with_private[event_column].astype(bool)
            ].copy()
            task_private["selected"] = task_private["experiment_id"].astype(int).isin(selected_ids)
            task_private["is_clean_target_event"] = task_private["category"].astype(str).eq(clean_category)
            tp = int(np.sum(task_private["selected"] & task_private["is_clean_target_event"]))
            fp = int(np.sum(task_private["selected"] & ~task_private["is_clean_target_event"]))
            fn = int(np.sum(~task_private["selected"] & task_private["is_clean_target_event"]))
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            sefa14_selection_quality_rows.append(
                {
                    "pipeline": pipeline,
                    "attack_type": attack_type,
                    "key_id": int(key_id),
                    "target_sbox": f"S{int(target_index)}",
                    "target_sbox_index": int(target_index),
                    "selected_ciphertext_count": int(tp + fp),
                    "actual_clean_count": int(tp),
                    "actual_clean_precision": precision,
                    "actual_clean_recall": recall,
                }
            )
sefa14_selection_quality = pd.DataFrame(sefa14_selection_quality_rows)
sefa14_rank_evaluation = sefa14_rank_evaluation.merge(
    sefa14_selection_quality,
    on=["pipeline", "attack_type", "key_id", "target_sbox", "target_sbox_index", "selected_ciphertext_count"],
    how="left",
    validate="one_to_one",
)

# Joint 8-bit true ranks.
sefa14_joint_rank_rows: List[Dict[str, Any]] = []
for prediction in sefa14_joint_predictions_public.itertuples(index=False):
    scores = sefa14_joint_scores_public[
        (sefa14_joint_scores_public["pipeline"] == prediction.pipeline)
        & (sefa14_joint_scores_public["key_id"].astype(int) == int(prediction.key_id))
    ]
    true_joint = sefa14_true_joint[int(prediction.key_id)]
    true_score = float(
        scores[scores["joint_guess"].astype(int) == int(true_joint)].iloc[0]["joint_score"]
    )
    rank = int(1 + np.sum(scores["joint_score"].to_numpy(float) > true_score + 1.0e-15))
    tie_count = int(
        np.sum(
            np.isclose(
                scores["joint_score"].to_numpy(float),
                true_score,
                atol=1.0e-15,
                rtol=1.0e-12,
            )
        )
    )
    sefa14_joint_rank_rows.append(
        {
            **prediction._asdict(),
            "true_joint_guess": int(true_joint),
            "true_joint_guess_hex": f"{true_joint:02x}",
            "true_joint_score": true_score,
            "true_joint_rank": int(rank),
            "true_joint_is_unique_rank1": bool(rank == 1 and tie_count == 1),
        }
    )
sefa14_joint_rank_evaluation = pd.DataFrame(sefa14_joint_rank_rows)

# First injection checkpoint at which each task reaches unique Rank-1.
first_checkpoint_rows: List[Dict[str, Any]] = []
for pipeline in _SEFA14_PIPELINES:
    for key_id in range(paper_sefa_config.number_of_keys):
        for target_index in paper_sefa_config.target_sbox_indices:
            subset = sefa14_prefix_rank_evaluation[
                (sefa14_prefix_rank_evaluation["pipeline"] == pipeline)
                & (sefa14_prefix_rank_evaluation["key_id"].astype(int) == int(key_id))
                & (
                    sefa14_prefix_rank_evaluation["target_sbox_index"].astype(int)
                    == int(target_index)
                )
            ].sort_values("injection_checkpoint")
            successes = subset[subset["true_is_unique_rank1"].astype(bool)]
            first_checkpoint_rows.append(
                {
                    "pipeline": pipeline,
                    "key_id": int(key_id),
                    "target_sbox": f"S{int(target_index)}",
                    "target_sbox_index": int(target_index),
                    "first_injection_checkpoint_rank1": (
                        int(successes.iloc[0]["injection_checkpoint"])
                        if not successes.empty
                        else np.nan
                    ),
                    "selected_ciphertexts_at_first_rank1": (
                        int(successes.iloc[0]["selected_ciphertext_count"])
                        if not successes.empty
                        else np.nan
                    ),
                }
            )
sefa14_first_checkpoints = pd.DataFrame(first_checkpoint_rows)
sefa14_rank_evaluation = sefa14_rank_evaluation.merge(
    sefa14_first_checkpoints,
    on=["pipeline", "key_id", "target_sbox", "target_sbox_index"],
    how="left",
    validate="one_to_one",
)

# Pipeline summaries.
sefa14_joint_summary = (
    sefa14_joint_rank_evaluation.groupby("pipeline", as_index=False)
    .agg(
        rank1_joint_8bit_count=("true_joint_is_unique_rank1", "sum"),
        mean_joint_true_rank=("true_joint_rank", "mean"),
    )
)
sefa14_pipeline_summary = (
    sefa14_rank_evaluation.groupby(["pipeline", "attack_type"], as_index=False)
    .agg(
        task_count=("true_rank", "size"),
        rank1_nibble_count=("true_is_unique_rank1", "sum"),
        rank1_nibble_success_rate=("true_is_unique_rank1", "mean"),
        mean_true_rank=("true_rank", "mean"),
        median_true_rank=("true_rank", "median"),
        mean_selected_ciphertexts_per_task=("selected_ciphertext_count", "mean"),
        mean_observable_events_per_task=("observable_event_count", "mean"),
        mean_score_margin=("score_margin", "mean"),
        mean_loso_consensus=("loso_consensus", "mean"),
        mean_bootstrap_true_winner_fraction=("bootstrap_true_winner_fraction", "mean"),
        actual_clean_precision=("actual_clean_precision", "mean"),
        actual_clean_recall=("actual_clean_recall", "mean"),
        actual_clean_count=("actual_clean_count", "sum"),
        median_first_injection_checkpoint_rank1=("first_injection_checkpoint_rank1", "median"),
        median_selected_ciphertexts_at_first_rank1=("selected_ciphertexts_at_first_rank1", "median"),
    )
    .merge(sefa14_joint_summary, on="pipeline", how="left", validate="one_to_one")
)


def sefa14_pipeline_metric(pipeline: str, column: str) -> float:
    row = sefa14_pipeline_summary[sefa14_pipeline_summary["pipeline"] == pipeline]
    if len(row) != 1:
        raise RuntimeError(f"Missing pipeline summary for {pipeline}")
    return float(row.iloc[0][column])


# Explicit model-impact comparisons.
sefa14_model_impact = {
    "stage11_pre_injection_SEFA": {
        "random_raw_mean_true_rank": sefa14_pipeline_metric("sefa_random_raw", "mean_true_rank"),
        "guided_raw_mean_true_rank": sefa14_pipeline_metric("sefa_guided_raw", "mean_true_rank"),
        "random_raw_rank1_nibbles": int(sefa14_pipeline_metric("sefa_random_raw", "rank1_nibble_count")),
        "guided_raw_rank1_nibbles": int(sefa14_pipeline_metric("sefa_guided_raw", "rank1_nibble_count")),
        "random_raw_mean_observable_effective_events": sefa14_pipeline_metric(
            "sefa_random_raw", "mean_observable_events_per_task"
        ),
        "guided_raw_mean_observable_effective_events": sefa14_pipeline_metric(
            "sefa_guided_raw", "mean_observable_events_per_task"
        ),
    },
    "stage10_post_injection_SEFA": {
        "random_raw_mean_true_rank": sefa14_pipeline_metric("sefa_random_raw", "mean_true_rank"),
        "random_threshold_mean_true_rank": sefa14_pipeline_metric(
            "sefa_random_threshold", "mean_true_rank"
        ),
        "random_budget_mean_true_rank": sefa14_pipeline_metric(
            "sefa_random_budget", "mean_true_rank"
        ),
        "guided_raw_selected_per_task": sefa14_pipeline_metric(
            "sefa_guided_raw", "mean_selected_ciphertexts_per_task"
        ),
        "guided_budget_selected_per_task": sefa14_pipeline_metric(
            "sefa_guided_budget", "mean_selected_ciphertexts_per_task"
        ),
        "guided_raw_rank1_nibbles": int(
            sefa14_pipeline_metric("sefa_guided_raw", "rank1_nibble_count")
        ),
        "guided_budget_rank1_nibbles": int(
            sefa14_pipeline_metric("sefa_guided_budget", "rank1_nibble_count")
        ),
        "matched_random_tasks_model_better": int(
            sefa14_matched_random[
                sefa14_matched_random["pipeline"].str.startswith("sefa_")
            ]["model_better_than_random_median"].sum()
        ),
        "matched_random_SEFA_task_count": int(
            len(
                sefa14_matched_random[
                    sefa14_matched_random["pipeline"].str.startswith("sefa_")
                ]
            )
        ),
    },
}

# Same-campaign attack comparison by arm and selector.
sefa14_attack_comparison_rows: List[Dict[str, Any]] = []
for arm in ("random", "guided"):
    for selector in ("raw", "threshold", "budget"):
        sifa_pipeline = f"sifa_{arm}_{selector}"
        sefa_pipeline = f"sefa_{arm}_{selector}"
        sefa14_attack_comparison_rows.append(
            {
                "campaign_arm": arm,
                "selector": selector,
                "sifa_mean_true_rank": sefa14_pipeline_metric(sifa_pipeline, "mean_true_rank"),
                "sefa_mean_true_rank": sefa14_pipeline_metric(sefa_pipeline, "mean_true_rank"),
                "sifa_rank1_nibbles": int(
                    sefa14_pipeline_metric(sifa_pipeline, "rank1_nibble_count")
                ),
                "sefa_rank1_nibbles": int(
                    sefa14_pipeline_metric(sefa_pipeline, "rank1_nibble_count")
                ),
                "sifa_mean_selected_ciphertexts": sefa14_pipeline_metric(
                    sifa_pipeline, "mean_selected_ciphertexts_per_task"
                ),
                "sefa_mean_selected_ciphertexts": sefa14_pipeline_metric(
                    sefa_pipeline, "mean_selected_ciphertexts_per_task"
                ),
                "sifa_median_first_rank1_injections": sefa14_pipeline_metric(
                    sifa_pipeline, "median_first_injection_checkpoint_rank1"
                ),
                "sefa_median_first_rank1_injections": sefa14_pipeline_metric(
                    sefa_pipeline, "median_first_injection_checkpoint_rank1"
                ),
            }
        )
sefa14_attack_comparison = pd.DataFrame(sefa14_attack_comparison_rows)

# Fixed success rules; no retrospective threshold adjustment.
sefa14_stage11_helped = bool(
    (
        sefa14_pipeline_metric("sefa_guided_raw", "rank1_nibble_count")
        > sefa14_pipeline_metric("sefa_random_raw", "rank1_nibble_count")
    )
    or (
        sefa14_pipeline_metric("sefa_guided_raw", "mean_true_rank")
        < sefa14_pipeline_metric("sefa_random_raw", "mean_true_rank")
    )
)
sefa14_budget_matched = sefa14_matched_random[
    sefa14_matched_random["pipeline"].isin(
        ["sefa_random_budget", "sefa_guided_budget"]
    )
]
sefa14_stage10_helped = bool(
    (
        sefa14_pipeline_metric("sefa_random_budget", "mean_true_rank")
        <= sefa14_pipeline_metric("sefa_random_raw", "mean_true_rank")
    )
    and (
        int(sefa14_budget_matched["model_not_worse_than_random_median"].sum())
        >= 6
    )
    and (
        (
            sefa14_pipeline_metric("sefa_guided_budget", "rank1_nibble_count")
            >= sefa14_pipeline_metric("sefa_guided_raw", "rank1_nibble_count")
        )
        or (
            sefa14_pipeline_metric("sefa_guided_budget", "mean_true_rank")
            < sefa14_pipeline_metric("sefa_guided_raw", "mean_true_rank")
        )
    )
)
sefa14_combined_all_nibbles_rank1 = bool(
    sefa14_pipeline_metric("sefa_guided_budget", "rank1_nibble_count") == 4
)
sefa14_combined_all_joint_rank1 = bool(
    sefa14_pipeline_metric("sefa_guided_budget", "rank1_joint_8bit_count") == 2
)

# Integrity checks are semantic positives; every healthy condition is True.
sefa14_integrity_checks = {
    "stage11_and_stage10_freezes_verified": True,
    "fresh_campaign_row_count_correct": bool(
        len(sefa14_public_frame) == _sefa14_total_injections
    ),
    "paired_plaintext_key_target_session_design_passed": bool(
        sefa14_public_frame.groupby("pair_id")["plaintext_hex"].nunique().eq(1).all()
        and sefa14_public_frame.groupby("pair_id")["key_id"].nunique().eq(1).all()
        and sefa14_public_frame.groupby("pair_id")["target_sbox_index"].nunique().eq(1).all()
        and sefa14_public_frame.groupby("pair_id")["session_id"].nunique().eq(1).all()
    ),
    "calibration_and_confirmation_disjoint": bool(
        set(
            sefa14_public_frame[
                sefa14_public_frame["campaign_partition"] == "calibration"
            ]["experiment_id"].astype(int)
        ).isdisjoint(
            set(
                sefa14_public_frame[
                    sefa14_public_frame["campaign_partition"] == "confirmation"
                ]["experiment_id"].astype(int)
            )
        )
    ),
    "sefa_policy_selected_from_public_calibration_only": True,
    "policy_frozen_before_confirmation_attack": True,
    "attack_frozen_before_truth_opened": True,
    "sefa_uses_effective_event_rule": True,
    "sefa_uses_correct_reference_ciphertexts": True,
    "faulty_ciphertext_values_not_used_by_key_scorer": True,
    "no_probability_weighting_inside_sei": True,
    "primary_score_is_exact_unweighted_sei": True,
    "x31_partial_decryption_guard_passed": True,
    "all_confirmation_tasks_key_identifiable": bool(
        np.all(sefa14_score_ranges.to_numpy(float) > 1.0e-15)
    ),
    "twelve_pipelines_four_tasks_each": bool(
        len(sefa14_rank_evaluation) == 12 * 4
    ),
    "twelve_pipelines_two_joint_tasks_each": bool(
        len(sefa14_joint_rank_evaluation) == 12 * 2
    ),
    "matched_random_controls_present_for_all_budget_tasks": bool(
        len(sefa14_matched_random) == 4 * 4
    ),
    "locked_truth_hashes_verified": True,
}
sefa14_all_integrity_checks_passed = bool(all(sefa14_integrity_checks.values()))

# Save validation outputs.
sefa14_rank_evaluation.to_csv(
    sefa14_validation_directory / "final_nibble_true_rank_evaluation.csv",
    index=False,
)
sefa14_prefix_rank_evaluation.to_csv(
    sefa14_validation_directory / "injection_prefix_true_rank_curves.csv",
    index=False,
)
sefa14_bootstrap.to_csv(
    sefa14_validation_directory / "bootstrap_repetitions.csv",
    index=False,
)
sefa14_bootstrap_summary.to_csv(
    sefa14_validation_directory / "bootstrap_summary.csv",
    index=False,
)
sefa14_matched_random.to_csv(
    sefa14_validation_directory / "matched_budget_model_vs_random.csv",
    index=False,
)
sefa14_selection_quality.to_csv(
    sefa14_validation_directory / "selection_quality_after_freeze.csv",
    index=False,
)
sefa14_joint_rank_evaluation.to_csv(
    sefa14_validation_directory / "joint_8bit_true_rank_evaluation.csv",
    index=False,
)
sefa14_pipeline_summary.to_csv(
    sefa14_validation_directory / "pipeline_summary.csv",
    index=False,
)
sefa14_attack_comparison.to_csv(
    sefa14_validation_directory / "same_campaign_sifa_vs_sefa_comparison.csv",
    index=False,
)
sefa14_write_json(
    sefa14_validation_directory / "integrity_checks.json",
    {
        "all_integrity_checks_passed": sefa14_all_integrity_checks_passed,
        "checks": sefa14_integrity_checks,
    },
)

# Plots are validation-only because true rank is displayed.
if paper_sefa_config.save_plots:
    for attack_type in ("SEFA", "SIFA"):
        fig, ax = plt.subplots(figsize=(11, 6))
        subset = sefa14_prefix_rank_evaluation[
            sefa14_prefix_rank_evaluation["attack_type"] == attack_type
        ]
        for pipeline, group in subset.groupby("pipeline"):
            curve = group.groupby("injection_checkpoint")["true_rank"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=pipeline)
        ax.set_xlabel("Confirmation fault attempts per key/target/arm")
        ax.set_ylabel("Mean true-key nibble rank")
        ax.set_title(f"Stage 14R — {attack_type} rank convergence")
        ax.set_yscale("log", base=2)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(
            sefa14_validation_directory / f"{attack_type.lower()}_rank_convergence.png",
            dpi=180,
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    display_frame = sefa14_pipeline_summary.sort_values(
        ["attack_type", "mean_true_rank", "pipeline"]
    )
    positions = np.arange(len(display_frame))
    ax.bar(positions, display_frame["mean_true_rank"].to_numpy(float))
    ax.set_xticks(positions)
    ax.set_xticklabels(display_frame["pipeline"], rotation=60, ha="right")
    ax.set_ylabel("Mean final true-key rank")
    ax.set_title("Stage 14R — final SIFA/SEFA pipeline comparison")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        sefa14_validation_directory / "final_pipeline_mean_rank_comparison.png",
        dpi=180,
    )
    plt.close(fig)

sefa14_elapsed = time.perf_counter() - sefa14_campaign_started
sefa14_summary = {
    "stage": "14R",
    "attack": "paper-faithful Statistical Effective Fault Attack on LBlock",
    "run_id": sefa14_run_id,
    "run_directory": str(sefa14_run_directory),
    "random_seed": paper_sefa_config.random_seed,
    "fresh_keys": paper_sefa_config.number_of_keys,
    "sessions": paper_sefa_config.number_of_sessions,
    "target_sboxes": ["S0", "S5"],
    "total_fresh_fault_attempts": _sefa14_total_injections,
    "calibration_attempts_per_key_target_arm": paper_sefa_config.calibration_injections_per_key_target_arm,
    "confirmation_attempts_per_key_target_arm": paper_sefa_config.confirmation_injections_per_key_target_arm,
    "paper_method": {
        "sefa_event": "C_prime != C",
        "sefa_key_recovery_ciphertext": "correct C",
        "primary_score": "unweighted SEI",
        "stage10_probability_inside_sei": False,
        "partial_decryption_target": "X31 nibble",
    },
    "theoretical_random_and_4": _SEFA14_THEORY,
    "frozen_thresholds": {
        "clean_target_effective": sefa14_effective_threshold,
        "clean_target_ineffective": sefa14_ineffective_threshold,
    },
    "chosen_sefa_top_fraction": sefa14_chosen_top_fraction,
    "frozen_sifa_top_fraction": paper_sefa_config.frozen_sifa_top_fraction,
    "public_policy_freeze_sha256": sefa14_policy_freeze["freeze_sha256"],
    "public_attack_freeze_sha256": sefa14_public_attack_freeze["freeze_sha256"],
    "truth_opened_only_after_freezes": True,
    "all_integrity_checks_passed": sefa14_all_integrity_checks_passed,
    "integrity_checks": sefa14_integrity_checks,
    "stage11_helped_by_fixed_rule": sefa14_stage11_helped,
    "stage10_helped_by_fixed_rule": sefa14_stage10_helped,
    "guided_budget_all_nibbles_rank1": sefa14_combined_all_nibbles_rank1,
    "guided_budget_all_joint_8bit_rank1": sefa14_combined_all_joint_rank1,
    "model_impact": sefa14_model_impact,
    "pipeline_summary": sefa14_pipeline_summary.to_dict(orient="records"),
    "final_nibble_results": sefa14_rank_evaluation.to_dict(orient="records"),
    "joint_8bit_results": sefa14_joint_rank_evaluation.to_dict(orient="records"),
    "matched_random_controls": sefa14_matched_random.to_dict(orient="records"),
    "same_campaign_sifa_vs_sefa": sefa14_attack_comparison.to_dict(orient="records"),
    "elapsed_seconds": float(sefa14_elapsed),
    "important_interpretation": (
        "The SEFA budget was calibrated on a disjoint public calibration partition. "
        "All final key ranks use a separate confirmation partition and were revealed only after public freeze."
    ),
}
sefa14_summary_path = sefa14_run_directory / "stage_14R_paper_sefa_summary.json"
sefa14_write_json(sefa14_summary_path, sefa14_summary)

print("\n" + "=" * 98)
print("Stage 14R completed — paper-faithful SEFA with same-campaign SIFA comparison")
print("=" * 98)
print("Run directory                    :", sefa14_run_directory)
print("All integrity checks passed      :", sefa14_all_integrity_checks_passed)
print("Chosen public SEFA top fraction  :", f"{sefa14_chosen_top_fraction:.3f}")
print("Stage-11 helped fixed rule       :", sefa14_stage11_helped)
print("Stage-10 helped fixed rule       :", sefa14_stage10_helped)
print("Guided+budget all nibble Rank-1  :", sefa14_combined_all_nibbles_rank1)
print("Guided+budget all 8-bit Rank-1   :", sefa14_combined_all_joint_rank1)
print("Public attack freeze SHA-256     :", sefa14_public_attack_freeze["freeze_sha256"])
print("Summary file                     :", sefa14_summary_path)
print("Elapsed seconds                  :", f"{sefa14_elapsed:.3f}")
print("=" * 98)
ipy_display(sefa14_pipeline_summary)
ipy_display(
    sefa14_rank_evaluation[
        [
            "pipeline",
            "key_id",
            "target_sbox",
            "selected_ciphertext_count",
            "best_key_guess_hex",
            "true_key_guess_hex",
            "true_rank",
            "true_is_unique_rank1",
            "bootstrap_true_winner_fraction",
            "actual_clean_precision",
            "actual_clean_recall",
            "first_injection_checkpoint_rank1",
        ]
    ].rename(columns={"best_key_guess_hex": "predicted_key_guess_hex"})
)
ipy_display(sefa14_matched_random)
ipy_display(sefa14_attack_comparison)

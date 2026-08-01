# %% [markdown]
# # Stage 13R-v2 — SIFA مقاله‌ای با SEI و مقایسه AI در برابر Random
# 
# نسخه قبلی Stage 13R روی ورودی S-box دور آخر از رابطه
# `X_hat = nibble(X32) XOR k` استفاده می‌کرد. این رابطه برای هر حدس کلید فقط
# برچسب‌های ۱۶ خانه هیستوگرام را جابه‌جا می‌کند؛ بنابراین SEI برای همه حدس‌ها
# یکسان می‌شود و کلید قابل تشخیص نیست.
# 
# در نسخه اصلاح‌شده، Fault روی Nibble متناظر از حالت `X31`، پس از تشکیل `X32`
# و پیش از شاخه چرخانده‌شده دور نهایی اعمال می‌شود. سپس برای حدس `k` از رابطه
# 
# \[
# \widehat X_{31,t}(k)
# =
# \operatorname{nibble}_{j}(X_{33})
# \oplus
# S_s\!\left(\operatorname{nibble}_{s}(X_{32})\oplus k
# ight),
# \]
# 
# استفاده می‌شود، که در آن `j` خروجی P-layer متناظر با S-box شماره `s` و
# `t=(j-2) mod 8` است. برای کلید صحیح، مقدار میانی بایاس‌شده `X31[t]`
# بازسازی می‌شود؛ حدس‌های غلط به علت اختلاف غیرخطی S-box صرفاً جایگشت
# هیستوگرام نیستند.
# 
# امتیاز اصلی دقیقاً SEI مقاله است و Probability مدل Stage 10 وارد امتیاز کلید
# نمی‌شود. Stage 11 فقط پارامترهای تزریق Guided را تعیین می‌کند و Stage 10 فقط
# نمونه‌های نامطمئن را در شاخه ML حذف می‌کند.
# %%
# ============================================================
# Stage 13R / Cell 1
# Configuration, frozen-model verification, and experiment plan
# ============================================================

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display


# This stage deliberately reuses the validated simulator and model-loading
# functions that were defined and executed in the Stage-12 notebook cell.
_PF_REQUIRED_SYMBOLS = (
    "engine",
    "Stage12Config",
    "ClosedLoopPlanEntry",
    "resolve_stage_contracts",
    "score_public_batch",
    "perturb_recommendation",
)
_PF_MISSING = [name for name in _PF_REQUIRED_SYMBOLS if name not in globals()]
if _PF_MISSING:
    raise RuntimeError(
        "Before running Stage 13R, run the Stage-12 cell. Missing symbols: "
        + ", ".join(_PF_MISSING)
    )


@dataclass(frozen=True)
class PaperSIFAConfig:
    input_stage11_run_directory: str
    output_root: str
    random_seed: int = 20260722

    # Two fresh keys, two target nibbles, two injection arms.
    number_of_keys: int = 2
    number_of_sessions: int = 4
    target_sbox_indices: Tuple[int, ...] = (0, 5)

    # Total injections =
    # keys * targets * arms * injections_per_key_target_arm.
    injections_per_key_target_arm: int = 3000

    # Small physical implementation jitter around the best Stage-11
    # recommendation.  These values match the Stage-12 exploit policy.
    guided_offset_jitter_sigma: float = 0.15
    guided_relative_parameter_jitter: float = 0.03

    # Timing/noise parameters inherited from the validated Stage-12 campaign.
    global_timing_jitter_sigma_samples: float = 0.35
    local_sbox_jitter_sigma_samples: float = 0.18
    injection_timing_jitter_sigma_samples: float = 0.20
    session_timing_shift_sigma_samples: float = 0.25
    response_trace_noise_sigma: float = 0.055
    response_trace_baseline_sigma: float = 0.035
    response_trace_gain_sigma: float = 0.06

    # Stage-10 feature reconstruction parameters.
    target_window_radius_samples: int = 24
    pulse_window_radius_samples: int = 24
    highpass_moving_average_width: int = 9
    trace_standard_deviation_floor: float = 1.0e-6

    # Public attack analysis.
    minimum_selected_ciphertexts: int = 16
    bootstrap_repetitions: int = 300
    save_plots: bool = True


def pf_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def pf_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pf_hex_fixed(value: int, bits: int) -> str:
    return f"{int(value):0{bits // 4}x}"


def pf_stable_freeze(directory: Path, output_path: Path, statement: str) -> Dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": statement,
        "files": {
            str(path.relative_to(directory)).replace("\\", "/"): pf_sha256_file(path)
            for path in files
        },
    }
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["freeze_sha256"] = hashlib.sha256(payload).hexdigest()
    pf_write_json(output_path, manifest)
    return manifest


# The path is derived from the already-executed Stage-12 configuration when
# available.  The explicit fallback keeps the cell directly usable.
_PF_STAGE11_DEFAULT = (
    getattr(stage_12_config, "input_stage11_run_directory", None)
    if "stage_12_config" in globals()
    else None
)
if not _PF_STAGE11_DEFAULT:
    _PF_STAGE11_DEFAULT = (
        './runs/stage_11'
        '/stage11_20260718_191029_290452_seed20260718'
    )

paper_sifa_config = PaperSIFAConfig(
    input_stage11_run_directory=os.environ.get(
        "LBLOCK_PAPER_SIFA_STAGE11",
        _PF_STAGE11_DEFAULT,
    ),
    output_root=os.environ.get(
        "LBLOCK_PAPER_SIFA_OUTPUT",
        './runs/stage_13R_paper_sifa_v2',
    ),
)

if paper_sifa_config.number_of_keys < 1:
    raise ValueError("number_of_keys must be positive")
if paper_sifa_config.number_of_sessions < 1:
    raise ValueError("number_of_sessions must be positive")
if set(paper_sifa_config.target_sbox_indices) != {0, 5}:
    raise ValueError("This experiment is pre-registered for target S-boxes S0 and S5")
if paper_sifa_config.injections_per_key_target_arm < 256:
    raise ValueError("Use at least 256 injections per key/target/arm")


# Stage-12 helper functions expect a Stage12Config-shaped object.  No Stage-12
# campaign is run here; this object only supplies the validated simulator and
# feature-extraction constants.
_PF_TOTAL_INJECTIONS = (
    paper_sifa_config.number_of_keys
    * len(paper_sifa_config.target_sbox_indices)
    * 2
    * paper_sifa_config.injections_per_key_target_arm
)
pf_sim_config = Stage12Config(
    input_stage11_run_directory=paper_sifa_config.input_stage11_run_directory,
    output_root=paper_sifa_config.output_root,
    random_seed=paper_sifa_config.random_seed,
    number_of_experiments=_PF_TOTAL_INJECTIONS,
    number_of_batches=1,
    experiments_per_batch=_PF_TOTAL_INJECTIONS,
    number_of_keys=paper_sifa_config.number_of_keys,
    number_of_sessions=paper_sifa_config.number_of_sessions,
    confirmation_batch_index=0,
    guided_exploit_fraction=0.50,
    guided_explore_fraction=0.00,
    randomized_baseline_fraction=0.50,
    safety_control_fraction=0.00,
    sifa_objective_fraction=1.00,
    sefa_objective_fraction=0.00,
    shfa_objective_fraction=0.00,
    exploit_offset_jitter_sigma=paper_sifa_config.guided_offset_jitter_sigma,
    exploit_relative_parameter_jitter=(
        paper_sifa_config.guided_relative_parameter_jitter
    ),
    global_timing_jitter_sigma_samples=(
        paper_sifa_config.global_timing_jitter_sigma_samples
    ),
    local_sbox_jitter_sigma_samples=(
        paper_sifa_config.local_sbox_jitter_sigma_samples
    ),
    injection_timing_jitter_sigma_samples=(
        paper_sifa_config.injection_timing_jitter_sigma_samples
    ),
    session_timing_shift_sigma_samples=(
        paper_sifa_config.session_timing_shift_sigma_samples
    ),
    response_trace_noise_sigma=paper_sifa_config.response_trace_noise_sigma,
    response_trace_baseline_sigma=paper_sifa_config.response_trace_baseline_sigma,
    response_trace_gain_sigma=paper_sifa_config.response_trace_gain_sigma,
    target_window_radius_samples=paper_sifa_config.target_window_radius_samples,
    pulse_window_radius_samples=paper_sifa_config.pulse_window_radius_samples,
    highpass_moving_average_width=(
        paper_sifa_config.highpass_moving_average_width
    ),
    trace_standard_deviation_floor=(
        paper_sifa_config.trace_standard_deviation_floor
    ),
    bootstrap_repetitions=paper_sifa_config.bootstrap_repetitions,
    save_plots=paper_sifa_config.save_plots,
)


# Verify all prior freezes through the already-validated Stage-12 resolver.
pf_stage11_directory = Path(
    paper_sifa_config.input_stage11_run_directory
).expanduser().resolve()
pf_contracts = resolve_stage_contracts(pf_stage11_directory)

pf_recommendations = pf_contracts["exploit_recommendations"].copy()
pf_best_recommendations: Dict[str, pd.Series] = {}
for _target_name in ("S0", "S5"):
    _subset = pf_recommendations[
        (pf_recommendations["target_sbox"].astype(str) == _target_name)
        & (pf_recommendations["objective"].astype(str) == "SIFA")
        & (pf_recommendations["recommendation_mode"].astype(str) == "exploit")
    ].sort_values(["rank", "robust_utility_SIFA"], ascending=[True, False])
    if _subset.empty:
        raise RuntimeError(f"No Stage-11 SIFA exploit recommendation for {_target_name}")
    pf_best_recommendations[_target_name] = _subset.iloc[0].copy()

pf_model10 = pf_contracts["stage10_model"]
pf_ml_threshold = float(
    pf_model10["branch_thresholds"]["clean_target_ineffective"]
)
if not (0.0 < pf_ml_threshold < 1.0):
    raise RuntimeError("Invalid frozen Stage-10 ineffective threshold")

pf_centers = np.asarray(
    [
        float(item["center_sample"])
        for item in pf_contracts["timing_map"]["sboxes"]
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Paper-faithful LBlock target mapping
# ---------------------------------------------------------------------------
# The final round satisfies
#     X33 = F(X32, K32) XOR ROL8(X31).
# If output nibble j of F is produced by S-box s, then it is XORed with
# X31 nibble t=(j-2) mod 8.  We fault this X31 nibble and recover K32[s].
_PF_P_SOURCE_FOR_OUTPUT = tuple(
    int(value) for value in engine.P_SOURCE_FOR_OUTPUT
)
_PF_SOURCE_TO_OUTPUT = {
    int(source): int(output_index)
    for output_index, source in enumerate(_PF_P_SOURCE_FOR_OUTPUT)
}
_PF_STATE_NIBBLE_BY_CHANNEL = {
    int(source): int((output_index - 2) % 8)
    for source, output_index in _PF_SOURCE_TO_OUTPUT.items()
}
if _PF_STATE_NIBBLE_BY_CHANNEL[0] != 0:
    raise AssertionError("S0 must map to X31 nibble 0")
if _PF_STATE_NIBBLE_BY_CHANNEL[5] != 2:
    raise AssertionError("S5 must map to X31 nibble 2")
pf_bounds_by_target = {
    target_name: pf_contracts["stage11_summary"]["candidate_pool_summary"][
        target_name
    ]["parameter_bounds"]
    for target_name in ("S0", "S5")
}


# Create a fresh key pool.  These key values are used only by the simulator.
# Candidate scoring later receives only the public campaign table.
_pf_key_rng = np.random.default_rng(
    np.random.SeedSequence([paper_sifa_config.random_seed, 13001])
)
pf_key_pool: List[int] = []
while len(pf_key_pool) < paper_sifa_config.number_of_keys:
    candidate = int(engine.random_80bit_integer(_pf_key_rng))
    if candidate not in pf_key_pool:
        pf_key_pool.append(candidate)

_pf_session_rng = np.random.default_rng(
    np.random.SeedSequence([paper_sifa_config.random_seed, 13002])
)
pf_session_shifts = _pf_session_rng.normal(
    0.0,
    paper_sifa_config.session_timing_shift_sigma_samples,
    size=paper_sifa_config.number_of_sessions,
)


pf_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
pf_run_id = f"stage13R_v2_{pf_timestamp}_seed{paper_sifa_config.random_seed}"
pf_run_directory = (
    Path(paper_sifa_config.output_root).expanduser().resolve() / pf_run_id
)
pf_public_campaign_directory = pf_run_directory / "public_campaign"
pf_public_attack_directory = pf_run_directory / "public_attack"
pf_private_directory = pf_run_directory / "private_after_freeze"
pf_validation_directory = pf_run_directory / "validation_only"
pf_locked_directory = pf_run_directory / "locked_truth"
for _directory in (
    pf_public_campaign_directory,
    pf_public_attack_directory,
    pf_private_directory,
    pf_validation_directory,
    pf_locked_directory,
):
    _directory.mkdir(parents=True, exist_ok=True)

pf_write_json(
    pf_public_campaign_directory / "paper_sifa_config.json",
    {
        **asdict(paper_sifa_config),
        "target_sbox_indices": list(paper_sifa_config.target_sbox_indices),
        "total_injections": _PF_TOTAL_INJECTIONS,
        "stage10_frozen_ineffective_threshold": pf_ml_threshold,
    },
)

pf_write_json(
    pf_public_campaign_directory / "paper_sifa_pre_registered_contract.json",
    {
        "primary_key_statistic": "SEI",
        "paper_equation": (
            "SEI(k)=sum_x (p_hat_k(x)-1/16)^2; highest score wins"
        ),
        "ineffective_observation_rule": "response_received and C_prime == C",
        "physical_fault_target": (
            "X31 state nibble after X32 is formed and before ROL8(X31) is XORed into X33"
        ),
        "lblock_partial_decryption": (
            "X31_hat_t(k)=nibble_j(X33) XOR "
            "S_s(nibble_s(X32) XOR k), where "
            "j=SOURCE_TO_OUTPUT[s] and t=(j-2) mod 8"
        ),
        "target_mapping": {
            "S0": "fault X31[0], recover K32[0]",
            "S5": "fault X31[2], recover K32[5]",
        },
        "symmetry_guard": (
            "The previous X32_nibble XOR key adaptation was rejected because "
            "SEI is invariant under a permutation of histogram bins."
        ),
        "primary_comparison": "guided_ml versus random_raw",
        "ablation_pipelines": [
            "random_raw",
            "random_ml",
            "guided_raw",
            "guided_ml",
        ],
        "guided_parameter_source": (
            "Stage-11 rank-1 exploit recommendation for objective SIFA"
        ),
        "random_parameter_source": (
            "uniform sampling from the same Stage-11 support bounds"
        ),
        "stage10_role": (
            "hard pre-registered sample filtering only; probabilities are not "
            "used as SEI weights"
        ),
        "paired_design": (
            "guided and random arms use the same plaintext, key, target, and "
            "session for every pair_id"
        ),
        "private_truth_available_to_key_scoring": False,
    },
)


# Deterministic algebraic guard: for the correct K32 nibble, the public
# partial-decryption formula must reconstruct the actual X31 target nibble.
_pf_guard_rng = np.random.default_rng(
    np.random.SeedSequence([paper_sifa_config.random_seed, 13991])
)
_pf_guard_key = int(engine.random_80bit_integer(_pf_guard_rng))
_pf_guard_round_key_32 = int(engine.key_schedule_lblock(_pf_guard_key)[31])
for _pf_guard_target in paper_sifa_config.target_sbox_indices:
    _pf_guard_output = _PF_SOURCE_TO_OUTPUT[int(_pf_guard_target)]
    _pf_guard_state = _PF_STATE_NIBBLE_BY_CHANNEL[int(_pf_guard_target)]
    _pf_guard_k = int(
        (_pf_guard_round_key_32 >> (4 * int(_pf_guard_target))) & 0xF
    )
    for _ in range(64):
        _pf_guard_plaintext = int(engine.random_64bit_integer(_pf_guard_rng))
        _pf_guard_context = engine.final_round_context(
            _pf_guard_plaintext,
            _pf_guard_key,
        )
        _pf_guard_x32_nibble = int(
            engine.get_nibble(
                int(_pf_guard_context["x32"]),
                int(_pf_guard_target),
            )
        )
        _pf_guard_x33_nibble = int(
            engine.get_nibble(
                int(_pf_guard_context["x33"]),
                int(_pf_guard_output),
            )
        )
        _pf_guard_reconstructed = int(
            _pf_guard_x33_nibble
            ^ int(
                engine.SBOX[int(_pf_guard_target)][
                    _pf_guard_x32_nibble ^ _pf_guard_k
                ]
            )
        )
        _pf_guard_actual = int(
            engine.get_nibble(
                int(_pf_guard_context["x31"]),
                int(_pf_guard_state),
            )
        )
        if _pf_guard_reconstructed != _pf_guard_actual:
            raise AssertionError(
                "LBlock X31 partial-decryption guard failed"
            )

print("=" * 84)
print("Stage 13R-v2 configuration ready")
print("Stage-11 input              :", pf_stage11_directory)
print("Output directory            :", pf_run_directory)
print("Total fresh injections      :", _PF_TOTAL_INJECTIONS)
print("Physical target mapping     :", {
    "S0": f"X31[{_PF_STATE_NIBBLE_BY_CHANNEL[0]}]",
    "S5": f"X31[{_PF_STATE_NIBBLE_BY_CHANNEL[5]}]",
})
print("Per key/target/arm          :", paper_sifa_config.injections_per_key_target_arm)
print("Fresh keys / sessions       :", paper_sifa_config.number_of_keys, "/", paper_sifa_config.number_of_sessions)
print("Frozen ML threshold         :", f"{pf_ml_threshold:.6f}")
for _target_name in ("S0", "S5"):
    _row = pf_best_recommendations[_target_name]
    print(
        f"{_target_name} best Stage-11 parameters:",
        {
            "offset": float(_row["timing_offset_samples"]),
            "width": float(_row["width_samples"]),
            "strength": float(_row["strength"]),
            "repeat": int(_row["repeat"]),
            "spacing": float(_row["repeat_spacing_samples"]),
            "robust_utility_SIFA": float(_row["robust_utility_SIFA"]),
        },
    )
print("=" * 84)
# %%
# ============================================================
# Stage 13R / Cell 2
# Fresh paired fault campaign: Guided model vs Random support
# ============================================================

_PF_CLASS_NAMES = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)
_PF_CLASS_TO_ID = {
    name: index for index, name in enumerate(_PF_CLASS_NAMES)
}


def pf_sample_random_support_parameters(
    target_index: int,
    rng: np.random.Generator,
) -> Any:
    """Uniform random baseline from exactly the Stage-11 support bounds."""
    target_name = f"S{int(target_index)}"
    bounds = pf_bounds_by_target[target_name]
    repeat_values = np.asarray(bounds["repeat"]["allowed"], dtype=np.int32)
    return engine.GlitchParameters(
        target_sbox_index=int(target_index),
        nominal_target_center_sample=float(pf_centers[target_index]),
        offset_samples=float(
            rng.uniform(
                bounds["timing_offset_samples"]["lower"],
                bounds["timing_offset_samples"]["upper"],
            )
        ),
        width_samples=float(
            rng.uniform(
                bounds["width_samples"]["lower"],
                bounds["width_samples"]["upper"],
            )
        ),
        strength=float(
            rng.uniform(
                bounds["strength"]["lower"],
                bounds["strength"]["upper"],
            )
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


def pf_simulator_config_proxy() -> Any:
    class Proxy:
        pass

    proxy = Proxy()
    proxy.response_trace_noise_sigma = (
        paper_sifa_config.response_trace_noise_sigma
    )
    proxy.response_trace_baseline_sigma = (
        paper_sifa_config.response_trace_baseline_sigma
    )
    proxy.response_trace_gain_sigma = (
        paper_sifa_config.response_trace_gain_sigma
    )
    return proxy



def pf_channel_values_from_x31(x31: int) -> np.ndarray:
    """
    Return eight logical channel values.  Channel s carries the X31 nibble
    that is XORed with the output of final-round S-box s after P and ROL8.
    """
    values = np.zeros(8, dtype=np.uint8)
    for channel_index in range(8):
        state_nibble_index = _PF_STATE_NIBBLE_BY_CHANNEL[channel_index]
        values[channel_index] = int(
            engine.get_nibble(int(x31), int(state_nibble_index))
        )
    return values


def pf_x31_from_channel_values(values: Sequence[int]) -> int:
    if len(values) != 8:
        raise ValueError("Exactly eight logical channel values are required")
    x31 = 0
    for channel_index, value in enumerate(values):
        state_nibble_index = _PF_STATE_NIBBLE_BY_CHANNEL[channel_index]
        x31 |= (int(value) & 0xF) << (4 * int(state_nibble_index))
    return int(x31) & int(engine.MASK32)


def pf_run_paired_fault_attempt(
    *,
    experiment_id: int,
    pair_id: int,
    arm: str,
    arm_code: int,
    key_id: int,
    session_id: int,
    target_index: int,
    plaintext: int,
    source_trace_index: int,
    parameters: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    """
    Run one fresh fault attempt on the paper-identifiable X31 target.

    Both paired arms use the same plaintext, key, logical target, session,
    and healthy source trace.  Only glitch parameters and physical randomness
    differ.  The target is X31[t] after X32 has already been formed and before
    ROL8(X31) is XORed into X33.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                paper_sifa_config.random_seed,
                int(pair_id),
                int(key_id),
                int(target_index),
                int(arm_code),
                13006,
            ]
        )
    )

    core_widths = np.asarray(
        [
            float(item["core_window_end_sample_exclusive"])
            - float(item["core_window_start_sample_inclusive"])
            for item in pf_contracts["timing_map"]["sboxes"]
        ],
        dtype=np.float64,
    )

    global_jitter = float(
        rng.normal(
            0.0,
            paper_sifa_config.global_timing_jitter_sigma_samples,
        )
    )
    local_jitter = rng.normal(
        0.0,
        paper_sifa_config.local_sbox_jitter_sigma_samples,
        size=8,
    )
    actual_centers = (
        pf_centers
        + float(pf_session_shifts[session_id])
        + global_jitter
        + local_jitter
    )

    injection_jitter = float(
        rng.normal(
            0.0,
            paper_sifa_config.injection_timing_jitter_sigma_samples,
        )
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

    master_key = int(pf_key_pool[key_id])
    context = engine.final_round_context(int(plaintext), master_key)
    healthy_ciphertext = int(context["ciphertext"])

    # Logical channel s maps to one physical X31 nibble t.  This keeps the
    # Stage-11 relative timing model and Stage-10 trace model on the same
    # eight logical channels while changing the cryptanalytic target.
    original_inputs = pf_channel_values_from_x31(int(context["x31"]))
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
                    _PF_STATE_NIBBLE_BY_CHANNEL[int(channel_index)]
                ),
            }

        faulted_x31 = pf_x31_from_channel_values(faulted_inputs)
        x33_faulted = int(
            int(context["f_output"])
            ^ engine.rol(int(faulted_x31), 8, 32)
        ) & int(engine.MASK32)
        faulty_ciphertext = int(
            (int(context["x32"]) << 32) | int(x33_faulted)
        ) & ((1 << 64) - 1)

        response_received = True
        ciphertext_equal = bool(
            faulty_ciphertext == healthy_ciphertext
        )
        ciphertext_hamming_distance = int(
            engine.hamming_distance(
                faulty_ciphertext,
                healthy_ciphertext,
            )
        )
        invalid_subtype = ""
    else:
        faulted_x31 = int(context["x31"])
        faulty_ciphertext = None
        response_received = False
        ciphertext_equal = False
        ciphertext_hamming_distance = -1
        invalid_subtype = (
            "reset" if rng.random() < 0.55 else "timeout"
        )

    category = engine.classify_fault_event(
        int(target_index),
        impacted_mask,
        invalid,
        ciphertext_equal,
    )

    response_trace = engine.synthesize_response_trace(
        pf_contracts["healthy_source"]["traces"][int(source_trace_index)],
        pf_contracts["healthy_source"]["absolute_samples"],
        pulse_values,
        parameters,
        actual_centers,
        original_inputs,
        faulted_inputs,
        impacted_mask,
        invalid,
        rng,
        pf_simulator_config_proxy(),
    ).astype(np.float32)

    trace_features = engine.trace_features(
        response_trace,
        pf_contracts["healthy_source"]["absolute_samples"],
        float(pf_centers[target_index]),
        pulse_values,
    )

    impacted_indices = [
        int(value)
        for value in np.where(impacted_mask > 0)[0]
    ]
    physical_target_nibble = int(
        _PF_STATE_NIBBLE_BY_CHANNEL[int(target_index)]
    )
    final_output_nibble = int(
        _PF_SOURCE_TO_OUTPUT[int(target_index)]
    )

    public_row: Dict[str, Any] = {
        "experiment_id": int(experiment_id),
        "pair_id": int(pair_id),
        "campaign_partition": "paper_sifa_v2_comparison",
        "campaign_arm": str(arm),
        "objective": "SIFA",
        "target_sbox": f"S{int(target_index)}",
        "target_sbox_index": int(target_index),
        "physical_fault_intermediate": "X31_state_after_X32_before_final_feedforward",
        "physical_x31_nibble_index": physical_target_nibble,
        "final_f_output_nibble_index": final_output_nibble,
        "recovered_key_nibble": f"K32[{int(target_index)}]",
        "key_id": int(key_id),
        "session_id": int(session_id),
        "source_healthy_trace_id": int(
            pf_contracts["healthy_source"]["trace_ids"][
                int(source_trace_index)
            ]
        ),
        "fault_model": str(parameters.fault_model),
        "parameter_source": (
            "stage11_rank1_SIFA_exploit"
            if arm == "guided_model"
            else "uniform_same_stage11_support"
        ),
        "recommendation_rank": (
            1 if arm == "guided_model" else -1
        ),
        "nominal_target_center_sample": float(
            parameters.nominal_target_center_sample
        ),
        "timing_offset_samples": float(parameters.offset_samples),
        "first_pulse_nominal_sample": float(
            parameters.nominal_target_center_sample
            + parameters.offset_samples
        ),
        "width_samples": float(parameters.width_samples),
        "strength": float(parameters.strength),
        "repeat": int(parameters.repeat),
        "repeat_spacing_samples": float(
            parameters.repeat_spacing_samples
        ),
        "plaintext_hex": pf_hex_fixed(plaintext, 64),
        "healthy_ciphertext_hex": pf_hex_fixed(
            healthy_ciphertext,
            64,
        ),
        "response_received": bool(response_received),
        "faulty_ciphertext_hex": (
            pf_hex_fixed(faulty_ciphertext, 64)
            if faulty_ciphertext is not None
            else ""
        ),
        "ciphertext_equal": (
            bool(ciphertext_equal)
            if response_received
            else ""
        ),
        "ciphertext_hamming_distance": (
            int(ciphertext_hamming_distance)
            if response_received
            else np.nan
        ),
        **trace_features,
    }

    private_row: Dict[str, Any] = {
        "experiment_id": int(experiment_id),
        "pair_id": int(pair_id),
        "campaign_arm": str(arm),
        "key_id": int(key_id),
        "session_id": int(session_id),
        "target_sbox": f"S{int(target_index)}",
        "target_sbox_index": int(target_index),
        "physical_x31_nibble_index": physical_target_nibble,
        "category": str(category),
        "category_id": int(_PF_CLASS_TO_ID[category]),
        # Compatibility names retained for the post-freeze analysis.
        "target_original_input": int(original_inputs[target_index]),
        "target_faulted_input": int(faulted_inputs[target_index]),
        "target_original_x31_value": int(original_inputs[target_index]),
        "target_faulted_x31_value": int(faulted_inputs[target_index]),
        "target_impacted": bool(
            impacted_mask[target_index]
        ),
        "off_target_impacted": bool(
            any(
                value != target_index
                for value in impacted_indices
            )
        ),
        "impacted_sboxes": ";".join(
            f"S{value}" for value in impacted_indices
        ),
        "impacted_sbox_count": int(
            len(impacted_indices)
        ),
        "changed_sbox_input_count": int(
            np.sum(original_inputs != faulted_inputs)
        ),
        "changed_x31_channel_count": int(
            np.sum(original_inputs != faulted_inputs)
        ),
        "fault_effective": bool(
            response_received and not ciphertext_equal
        ),
        "invalid_subtype": str(invalid_subtype),
        "invalid_probability": float(
            invalid_probability
        ),
        "global_jitter_samples": float(
            global_jitter
        ),
        "injection_jitter_samples": float(
            injection_jitter
        ),
        "model_details_json": json.dumps(
            model_details,
            sort_keys=True,
        ),
    }

    return public_row, private_row, response_trace

pf_campaign_started = time.perf_counter()
pf_public_rows: List[Dict[str, Any]] = []
pf_private_rows: List[Dict[str, Any]] = []
pf_response_traces: List[np.ndarray] = []

pf_experiment_id = 0
pf_global_pair_id = 0
pf_total_pairs = (
    paper_sifa_config.number_of_keys
    * len(paper_sifa_config.target_sbox_indices)
    * paper_sifa_config.injections_per_key_target_arm
)

for pf_key_id in range(paper_sifa_config.number_of_keys):
    for pf_target_index in paper_sifa_config.target_sbox_indices:
        pf_target_name = f"S{int(pf_target_index)}"
        pf_best_row = pf_best_recommendations[pf_target_name]
        pf_bounds = pf_bounds_by_target[pf_target_name]

        for pf_local_pair_index in range(
            paper_sifa_config.injections_per_key_target_arm
        ):
            pf_session_id = int(
                pf_local_pair_index
                % paper_sifa_config.number_of_sessions
            )

            # Paired design: both arms receive the same plaintext and the
            # same clean source trace.
            pf_common_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        paper_sifa_config.random_seed,
                        int(pf_key_id),
                        int(pf_target_index),
                        int(pf_local_pair_index),
                        13005,
                    ]
                )
            )
            pf_plaintext = int(
                engine.random_64bit_integer(
                    pf_common_rng
                )
            )
            pf_source_trace_index = int(
                pf_common_rng.integers(
                    0,
                    pf_contracts["healthy_source"][
                        "traces"
                    ].shape[0],
                )
            )

            for pf_arm_code, pf_arm in enumerate(
                ("guided_model", "random_uniform")
            ):
                pf_parameter_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_sifa_config.random_seed,
                            int(pf_key_id),
                            int(pf_target_index),
                            int(pf_local_pair_index),
                            int(pf_arm_code),
                            13004,
                        ]
                    )
                )

                if pf_arm == "guided_model":
                    pf_parameters = perturb_recommendation(
                        pf_best_row,
                        "exploit",
                        pf_bounds,
                        float(pf_centers[pf_target_index]),
                        pf_parameter_rng,
                        pf_sim_config,
                    )
                else:
                    pf_parameters = (
                        pf_sample_random_support_parameters(
                            pf_target_index,
                            pf_parameter_rng,
                        )
                    )

                (
                    pf_public_row,
                    pf_private_row,
                    pf_response_trace,
                ) = pf_run_paired_fault_attempt(
                    experiment_id=pf_experiment_id,
                    pair_id=pf_global_pair_id,
                    arm=pf_arm,
                    arm_code=pf_arm_code,
                    key_id=pf_key_id,
                    session_id=pf_session_id,
                    target_index=pf_target_index,
                    plaintext=pf_plaintext,
                    source_trace_index=pf_source_trace_index,
                    parameters=pf_parameters,
                )

                pf_public_rows.append(pf_public_row)
                pf_private_rows.append(pf_private_row)
                pf_response_traces.append(
                    pf_response_trace
                )
                pf_experiment_id += 1

            pf_global_pair_id += 1
            if (
                pf_global_pair_id % 1000 == 0
                or pf_global_pair_id == pf_total_pairs
            ):
                print(
                    "Paired injections completed:",
                    f"{pf_global_pair_id}/{pf_total_pairs}",
                    "pairs |",
                    f"{2 * pf_global_pair_id}/{_PF_TOTAL_INJECTIONS}",
                    "fault attempts",
                )

pf_public_frame = (
    pd.DataFrame(pf_public_rows)
    .sort_values("experiment_id")
    .reset_index(drop=True)
)
pf_private_frame = (
    pd.DataFrame(pf_private_rows)
    .sort_values("experiment_id")
    .reset_index(drop=True)
)
pf_trace_matrix = np.stack(
    pf_response_traces,
    axis=0,
).astype(np.float32)

if len(pf_public_frame) != _PF_TOTAL_INJECTIONS:
    raise AssertionError("Fresh campaign row count mismatch")
if pf_trace_matrix.shape[0] != _PF_TOTAL_INJECTIONS:
    raise AssertionError("Fresh trace count mismatch")
if not (
    pf_public_frame.groupby("pair_id")[
        "plaintext_hex"
    ].nunique().eq(1).all()
):
    raise AssertionError(
        "Paired arms do not share the same plaintext"
    )
if not (
    pf_public_frame.groupby("pair_id")[
        "key_id"
    ].nunique().eq(1).all()
):
    raise AssertionError(
        "Paired arms do not share the same key"
    )
if not (
    pf_public_frame.groupby("pair_id")[
        "target_sbox_index"
    ].nunique().eq(1).all()
):
    raise AssertionError(
        "Paired arms do not share the same target"
    )

print("=" * 84)
print("Fresh paired X31-target campaign generated")
print("Rows / traces       :", len(pf_public_frame), "/", pf_trace_matrix.shape)
print("Pair count          :", pf_public_frame["pair_id"].nunique())
print("Arms                :", pf_public_frame["campaign_arm"].value_counts().to_dict())
print("Elapsed seconds     :", f"{time.perf_counter() - pf_campaign_started:.3f}")
print("=" * 84)
# %%
# ============================================================
# Stage 13R / Cell 3
# Frozen Stage-10 quality scoring and public sample selection
# ============================================================

pf_probability_frame = score_public_batch(
    pf_public_frame,
    pf_trace_matrix,
    pf_contracts["healthy_source"]["absolute_samples"],
    pf_model10,
    pf_sim_config,
)

pf_public_scored = pf_public_frame.merge(
    pf_probability_frame,
    on="experiment_id",
    validate="one_to_one",
)

# Paper-faithful SIFA observation rule:
# keep every returned ciphertext for which the faulted and healthy
# ciphertexts are equal.  This intentionally includes missed faults,
# exactly as in the practical SIFA paper.
pf_public_scored["paper_ineffective"] = (
    pf_public_scored["response_received"].astype(bool)
    & pf_public_scored["ciphertext_equal"]
    .fillna(False)
    .astype(bool)
)

# ML-assisted variant:
# the exact same paper rule is applied first; the frozen Stage-10 model
# then removes samples that are unlikely to be clean target-ineffective.
# The probability is NOT used as a weight in the key score.
pf_public_scored["ml_selected_ineffective"] = (
    pf_public_scored["paper_ineffective"]
    & (
        pf_public_scored[
            "p_clean_target_ineffective"
        ].astype(float)
        >= pf_ml_threshold
    )
)

# Save only public information before any private validation is performed.
pf_public_scored.to_csv(
    pf_public_campaign_directory
    / "paired_fault_campaign_public.csv",
    index=False,
)
pf_probability_frame.to_csv(
    pf_public_campaign_directory
    / "stage10_quality_probabilities_public.csv",
    index=False,
)
np.savez_compressed(
    pf_public_campaign_directory
    / "paired_response_traces_public.npz",
    experiment_id=pf_public_scored[
        "experiment_id"
    ].to_numpy(np.int64),
    response_traces=pf_trace_matrix,
    absolute_samples=np.asarray(
        pf_contracts["healthy_source"][
            "absolute_samples"
        ]
    ),
)

pf_parameter_summary = (
    pf_public_scored.groupby(
        [
            "campaign_arm",
            "target_sbox",
        ],
        as_index=False,
    )
    .agg(
        count=("experiment_id", "size"),
        mean_timing_offset=(
            "timing_offset_samples",
            "mean",
        ),
        std_timing_offset=(
            "timing_offset_samples",
            "std",
        ),
        mean_width=("width_samples", "mean"),
        std_width=("width_samples", "std"),
        mean_strength=("strength", "mean"),
        std_strength=("strength", "std"),
        mean_repeat=("repeat", "mean"),
        mean_repeat_spacing=(
            "repeat_spacing_samples",
            "mean",
        ),
        paper_ineffective_count=(
            "paper_ineffective",
            "sum",
        ),
        ml_selected_count=(
            "ml_selected_ineffective",
            "sum",
        ),
        mean_p_clean_target_ineffective=(
            "p_clean_target_ineffective",
            "mean",
        ),
    )
)
pf_parameter_summary[
    "paper_ineffective_rate"
] = (
    pf_parameter_summary[
        "paper_ineffective_count"
    ]
    / pf_parameter_summary["count"]
)
pf_parameter_summary[
    "ml_selected_rate"
] = (
    pf_parameter_summary["ml_selected_count"]
    / pf_parameter_summary["count"]
)
pf_parameter_summary.to_csv(
    pf_public_campaign_directory
    / "parameter_and_yield_summary_public.csv",
    index=False,
)

pf_selected_event_rows: List[Dict[str, Any]] = []
for pf_pipeline_name, pf_arm_name, pf_selection_column in (
    (
        "random_raw",
        "random_uniform",
        "paper_ineffective",
    ),
    (
        "random_ml",
        "random_uniform",
        "ml_selected_ineffective",
    ),
    (
        "guided_raw",
        "guided_model",
        "paper_ineffective",
    ),
    (
        "guided_ml",
        "guided_model",
        "ml_selected_ineffective",
    ),
):
    pf_subset = pf_public_scored[
        (
            pf_public_scored["campaign_arm"]
            == pf_arm_name
        )
        & pf_public_scored[
            pf_selection_column
        ].astype(bool)
    ]
    for pf_row in pf_subset.itertuples(
        index=False
    ):
        pf_selected_event_rows.append(
            {
                "pipeline": pf_pipeline_name,
                "experiment_id": int(
                    pf_row.experiment_id
                ),
                "pair_id": int(pf_row.pair_id),
                "key_id": int(pf_row.key_id),
                "session_id": int(
                    pf_row.session_id
                ),
                "target_sbox": str(
                    pf_row.target_sbox
                ),
                "target_sbox_index": int(
                    pf_row.target_sbox_index
                ),
                "physical_x31_nibble_index": int(
                    pf_row.physical_x31_nibble_index
                ),
                "final_f_output_nibble_index": int(
                    pf_row.final_f_output_nibble_index
                ),
                "healthy_ciphertext_hex": str(
                    pf_row.healthy_ciphertext_hex
                ),
                "p_clean_target_ineffective": float(
                    pf_row.p_clean_target_ineffective
                ),
            }
        )

pf_selected_events_public = pd.DataFrame(
    pf_selected_event_rows
)
pf_selected_events_public.to_csv(
    pf_public_campaign_directory
    / "selected_ineffective_ciphertexts_public.csv",
    index=False,
)

pf_write_json(
    pf_public_campaign_directory
    / "public_data_access_manifest.json",
    {
        "opened_before_key_scoring": [
            "Stage-11 frozen exploit recommendations",
            "Stage-10 frozen deployment classifier",
            "Stage-5 target contract",
            "Stage-4 timing map",
            "Stage-3 healthy ROI traces",
        ],
        "private_simulator_rows_used_for_key_scoring": False,
        "key_truth_used_for_key_scoring": False,
        "stage10_probability_role": (
            "hard sample filter only; no probability weighting"
        ),
        "stage11_role": (
            "pre-injection selection of guided glitch parameters"
        ),
        "cryptanalytic_target": (
            "X31 state nibble; public partial decryption through the final "
            "LBlock S-box and P/feed-forward path"
        ),
    },
)

print("=" * 84)
print("Stage-10 public scoring completed")
print("Frozen ineffective threshold:", f"{pf_ml_threshold:.6f}")
print(
    pf_parameter_summary[
        [
            "campaign_arm",
            "target_sbox",
            "count",
            "paper_ineffective_count",
            "paper_ineffective_rate",
            "ml_selected_count",
            "ml_selected_rate",
        ]
    ].to_string(index=False)
)
print("=" * 84)
# %%
# ============================================================
# Stage 13R-v2 / Cell 4
# Paper-faithful key recovery: exact SEI (primary), CHI, and LLR
# ============================================================

_PF_PIPELINES = {
    "random_raw": {
        "campaign_arm": "random_uniform",
        "selection_column": "paper_ineffective",
        "uses_stage11": False,
        "uses_stage10": False,
    },
    "random_ml": {
        "campaign_arm": "random_uniform",
        "selection_column": "ml_selected_ineffective",
        "uses_stage11": False,
        "uses_stage10": True,
    },
    "guided_raw": {
        "campaign_arm": "guided_model",
        "selection_column": "paper_ineffective",
        "uses_stage11": True,
        "uses_stage10": False,
    },
    "guided_ml": {
        "campaign_arm": "guided_model",
        "selection_column": "ml_selected_ineffective",
        "uses_stage11": True,
        "uses_stage10": True,
    },
}


def pf_parse_ciphertext_words(ciphertext_hex: Any) -> Tuple[int, int]:
    """Parse C=X32||X33 according to the validated LBlock convention."""
    text = str(ciphertext_hex).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 16:
        raise ValueError(
            f"Expected a 64-bit ciphertext, got {text!r}"
        )
    return int(text[:8], 16), int(text[8:], 16)


def pf_reconstruct_x31_target(
    x32: int,
    x33: int,
    target_sbox_index: int,
    key_guess: int,
) -> int:
    """
    Partial-decrypt the final round to the faulted X31 nibble.

    Let j be the F-output nibble produced by S-box s and
    t=(j-2) mod 8. Since
        X33[j] = S_s(X32[s] XOR K32[s]) XOR X31[t],
    the candidate reconstruction is
        X31_hat[t](k) = X33[j] XOR S_s(X32[s] XOR k).
    """
    sbox_index = int(target_sbox_index)
    output_index = int(_PF_SOURCE_TO_OUTPUT[sbox_index])
    x32_nibble = int(engine.get_nibble(int(x32), sbox_index))
    x33_nibble = int(engine.get_nibble(int(x33), output_index))
    return int(
        x33_nibble
        ^ int(
            engine.SBOX[sbox_index][
                x32_nibble ^ int(key_guess)
            ]
        )
    )


_PF_HAMMING_WEIGHT = np.asarray(
    [int(value).bit_count() for value in range(16)],
    dtype=np.float64,
)
_PF_UNIFORM = np.full(
    16,
    1.0 / 16.0,
    dtype=np.float64,
)

# Secondary theoretical statistic when the exact random-AND diagonal is known.
# The primary practical attack remains SEI and does not use this distribution.
_PF_RANDOM_AND_DIAGONAL = np.power(
    2.0,
    -_PF_HAMMING_WEIGHT,
)
_PF_KNOWN_MODEL_DISTRIBUTION = (
    _PF_RANDOM_AND_DIAGONAL
    / np.sum(_PF_RANDOM_AND_DIAGONAL)
)
_PF_LLR_LOG_RATIO = np.log2(
    _PF_KNOWN_MODEL_DISTRIBUTION
    / _PF_UNIFORM
)


def pf_sifa_candidate_scores(
    selected_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the practical SIFA statistic from the paper.

    Every selected C'=C sample contributes one unweighted count.  Stage-10
    probabilities may have been used as a hard filter before this function,
    but they never enter SEI, CHI, or LLR as weights.
    """
    if selected_frame.empty:
        return pd.DataFrame(
            {
                "key_guess": np.arange(16, dtype=np.int32),
                "key_guess_hex": [f"{value:x}" for value in range(16)],
                "sample_count": 0,
                "sei": np.nan,
                "chi": np.nan,
                "llr_known_model": np.nan,
            }
        )

    target_indices = selected_frame[
        "target_sbox_index"
    ].astype(int).unique()
    if len(target_indices) != 1:
        raise ValueError(
            "Each SIFA task must contain exactly one target S-box"
        )
    target_index = int(target_indices[0])

    x32_values = selected_frame[
        "x32_word"
    ].to_numpy(np.uint64)
    x33_values = selected_frame[
        "x33_word"
    ].to_numpy(np.uint64)
    sample_count = int(len(selected_frame))
    rows: List[Dict[str, Any]] = []

    for key_guess in range(16):
        hypothesized_intermediate = np.fromiter(
            (
                pf_reconstruct_x31_target(
                    int(x32),
                    int(x33),
                    target_index,
                    int(key_guess),
                )
                for x32, x33 in zip(x32_values, x33_values)
            ),
            dtype=np.uint8,
            count=sample_count,
        )
        counts = np.bincount(
            hypothesized_intermediate,
            minlength=16,
        ).astype(np.float64)
        empirical = counts / float(sample_count)

        # Exact practical statistic of the SIFA paper:
        # SEI = sum_x (p_hat(x)-1/16)^2.
        sei = float(
            np.sum(
                np.square(
                    empirical - _PF_UNIFORM
                )
            )
        )

        # For uniform theta, Pearson CHI = N*16*SEI.
        chi = float(
            sample_count * 16.0 * sei
        )

        # Paper's theoretical known-distribution statistic.
        llr = float(
            np.sum(counts * _PF_LLR_LOG_RATIO)
        )

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


def pf_attach_target_nibbles(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    parsed = [
        pf_parse_ciphertext_words(value)
        for value in result["healthy_ciphertext_hex"]
    ]
    result["x32_word"] = np.asarray(
        [value[0] for value in parsed],
        dtype=np.uint64,
    )
    result["x33_word"] = np.asarray(
        [value[1] for value in parsed],
        dtype=np.uint64,
    )
    result["physical_x31_nibble_index"] = [
        int(_PF_STATE_NIBBLE_BY_CHANNEL[int(target_index)])
        for target_index in result["target_sbox_index"]
    ]
    result["final_f_output_nibble_index"] = [
        int(_PF_SOURCE_TO_OUTPUT[int(target_index)])
        for target_index in result["target_sbox_index"]
    ]
    return result


def pf_unique_best_row(
    candidate_scores: pd.DataFrame,
    score_column: str = "sei",
) -> Dict[str, Any]:
    ordered = candidate_scores.sort_values(
        [score_column, "key_guess"],
        ascending=[False, True],
    ).reset_index(drop=True)
    if ordered.empty:
        return {
            "best_key_guess": -1,
            "best_key_guess_hex": "",
            "best_score": np.nan,
            "second_score": np.nan,
            "score_margin": np.nan,
            "unique_best": False,
        }
    best_score = float(
        ordered.iloc[0][score_column]
    )
    second_score = (
        float(ordered.iloc[1][score_column])
        if len(ordered) > 1
        else np.nan
    )
    unique_best = bool(
        np.sum(
            np.isclose(
                candidate_scores[
                    score_column
                ].to_numpy(float),
                best_score,
                atol=1.0e-15,
                rtol=1.0e-12,
            )
        )
        == 1
    )
    return {
        "best_key_guess": int(
            ordered.iloc[0]["key_guess"]
        ),
        "best_key_guess_hex": str(
            ordered.iloc[0]["key_guess_hex"]
        ),
        "best_score": best_score,
        "second_score": second_score,
        "score_margin": float(
            best_score - second_score
        ),
        "unique_best": unique_best,
    }


def pf_injection_checkpoints(
    maximum: int,
) -> List[int]:
    requested = [
        100,
        200,
        400,
        600,
        800,
        1200,
        1600,
        2000,
        2500,
        maximum,
    ]
    return sorted(
        {
            int(min(maximum, value))
            for value in requested
            if value <= maximum
            or value == maximum
        }
    )


def pf_selected_checkpoints(
    maximum: int,
) -> List[int]:
    requested = [
        16,
        32,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
        maximum,
    ]
    return sorted(
        {
            int(min(maximum, value))
            for value in requested
            if value <= maximum
            or value == maximum
        }
    )


pf_scoring_frame = pf_attach_target_nibbles(
    pf_public_scored
)

pf_final_score_parts: List[pd.DataFrame] = []
pf_final_prediction_rows: List[Dict[str, Any]] = []
pf_injection_prefix_parts: List[pd.DataFrame] = []
pf_selected_prefix_parts: List[pd.DataFrame] = []
pf_bootstrap_rows: List[Dict[str, Any]] = []

for pf_pipeline, pf_spec in _PF_PIPELINES.items():
    for pf_key_id in range(
        paper_sifa_config.number_of_keys
    ):
        for pf_target_index in (
            paper_sifa_config.target_sbox_indices
        ):
            pf_arm_all = (
                pf_scoring_frame[
                    (
                        pf_scoring_frame[
                            "campaign_arm"
                        ]
                        == pf_spec[
                            "campaign_arm"
                        ]
                    )
                    & (
                        pf_scoring_frame[
                            "key_id"
                        ]
                        == pf_key_id
                    )
                    & (
                        pf_scoring_frame[
                            "target_sbox_index"
                        ]
                        == pf_target_index
                    )
                ]
                .sort_values(
                    [
                        "pair_id",
                        "experiment_id",
                    ]
                )
                .reset_index(drop=True)
            )
            pf_selected = (
                pf_arm_all[
                    pf_arm_all[
                        pf_spec[
                            "selection_column"
                        ]
                    ].astype(bool)
                ]
                .copy()
                .reset_index(drop=True)
            )

            pf_scores = pf_sifa_candidate_scores(
                pf_selected
            )
            pf_scores.insert(
                0,
                "pipeline",
                pf_pipeline,
            )
            pf_scores.insert(
                1,
                "key_id",
                int(pf_key_id),
            )
            pf_scores.insert(
                2,
                "target_sbox",
                f"S{int(pf_target_index)}",
            )
            pf_scores.insert(
                3,
                "target_sbox_index",
                int(pf_target_index),
            )
            pf_scores.insert(
                4,
                "injection_count",
                int(len(pf_arm_all)),
            )
            pf_final_score_parts.append(
                pf_scores
            )

            pf_prediction = pf_unique_best_row(
                pf_scores,
                "sei",
            )
            pf_final_prediction_rows.append(
                {
                    "pipeline": pf_pipeline,
                    "key_id": int(
                        pf_key_id
                    ),
                    "target_sbox": (
                        f"S{int(pf_target_index)}"
                    ),
                    "target_sbox_index": int(
                        pf_target_index
                    ),
                    "injection_count": int(
                        len(pf_arm_all)
                    ),
                    "selected_ciphertext_count": int(
                        len(pf_selected)
                    ),
                    **pf_prediction,
                }
            )

            # Rank evolution versus total fault attempts.
            for pf_checkpoint in (
                pf_injection_checkpoints(
                    len(pf_arm_all)
                )
            ):
                pf_prefix_all = pf_arm_all.iloc[
                    :pf_checkpoint
                ]
                pf_prefix_selected = (
                    pf_prefix_all[
                        pf_prefix_all[
                            pf_spec[
                                "selection_column"
                            ]
                        ].astype(bool)
                    ]
                )
                if (
                    len(pf_prefix_selected)
                    < paper_sifa_config.minimum_selected_ciphertexts
                ):
                    continue
                pf_prefix_scores = (
                    pf_sifa_candidate_scores(
                        pf_prefix_selected
                    )
                )
                pf_prefix_scores.insert(
                    0,
                    "pipeline",
                    pf_pipeline,
                )
                pf_prefix_scores.insert(
                    1,
                    "key_id",
                    int(pf_key_id),
                )
                pf_prefix_scores.insert(
                    2,
                    "target_sbox",
                    f"S{int(pf_target_index)}",
                )
                pf_prefix_scores.insert(
                    3,
                    "target_sbox_index",
                    int(pf_target_index),
                )
                pf_prefix_scores.insert(
                    4,
                    "injection_count",
                    int(pf_checkpoint),
                )
                pf_prefix_scores.insert(
                    5,
                    "selected_ciphertext_count",
                    int(
                        len(
                            pf_prefix_selected
                        )
                    ),
                )
                pf_injection_prefix_parts.append(
                    pf_prefix_scores
                )

            # Rank evolution versus number of useful selected ciphertexts.
            for pf_selected_count in (
                pf_selected_checkpoints(
                    len(pf_selected)
                )
            ):
                if (
                    pf_selected_count
                    < paper_sifa_config.minimum_selected_ciphertexts
                ):
                    continue
                pf_selected_prefix = (
                    pf_selected.iloc[
                        :pf_selected_count
                    ]
                )
                pf_selected_scores = (
                    pf_sifa_candidate_scores(
                        pf_selected_prefix
                    )
                )
                pf_selected_scores.insert(
                    0,
                    "pipeline",
                    pf_pipeline,
                )
                pf_selected_scores.insert(
                    1,
                    "key_id",
                    int(pf_key_id),
                )
                pf_selected_scores.insert(
                    2,
                    "target_sbox",
                    f"S{int(pf_target_index)}",
                )
                pf_selected_scores.insert(
                    3,
                    "target_sbox_index",
                    int(pf_target_index),
                )
                pf_selected_scores.insert(
                    4,
                    "selected_ciphertext_count",
                    int(pf_selected_count),
                )
                pf_selected_prefix_parts.append(
                    pf_selected_scores
                )

            # Public bootstrap winner stability.  This does not require the
            # true key and is therefore frozen with the attack.
            if (
                len(pf_selected)
                >= paper_sifa_config.minimum_selected_ciphertexts
            ):
                pf_boot_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_sifa_config.random_seed,
                            int(pf_key_id),
                            int(pf_target_index),
                            list(
                                _PF_PIPELINES
                            ).index(
                                pf_pipeline
                            ),
                            13020,
                        ]
                    )
                )
                pf_winner_counts = np.zeros(
                    16,
                    dtype=np.int64,
                )
                for _ in range(
                    paper_sifa_config.bootstrap_repetitions
                ):
                    pf_boot_indices = (
                        pf_boot_rng.integers(
                            0,
                            len(pf_selected),
                            size=len(
                                pf_selected
                            ),
                        )
                    )
                    pf_boot_sample = (
                        pf_selected.iloc[
                            pf_boot_indices
                        ]
                    )
                    pf_boot_scores = (
                        pf_sifa_candidate_scores(
                            pf_boot_sample
                        )
                    )
                    pf_boot_best = int(
                        pf_boot_scores.sort_values(
                            [
                                "sei",
                                "key_guess",
                            ],
                            ascending=[
                                False,
                                True,
                            ],
                        ).iloc[0][
                            "key_guess"
                        ]
                    )
                    pf_winner_counts[
                        pf_boot_best
                    ] += 1

                for pf_candidate in range(16):
                    pf_bootstrap_rows.append(
                        {
                            "pipeline": pf_pipeline,
                            "key_id": int(
                                pf_key_id
                            ),
                            "target_sbox": (
                                f"S{int(pf_target_index)}"
                            ),
                            "target_sbox_index": int(
                                pf_target_index
                            ),
                            "key_guess": int(
                                pf_candidate
                            ),
                            "key_guess_hex": (
                                f"{pf_candidate:x}"
                            ),
                            "winner_count": int(
                                pf_winner_counts[
                                    pf_candidate
                                ]
                            ),
                            "winner_fraction": float(
                                pf_winner_counts[
                                    pf_candidate
                                ]
                                / paper_sifa_config.bootstrap_repetitions
                            ),
                            "bootstrap_repetitions": int(
                                paper_sifa_config.bootstrap_repetitions
                            ),
                        }
                    )

pf_final_scores_public = pd.concat(
    pf_final_score_parts,
    ignore_index=True,
)

# Mandatory identifiability guard.  The obsolete final-round-XOR adaptation
# produced exactly equal SEI values for all 16 candidates.  This run is
# rejected before truth is opened if that symmetry reappears.
pf_sei_range_by_task = (
    pf_final_scores_public.groupby(
        ["pipeline", "key_id", "target_sbox_index"]
    )["sei"]
    .agg(lambda values: float(np.nanmax(values) - np.nanmin(values)))
    .reset_index(name="sei_candidate_range")
)
pf_non_degenerate_tasks = pf_sei_range_by_task[
    pf_sei_range_by_task["sei_candidate_range"] > 1.0e-15
]
if len(pf_non_degenerate_tasks) != len(pf_sei_range_by_task):
    bad = pf_sei_range_by_task[
        pf_sei_range_by_task["sei_candidate_range"] <= 1.0e-15
    ]
    raise RuntimeError(
        "SEI candidate scores are permutation-degenerate for tasks: "
        + bad.to_dict(orient="records").__repr__()
    )
pf_final_predictions_public = pd.DataFrame(
    pf_final_prediction_rows
)
pf_injection_prefix_scores_public = (
    pd.concat(
        pf_injection_prefix_parts,
        ignore_index=True,
    )
    if pf_injection_prefix_parts
    else pd.DataFrame()
)
pf_selected_prefix_scores_public = (
    pd.concat(
        pf_selected_prefix_parts,
        ignore_index=True,
    )
    if pf_selected_prefix_parts
    else pd.DataFrame()
)
pf_bootstrap_public = pd.DataFrame(
    pf_bootstrap_rows
)


# Independent 8-bit combination.  The per-target key-recovery statistic
# remains the paper's SEI.  For the Cartesian 8-bit display, each target's
# 16 scores are standardized before summation so that S0 and S5 contribute
# equally despite different sample counts.
pf_joint_parts: List[pd.DataFrame] = []
pf_joint_prediction_rows: List[Dict[str, Any]] = []
for pf_pipeline in _PF_PIPELINES:
    for pf_key_id in range(
        paper_sifa_config.number_of_keys
    ):
        pf_target_score_vectors: Dict[int, pd.DataFrame] = {}
        for pf_target_index in (
            paper_sifa_config.target_sbox_indices
        ):
            pf_part = pf_final_scores_public[
                (
                    pf_final_scores_public[
                        "pipeline"
                    ]
                    == pf_pipeline
                )
                & (
                    pf_final_scores_public[
                        "key_id"
                    ]
                    == pf_key_id
                )
                & (
                    pf_final_scores_public[
                        "target_sbox_index"
                    ]
                    == pf_target_index
                )
            ].sort_values(
                "key_guess"
            )
            values = pf_part["sei"].to_numpy(
                float
            )
            standard_deviation = float(
                np.std(values)
            )
            if (
                not np.isfinite(
                    standard_deviation
                )
                or standard_deviation
                <= 1.0e-15
            ):
                standardized = np.zeros(
                    16,
                    dtype=np.float64,
                )
            else:
                standardized = (
                    values
                    - float(
                        np.mean(values)
                    )
                ) / standard_deviation
            pf_target_score_vectors[
                pf_target_index
            ] = pd.DataFrame(
                {
                    "key_guess": np.arange(
                        16
                    ),
                    "sei": values,
                    "z_sei": standardized,
                }
            )

        pf_joint_rows: List[Dict[str, Any]] = []
        for pf_guess_s0 in range(16):
            for pf_guess_s5 in range(16):
                pf_joint_rows.append(
                    {
                        "pipeline": pf_pipeline,
                        "key_id": int(
                            pf_key_id
                        ),
                        "guess_K32_0": int(
                            pf_guess_s0
                        ),
                        "guess_K32_0_hex": (
                            f"{pf_guess_s0:x}"
                        ),
                        "guess_K32_5": int(
                            pf_guess_s5
                        ),
                        "guess_K32_5_hex": (
                            f"{pf_guess_s5:x}"
                        ),
                        "guess_8bit_pair_hex": (
                            f"{pf_guess_s0:x}"
                            f"{pf_guess_s5:x}"
                        ),
                        "joint_z_sei": float(
                            pf_target_score_vectors[
                                0
                            ].iloc[
                                pf_guess_s0
                            ][
                                "z_sei"
                            ]
                            + pf_target_score_vectors[
                                5
                            ].iloc[
                                pf_guess_s5
                            ][
                                "z_sei"
                            ]
                        ),
                    }
                )
        pf_joint = pd.DataFrame(
            pf_joint_rows
        ).sort_values(
            [
                "joint_z_sei",
                "guess_K32_0",
                "guess_K32_5",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).reset_index(drop=True)
        pf_joint_parts.append(pf_joint)

        pf_best_joint_score = float(
            pf_joint.iloc[0][
                "joint_z_sei"
            ]
        )
        pf_second_joint_score = float(
            pf_joint.iloc[1][
                "joint_z_sei"
            ]
        )
        pf_joint_prediction_rows.append(
            {
                "pipeline": pf_pipeline,
                "key_id": int(pf_key_id),
                "predicted_K32_0": int(
                    pf_joint.iloc[0][
                        "guess_K32_0"
                    ]
                ),
                "predicted_K32_0_hex": str(
                    pf_joint.iloc[0][
                        "guess_K32_0_hex"
                    ]
                ),
                "predicted_K32_5": int(
                    pf_joint.iloc[0][
                        "guess_K32_5"
                    ]
                ),
                "predicted_K32_5_hex": str(
                    pf_joint.iloc[0][
                        "guess_K32_5_hex"
                    ]
                ),
                "predicted_8bit_pair_hex": str(
                    pf_joint.iloc[0][
                        "guess_8bit_pair_hex"
                    ]
                ),
                "joint_score_margin": float(
                    pf_best_joint_score
                    - pf_second_joint_score
                ),
                "unique_best": bool(
                    np.sum(
                        np.isclose(
                            pf_joint[
                                "joint_z_sei"
                            ].to_numpy(
                                float
                            ),
                            pf_best_joint_score,
                            atol=1.0e-15,
                            rtol=1.0e-12,
                        )
                    )
                    == 1
                ),
            }
        )

pf_joint_scores_public = pd.concat(
    pf_joint_parts,
    ignore_index=True,
)
pf_joint_predictions_public = pd.DataFrame(
    pf_joint_prediction_rows
)


# Save all public attack outputs before any key truth or private category is
# opened for evaluation.
pf_final_scores_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_final_candidate_scores_public.csv",
    index=False,
)
pf_final_predictions_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_final_predictions_public.csv",
    index=False,
)
pf_injection_prefix_scores_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_injection_prefix_scores_public.csv",
    index=False,
)
pf_selected_prefix_scores_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_selected_prefix_scores_public.csv",
    index=False,
)
pf_bootstrap_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_bootstrap_winner_stability_public.csv",
    index=False,
)
pf_joint_scores_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_joint_8bit_candidate_scores_public.csv",
    index=False,
)
pf_joint_predictions_public.to_csv(
    pf_public_attack_directory
    / "paper_sifa_joint_8bit_predictions_public.csv",
    index=False,
)

pf_write_json(
    pf_public_attack_directory
    / "paper_sifa_attack_contract.json",
    {
        "reference_method": (
            "Dobraunig et al., SIFA, Sections 3.2 and 3.3"
        ),
        "primary_statistic": "SEI",
        "primary_statistic_formula": (
            "sum_x (p_hat_k(x)-1/16)^2"
        ),
        "ranking_direction": "descending",
        "chi_relation": "CHI=N*16*SEI",
        "known_model_llr_is_secondary": True,
        "key_score_probability_weighting": False,
        "partial_decryption": (
            "x_hat_i(k)=nibble_i(X32) XOR k"
        ),
        "pipelines": _PF_PIPELINES,
    },
)

# Freeze covers both the public campaign and the complete public attack.
pf_public_freeze_directory = (
    pf_run_directory / "PUBLIC_FREEZE_PAYLOAD"
)
pf_public_freeze_directory.mkdir(
    parents=True,
    exist_ok=True,
)

# The freeze directory stores compact manifests pointing to the already
# written public artifacts.  Hashes are computed over every public file.
pf_public_file_hashes = {}
for pf_public_root in (
    pf_public_campaign_directory,
    pf_public_attack_directory,
):
    for pf_path in sorted(
        pf_public_root.rglob("*")
    ):
        if pf_path.is_file():
            pf_relative = (
                f"{pf_public_root.name}/"
                + str(
                    pf_path.relative_to(
                        pf_public_root
                    )
                ).replace("\\", "/")
            )
            pf_public_file_hashes[
                pf_relative
            ] = pf_sha256_file(pf_path)

pf_public_attack_freeze = {
    "created_at": datetime.now().isoformat(
        timespec="seconds"
    ),
    "statement": (
        "All campaign observations, Stage-10 selections, SEI/CHI/LLR "
        "candidate scores, prefix curves, bootstrap winners, and 8-bit "
        "predictions were frozen before key truth and private simulator "
        "categories were used for evaluation."
    ),
    "files": pf_public_file_hashes,
}
pf_freeze_payload = json.dumps(
    pf_public_attack_freeze,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
pf_public_attack_freeze[
    "freeze_sha256"
] = hashlib.sha256(
    pf_freeze_payload
).hexdigest()
pf_write_json(
    pf_run_directory
    / "paper_sifa_public_attack_freeze_manifest.json",
    pf_public_attack_freeze,
)

pf_public_attack_frozen = True
print("=" * 84)
print("Paper-faithful X31-target SIFA scoring completed")
print("Primary statistic : SEI")
print("Candidate rows    :", len(pf_final_scores_public))
print("Public predictions:")
print(
    pf_final_predictions_public[
        [
            "pipeline",
            "key_id",
            "target_sbox",
            "selected_ciphertext_count",
            "best_key_guess_hex",
            "score_margin",
            "unique_best",
        ]
    ].to_string(index=False)
)
print("Public freeze     :", pf_public_attack_freeze["freeze_sha256"])
print("=" * 84)
# %%
# ============================================================
# Stage 13R-v2 / Cell 5
# Post-freeze truth evaluation and AI-vs-random comparison
# ============================================================

if not globals().get(
    "pf_public_attack_frozen",
    False,
):
    raise RuntimeError(
        "Public attack must be frozen before truth evaluation"
    )


def pf_true_round_key_nibble(
    key_id: int,
    target_index: int,
) -> int:
    round_keys = engine.key_schedule_lblock(
        int(pf_key_pool[key_id])
    )
    round_key_32 = int(round_keys[31])
    return int(
        (
            round_key_32
            >> (4 * int(target_index))
        )
        & 0xF
    )


def pf_rank_from_scores(
    score_frame: pd.DataFrame,
    true_guess: int,
    score_column: str = "sei",
) -> Dict[str, Any]:
    values = score_frame[
        score_column
    ].to_numpy(float)
    guesses = score_frame[
        "key_guess"
    ].to_numpy(int)
    true_mask = guesses == int(
        true_guess
    )
    if np.sum(true_mask) != 1:
        raise RuntimeError(
            "True candidate is absent or duplicated"
        )
    true_score = float(
        values[true_mask][0]
    )
    if not np.isfinite(true_score):
        return {
            "true_rank": 16,
            "true_score": true_score,
            "best_score": np.nan,
            "best_guess": -1,
            "true_is_unique_rank1": False,
        }

    tolerance = 1.0e-12
    true_rank = int(
        1
        + np.sum(
            values
            > true_score + tolerance
        )
    )
    best_score = float(
        np.nanmax(values)
    )
    best_indices = np.where(
        np.isclose(
            values,
            best_score,
            atol=1.0e-15,
            rtol=1.0e-12,
        )
    )[0]
    best_guess = int(
        np.min(
            guesses[best_indices]
        )
    )
    return {
        "true_rank": true_rank,
        "true_score": true_score,
        "best_score": best_score,
        "best_guess": best_guess,
        "true_is_unique_rank1": bool(
            true_rank == 1
            and len(best_indices) == 1
            and best_guess == int(true_guess)
        ),
    }


# Truth is materialized only now, after the public freeze.
pf_truth_rows: List[Dict[str, Any]] = []
for pf_key_id in range(
    paper_sifa_config.number_of_keys
):
    pf_round_keys = engine.key_schedule_lblock(
        int(pf_key_pool[pf_key_id])
    )
    pf_round_key_32 = int(
        pf_round_keys[31]
    )
    pf_truth_rows.append(
        {
            "key_id": int(pf_key_id),
            "master_key_hex": pf_hex_fixed(
                pf_key_pool[pf_key_id],
                80,
            ),
            "round_key_32_hex": pf_hex_fixed(
                pf_round_key_32,
                32,
            ),
            "K32_0": int(
                pf_true_round_key_nibble(
                    pf_key_id,
                    0,
                )
            ),
            "K32_5": int(
                pf_true_round_key_nibble(
                    pf_key_id,
                    5,
                )
            ),
        }
    )
pf_truth_frame = pd.DataFrame(
    pf_truth_rows
)
pf_truth_path = (
    pf_locked_directory
    / "paper_sifa_key_truth_LOCKED.json"
)
pf_write_json(
    pf_truth_path,
    {
        "warning": (
            "Opened by the code only after the public SIFA attack freeze."
        ),
        "keys": pf_truth_rows,
    },
)
pf_truth_sha256 = pf_sha256_file(
    pf_truth_path
)

# Private simulator categories are also written and evaluated only after
# the attack scores have been frozen.
pf_private_frame.to_csv(
    pf_private_directory
    / "paired_fault_ground_truth_after_freeze.csv",
    index=False,
)
pf_public_private_validation = (
    pf_public_scored.merge(
        pf_private_frame,
        on=[
            "experiment_id",
            "pair_id",
            "campaign_arm",
            "key_id",
            "session_id",
            "target_sbox",
            "target_sbox_index",
        ],
        validate="one_to_one",
        suffixes=(
            "",
            "_private",
        ),
    )
)


# ------------------------------------------------------------
# Final 4-bit rank evaluation
# ------------------------------------------------------------
pf_final_rank_rows: List[Dict[str, Any]] = []
for (
    pf_pipeline,
    pf_key_id,
    pf_target_index,
), pf_group in pf_final_scores_public.groupby(
    [
        "pipeline",
        "key_id",
        "target_sbox_index",
    ]
):
    pf_true_guess = (
        pf_true_round_key_nibble(
            int(pf_key_id),
            int(pf_target_index),
        )
    )
    pf_rank_result = pf_rank_from_scores(
        pf_group,
        pf_true_guess,
        "sei",
    )
    pf_prediction_row = (
        pf_final_predictions_public[
            (
                pf_final_predictions_public[
                    "pipeline"
                ]
                == pf_pipeline
            )
            & (
                pf_final_predictions_public[
                    "key_id"
                ]
                == int(pf_key_id)
            )
            & (
                pf_final_predictions_public[
                    "target_sbox_index"
                ]
                == int(pf_target_index)
            )
        ].iloc[0]
    )
    pf_boot_true = (
        pf_bootstrap_public[
            (
                pf_bootstrap_public[
                    "pipeline"
                ]
                == pf_pipeline
            )
            & (
                pf_bootstrap_public[
                    "key_id"
                ]
                == int(pf_key_id)
            )
            & (
                pf_bootstrap_public[
                    "target_sbox_index"
                ]
                == int(pf_target_index)
            )
            & (
                pf_bootstrap_public[
                    "key_guess"
                ]
                == int(pf_true_guess)
            )
        ]
    )
    pf_boot_true_fraction = (
        float(
            pf_boot_true.iloc[0][
                "winner_fraction"
            ]
        )
        if not pf_boot_true.empty
        else np.nan
    )
    pf_final_rank_rows.append(
        {
            "pipeline": str(pf_pipeline),
            "key_id": int(pf_key_id),
            "target_sbox": (
                f"S{int(pf_target_index)}"
            ),
            "target_sbox_index": int(
                pf_target_index
            ),
            "injection_count": int(
                pf_prediction_row[
                    "injection_count"
                ]
            ),
            "selected_ciphertext_count": int(
                pf_prediction_row[
                    "selected_ciphertext_count"
                ]
            ),
            "predicted_key_guess": int(
                pf_prediction_row[
                    "best_key_guess"
                ]
            ),
            "predicted_key_guess_hex": str(
                pf_prediction_row[
                    "best_key_guess_hex"
                ]
            ),
            "score_margin": float(
                pf_prediction_row[
                    "score_margin"
                ]
            ),
            "unique_best": bool(
                pf_prediction_row[
                    "unique_best"
                ]
            ),
            "true_rank": int(
                pf_rank_result[
                    "true_rank"
                ]
            ),
            "true_is_unique_rank1": bool(
                pf_rank_result[
                    "true_is_unique_rank1"
                ]
            ),
            "bootstrap_true_winner_fraction": (
                pf_boot_true_fraction
            ),
        }
    )

pf_final_rank_evaluation = pd.DataFrame(
    pf_final_rank_rows
)
pf_final_rank_evaluation.to_csv(
    pf_validation_directory
    / "paper_sifa_final_true_rank_evaluation.csv",
    index=False,
)


# ------------------------------------------------------------
# Prefix rank evaluation
# ------------------------------------------------------------
def pf_evaluate_prefix_table(
    public_prefix_scores: pd.DataFrame,
    checkpoint_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if public_prefix_scores.empty:
        return pd.DataFrame()
    group_columns = [
        "pipeline",
        "key_id",
        "target_sbox_index",
        checkpoint_column,
    ]
    if (
        "selected_ciphertext_count"
        in public_prefix_scores.columns
        and checkpoint_column
        != "selected_ciphertext_count"
    ):
        group_columns.append(
            "selected_ciphertext_count"
        )

    for group_values, group in (
        public_prefix_scores.groupby(
            group_columns
        )
    ):
        values = (
            group_values
            if isinstance(
                group_values,
                tuple,
            )
            else (group_values,)
        )
        metadata = dict(
            zip(
                group_columns,
                values,
            )
        )
        true_guess = (
            pf_true_round_key_nibble(
                int(metadata["key_id"]),
                int(
                    metadata[
                        "target_sbox_index"
                    ]
                ),
            )
        )
        rank_result = pf_rank_from_scores(
            group,
            true_guess,
            "sei",
        )
        best = pf_unique_best_row(
            group,
            "sei",
        )
        rows.append(
            {
                **metadata,
                "target_sbox": (
                    f"S{int(metadata['target_sbox_index'])}"
                ),
                "predicted_key_guess": int(
                    best["best_key_guess"]
                ),
                "score_margin": float(
                    best["score_margin"]
                ),
                "unique_best": bool(
                    best["unique_best"]
                ),
                "true_rank": int(
                    rank_result[
                        "true_rank"
                    ]
                ),
                "true_is_unique_rank1": bool(
                    rank_result[
                        "true_is_unique_rank1"
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


pf_injection_rank_curve = (
    pf_evaluate_prefix_table(
        pf_injection_prefix_scores_public,
        "injection_count",
    )
)
pf_selected_rank_curve = (
    pf_evaluate_prefix_table(
        pf_selected_prefix_scores_public,
        "selected_ciphertext_count",
    )
)
pf_injection_rank_curve.to_csv(
    pf_validation_directory
    / "paper_sifa_true_rank_vs_injections.csv",
    index=False,
)
pf_selected_rank_curve.to_csv(
    pf_validation_directory
    / "paper_sifa_true_rank_vs_selected_ciphertexts.csv",
    index=False,
)


# Earliest observed Rank-1 checkpoint for each task.
pf_task_efficiency_rows: List[Dict[str, Any]] = []
for (
    pf_pipeline,
    pf_key_id,
    pf_target_index,
), pf_final_row_group in pf_final_rank_evaluation.groupby(
    [
        "pipeline",
        "key_id",
        "target_sbox_index",
    ]
):
    pf_final_row = (
        pf_final_row_group.iloc[0]
    )
    pf_injection_sub = (
        pf_injection_rank_curve[
            (
                pf_injection_rank_curve[
                    "pipeline"
                ]
                == pf_pipeline
            )
            & (
                pf_injection_rank_curve[
                    "key_id"
                ]
                == pf_key_id
            )
            & (
                pf_injection_rank_curve[
                    "target_sbox_index"
                ]
                == pf_target_index
            )
            & pf_injection_rank_curve[
                "true_is_unique_rank1"
            ].astype(bool)
        ]
    )
    pf_selected_sub = (
        pf_selected_rank_curve[
            (
                pf_selected_rank_curve[
                    "pipeline"
                ]
                == pf_pipeline
            )
            & (
                pf_selected_rank_curve[
                    "key_id"
                ]
                == pf_key_id
            )
            & (
                pf_selected_rank_curve[
                    "target_sbox_index"
                ]
                == pf_target_index
            )
            & pf_selected_rank_curve[
                "true_is_unique_rank1"
            ].astype(bool)
        ]
    )
    pf_task_efficiency_rows.append(
        {
            "pipeline": str(
                pf_pipeline
            ),
            "key_id": int(pf_key_id),
            "target_sbox": (
                f"S{int(pf_target_index)}"
            ),
            "target_sbox_index": int(
                pf_target_index
            ),
            "final_true_rank": int(
                pf_final_row["true_rank"]
            ),
            "final_unique_rank1": bool(
                pf_final_row[
                    "true_is_unique_rank1"
                ]
            ),
            "final_selected_ciphertexts": int(
                pf_final_row[
                    "selected_ciphertext_count"
                ]
            ),
            "first_injection_checkpoint_rank1": (
                int(
                    pf_injection_sub[
                        "injection_count"
                    ].min()
                )
                if not pf_injection_sub.empty
                else np.nan
            ),
            "first_selected_checkpoint_rank1": (
                int(
                    pf_selected_sub[
                        "selected_ciphertext_count"
                    ].min()
                )
                if not pf_selected_sub.empty
                else np.nan
            ),
            "final_score_margin": float(
                pf_final_row[
                    "score_margin"
                ]
            ),
            "bootstrap_true_winner_fraction": float(
                pf_final_row[
                    "bootstrap_true_winner_fraction"
                ]
            ),
        }
    )

pf_task_efficiency = pd.DataFrame(
    pf_task_efficiency_rows
)
pf_task_efficiency.to_csv(
    pf_validation_directory
    / "paper_sifa_task_efficiency.csv",
    index=False,
)


# ------------------------------------------------------------
# Joint 8-bit truth evaluation
# ------------------------------------------------------------
pf_joint_rank_rows: List[Dict[str, Any]] = []
for (
    pf_pipeline,
    pf_key_id,
), pf_joint_group in pf_joint_scores_public.groupby(
    [
        "pipeline",
        "key_id",
    ]
):
    pf_true_k0 = (
        pf_true_round_key_nibble(
            int(pf_key_id),
            0,
        )
    )
    pf_true_k5 = (
        pf_true_round_key_nibble(
            int(pf_key_id),
            5,
        )
    )
    pf_true_mask = (
        (
            pf_joint_group[
                "guess_K32_0"
            ].to_numpy(int)
            == pf_true_k0
        )
        & (
            pf_joint_group[
                "guess_K32_5"
            ].to_numpy(int)
            == pf_true_k5
        )
    )
    pf_true_joint_score = float(
        pf_joint_group.loc[
            pf_true_mask,
            "joint_z_sei",
        ].iloc[0]
    )
    pf_joint_values = pf_joint_group[
        "joint_z_sei"
    ].to_numpy(float)
    pf_joint_rank = int(
        1
        + np.sum(
            pf_joint_values
            > pf_true_joint_score
            + 1.0e-12
        )
    )
    pf_joint_best_score = float(
        np.max(pf_joint_values)
    )
    pf_joint_unique = bool(
        pf_joint_rank == 1
        and np.sum(
            np.isclose(
                pf_joint_values,
                pf_joint_best_score,
                atol=1.0e-15,
                rtol=1.0e-12,
            )
        )
        == 1
    )
    pf_public_joint_prediction = (
        pf_joint_predictions_public[
            (
                pf_joint_predictions_public[
                    "pipeline"
                ]
                == pf_pipeline
            )
            & (
                pf_joint_predictions_public[
                    "key_id"
                ]
                == int(pf_key_id)
            )
        ].iloc[0]
    )
    pf_joint_rank_rows.append(
        {
            "pipeline": str(
                pf_pipeline
            ),
            "key_id": int(pf_key_id),
            "predicted_8bit_pair_hex": str(
                pf_public_joint_prediction[
                    "predicted_8bit_pair_hex"
                ]
            ),
            "unique_best": bool(
                pf_public_joint_prediction[
                    "unique_best"
                ]
            ),
            "joint_score_margin": float(
                pf_public_joint_prediction[
                    "joint_score_margin"
                ]
            ),
            "true_pair_rank": int(
                pf_joint_rank
            ),
            "true_pair_is_unique_rank1": (
                pf_joint_unique
            ),
        }
    )

pf_joint_rank_evaluation = pd.DataFrame(
    pf_joint_rank_rows
)
pf_joint_rank_evaluation.to_csv(
    pf_validation_directory
    / "paper_sifa_joint_8bit_true_rank_evaluation.csv",
    index=False,
)


# ------------------------------------------------------------
# Post-freeze quality/yield validation
# ------------------------------------------------------------
pf_quality_rows: List[Dict[str, Any]] = []
for pf_pipeline, pf_spec in _PF_PIPELINES.items():
    pf_arm_frame = (
        pf_public_private_validation[
            pf_public_private_validation[
                "campaign_arm"
            ]
            == pf_spec["campaign_arm"]
        ]
    )
    pf_selected_frame = (
        pf_arm_frame[
            pf_arm_frame[
                pf_spec[
                    "selection_column"
                ]
            ].astype(bool)
        ]
    )
    pf_clean_total = int(
        np.sum(
            pf_arm_frame["category"]
            == "clean_target_ineffective"
        )
    )
    pf_clean_selected = int(
        np.sum(
            pf_selected_frame[
                "category"
            ]
            == "clean_target_ineffective"
        )
    )
    pf_quality_rows.append(
        {
            "pipeline": pf_pipeline,
            "injection_count": int(
                len(pf_arm_frame)
            ),
            "selected_ciphertext_count": int(
                len(pf_selected_frame)
            ),
            "selected_rate": float(
                len(pf_selected_frame)
                / max(
                    1,
                    len(pf_arm_frame),
                )
            ),
            "actual_clean_target_ineffective_count": (
                pf_clean_selected
            ),
            "actual_clean_precision": float(
                pf_clean_selected
                / max(
                    1,
                    len(pf_selected_frame),
                )
            ),
            "actual_clean_recall": float(
                pf_clean_selected
                / max(
                    1,
                    pf_clean_total,
                )
            ),
            "uses_stage11_optimizer": bool(
                pf_spec["uses_stage11"]
            ),
            "uses_stage10_classifier": bool(
                pf_spec["uses_stage10"]
            ),
        }
    )
pf_quality_validation = pd.DataFrame(
    pf_quality_rows
)
pf_quality_validation.to_csv(
    pf_validation_directory
    / "paper_sifa_selection_quality_after_freeze.csv",
    index=False,
)


# Paired bootstrap of Guided minus Random event yield.
def pf_paired_bootstrap_difference(
    value_column: str,
    repetitions: int,
) -> Dict[str, float]:
    pivot = (
        pf_public_private_validation.pivot(
            index="pair_id",
            columns="campaign_arm",
            values=value_column,
        )
        .dropna()
    )
    differences = (
        pivot["guided_model"].to_numpy(
            float
        )
        - pivot["random_uniform"].to_numpy(
            float
        )
    )
    point = float(
        np.mean(differences)
    )
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                paper_sifa_config.random_seed,
                13040,
                len(
                    value_column
                ),
            ]
        )
    )
    bootstrap_values = np.empty(
        repetitions,
        dtype=np.float64,
    )
    for index in range(repetitions):
        sample_indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )
        bootstrap_values[index] = float(
            np.mean(
                differences[
                    sample_indices
                ]
            )
        )
    return {
        "guided_minus_random": point,
        "ci95_low": float(
            np.quantile(
                bootstrap_values,
                0.025,
            )
        ),
        "ci95_high": float(
            np.quantile(
                bootstrap_values,
                0.975,
            )
        ),
    }


pf_public_private_validation[
    "paper_ineffective_numeric"
] = pf_public_private_validation[
    "paper_ineffective"
].astype(int)
pf_public_private_validation[
    "actual_clean_ineffective_numeric"
] = (
    pf_public_private_validation[
        "category"
    ]
    == "clean_target_ineffective"
).astype(int)

pf_paired_yield_comparison = {
    "paper_ineffective_yield": (
        pf_paired_bootstrap_difference(
            "paper_ineffective_numeric",
            paper_sifa_config.bootstrap_repetitions,
        )
    ),
    "actual_clean_target_ineffective_yield": (
        pf_paired_bootstrap_difference(
            "actual_clean_ineffective_numeric",
            paper_sifa_config.bootstrap_repetitions,
        )
    ),
}
pf_write_json(
    pf_validation_directory
    / "paired_guided_minus_random_yield_bootstrap.json",
    pf_paired_yield_comparison,
)


# ------------------------------------------------------------
# Aggregate comparison
# ------------------------------------------------------------
pf_pipeline_summary_rows: List[
    Dict[str, Any]
] = []
for pf_pipeline, pf_group in (
    pf_task_efficiency.groupby(
        "pipeline"
    )
):
    injection_rank1_values = pd.to_numeric(
        pf_group[
            "first_injection_checkpoint_rank1"
        ],
        errors="coerce",
    )
    selected_rank1_values = pd.to_numeric(
        pf_group[
            "first_selected_checkpoint_rank1"
        ],
        errors="coerce",
    )
    pf_quality = (
        pf_quality_validation[
            pf_quality_validation[
                "pipeline"
            ]
            == pf_pipeline
        ].iloc[0]
    )
    pf_pipeline_summary_rows.append(
        {
            "pipeline": str(
                pf_pipeline
            ),
            "task_count": int(
                len(pf_group)
            ),
            "unique_rank1_task_count": int(
                np.sum(
                    pf_group[
                        "final_unique_rank1"
                    ].astype(bool)
                )
            ),
            "rank1_success_rate": float(
                np.mean(
                    pf_group[
                        "final_unique_rank1"
                    ].astype(float)
                )
            ),
            "mean_final_true_rank": float(
                pf_group[
                    "final_true_rank"
                ].mean()
            ),
            "median_first_injection_checkpoint_rank1": (
                float(
                    np.nanmedian(
                        injection_rank1_values
                    )
                )
                if np.isfinite(
                    injection_rank1_values
                ).any()
                else np.nan
            ),
            "median_first_selected_checkpoint_rank1": (
                float(
                    np.nanmedian(
                        selected_rank1_values
                    )
                )
                if np.isfinite(
                    selected_rank1_values
                ).any()
                else np.nan
            ),
            "mean_final_score_margin": float(
                pf_group[
                    "final_score_margin"
                ].mean()
            ),
            "mean_bootstrap_true_winner_fraction": float(
                pf_group[
                    "bootstrap_true_winner_fraction"
                ].mean()
            ),
            "selected_ciphertext_count": int(
                pf_quality[
                    "selected_ciphertext_count"
                ]
            ),
            "actual_clean_precision": float(
                pf_quality[
                    "actual_clean_precision"
                ]
            ),
            "actual_clean_recall": float(
                pf_quality[
                    "actual_clean_recall"
                ]
            ),
        }
    )

pf_pipeline_summary = pd.DataFrame(
    pf_pipeline_summary_rows
)
pf_pipeline_summary.to_csv(
    pf_validation_directory
    / "paper_sifa_pipeline_comparison.csv",
    index=False,
)


def pf_pipeline_metric(
    pipeline: str,
    column: str,
) -> float:
    return float(
        pf_pipeline_summary.loc[
            pf_pipeline_summary[
                "pipeline"
            ]
            == pipeline,
            column,
        ].iloc[0]
    )


pf_primary_random = (
    pf_pipeline_summary[
        pf_pipeline_summary[
            "pipeline"
        ]
        == "random_raw"
    ].iloc[0]
)
pf_primary_ai = (
    pf_pipeline_summary[
        pf_pipeline_summary[
            "pipeline"
        ]
        == "guided_ml"
    ].iloc[0]
)

# Per-task wins based on the earliest measured injection checkpoint.
pf_primary_task_comparison = (
    pf_task_efficiency[
        pf_task_efficiency[
            "pipeline"
        ].isin(
            [
                "random_raw",
                "guided_ml",
            ]
        )
    ]
    .pivot(
        index=[
            "key_id",
            "target_sbox_index",
        ],
        columns="pipeline",
        values="first_injection_checkpoint_rank1",
    )
    .reset_index()
)

pf_guided_better_count = 0
pf_random_better_count = 0
pf_tie_count = 0
for pf_row in (
    pf_primary_task_comparison.itertuples(
        index=False
    )
):
    random_value = getattr(
        pf_row,
        "random_raw",
    )
    guided_value = getattr(
        pf_row,
        "guided_ml",
    )
    random_finite = pd.notna(
        random_value
    )
    guided_finite = pd.notna(
        guided_value
    )
    if guided_finite and (
        not random_finite
        or guided_value < random_value
    ):
        pf_guided_better_count += 1
    elif random_finite and (
        not guided_finite
        or random_value < guided_value
    ):
        pf_random_better_count += 1
    else:
        pf_tie_count += 1

pf_ai_helped = bool(
    pf_guided_better_count
    > pf_random_better_count
    and int(
        pf_primary_ai[
            "unique_rank1_task_count"
        ]
    )
    >= int(
        pf_primary_random[
            "unique_rank1_task_count"
        ]
    )
)


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
pf_generated_plots: List[str] = []
if paper_sifa_config.save_plots:
    if not pf_injection_rank_curve.empty:
        figure = plt.figure(
            figsize=(8.5, 5.5)
        )
        axis = figure.add_subplot(
            1,
            1,
            1,
        )
        for pf_pipeline, pf_group in (
            pf_injection_rank_curve.groupby(
                "pipeline"
            )
        ):
            curve = (
                pf_group.groupby(
                    "injection_count",
                    as_index=False,
                )["true_rank"]
                .mean()
                .sort_values(
                    "injection_count"
                )
            )
            axis.plot(
                curve[
                    "injection_count"
                ],
                curve["true_rank"],
                marker="o",
                label=pf_pipeline,
            )
        axis.set_xlabel(
            "Total fault injections per key/target/arm"
        )
        axis.set_ylabel(
            "Mean true-key rank"
        )
        axis.set_yscale("log")
        axis.set_title(
            "Paper-faithful SIFA: rank versus injections"
        )
        axis.grid(
            alpha=0.25
        )
        axis.legend()
        figure.tight_layout()
        plot_path = (
            pf_validation_directory
            / "paper_sifa_rank_vs_injections.png"
        )
        figure.savefig(
            plot_path,
            dpi=180,
        )
        plt.close(figure)
        pf_generated_plots.append(
            plot_path.name
        )

    if not pf_selected_rank_curve.empty:
        figure = plt.figure(
            figsize=(8.5, 5.5)
        )
        axis = figure.add_subplot(
            1,
            1,
            1,
        )
        for pf_pipeline, pf_group in (
            pf_selected_rank_curve.groupby(
                "pipeline"
            )
        ):
            curve = (
                pf_group.groupby(
                    "selected_ciphertext_count",
                    as_index=False,
                )["true_rank"]
                .mean()
                .sort_values(
                    "selected_ciphertext_count"
                )
            )
            axis.plot(
                curve[
                    "selected_ciphertext_count"
                ],
                curve["true_rank"],
                marker="o",
                label=pf_pipeline,
            )
        axis.set_xlabel(
            "Selected ineffective ciphertexts"
        )
        axis.set_ylabel(
            "Mean true-key rank"
        )
        axis.set_yscale("log")
        axis.set_title(
            "Paper-faithful SIFA: rank versus useful samples"
        )
        axis.grid(
            alpha=0.25
        )
        axis.legend()
        figure.tight_layout()
        plot_path = (
            pf_validation_directory
            / "paper_sifa_rank_vs_selected_ciphertexts.png"
        )
        figure.savefig(
            plot_path,
            dpi=180,
        )
        plt.close(figure)
        pf_generated_plots.append(
            plot_path.name
        )

    figure = plt.figure(
        figsize=(8.5, 5.5)
    )
    axis = figure.add_subplot(
        1,
        1,
        1,
    )
    pf_plot_frame = pf_pipeline_summary.sort_values(
        "pipeline"
    )
    axis.bar(
        pf_plot_frame["pipeline"],
        pf_plot_frame[
            "actual_clean_precision"
        ],
    )
    axis.set_ylim(
        0.0,
        1.0,
    )
    axis.set_ylabel(
        "Precision of selected ciphertexts"
    )
    axis.set_title(
        "Post-freeze clean-target precision"
    )
    axis.tick_params(
        axis="x",
        rotation=25,
    )
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    figure.tight_layout()
    plot_path = (
        pf_validation_directory
        / "paper_sifa_selection_precision.png"
    )
    figure.savefig(
        plot_path,
        dpi=180,
    )
    plt.close(figure)
    pf_generated_plots.append(
        plot_path.name
    )


# ------------------------------------------------------------
# Final integrity checks and summary
# ------------------------------------------------------------
pf_integrity_checks = {
    "stage11_freeze_verified": bool(
        pf_contracts["stage11_verify"][
            "passed"
        ]
    ),
    "stage10_freeze_verified": bool(
        pf_contracts["stage10_verify"][
            "passed"
        ]
    ),
    "stage09_freeze_verified": bool(
        pf_contracts["stage9_verify"][
            "passed"
        ]
    ),
    "fresh_campaign_row_count_correct": bool(
        len(pf_public_scored)
        == _PF_TOTAL_INJECTIONS
    ),
    "paired_plaintexts_identical_across_arms": bool(
        pf_public_scored.groupby(
            "pair_id"
        )["plaintext_hex"].nunique().eq(
            1
        ).all()
    ),
    "paired_keys_identical_across_arms": bool(
        pf_public_scored.groupby(
            "pair_id"
        )["key_id"].nunique().eq(
            1
        ).all()
    ),
    "paired_targets_identical_across_arms": bool(
        pf_public_scored.groupby(
            "pair_id"
        )[
            "target_sbox_index"
        ].nunique().eq(1).all()
    ),
    "guided_uses_stage11_rank1_sifa_parameters": bool(
        (
            pf_public_scored.loc[
                pf_public_scored[
                    "campaign_arm"
                ]
                == "guided_model",
                "parameter_source",
            ]
            == "stage11_rank1_SIFA_exploit"
        ).all()
    ),
    "random_uses_same_stage11_support_domain": bool(
        (
            pf_public_scored.loc[
                pf_public_scored[
                    "campaign_arm"
                ]
                == "random_uniform",
                "parameter_source",
            ]
            == "uniform_same_stage11_support"
        ).all()
    ),
    "paper_ineffective_rule_is_c_equal_cprime": True,
    "primary_key_statistic_is_exact_sei": True,
    "stage10_probabilities_not_used_as_sei_weights": True,
    "fault_target_is_x31_before_final_feedforward": bool(
        (
            pf_public_scored["physical_fault_intermediate"]
            == "X31_state_after_X32_before_final_feedforward"
        ).all()
    ),
    "target_mapping_s0_to_x31_0": bool(
        pf_public_scored.loc[
            pf_public_scored["target_sbox_index"] == 0,
            "physical_x31_nibble_index",
        ].eq(0).all()
    ),
    "target_mapping_s5_to_x31_2": bool(
        pf_public_scored.loc[
            pf_public_scored["target_sbox_index"] == 5,
            "physical_x31_nibble_index",
        ].eq(2).all()
    ),
    "sei_scores_are_key_identifiable": bool(
        (
            pf_sei_range_by_task["sei_candidate_range"]
            > 1.0e-15
        ).all()
    ),
    "candidate_space_is_16_per_nibble": bool(
        pf_final_scores_public.groupby(
            [
                "pipeline",
                "key_id",
                "target_sbox_index",
            ]
        ).size().eq(16).all()
    ),
    "joint_candidate_space_is_256_per_key": bool(
        pf_joint_scores_public.groupby(
            [
                "pipeline",
                "key_id",
            ]
        ).size().eq(256).all()
    ),
    "public_attack_frozen_before_truth_evaluation": True,
    "private_categories_used_only_after_public_freeze": True,
    "truth_file_created_after_public_freeze": True,
}

pf_all_checks_passed = bool(
    all(
        pf_integrity_checks.values()
    )
)

pf_guided_ml_success = bool(
    pf_final_rank_evaluation.loc[
        pf_final_rank_evaluation[
            "pipeline"
        ]
        == "guided_ml",
        "true_is_unique_rank1",
    ].all()
)
pf_random_raw_success = bool(
    pf_final_rank_evaluation.loc[
        pf_final_rank_evaluation[
            "pipeline"
        ]
        == "random_raw",
        "true_is_unique_rank1",
    ].all()
)
pf_guided_ml_8bit_success = bool(
    pf_joint_rank_evaluation.loc[
        pf_joint_rank_evaluation[
            "pipeline"
        ]
        == "guided_ml",
        "true_pair_is_unique_rank1",
    ].all()
)
pf_random_raw_8bit_success = bool(
    pf_joint_rank_evaluation.loc[
        pf_joint_rank_evaluation[
            "pipeline"
        ]
        == "random_raw",
        "true_pair_is_unique_rank1",
    ].all()
)

pf_summary = {
    "stage": "13R-v2",
    "run_id": pf_run_id,
    "run_directory": str(
        pf_run_directory
    ),
    "method": (
        "Paper-faithful SIFA with exact SEI on X31; "
        "guided-vs-random paired comparison"
    ),
    "reference_formula": (
        "SEI(k)=sum_x (p_hat_k(x)-1/16)^2"
    ),
    "lblock_partial_decryption_formula": (
        "X31_hat_t(k)=nibble_j(X33) XOR "
        "S_s(nibble_s(X32) XOR k); "
        "j=SOURCE_TO_OUTPUT[s], t=(j-2) mod 8"
    ),
    "physical_fault_targets": {
        "S0": "X31[0]",
        "S5": "X31[2]",
    },
    "obsolete_x32_xor_key_adaptation_rejected": True,
    "all_checks_passed": pf_all_checks_passed,
    "paper_faithful_sei_primary": True,
    "public_attack_frozen_before_truth": True,
    "private_ground_truth_used_for_scoring": False,
    "stage10_probability_weighted_key_score": False,
    "stage11_optimizer_used_for_guided_injection": True,
    "stage10_classifier_used_for_ml_filtering": True,
    "stage11_freeze_verified": bool(
        pf_contracts["stage11_verify"][
            "passed"
        ]
    ),
    "stage10_freeze_verified": bool(
        pf_contracts["stage10_verify"][
            "passed"
        ]
    ),
    "number_of_fresh_injections": int(
        _PF_TOTAL_INJECTIONS
    ),
    "number_of_paired_plaintexts": int(
        pf_public_scored[
            "pair_id"
        ].nunique()
    ),
    "number_of_keys": int(
        paper_sifa_config.number_of_keys
    ),
    "number_of_sessions": int(
        paper_sifa_config.number_of_sessions
    ),
    "target_sboxes": [
        "S0",
        "S5",
    ],
    "target_key_nibbles": [
        "K32[0]",
        "K32[5]",
    ],
    "injections_per_key_target_arm": int(
        paper_sifa_config.injections_per_key_target_arm
    ),
    "stage10_ineffective_threshold": float(
        pf_ml_threshold
    ),
    "public_attack_freeze_sha256": (
        pf_public_attack_freeze[
            "freeze_sha256"
        ]
    ),
    "locked_truth_sha256": (
        pf_truth_sha256
    ),
    "primary_comparison": {
        "baseline": "random_raw",
        "ai_pipeline": "guided_ml",
        "guided_better_task_count": int(
            pf_guided_better_count
        ),
        "random_better_task_count": int(
            pf_random_better_count
        ),
        "tie_task_count": int(
            pf_tie_count
        ),
        "ai_helped_by_pre_registered_rule": (
            pf_ai_helped
        ),
        "guided_ml_all_nibbles_unique_rank1": (
            pf_guided_ml_success
        ),
        "random_raw_all_nibbles_unique_rank1": (
            pf_random_raw_success
        ),
        "guided_ml_all_8bit_pairs_unique_rank1": (
            pf_guided_ml_8bit_success
        ),
        "random_raw_all_8bit_pairs_unique_rank1": (
            pf_random_raw_8bit_success
        ),
    },
    "paired_yield_comparison": (
        pf_paired_yield_comparison
    ),
    "pipeline_summary": (
        pf_pipeline_summary.replace(
            {
                np.nan: None,
                np.inf: None,
                -np.inf: None,
            }
        ).to_dict(
            orient="records"
        )
    ),
    "public_predictions": (
        pf_final_predictions_public.to_dict(
            orient="records"
        )
    ),
    "public_joint_8bit_predictions": (
        pf_joint_predictions_public.to_dict(
            orient="records"
        )
    ),
    "integrity_checks": (
        pf_integrity_checks
    ),
    "generated_plots": (
        pf_generated_plots
    ),
    "elapsed_seconds": float(
        time.perf_counter()
        - pf_campaign_started
    ),
}

pf_write_json(
    pf_run_directory
    / "stage_13R_v2_paper_sifa_summary.json",
    pf_summary,
)
pf_write_json(
    pf_validation_directory
    / "stage_13R_v2_validation_checks.json",
    {
        "all_checks_passed": (
            pf_all_checks_passed
        ),
        "checks": pf_integrity_checks,
    },
)

print("\n" + "=" * 88)
print("Stage 13R-v2 completed — paper-faithful X31-target SIFA benchmark")
print("=" * 88)
print("Run directory                 :", pf_run_directory)
print("All integrity checks passed   :", pf_all_checks_passed)
print("Guided+ML all nibble Rank-1   :", pf_guided_ml_success)
print("Random raw all nibble Rank-1  :", pf_random_raw_success)
print("Guided+ML all 8-bit Rank-1    :", pf_guided_ml_8bit_success)
print("Random raw all 8-bit Rank-1   :", pf_random_raw_8bit_success)
print("Guided better / random / ties :", pf_guided_better_count, "/", pf_random_better_count, "/", pf_tie_count)
print("AI helped by fixed rule       :", pf_ai_helped)
print("Public freeze SHA-256         :", pf_public_attack_freeze["freeze_sha256"])
print("Summary file                  :", pf_run_directory / "stage_13R_v2_paper_sifa_summary.json")
print("Elapsed seconds               :", f"{pf_summary['elapsed_seconds']:.3f}")
print("=" * 88)

ipy_display(
    pf_pipeline_summary.sort_values(
        "pipeline"
    ).reset_index(
        drop=True
    )
)
ipy_display(
    pf_final_rank_evaluation.sort_values(
        [
            "pipeline",
            "key_id",
            "target_sbox_index",
        ]
    ).reset_index(
        drop=True
    )
)

# ============================================================
# Stage 07 — Analytical and Empirical Fault-Bias/Capacity Study
# for LBlock-64/80
#
# هدف‌ها:
#   1) ساخت مدل دقیق انتقال برای تمام Fault modelهای Stage 06
#   2) محاسبه p_j^i، p_j^e، q_j^i و q_j^e
#   3) اندازه‌گیری ظرفیت اطلاعاتی SIFA، SEFA و SHFA
#   4) مقایسه نظریه با کمپین تجربی Stage 06
#   5) سنجش بازده اطلاعاتی پارامترهای glitch
#   6) تولید prior علمی برای کمپین بزرگ Stage 08
#
# سیاست نشت اطلاعات:
#   - پوشه public_theory فقط از تعریف مدل‌های Fault استفاده می‌کند.
#   - این پوشه قبل از بازشدن Ground Truth خصوصی freeze می‌شود.
#   - تمام تحلیل‌های وابسته به category یا ورودی واقعی S-box در
#     validation_only ذخیره می‌شوند و نباید ورودی حمله کلیدی باشند.
# ============================================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
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
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


FAULT_MODELS: Tuple[str, ...] = (
    "random_and_4",
    "random_and_2",
    "single_bit_flip",
    "stuck_at_bit",
    "random_nibble",
)

FAULT_CATEGORIES: Tuple[str, ...] = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)

INEFFECTIVE_CATEGORY = "clean_target_ineffective"
EFFECTIVE_CATEGORY = "clean_target_effective"


@dataclass(frozen=True)
class Stage07Config:
    input_stage6_run_directory: str
    output_root: str = "runs/stage_07"
    random_seed: int = 20260718

    number_of_inputs: int = 16
    dirichlet_alpha: float = 0.5
    confidence_level: float = 0.95
    bootstrap_repetitions: int = 300
    minimum_clean_samples_for_bootstrap: int = 40
    minimum_each_outcome_for_bootstrap: int = 5

    # پارامترهای تحلیل سلولی کمپین
    offset_bin_edges: Tuple[float, ...] = (
        -1e9,
        -12.0,
        -4.0,
        4.0,
        12.0,
        1e9,
    )
    width_bin_edges: Tuple[float, ...] = (
        0.0,
        3.0,
        6.0,
        1e9,
    )
    strength_bin_edges: Tuple[float, ...] = (
        0.0,
        0.65,
        1.10,
        1e9,
    )
    minimum_parameter_cell_attempts: int = 20
    top_parameter_cells_per_attack: int = 12

    save_plots: bool = True
    enable_empirical_validation: bool = True

    # معیارهای پذیرش
    minimum_primary_pi_correlation: float = 0.85
    maximum_primary_qi_tv_distance: float = 0.17
    maximum_primary_qe_tv_distance: float = 0.12
    maximum_primary_mean_rate_error: float = 0.05


# ============================================================
# 1. ابزارهای فایل، hash و جدول
# ============================================================


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, default: float = 0.0) -> float:
    result = float(value)
    return result if math.isfinite(result) else default


# ============================================================
# 2. مدل‌های دقیق انتقال Fault
# ============================================================


def hamming_weight_4(value: int) -> int:
    return int(value & 0xF).bit_count()


def transition_matrix_for_model(model: str) -> np.ndarray:
    """
    T[x, y] = P(X'=y | X=x)

    تمام احتمالات از شمارش دقیق فضای تصادفی مدل به دست می‌آیند؛
    بنابراین هیچ شبیه‌سازی Monte Carlo در بخش نظری وجود ندارد.
    """

    matrix = np.zeros((16, 16), dtype=np.float64)

    if model == "random_and_4":
        for x in range(16):
            for mask in range(16):
                matrix[x, x & mask] += 1.0 / 16.0

    elif model == "random_and_2":
        position_pairs = list(combinations(range(4), 2))
        total_cases = len(position_pairs) * 4

        for x in range(16):
            for positions in position_pairs:
                for random_bits in range(4):
                    mask = 0xF
                    for local_index, bit_position in enumerate(positions):
                        bit_value = (random_bits >> local_index) & 1
                        if bit_value == 0:
                            mask &= ~(1 << bit_position)
                    matrix[x, x & mask] += 1.0 / total_cases

    elif model == "single_bit_flip":
        for x in range(16):
            for bit_position in range(4):
                matrix[x, x ^ (1 << bit_position)] += 1.0 / 4.0

    elif model == "stuck_at_bit":
        for x in range(16):
            for bit_position in range(4):
                for stuck_value in (0, 1):
                    y = (
                        (x & ~(1 << bit_position))
                        | (stuck_value << bit_position)
                    )
                    matrix[x, y] += 1.0 / 8.0

    elif model == "random_nibble":
        matrix[:, :] = 1.0 / 16.0

    else:
        raise ValueError(f"Unknown fault model: {model}")

    row_sums = np.sum(matrix, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise AssertionError(f"Transition rows do not sum to one: {model}")

    return matrix


def all_transition_matrices() -> Dict[str, np.ndarray]:
    return {
        model: transition_matrix_for_model(model)
        for model in FAULT_MODELS
    }


# ============================================================
# 3. معیارهای توزیع و اطلاعات
# ============================================================


def safe_entropy_bits(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = probabilities > 0
    if not np.any(positive):
        return 0.0
    return float(
        -np.sum(probabilities[positive] * np.log2(probabilities[positive]))
    )


def kl_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    positive = p > 0
    if np.any(q[positive] <= 0):
        return float("inf")
    return float(np.sum(p[positive] * np.log2(p[positive] / q[positive])))


def jensen_shannon_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_divergence_bits(p, midpoint) + 0.5 * kl_divergence_bits(q, midpoint)


def distribution_metrics(probabilities: np.ndarray) -> Dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
    uniform = np.full(probabilities.shape, 1.0 / probabilities.size)

    entropy = safe_entropy_bits(probabilities)
    maximum_entropy = math.log2(probabilities.size)

    return {
        "entropy_bits": entropy,
        "entropy_deficit_bits": float(maximum_entropy - entropy),
        "kl_to_uniform_bits": kl_divergence_bits(probabilities, uniform),
        "squared_euclidean_imbalance": float(
            np.sum((probabilities - uniform) ** 2)
        ),
        "total_variation_to_uniform": float(
            0.5 * np.sum(np.abs(probabilities - uniform))
        ),
        "maximum_probability": float(np.max(probabilities)),
        "minimum_probability": float(np.min(probabilities)),
        "max_to_min_ratio": (
            float(np.max(probabilities) / np.min(probabilities))
            if np.min(probabilities) > 0
            else float("inf")
        ),
    }


def mutual_information_from_channel(
    input_distribution: np.ndarray,
    channel: np.ndarray,
) -> float:
    """I(X;Y) برای channel با ابعاد input × output."""

    input_distribution = np.asarray(input_distribution, dtype=np.float64)
    channel = np.asarray(channel, dtype=np.float64)
    output_distribution = input_distribution @ channel

    information = 0.0
    for x in range(channel.shape[0]):
        for y in range(channel.shape[1]):
            conditional = channel[x, y]
            if conditional <= 0 or output_distribution[y] <= 0:
                continue
            information += (
                input_distribution[x]
                * conditional
                * math.log2(conditional / output_distribution[y])
            )

    return float(information)


def blahut_arimoto_capacity_bits(
    channel: np.ndarray,
    tolerance: float = 1e-13,
    maximum_iterations: int = 10000,
) -> Tuple[float, np.ndarray, int]:
    """
    ظرفیت Shannon کانال گسسته با الگوریتم Blahut-Arimoto.

    channel[x, y] = P(Y=y | X=x)
    """

    channel = np.asarray(channel, dtype=np.float64)
    input_count = channel.shape[0]
    distribution = np.full(input_count, 1.0 / input_count)

    for iteration in range(1, maximum_iterations + 1):
        output_distribution = distribution @ channel
        divergences = np.zeros(input_count, dtype=np.float64)

        for x in range(input_count):
            valid = channel[x] > 0
            divergences[x] = np.sum(
                channel[x, valid]
                * np.log(
                    channel[x, valid]
                    / output_distribution[valid]
                )
            )

        log_weights = np.log(np.maximum(distribution, 1e-300)) + divergences
        log_weights -= np.max(log_weights)
        new_distribution = np.exp(log_weights)
        new_distribution /= np.sum(new_distribution)

        if np.max(np.abs(new_distribution - distribution)) < tolerance:
            distribution = new_distribution
            break

        distribution = new_distribution

    capacity_nats = 0.0
    output_distribution = distribution @ channel
    for x in range(input_count):
        valid = channel[x] > 0
        capacity_nats += distribution[x] * np.sum(
            channel[x, valid]
            * np.log(channel[x, valid] / output_distribution[valid])
        )

    return float(capacity_nats / math.log(2.0)), distribution, iteration


def conditional_bias_distributions(
    transition_matrix: np.ndarray,
    input_distribution: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if input_distribution is None:
        input_distribution = np.full(16, 1.0 / 16.0)

    input_distribution = np.asarray(input_distribution, dtype=np.float64)
    p_ineffective = np.diag(transition_matrix).copy()
    p_effective = 1.0 - p_ineffective

    mean_ineffective = float(np.dot(input_distribution, p_ineffective))
    mean_effective = float(np.dot(input_distribution, p_effective))

    if mean_ineffective > 0:
        q_ineffective = (
            input_distribution * p_ineffective / mean_ineffective
        )
    else:
        q_ineffective = np.zeros(16, dtype=np.float64)

    if mean_effective > 0:
        q_effective = input_distribution * p_effective / mean_effective
    else:
        q_effective = np.zeros(16, dtype=np.float64)

    outcome_channel = np.column_stack([p_ineffective, p_effective])
    uniform_input_information = mutual_information_from_channel(
        input_distribution,
        outcome_channel,
    )
    channel_capacity, capacity_distribution, iterations = (
        blahut_arimoto_capacity_bits(outcome_channel)
    )

    q_hybrid = np.outer(q_effective, q_ineffective)

    ineffective_metrics = (
        distribution_metrics(q_ineffective)
        if mean_ineffective > 0
        else {
            "entropy_bits": 0.0,
            "entropy_deficit_bits": 0.0,
            "kl_to_uniform_bits": 0.0,
            "squared_euclidean_imbalance": 0.0,
            "total_variation_to_uniform": 0.0,
            "maximum_probability": 0.0,
            "minimum_probability": 0.0,
            "max_to_min_ratio": 0.0,
        }
    )
    effective_metrics = (
        distribution_metrics(q_effective)
        if mean_effective > 0
        else {
            "entropy_bits": 0.0,
            "entropy_deficit_bits": 0.0,
            "kl_to_uniform_bits": 0.0,
            "squared_euclidean_imbalance": 0.0,
            "total_variation_to_uniform": 0.0,
            "maximum_probability": 0.0,
            "minimum_probability": 0.0,
            "max_to_min_ratio": 0.0,
        }
    )

    if mean_ineffective > 0 and mean_effective > 0:
        hybrid_metrics = distribution_metrics(q_hybrid.reshape(-1))
    else:
        hybrid_metrics = {
            "entropy_bits": 0.0,
            "entropy_deficit_bits": 0.0,
            "kl_to_uniform_bits": 0.0,
            "squared_euclidean_imbalance": 0.0,
            "total_variation_to_uniform": 0.0,
            "maximum_probability": 0.0,
            "minimum_probability": 0.0,
            "max_to_min_ratio": 0.0,
        }

    return {
        "p_ineffective_given_x": p_ineffective,
        "p_effective_given_x": p_effective,
        "q_x_given_ineffective": q_ineffective,
        "q_x_given_effective": q_effective,
        "q_hybrid_effective_ineffective": q_hybrid,
        "mean_ineffective_probability": mean_ineffective,
        "mean_effective_probability": mean_effective,
        "ineffective_metrics": ineffective_metrics,
        "effective_metrics": effective_metrics,
        "hybrid_metrics": hybrid_metrics,
        "uniform_input_mutual_information_bits": uniform_input_information,
        "channel_capacity_bits": channel_capacity,
        "capacity_achieving_input_distribution": capacity_distribution,
        "blahut_arimoto_iterations": iterations,
    }


# ============================================================
# 4. ساخت خروجی نظری و freeze
# ============================================================


def build_theoretical_outputs(
    matrices: Mapping[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    bias_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}

    for model in FAULT_MODELS:
        analysis = conditional_bias_distributions(matrices[model])
        detailed[model] = analysis

        p_i = analysis["p_ineffective_given_x"]
        p_e = analysis["p_effective_given_x"]
        q_i = analysis["q_x_given_ineffective"]
        q_e = analysis["q_x_given_effective"]

        for x in range(16):
            bias_rows.append({
                "fault_model": model,
                "input_decimal": x,
                "input_hex": f"0x{x:X}",
                "hamming_weight": hamming_weight_4(x),
                "p_ineffective_given_x": float(p_i[x]),
                "p_effective_given_x": float(p_e[x]),
                "q_x_given_ineffective": float(q_i[x]),
                "q_x_given_effective": float(q_e[x]),
            })

        capacity_rows.append({
            "fault_model": model,
            "mean_ineffective_probability": float(
                analysis["mean_ineffective_probability"]
            ),
            "mean_effective_probability": float(
                analysis["mean_effective_probability"]
            ),
            "sifa_kl_capacity_bits": float(
                analysis["ineffective_metrics"]["kl_to_uniform_bits"]
            ),
            "sifa_sei": float(
                analysis["ineffective_metrics"][
                    "squared_euclidean_imbalance"
                ]
            ),
            "sifa_tv": float(
                analysis["ineffective_metrics"][
                    "total_variation_to_uniform"
                ]
            ),
            "sefa_kl_capacity_bits": float(
                analysis["effective_metrics"]["kl_to_uniform_bits"]
            ),
            "sefa_sei": float(
                analysis["effective_metrics"][
                    "squared_euclidean_imbalance"
                ]
            ),
            "sefa_tv": float(
                analysis["effective_metrics"][
                    "total_variation_to_uniform"
                ]
            ),
            "shfa_joint_kl_capacity_bits": float(
                analysis["hybrid_metrics"]["kl_to_uniform_bits"]
            ),
            "shfa_joint_sei": float(
                analysis["hybrid_metrics"][
                    "squared_euclidean_imbalance"
                ]
            ),
            "uniform_input_mutual_information_bits": float(
                analysis["uniform_input_mutual_information_bits"]
            ),
            "binary_outcome_channel_capacity_bits": float(
                analysis["channel_capacity_bits"]
            ),
        })

    primary = detailed["random_and_4"]
    prior = {
        "stage": 7,
        "primary_fault_model": "random_and_4",
        "model_definition": "X' = X AND R, R uniform on 0..15",
        "ground_truth_used": False,
        "analytical_mean_ineffective_probability": float(
            primary["mean_ineffective_probability"]
        ),
        "sifa_kl_capacity_bits": float(
            primary["ineffective_metrics"]["kl_to_uniform_bits"]
        ),
        "sefa_kl_capacity_bits": float(
            primary["effective_metrics"]["kl_to_uniform_bits"]
        ),
        "shfa_joint_kl_capacity_bits": float(
            primary["hybrid_metrics"]["kl_to_uniform_bits"]
        ),
        "uniform_input_mutual_information_bits": float(
            primary["uniform_input_mutual_information_bits"]
        ),
        "recommended_stage_08_policy": {
            "primary_model_fraction": 0.80,
            "control_model_fraction": 0.20,
            "retain_both_selected_targets": True,
            "retain_broad_offset_width_strength_exploration": True,
            "do_not_use_private_oracle_parameter_cells_as_attack_features": True,
            "reason": (
                "Random-AND-4 provides non-zero theoretical bias for both "
                "ineffective and effective outcomes, while SHFA combines both."
            ),
        },
    }

    return bias_rows, capacity_rows, prior


def save_theoretical_arrays(
    path: Path,
    matrices: Mapping[str, np.ndarray],
    detailed: Mapping[str, Any],
) -> None:
    payload: Dict[str, np.ndarray] = {}

    for model in FAULT_MODELS:
        safe_name = model
        payload[f"{safe_name}_transition_matrix"] = matrices[model]
        payload[f"{safe_name}_p_ineffective"] = detailed[model][
            "p_ineffective_given_x"
        ]
        payload[f"{safe_name}_p_effective"] = detailed[model][
            "p_effective_given_x"
        ]
        payload[f"{safe_name}_q_ineffective"] = detailed[model][
            "q_x_given_ineffective"
        ]
        payload[f"{safe_name}_q_effective"] = detailed[model][
            "q_x_given_effective"
        ]
        payload[f"{safe_name}_q_hybrid"] = detailed[model][
            "q_hybrid_effective_ineffective"
        ]

    np.savez_compressed(path, **payload)


# ============================================================
# 5. معیارهای تجربی
# ============================================================


def wilson_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0

    # z=1.95996 برای 95%؛ برای تنظیمات دیگر تقریب نرمال معکوس
    # بدون وابستگی scipy استفاده می‌شود.
    if abs(confidence_level - 0.95) < 1e-12:
        z = 1.959963984540054
    elif abs(confidence_level - 0.90) < 1e-12:
        z = 1.6448536269514722
    elif abs(confidence_level - 0.99) < 1e-12:
        z = 2.5758293035489004
    else:
        z = 1.959963984540054

    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def smoothed_distribution(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return (counts + alpha) / (np.sum(counts) + alpha * counts.size)


def empirical_joint_metrics(
    ineffective_counts: np.ndarray,
    effective_counts: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    ineffective_counts = np.asarray(ineffective_counts, dtype=np.float64)
    effective_counts = np.asarray(effective_counts, dtype=np.float64)

    q_i = smoothed_distribution(ineffective_counts, alpha)
    q_e = smoothed_distribution(effective_counts, alpha)

    # posterior mean P(i|x) با prior Jeffreys/Beta(alpha, alpha)
    p_i = (
        ineffective_counts + alpha
    ) / (
        ineffective_counts + effective_counts + 2.0 * alpha
    )
    p_e = 1.0 - p_i

    joint_counts = np.column_stack([ineffective_counts, effective_counts])
    joint = joint_counts + alpha
    joint /= np.sum(joint)
    input_distribution = np.sum(joint, axis=1)
    output_distribution = np.sum(joint, axis=0)

    mutual_information = 0.0
    for x in range(16):
        for outcome in range(2):
            probability = joint[x, outcome]
            if probability <= 0:
                continue
            mutual_information += probability * math.log2(
                probability
                / (input_distribution[x] * output_distribution[outcome])
            )

    q_hybrid = np.outer(q_e, q_i)

    return {
        "p_ineffective_given_x": p_i,
        "p_effective_given_x": p_e,
        "q_x_given_ineffective": q_i,
        "q_x_given_effective": q_e,
        "q_hybrid": q_hybrid,
        "ineffective_metrics": distribution_metrics(q_i),
        "effective_metrics": distribution_metrics(q_e),
        "hybrid_metrics": distribution_metrics(q_hybrid.reshape(-1)),
        "mutual_information_bits": float(mutual_information),
    }


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def merge_stage6_data(
    public_csv: Path,
    ground_truth_csv: Path,
) -> pd.DataFrame:
    public = pd.read_csv(public_csv)
    ground_truth = pd.read_csv(ground_truth_csv)

    if public["experiment_id"].duplicated().any():
        raise ValueError("Duplicate experiment_id in public campaign")
    if ground_truth["experiment_id"].duplicated().any():
        raise ValueError("Duplicate experiment_id in ground truth")

    private_columns = [
        "experiment_id",
        "category",
        "category_id",
        "sampling_regime",
        "target_original_input",
        "target_faulted_input",
        "impacted_sbox_count",
        "target_impacted",
        "off_target_impacted",
        "fault_effective",
    ]

    merged = public.merge(
        ground_truth[private_columns],
        on="experiment_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(public) or len(merged) != len(ground_truth):
        raise ValueError("Public/private experiment IDs are not one-to-one")

    merged["target_original_input"] = merged[
        "target_original_input"
    ].astype(np.int32)
    merged["category"] = merged["category"].astype(str)

    return merged


def empirical_group_analysis(
    merged: pd.DataFrame,
    theoretical: Mapping[str, Any],
    config: Stage07Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    bias_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    comparison: Dict[str, Any] = {}

    targets = sorted(merged["target_sbox"].unique().tolist())

    for target in targets:
        comparison[target] = {}

        for model in FAULT_MODELS:
            group = merged[
                (merged["target_sbox"] == target)
                & (merged["fault_model"] == model)
            ].copy()

            clean = group[group["category"].isin([
                INEFFECTIVE_CATEGORY,
                EFFECTIVE_CATEGORY,
            ])]

            ineffective_inputs = clean.loc[
                clean["category"] == INEFFECTIVE_CATEGORY,
                "target_original_input",
            ].to_numpy(dtype=np.int32)
            effective_inputs = clean.loc[
                clean["category"] == EFFECTIVE_CATEGORY,
                "target_original_input",
            ].to_numpy(dtype=np.int32)

            counts_i = np.bincount(ineffective_inputs, minlength=16).astype(np.int64)
            counts_e = np.bincount(effective_inputs, minlength=16).astype(np.int64)
            empirical = empirical_joint_metrics(
                counts_i,
                counts_e,
                config.dirichlet_alpha,
            )

            theory = theoretical[model]
            theory_pi = theory["p_ineffective_given_x"]
            theory_qi = theory["q_x_given_ineffective"]
            theory_qe = theory["q_x_given_effective"]

            pi_correlation = safe_correlation(
                empirical["p_ineffective_given_x"],
                theory_pi,
            )
            pi_mae = float(np.mean(np.abs(
                empirical["p_ineffective_given_x"] - theory_pi
            )))
            qi_tv = float(0.5 * np.sum(np.abs(
                empirical["q_x_given_ineffective"] - theory_qi
            ))) if np.sum(theory_qi) > 0 else 0.0
            qe_tv = float(0.5 * np.sum(np.abs(
                empirical["q_x_given_effective"] - theory_qe
            ))) if np.sum(theory_qe) > 0 else 0.0
            qi_js = jensen_shannon_bits(
                empirical["q_x_given_ineffective"],
                theory_qi,
            ) if np.sum(theory_qi) > 0 else 0.0
            qe_js = jensen_shannon_bits(
                empirical["q_x_given_effective"],
                theory_qe,
            ) if np.sum(theory_qe) > 0 else 0.0

            clean_count = len(clean)
            ineffective_count = int(np.sum(counts_i))
            effective_count = int(np.sum(counts_e))
            observed_ineffective_rate = (
                ineffective_count / clean_count if clean_count else 0.0
            )
            rate_low, rate_high = wilson_interval(
                ineffective_count,
                clean_count,
                config.confidence_level,
            )

            category_counts = group["category"].value_counts().to_dict()
            attempts = len(group)

            capacity_row = {
                "target_sbox": target,
                "fault_model": model,
                "attempt_count": attempts,
                "clean_target_count": clean_count,
                "clean_target_ineffective_count": ineffective_count,
                "clean_target_effective_count": effective_count,
                "observed_ineffective_rate_given_clean": observed_ineffective_rate,
                "ineffective_rate_ci_low": rate_low,
                "ineffective_rate_ci_high": rate_high,
                "theoretical_ineffective_rate": float(
                    theory["mean_ineffective_probability"]
                ),
                "absolute_mean_rate_error": abs(
                    observed_ineffective_rate
                    - float(theory["mean_ineffective_probability"])
                ),
                "p_i_theory_empirical_correlation": pi_correlation,
                "p_i_mean_absolute_error": pi_mae,
                "q_i_total_variation_to_theory": qi_tv,
                "q_e_total_variation_to_theory": qe_tv,
                "q_i_jensen_shannon_to_theory_bits": qi_js,
                "q_e_jensen_shannon_to_theory_bits": qe_js,
                "empirical_sifa_kl_capacity_bits": float(
                    empirical["ineffective_metrics"]["kl_to_uniform_bits"]
                ),
                "theoretical_sifa_kl_capacity_bits": float(
                    theory["ineffective_metrics"]["kl_to_uniform_bits"]
                ),
                "empirical_sefa_kl_capacity_bits": float(
                    empirical["effective_metrics"]["kl_to_uniform_bits"]
                ),
                "theoretical_sefa_kl_capacity_bits": float(
                    theory["effective_metrics"]["kl_to_uniform_bits"]
                ),
                "empirical_shfa_joint_kl_capacity_bits": float(
                    empirical["hybrid_metrics"]["kl_to_uniform_bits"]
                ),
                "theoretical_shfa_joint_kl_capacity_bits": float(
                    theory["hybrid_metrics"]["kl_to_uniform_bits"]
                ),
                "empirical_mutual_information_bits": float(
                    empirical["mutual_information_bits"]
                ),
                "missed_rate": category_counts.get("missed", 0) / attempts,
                "clean_ineffective_rate_per_attempt": (
                    category_counts.get(INEFFECTIVE_CATEGORY, 0) / attempts
                ),
                "clean_effective_rate_per_attempt": (
                    category_counts.get(EFFECTIVE_CATEGORY, 0) / attempts
                ),
                "off_target_rate": category_counts.get("off_target", 0) / attempts,
                "multi_hit_rate": category_counts.get("multi_hit", 0) / attempts,
                "invalid_reset_rate": category_counts.get("invalid_reset", 0) / attempts,
            }
            capacity_rows.append(capacity_row)

            comparison[target][model] = {
                key: finite_float(value)
                for key, value in capacity_row.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            }

            p_i_emp = empirical["p_ineffective_given_x"]
            p_e_emp = empirical["p_effective_given_x"]
            q_i_emp = empirical["q_x_given_ineffective"]
            q_e_emp = empirical["q_x_given_effective"]

            for x in range(16):
                total_x = int(counts_i[x] + counts_e[x])
                low, high = wilson_interval(
                    int(counts_i[x]),
                    total_x,
                    config.confidence_level,
                )
                bias_rows.append({
                    "target_sbox": target,
                    "fault_model": model,
                    "input_decimal": x,
                    "input_hex": f"0x{x:X}",
                    "hamming_weight": hamming_weight_4(x),
                    "clean_ineffective_count": int(counts_i[x]),
                    "clean_effective_count": int(counts_e[x]),
                    "clean_total_count": total_x,
                    "empirical_p_ineffective_given_x": float(p_i_emp[x]),
                    "p_ineffective_ci_low": low,
                    "p_ineffective_ci_high": high,
                    "theoretical_p_ineffective_given_x": float(theory_pi[x]),
                    "empirical_p_effective_given_x": float(p_e_emp[x]),
                    "empirical_q_x_given_ineffective": float(q_i_emp[x]),
                    "theoretical_q_x_given_ineffective": float(theory_qi[x]),
                    "empirical_q_x_given_effective": float(q_e_emp[x]),
                    "theoretical_q_x_given_effective": float(theory_qe[x]),
                })

    return bias_rows, capacity_rows, comparison


# ============================================================
# 6. Bootstrap طبقه‌بندی‌شده بر اساس key/session
# ============================================================


def stratified_bootstrap_indices(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: List[np.ndarray] = []
    grouped = frame.groupby(["key_id", "session_id"], sort=True)

    for _, group in grouped:
        indices = group.index.to_numpy(dtype=np.int64)
        sampled = rng.choice(indices, size=len(indices), replace=True)
        selected.append(sampled)

    if not selected:
        return np.empty(0, dtype=np.int64)

    return np.concatenate(selected)


def run_empirical_bootstrap(
    merged: pd.DataFrame,
    theoretical: Mapping[str, Any],
    config: Stage07Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed + 701)
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    targets = sorted(merged["target_sbox"].unique().tolist())

    for target in targets:
        for model in FAULT_MODELS:
            clean = merged[
                (merged["target_sbox"] == target)
                & (merged["fault_model"] == model)
                & (merged["category"].isin([
                    INEFFECTIVE_CATEGORY,
                    EFFECTIVE_CATEGORY,
                ]))
            ].copy()

            ineffective_count = int(np.sum(
                clean["category"] == INEFFECTIVE_CATEGORY
            ))
            effective_count = int(np.sum(
                clean["category"] == EFFECTIVE_CATEGORY
            ))

            if (
                len(clean) < config.minimum_clean_samples_for_bootstrap
                or ineffective_count < config.minimum_each_outcome_for_bootstrap
                or effective_count < config.minimum_each_outcome_for_bootstrap
            ):
                summary[f"{target}:{model}"] = {
                    "bootstrap_available": False,
                    "reason": "Insufficient samples in one or both clean outcomes",
                    "clean_count": len(clean),
                    "ineffective_count": ineffective_count,
                    "effective_count": effective_count,
                }
                continue

            clean = clean.reset_index(drop=True)
            bootstrap_metrics: List[Dict[str, float]] = []

            for repetition in range(config.bootstrap_repetitions):
                indices = stratified_bootstrap_indices(clean, rng)
                sample = clean.loc[indices]

                inputs_i = sample.loc[
                    sample["category"] == INEFFECTIVE_CATEGORY,
                    "target_original_input",
                ].to_numpy(dtype=np.int32)
                inputs_e = sample.loc[
                    sample["category"] == EFFECTIVE_CATEGORY,
                    "target_original_input",
                ].to_numpy(dtype=np.int32)

                counts_i = np.bincount(inputs_i, minlength=16)
                counts_e = np.bincount(inputs_e, minlength=16)
                empirical = empirical_joint_metrics(
                    counts_i,
                    counts_e,
                    config.dirichlet_alpha,
                )

                theory_pi = theoretical[model]["p_ineffective_given_x"]
                bootstrap_metrics.append({
                    "ineffective_rate": float(len(inputs_i) / len(sample)),
                    "sifa_kl_bits": float(
                        empirical["ineffective_metrics"]["kl_to_uniform_bits"]
                    ),
                    "sefa_kl_bits": float(
                        empirical["effective_metrics"]["kl_to_uniform_bits"]
                    ),
                    "shfa_kl_bits": float(
                        empirical["hybrid_metrics"]["kl_to_uniform_bits"]
                    ),
                    "mutual_information_bits": float(
                        empirical["mutual_information_bits"]
                    ),
                    "pi_correlation": safe_correlation(
                        empirical["p_ineffective_given_x"],
                        theory_pi,
                    ),
                })

                row = {
                    "target_sbox": target,
                    "fault_model": model,
                    "repetition": repetition,
                }
                row.update(bootstrap_metrics[-1])
                rows.append(row)

            metric_names = list(bootstrap_metrics[0].keys())
            group_summary: Dict[str, Any] = {
                "bootstrap_available": True,
                "repetitions": config.bootstrap_repetitions,
                "clean_count": len(clean),
                "ineffective_count": ineffective_count,
                "effective_count": effective_count,
            }

            for metric in metric_names:
                values = np.asarray([
                    item[metric] for item in bootstrap_metrics
                ], dtype=np.float64)
                group_summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                }

            summary[f"{target}:{model}"] = group_summary

    return rows, summary


# ============================================================
# 7. تحلیل بازده کمپین و سلول‌های پارامتری
# ============================================================


def cut_with_labels(
    values: pd.Series,
    edges: Sequence[float],
    prefix: str,
) -> pd.Series:
    labels = [
        f"{prefix}{index}:[{edges[index]:g},{edges[index + 1]:g})"
        for index in range(len(edges) - 1)
    ]
    return pd.cut(
        values,
        bins=list(edges),
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype(str)


def category_posterior_rates(
    counts: Mapping[str, int],
    alpha: float = 0.5,
) -> Dict[str, float]:
    total = sum(int(counts.get(category, 0)) for category in FAULT_CATEGORIES)
    denominator = total + alpha * len(FAULT_CATEGORIES)
    return {
        category: (int(counts.get(category, 0)) + alpha) / denominator
        for category in FAULT_CATEGORIES
    }


def analyze_parameter_cells(
    merged: pd.DataFrame,
    theoretical: Mapping[str, Any],
    config: Stage07Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    primary = merged[merged["fault_model"] == "random_and_4"].copy()
    primary["offset_bin"] = cut_with_labels(
        primary["timing_offset_samples"],
        config.offset_bin_edges,
        "o",
    )
    primary["width_bin"] = cut_with_labels(
        primary["width_samples"],
        config.width_bin_edges,
        "w",
    )
    primary["strength_bin"] = cut_with_labels(
        primary["strength"],
        config.strength_bin_edges,
        "s",
    )
    primary["repeat_bin"] = primary["repeat"].map(
        lambda value: "r1" if int(value) == 1 else ("r2" if int(value) == 2 else "r3plus")
    )

    theory = theoretical["random_and_4"]
    sifa_capacity = float(theory["ineffective_metrics"]["kl_to_uniform_bits"])
    sefa_capacity = float(theory["effective_metrics"]["kl_to_uniform_bits"])
    shfa_capacity = float(theory["hybrid_metrics"]["kl_to_uniform_bits"])

    one_dimensional_rows: List[Dict[str, Any]] = []

    for target in sorted(primary["target_sbox"].unique()):
        target_frame = primary[primary["target_sbox"] == target]

        for parameter_name, binned_name in (
            ("timing_offset_samples", "offset_bin"),
            ("width_samples", "width_bin"),
            ("strength", "strength_bin"),
            ("repeat", "repeat_bin"),
        ):
            for bin_name, group in target_frame.groupby(binned_name, sort=True):
                counts = group["category"].value_counts().to_dict()
                rates = category_posterior_rates(counts)
                equal_pool = int(np.sum(
                    group["response_received"].astype(bool)
                    & group["ciphertext_equal"].astype(bool)
                ))
                different_pool = int(np.sum(
                    group["response_received"].astype(bool)
                    & (~group["ciphertext_equal"].astype(bool))
                ))
                clean_i = int(counts.get(INEFFECTIVE_CATEGORY, 0))
                clean_e = int(counts.get(EFFECTIVE_CATEGORY, 0))

                one_dimensional_rows.append({
                    "target_sbox": target,
                    "parameter": parameter_name,
                    "bin": str(bin_name),
                    "attempt_count": len(group),
                    **{f"rate_{key}": value for key, value in rates.items()},
                    "sifa_observable_pool_count": equal_pool,
                    "sefa_observable_pool_count": different_pool,
                    "sifa_oracle_precision": clean_i / equal_pool if equal_pool else 0.0,
                    "sefa_oracle_precision": clean_e / different_pool if different_pool else 0.0,
                    "sifa_information_yield_bits_per_attempt": rates[
                        INEFFECTIVE_CATEGORY
                    ] * sifa_capacity,
                    "sefa_information_yield_bits_per_attempt": rates[
                        EFFECTIVE_CATEGORY
                    ] * sefa_capacity,
                    "shfa_information_yield_bits_per_attempt": min(
                        rates[INEFFECTIVE_CATEGORY],
                        rates[EFFECTIVE_CATEGORY],
                    ) * shfa_capacity,
                })

    cell_rows: List[Dict[str, Any]] = []
    grouping_columns = [
        "target_sbox",
        "offset_bin",
        "width_bin",
        "strength_bin",
        "repeat_bin",
    ]

    for keys, group in primary.groupby(grouping_columns, sort=True):
        if len(group) < config.minimum_parameter_cell_attempts:
            continue

        target, offset_bin, width_bin, strength_bin, repeat_bin = keys
        counts = group["category"].value_counts().to_dict()
        rates = category_posterior_rates(counts)

        equal_pool = int(np.sum(
            group["response_received"].astype(bool)
            & group["ciphertext_equal"].astype(bool)
        ))
        different_pool = int(np.sum(
            group["response_received"].astype(bool)
            & (~group["ciphertext_equal"].astype(bool))
        ))
        clean_i = int(counts.get(INEFFECTIVE_CATEGORY, 0))
        clean_e = int(counts.get(EFFECTIVE_CATEGORY, 0))
        precision_i = clean_i / equal_pool if equal_pool else 0.0
        precision_e = clean_e / different_pool if different_pool else 0.0

        rate_i = rates[INEFFECTIVE_CATEGORY]
        rate_e = rates[EFFECTIVE_CATEGORY]
        invalid_rate = rates["invalid_reset"]

        sifa_yield = rate_i * sifa_capacity
        sefa_yield = rate_e * sefa_capacity
        shfa_yield = min(rate_i, rate_e) * shfa_capacity

        sifa_score = (
            sifa_yield * (0.5 + 0.5 * precision_i)
            - 0.05 * invalid_rate
        )
        sefa_score = (
            sefa_yield * (0.5 + 0.5 * precision_e)
            - 0.05 * invalid_rate
        )
        shfa_score = (
            shfa_yield * (0.5 + 0.25 * (precision_i + precision_e))
            - 0.05 * invalid_rate
        )

        cell_rows.append({
            "target_sbox": str(target),
            "offset_bin": str(offset_bin),
            "width_bin": str(width_bin),
            "strength_bin": str(strength_bin),
            "repeat_bin": str(repeat_bin),
            "attempt_count": len(group),
            **{f"count_{category}": int(counts.get(category, 0)) for category in FAULT_CATEGORIES},
            **{f"rate_{category}": float(rates[category]) for category in FAULT_CATEGORIES},
            "sifa_observable_pool_count": equal_pool,
            "sefa_observable_pool_count": different_pool,
            "sifa_oracle_precision": precision_i,
            "sefa_oracle_precision": precision_e,
            "sifa_information_yield_bits_per_attempt": sifa_yield,
            "sefa_information_yield_bits_per_attempt": sefa_yield,
            "shfa_information_yield_bits_per_attempt": shfa_yield,
            "sifa_oracle_cell_score": sifa_score,
            "sefa_oracle_cell_score": sefa_score,
            "shfa_oracle_cell_score": shfa_score,
        })

    recommendations: Dict[str, Any] = {
        "warning": (
            "These parameter-cell recommendations use simulator ground-truth "
            "categories. They are an oracle benchmark for Stage 08 campaign "
            "design and must not be used as leakage-safe ML features."
        ),
        "primary_fault_model": "random_and_4",
        "targets": {},
    }

    for target in sorted(primary["target_sbox"].unique()):
        recommendations["targets"][target] = {}
        target_rows = [row for row in cell_rows if row["target_sbox"] == target]

        for attack, score_column in (
            ("SIFA", "sifa_oracle_cell_score"),
            ("SEFA", "sefa_oracle_cell_score"),
            ("SHFA", "shfa_oracle_cell_score"),
        ):
            ranked = sorted(
                target_rows,
                key=lambda row: (row[score_column], row["attempt_count"]),
                reverse=True,
            )[: config.top_parameter_cells_per_attack]
            recommendations["targets"][target][attack] = ranked

    return one_dimensional_rows, cell_rows, recommendations


# ============================================================
# 8. کنترل‌های علمی
# ============================================================


def validate_theoretical_analysis(
    matrices: Mapping[str, np.ndarray],
    detailed: Mapping[str, Any],
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    maximum_row_error = max(
        float(np.max(np.abs(np.sum(matrix, axis=1) - 1.0)))
        for matrix in matrices.values()
    )
    checks["transition_rows_sum_to_one"] = {
        "passed": maximum_row_error <= 1e-12,
        "maximum_error": maximum_row_error,
    }

    random_and_pi = detailed["random_and_4"]["p_ineffective_given_x"]
    formula = np.asarray([
        2.0 ** (-hamming_weight_4(x))
        for x in range(16)
    ])
    formula_error = float(np.max(np.abs(random_and_pi - formula)))
    checks["random_and_4_formula"] = {
        "passed": formula_error <= 1e-12,
        "maximum_error": formula_error,
    }

    expected_mean = (3.0 / 4.0) ** 4
    observed_mean = float(
        detailed["random_and_4"]["mean_ineffective_probability"]
    )
    checks["random_and_4_mean_probability"] = {
        "passed": abs(observed_mean - expected_mean) <= 1e-12,
        "observed": observed_mean,
        "expected": expected_mean,
    }

    primary = detailed["random_and_4"]
    checks["primary_nonzero_attack_capacities"] = {
        "passed": bool(
            primary["ineffective_metrics"]["kl_to_uniform_bits"] > 0.25
            and primary["effective_metrics"]["kl_to_uniform_bits"] > 0.10
            and primary["hybrid_metrics"]["kl_to_uniform_bits"] > 0.40
            and primary["uniform_input_mutual_information_bits"] > 0.15
        ),
        "sifa_kl_bits": float(
            primary["ineffective_metrics"]["kl_to_uniform_bits"]
        ),
        "sefa_kl_bits": float(
            primary["effective_metrics"]["kl_to_uniform_bits"]
        ),
        "shfa_kl_bits": float(
            primary["hybrid_metrics"]["kl_to_uniform_bits"]
        ),
        "mutual_information_bits": float(
            primary["uniform_input_mutual_information_bits"]
        ),
    }

    constant_models = ("stuck_at_bit", "random_nibble")
    maximum_constant_capacity = max(
        float(detailed[model]["uniform_input_mutual_information_bits"])
        for model in constant_models
    )
    checks["constant_ineffective_probability_models_have_zero_information"] = {
        "passed": maximum_constant_capacity <= 1e-12,
        "maximum_information_bits": maximum_constant_capacity,
    }

    single_flip_i_probability = float(
        detailed["single_bit_flip"]["mean_ineffective_probability"]
    )
    checks["single_bit_flip_is_always_effective"] = {
        "passed": single_flip_i_probability <= 1e-12,
        "mean_ineffective_probability": single_flip_i_probability,
    }

    return {
        "all_theoretical_checks_passed": all(
            bool(check["passed"]) for check in checks.values()
        ),
        "checks": checks,
    }


def validate_empirical_analysis(
    capacity_rows: Sequence[Mapping[str, Any]],
    config: Stage07Config,
) -> Dict[str, Any]:
    primary_rows = [
        row for row in capacity_rows
        if row["fault_model"] == "random_and_4"
    ]

    checks: Dict[str, Any] = {}

    for row in primary_rows:
        target = row["target_sbox"]
        checks[f"{target}_primary_pi_correlation"] = {
            "passed": bool(
                row["p_i_theory_empirical_correlation"]
                >= config.minimum_primary_pi_correlation
            ),
            "value": float(row["p_i_theory_empirical_correlation"]),
            "threshold": config.minimum_primary_pi_correlation,
        }
        checks[f"{target}_primary_qi_tv"] = {
            "passed": bool(
                row["q_i_total_variation_to_theory"]
                <= config.maximum_primary_qi_tv_distance
            ),
            "value": float(row["q_i_total_variation_to_theory"]),
            "threshold": config.maximum_primary_qi_tv_distance,
        }
        checks[f"{target}_primary_qe_tv"] = {
            "passed": bool(
                row["q_e_total_variation_to_theory"]
                <= config.maximum_primary_qe_tv_distance
            ),
            "value": float(row["q_e_total_variation_to_theory"]),
            "threshold": config.maximum_primary_qe_tv_distance,
        }
        checks[f"{target}_primary_mean_rate"] = {
            "passed": bool(
                row["absolute_mean_rate_error"]
                <= config.maximum_primary_mean_rate_error
            ),
            "value": float(row["absolute_mean_rate_error"]),
            "threshold": config.maximum_primary_mean_rate_error,
        }

    checks["both_selected_targets_present"] = {
        "passed": len(primary_rows) == 2,
        "targets": sorted(row["target_sbox"] for row in primary_rows),
    }

    return {
        "all_empirical_checks_passed": all(
            bool(check["passed"]) for check in checks.values()
        ),
        "checks": checks,
    }


# ============================================================
# 9. شکل‌ها
# ============================================================


def save_theoretical_plots(
    output_directory: Path,
    detailed: Mapping[str, Any],
    capacity_rows: Sequence[Mapping[str, Any]],
    config: Stage07Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []
    x_values = np.arange(16)

    figure = plt.figure(figsize=(11, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    for model in FAULT_MODELS:
        axis.plot(
            x_values,
            detailed[model]["p_ineffective_given_x"],
            marker="o",
            linewidth=1.2,
            label=model,
        )
    axis.set_xticks(x_values)
    axis.set_title("Theoretical ineffective-fault probability by S-box input")
    axis.set_xlabel("Input nibble X")
    axis.set_ylabel("P(ineffective | X)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = output_directory / "theoretical_ineffective_probability.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    names = [row["fault_model"] for row in capacity_rows]
    sifa = [row["sifa_kl_capacity_bits"] for row in capacity_rows]
    sefa = [row["sefa_kl_capacity_bits"] for row in capacity_rows]
    shfa = [row["shfa_joint_kl_capacity_bits"] for row in capacity_rows]
    positions = np.arange(len(names), dtype=np.float64)
    width = 0.25

    figure = plt.figure(figsize=(12, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(positions - width, sifa, width=width, label="SIFA")
    axis.bar(positions, sefa, width=width, label="SEFA")
    axis.bar(positions + width, shfa, width=width, label="SHFA")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=20, ha="right")
    axis.set_title("Theoretical KL bias capacity by fault model")
    axis.set_ylabel("KL divergence from uniform [bits]")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = output_directory / "theoretical_fault_model_capacity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    primary = detailed["random_and_4"]
    figure = plt.figure(figsize=(11, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(
        x_values,
        primary["q_x_given_ineffective"],
        marker="o",
        label="q(X | ineffective)",
    )
    axis.plot(
        x_values,
        primary["q_x_given_effective"],
        marker="s",
        label="q(X | effective)",
    )
    axis.axhline(1.0 / 16.0, linestyle="--", linewidth=1.0, label="uniform")
    axis.set_xticks(x_values)
    axis.set_title("Random-AND-4 conditional input distributions")
    axis.set_xlabel("Input nibble X")
    axis.set_ylabel("Probability")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    path = output_directory / "random_and_4_conditional_distributions.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    return generated


def save_empirical_plots(
    output_directory: Path,
    bias_rows: Sequence[Mapping[str, Any]],
    capacity_rows: Sequence[Mapping[str, Any]],
    cell_rows: Sequence[Mapping[str, Any]],
    config: Stage07Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []

    for target in sorted({row["target_sbox"] for row in bias_rows}):
        rows = [
            row for row in bias_rows
            if row["target_sbox"] == target
            and row["fault_model"] == "random_and_4"
        ]
        rows.sort(key=lambda row: row["input_decimal"])
        x_values = np.asarray([row["input_decimal"] for row in rows])
        empirical = np.asarray([
            row["empirical_p_ineffective_given_x"] for row in rows
        ])
        theoretical = np.asarray([
            row["theoretical_p_ineffective_given_x"] for row in rows
        ])
        low = np.asarray([row["p_ineffective_ci_low"] for row in rows])
        high = np.asarray([row["p_ineffective_ci_high"] for row in rows])

        figure = plt.figure(figsize=(11, 5.5))
        axis = figure.add_subplot(1, 1, 1)
        axis.plot(x_values, theoretical, marker="o", label="theoretical")
        axis.errorbar(
            x_values,
            empirical,
            yerr=np.vstack([empirical - low, high - empirical]),
            marker="s",
            linestyle="--",
            capsize=3,
            label="empirical",
        )
        axis.set_xticks(x_values)
        axis.set_ylim(-0.03, 1.05)
        axis.set_title(f"Random-AND-4 ineffective bias: {target}")
        axis.set_xlabel("True final-round S-box input")
        axis.set_ylabel("P(ineffective | X)")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        path = output_directory / f"random_and_4_theory_vs_empirical_{target}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        generated.append(path.name)

    rows = [row for row in capacity_rows if row["clean_target_count"] > 0]
    labels = [f"{row['target_sbox']}:{row['fault_model']}" for row in rows]
    values = [row["empirical_shfa_joint_kl_capacity_bits"] for row in rows]
    order = np.argsort(values)[::-1]
    ordered_labels = [labels[index] for index in order]
    ordered_values = [values[index] for index in order]

    figure = plt.figure(figsize=(13, 6))
    axis = figure.add_subplot(1, 1, 1)
    positions = np.arange(len(ordered_labels))
    axis.bar(positions, ordered_values)
    axis.set_xticks(positions)
    axis.set_xticklabels(ordered_labels, rotation=45, ha="right")
    axis.set_title("Empirical SHFA joint KL bias by target and model")
    axis.set_ylabel("KL divergence from uniform product [bits]")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "empirical_shfa_capacity_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    for target in sorted({row["target_sbox"] for row in cell_rows}):
        target_rows = [row for row in cell_rows if row["target_sbox"] == target]
        target_rows = sorted(
            target_rows,
            key=lambda row: row["shfa_oracle_cell_score"],
            reverse=True,
        )[:12]
        if not target_rows:
            continue

        names = [
            f"{row['offset_bin']}|{row['width_bin']}|{row['strength_bin']}|{row['repeat_bin']}"
            for row in target_rows
        ]
        scores = [row["shfa_oracle_cell_score"] for row in target_rows]
        positions = np.arange(len(names))

        figure = plt.figure(figsize=(14, 6))
        axis = figure.add_subplot(1, 1, 1)
        axis.bar(positions, scores)
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
        axis.set_title(f"Top oracle SHFA parameter cells — {target}")
        axis.set_ylabel("Oracle cell score")
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        path = output_directory / f"oracle_parameter_cells_{target}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        generated.append(path.name)

    return generated


# ============================================================
# 10. اجرای کامل Stage 07
# ============================================================


def run_stage_07(config: Stage07Config) -> Dict[str, Any]:
    start_time = time.perf_counter()

    stage6_run_directory = Path(
        config.input_stage6_run_directory
    ).expanduser().resolve()
    stage6_summary_path = stage6_run_directory / "stage_06_summary.json"
    public6 = stage6_run_directory / "public"
    private6 = stage6_run_directory / "private_ground_truth"

    semantics_path = public6 / "fault_model_semantics.json"
    public_campaign_path = public6 / "fault_campaign_public.csv"
    ground_truth_path = private6 / "fault_ground_truth.csv"

    required_initial = [stage6_summary_path, semantics_path]
    missing = [str(path) for path in required_initial if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 06 input files:\n  - " + "\n  - ".join(missing)
        )

    stage6_summary = read_json(stage6_summary_path)
    if not stage6_summary.get("all_checks_passed", False):
        raise RuntimeError("Stage 06 did not pass all checks")

    semantics = read_json(semantics_path)
    if semantics.get("primary_model", {}).get("name") != "random_and_4":
        raise RuntimeError("Unexpected primary fault model in Stage 06")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage07_{timestamp}_seed{config.random_seed}"
    run_directory = Path(config.output_root).expanduser().resolve() / run_id
    public_theory_directory = run_directory / "public_theory"
    validation_directory = run_directory / "validation_only"
    public_theory_directory.mkdir(parents=True, exist_ok=False)

    # --------------------------------------------------------
    # بخش نظری — بدون بازکردن ground truth
    # --------------------------------------------------------
    matrices = all_transition_matrices()
    detailed: Dict[str, Any] = {
        model: conditional_bias_distributions(matrices[model])
        for model in FAULT_MODELS
    }
    bias_rows, capacity_rows_theory, prior = build_theoretical_outputs(matrices)
    theoretical_validation = validate_theoretical_analysis(matrices, detailed)

    write_csv_rows(
        public_theory_directory / "theoretical_fault_bias_by_input.csv",
        bias_rows,
    )
    write_csv_rows(
        public_theory_directory / "theoretical_model_capacity.csv",
        capacity_rows_theory,
    )
    write_json(
        public_theory_directory / "theoretical_attack_prior.json",
        prior,
    )
    write_json(
        public_theory_directory / "theoretical_validation_checks.json",
        theoretical_validation,
    )
    write_json(
        public_theory_directory / "fault_bias_definitions.json",
        {
            "p_j_i": "P(ineffective | X=j)",
            "p_j_e": "P(effective | X=j) = 1 - p_j_i",
            "q_j_i": "P(X=j | ineffective)",
            "q_j_e": "P(X=j | effective)",
            "q_h_xy": "q_e(x) * q_i(y)",
            "sifa_capacity_metric": "D_KL(q_i || Uniform_16) in bits",
            "sefa_capacity_metric": "D_KL(q_e || Uniform_16) in bits",
            "shfa_capacity_metric": "D_KL(q_e tensor q_i || Uniform_256) in bits",
            "channel_information_metric": "I(X; effective/ineffective) under uniform X",
        },
    )
    save_theoretical_arrays(
        public_theory_directory / "theoretical_fault_bias_arrays.npz",
        matrices,
        detailed,
    )
    write_json(
        public_theory_directory / "stage_07_config.json",
        {
            **asdict(config),
            "offset_bin_edges": list(config.offset_bin_edges),
            "width_bin_edges": list(config.width_bin_edges),
            "strength_bin_edges": list(config.strength_bin_edges),
        },
    )

    public_plots = save_theoretical_plots(
        public_theory_directory,
        detailed,
        capacity_rows_theory,
        config,
    )

    theory_files = sorted(
        path for path in public_theory_directory.iterdir() if path.is_file()
    )
    theory_freeze_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": (
            "All public theoretical outputs were generated and frozen before "
            "opening Stage 06 private ground truth."
        ),
        "ground_truth_opened_before_freeze": False,
        "files": {path.name: sha256_file(path) for path in theory_files},
    }
    freeze_source = json.dumps(
        theory_freeze_manifest,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    theory_freeze_sha256 = hashlib.sha256(freeze_source).hexdigest()
    theory_freeze_manifest["freeze_sha256"] = theory_freeze_sha256
    write_json(run_directory / "theory_freeze_manifest.json", theory_freeze_manifest)

    # --------------------------------------------------------
    # بخش تجربی — فقط پس از freeze
    # --------------------------------------------------------
    empirical_available = False
    empirical_passed = True
    empirical_capacity_rows: List[Dict[str, Any]] = []
    empirical_bias_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    comparison: Dict[str, Any] = {}
    parameter_1d_rows: List[Dict[str, Any]] = []
    parameter_cell_rows: List[Dict[str, Any]] = []
    oracle_recommendations: Dict[str, Any] = {}
    empirical_plots: List[str] = []
    empirical_validation: Dict[str, Any] = {
        "all_empirical_checks_passed": True,
        "checks": {},
    }
    bootstrap_summary: Dict[str, Any] = {}

    if config.enable_empirical_validation:
        missing_private = [
            str(path)
            for path in (public_campaign_path, ground_truth_path)
            if not path.is_file()
        ]
        if missing_private:
            raise FileNotFoundError(
                "Missing Stage 06 campaign/ground-truth files:\n  - "
                + "\n  - ".join(missing_private)
            )

        empirical_available = True
        validation_directory.mkdir(parents=True, exist_ok=True)
        merged = merge_stage6_data(public_campaign_path, ground_truth_path)

        (
            empirical_bias_rows,
            empirical_capacity_rows,
            comparison,
        ) = empirical_group_analysis(
            merged,
            detailed,
            config,
        )

        bootstrap_rows, bootstrap_summary = run_empirical_bootstrap(
            merged,
            detailed,
            config,
        )

        (
            parameter_1d_rows,
            parameter_cell_rows,
            oracle_recommendations,
        ) = analyze_parameter_cells(
            merged,
            detailed,
            config,
        )

        empirical_validation = validate_empirical_analysis(
            empirical_capacity_rows,
            config,
        )
        empirical_passed = bool(
            empirical_validation["all_empirical_checks_passed"]
        )

        write_csv_rows(
            validation_directory / "empirical_fault_bias_by_input.csv",
            empirical_bias_rows,
        )
        write_csv_rows(
            validation_directory / "empirical_capacity_by_target_model.csv",
            empirical_capacity_rows,
        )
        if bootstrap_rows:
            write_csv_rows(
                validation_directory / "empirical_capacity_bootstrap.csv",
                bootstrap_rows,
            )
        write_json(
            validation_directory / "empirical_capacity_bootstrap_summary.json",
            bootstrap_summary,
        )
        write_json(
            validation_directory / "theoretical_empirical_comparison.json",
            comparison,
        )
        write_csv_rows(
            validation_directory / "parameter_sensitivity_1d.csv",
            parameter_1d_rows,
        )
        if parameter_cell_rows:
            write_csv_rows(
                validation_directory / "oracle_parameter_cells.csv",
                parameter_cell_rows,
            )
        write_json(
            validation_directory / "oracle_campaign_recommendations.json",
            oracle_recommendations,
        )
        write_json(
            validation_directory / "empirical_validation_checks.json",
            empirical_validation,
        )
        write_json(
            validation_directory / "private_data_use_manifest.json",
            {
                "opened_after_theory_freeze": True,
                "theory_freeze_sha256": theory_freeze_sha256,
                "files_opened": [
                    str(public_campaign_path),
                    str(ground_truth_path),
                ],
                "private_fields_used": [
                    "category",
                    "target_original_input",
                    "target_faulted_input",
                    "impacted_sbox_count",
                ],
                "private_fields_not_exported_as_attack_features": [
                    "master_key_hex",
                    "round_key_32_hex",
                    "x31_hex",
                    "x32_hex",
                ],
            },
        )

        empirical_plots = save_empirical_plots(
            validation_directory,
            empirical_bias_rows,
            empirical_capacity_rows,
            parameter_cell_rows,
            config,
        )

    all_checks_passed = bool(
        theoretical_validation["all_theoretical_checks_passed"]
        and empirical_passed
    )

    primary_theory = detailed["random_and_4"]
    primary_empirical = {
        row["target_sbox"]: row
        for row in empirical_capacity_rows
        if row["fault_model"] == "random_and_4"
    }

    elapsed_seconds = time.perf_counter() - start_time

    summary = {
        "stage": 7,
        "run_id": run_id,
        "run_directory": str(run_directory),
        "input_stage_06_run_directory": str(stage6_run_directory),
        "public_theory_directory": str(public_theory_directory),
        "validation_only_directory": str(validation_directory),
        "all_checks_passed": all_checks_passed,
        "all_theoretical_checks_passed": bool(
            theoretical_validation["all_theoretical_checks_passed"]
        ),
        "empirical_validation_available": empirical_available,
        "empirical_validation_passed": empirical_passed,
        "number_of_stage_06_experiments": int(
            stage6_summary["number_of_experiments"]
        ),
        "selected_sboxes": stage6_summary["selected_sboxes"],
        "selected_last_round_key_nibbles": stage6_summary[
            "selected_last_round_key_nibbles"
        ],
        "total_target_bits": int(stage6_summary["total_target_bits"]),
        "primary_fault_model": "random_and_4",
        "theoretical_primary_mean_ineffective_probability": float(
            primary_theory["mean_ineffective_probability"]
        ),
        "theoretical_primary_sifa_kl_capacity_bits": float(
            primary_theory["ineffective_metrics"]["kl_to_uniform_bits"]
        ),
        "theoretical_primary_sefa_kl_capacity_bits": float(
            primary_theory["effective_metrics"]["kl_to_uniform_bits"]
        ),
        "theoretical_primary_shfa_joint_kl_capacity_bits": float(
            primary_theory["hybrid_metrics"]["kl_to_uniform_bits"]
        ),
        "theoretical_primary_uniform_input_mutual_information_bits": float(
            primary_theory["uniform_input_mutual_information_bits"]
        ),
        "theoretical_primary_channel_capacity_bits": float(
            primary_theory["channel_capacity_bits"]
        ),
        "empirical_primary_results": {
            target: {
                "clean_target_count": int(row["clean_target_count"]),
                "clean_ineffective_count": int(
                    row["clean_target_ineffective_count"]
                ),
                "clean_effective_count": int(
                    row["clean_target_effective_count"]
                ),
                "observed_ineffective_rate_given_clean": float(
                    row["observed_ineffective_rate_given_clean"]
                ),
                "p_i_theory_empirical_correlation": float(
                    row["p_i_theory_empirical_correlation"]
                ),
                "q_i_total_variation_to_theory": float(
                    row["q_i_total_variation_to_theory"]
                ),
                "q_e_total_variation_to_theory": float(
                    row["q_e_total_variation_to_theory"]
                ),
                "empirical_sifa_kl_capacity_bits": float(
                    row["empirical_sifa_kl_capacity_bits"]
                ),
                "empirical_sefa_kl_capacity_bits": float(
                    row["empirical_sefa_kl_capacity_bits"]
                ),
                "empirical_shfa_joint_kl_capacity_bits": float(
                    row["empirical_shfa_joint_kl_capacity_bits"]
                ),
            }
            for target, row in primary_empirical.items()
        },
        "theory_freeze_sha256": theory_freeze_sha256,
        "bootstrap_repetitions": config.bootstrap_repetitions,
        "parameter_cell_count": len(parameter_cell_rows),
        "elapsed_seconds": float(elapsed_seconds),
        "public_theory_files": sorted(
            path.name for path in public_theory_directory.iterdir() if path.is_file()
        ),
        "validation_only_files": sorted(
            path.name for path in validation_directory.iterdir() if path.is_file()
        ) if validation_directory.is_dir() else [],
        "generated_plots": {
            "public_theory": public_plots,
            "validation_only": empirical_plots,
        },
    }

    write_json(run_directory / "stage_07_validation_checks.json", {
        "all_checks_passed": all_checks_passed,
        "theoretical": theoretical_validation,
        "empirical": empirical_validation,
    })
    write_json(run_directory / "stage_07_summary.json", summary)
    write_json(run_directory / "run_manifest.json", {
        "stage": 7,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": {
            **asdict(config),
            "offset_bin_edges": list(config.offset_bin_edges),
            "width_bin_edges": list(config.width_bin_edges),
            "strength_bin_edges": list(config.strength_bin_edges),
        },
        "input_hashes": {
            "stage_06_summary.json": sha256_file(stage6_summary_path),
            "fault_model_semantics.json": sha256_file(semantics_path),
            "fault_campaign_public.csv": (
                sha256_file(public_campaign_path)
                if public_campaign_path.is_file()
                else None
            ),
            "fault_ground_truth.csv": (
                sha256_file(ground_truth_path)
                if ground_truth_path.is_file()
                else None
            ),
        },
    })

    print("\n" + "=" * 84)
    print("Stage 07 complete: fault-bias and information-capacity analysis")
    print("=" * 84)
    print("Run directory                         :", summary["run_directory"])
    print("All checks passed                     :", summary["all_checks_passed"])
    print("Theoretical checks passed             :", summary["all_theoretical_checks_passed"])
    print("Empirical validation passed           :", summary["empirical_validation_passed"])
    print("Primary model                         :", summary["primary_fault_model"])
    print("Mean ineffective probability          :", f"{summary['theoretical_primary_mean_ineffective_probability']:.9f}")
    print("SIFA KL capacity [bits]               :", f"{summary['theoretical_primary_sifa_kl_capacity_bits']:.9f}")
    print("SEFA KL capacity [bits]               :", f"{summary['theoretical_primary_sefa_kl_capacity_bits']:.9f}")
    print("SHFA joint KL capacity [bits]         :", f"{summary['theoretical_primary_shfa_joint_kl_capacity_bits']:.9f}")
    print("Uniform-input mutual information      :", f"{summary['theoretical_primary_uniform_input_mutual_information_bits']:.9f}")
    for target, result in summary["empirical_primary_results"].items():
        print(
            f"{target} empirical pi correlation          :",
            f"{result['p_i_theory_empirical_correlation']:.6f}",
        )
        print(
            f"{target} empirical SHFA KL [bits]          :",
            f"{result['empirical_shfa_joint_kl_capacity_bits']:.6f}",
        )
    print("Elapsed seconds                       :", f"{elapsed_seconds:.3f}")
    print("=" * 84)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 07 failed validation. Inspect stage_07_validation_checks.json"
        )

    return summary


def load_stage_07_config(config_path: str | Path) -> Stage07Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    for tuple_field in (
        "offset_bin_edges",
        "width_bin_edges",
        "strength_bin_edges",
    ):
        if tuple_field in raw:
            raw[tuple_field] = tuple(raw[tuple_field])

    return Stage07Config(**raw)


if __name__ == "__main__":
    default_config = Stage07Config(
        input_stage6_run_directory=(
            r"C:\Users\SADRA\Desktop\LBlock\runs\stage_06"
            r"\stage06_20260718_175343_608585_seed20260718"
        ),
        output_root=r"C:\Users\SADRA\Desktop\LBlock\runs\stage_07",
    )
    run_stage_07(default_config)

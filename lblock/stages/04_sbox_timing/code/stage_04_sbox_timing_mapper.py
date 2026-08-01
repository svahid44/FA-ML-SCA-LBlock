# ============================================================
# Stage 04 — Public-only S-box Timing Mapper for LBlock
#
# هدف:
#   - تفکیک زمان S0 تا S7 در ROI دور آخر
#   - استفاده فقط از تریس‌ها، ciphertext و شناسه‌های عمومی
#   - عدم استفاده از کلید اصلی، round key یا timing مخفی
#   - تولید نقشه زمان مناسب برای موتور fault مرحله‌های بعد
#
# ایده اصلی:
#   نیمه چپ ciphertext همان X32 است. برای هر S-box شماره i:
#
#       input_i = nibble_i(X32) XOR nibble_i(K32)
#
#   مقدار nibble کلید ناشناخته است، اما در هر key_id ثابت است.
#   بنابراین وابستگی آماری تریس به کلاس nibble_i(X32)، حتی بدون
#   دانستن کلید، در زمان اجرای S_i باقی می‌ماند.
#
#   برای هر nibble:
#     1) پروفایل eta-squared به‌صورت جداگانه در هر key_id محاسبه می‌شود.
#     2) baseline با permutation داخل key/session تخمین زده می‌شود.
#     3) هشت پروفایل وابستگی داده ساخته می‌شوند.
#     4) یک comb منظم هشت‌رویدادی روی ROI جست‌وجو می‌شود.
#
# ارزیابی private فقط بعد از freeze شدن خروجی public انجام می‌شود.
# ============================================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
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


@dataclass(frozen=True)
class Stage04Config:
    """تنظیمات استخراج نقشه زمانی هشت S-box دور آخر."""

    input_stage3_run_directory: str
    output_root: str = "runs/stage_04"
    random_seed: int = 20260718

    number_of_sboxes: int = 8
    number_of_classes: int = 16
    minimum_class_count: int = 2

    # پروفایل آماری
    permutation_count: int = 48
    smoothing_radius_samples: int = 2
    specificity_weight: float = 0.35

    # جست‌وجوی comb هشت‌رویدادی
    minimum_spacing_fraction_of_round: float = 0.04
    maximum_spacing_fraction_of_round: float = 0.12
    earliest_first_center_fraction_of_round: float = 0.04
    latest_first_center_fraction_of_round: float = 0.42
    latest_last_center_fraction_of_round: float = 0.88
    prominence_weight: float = 0.15
    profile_balance_penalty: float = 0.05

    # پایداری آماری
    bootstrap_iterations: int = 64
    bootstrap_confidence_percent: float = 95.0
    minimum_subset_traces: int = 64

    # پنجره‌های زمانی
    exploration_half_width_fraction_of_spacing: float = 0.75

    # خروجی و ارزیابی
    save_plots: bool = True
    enable_private_evaluation: bool = True


# ============================================================
# 1. توابع عمومی فایل و ثبت نتایج
# ============================================================

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError("CSV rows cannot be empty")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ============================================================
# 2. پردازش metadata عمومی و ساخت برچسب‌های قابل مشاهده
# ============================================================

def reorder_metadata_by_trace_id(
    metadata_rows: Sequence[Mapping[str, str]],
    trace_ids: np.ndarray,
) -> List[Dict[str, str]]:
    """
    ترتیب metadata را با آرایه trace_ids داخل فایل NPZ یکسان می‌کند.
    """

    by_id = {
        int(row["trace_id"]): dict(row)
        for row in metadata_rows
    }

    missing = [
        int(trace_id)
        for trace_id in trace_ids
        if int(trace_id) not in by_id
    ]

    if missing:
        raise ValueError(
            "Metadata is missing trace IDs: "
            + ", ".join(map(str, missing[:10]))
        )

    return [
        by_id[int(trace_id)]
        for trace_id in trace_ids
    ]


def extract_public_observable_labels(
    ordered_metadata: Sequence[Mapping[str, str]],
    number_of_sboxes: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    استخراج X32 از نیمه چپ ciphertext.

    تنها داده‌های مورد استفاده:
      - ciphertext_hex
      - key_id
      - session_id
      - trace_id

    هیچ مقدار کلید یا state مخفی خوانده نمی‌شود.
    """

    trace_ids = np.asarray([
        int(row["trace_id"])
        for row in ordered_metadata
    ], dtype=np.int32)

    key_ids = np.asarray([
        int(row["key_id"])
        for row in ordered_metadata
    ], dtype=np.int32)

    session_ids = np.asarray([
        int(row["session_id"])
        for row in ordered_metadata
    ], dtype=np.int32)

    x32_values = np.asarray([
        (
            int(str(row["ciphertext_hex"]), 16)
            >> 32
        ) & 0xFFFFFFFF
        for row in ordered_metadata
    ], dtype=np.uint32)

    labels = np.empty(
        (len(ordered_metadata), number_of_sboxes),
        dtype=np.uint8,
    )

    for sbox_index in range(number_of_sboxes):
        labels[:, sbox_index] = (
            x32_values >> (4 * sbox_index)
        ) & 0xF

    return trace_ids, key_ids, session_ids, labels


def build_class_coverage_report(
    labels: np.ndarray,
    key_ids: np.ndarray,
    number_of_classes: int,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "number_of_keys": int(np.unique(key_ids).size),
        "targets": {},
    }

    for sbox_index in range(labels.shape[1]):
        target_rows = []
        minimum_nonzero_count = None
        minimum_observed_classes = number_of_classes

        for key_id in np.unique(key_ids):
            indices = np.where(key_ids == key_id)[0]
            counts = np.bincount(
                labels[indices, sbox_index],
                minlength=number_of_classes,
            )

            nonzero = counts[counts > 0]

            if nonzero.size:
                local_minimum = int(np.min(nonzero))

                if minimum_nonzero_count is None:
                    minimum_nonzero_count = local_minimum
                else:
                    minimum_nonzero_count = min(
                        minimum_nonzero_count,
                        local_minimum,
                    )

            observed_classes = int(np.sum(counts > 0))
            minimum_observed_classes = min(
                minimum_observed_classes,
                observed_classes,
            )

            target_rows.append({
                "key_id": int(key_id),
                "trace_count": int(indices.size),
                "observed_class_count": observed_classes,
                "class_counts": counts.astype(int).tolist(),
            })

        report["targets"][f"S{sbox_index}"] = {
            "minimum_observed_classes_per_key": int(
                minimum_observed_classes
            ),
            "minimum_nonzero_class_count": int(
                minimum_nonzero_count
                if minimum_nonzero_count is not None
                else 0
            ),
            "per_key": target_rows,
        }

    return report


# ============================================================
# 3. حذف اثر session بدون حذف وابستگی داده
# ============================================================

def residualize_and_balance_sessions(
    traces: np.ndarray,
    session_ids: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    برای هر session:
      - میانگین زمانی آن session از traceها کم می‌شود.
      - یک scale مقاوم سراسری برای برابرکردن gain session اعمال می‌شود.

    این عملیات زمان یا کلید را وارد مدل نمی‌کند.
    """

    traces = np.asarray(traces, dtype=np.float64)

    if traces.ndim != 2:
        raise ValueError("traces must be a 2-D matrix")

    output = np.empty_like(traces)
    rows = []

    for session_id in np.unique(session_ids):
        indices = np.where(session_ids == session_id)[0]
        block = traces[indices]

        mean_profile = np.mean(block, axis=0)
        residual = block - mean_profile[None, :]

        residual_median = float(np.median(residual))
        residual_mad = float(
            1.4826
            * np.median(
                np.abs(residual - residual_median)
            )
        )

        scale = (
            residual_mad
            if residual_mad > 1e-10
            else float(np.std(residual))
        )

        if scale <= 1e-10:
            scale = 1.0

        output[indices] = residual / scale

        rows.append({
            "session_id": int(session_id),
            "trace_count": int(indices.size),
            "removed_mean_profile_average": float(
                np.mean(mean_profile)
            ),
            "removed_mean_profile_std": float(
                np.std(mean_profile)
            ),
            "robust_scale": float(scale),
        })

    return output, {"sessions": rows}


# ============================================================
# 4. ساخت پروفایل eta-squared به‌صورت key-stratified
# ============================================================

def eta_squared_profile_stratified(
    traces: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    minimum_class_count: int,
) -> np.ndarray:
    """
    eta² را در هر stratum به‌صورت مستقل محاسبه و سپس با تعداد
    trace وزن‌دهی می‌کند.

    در اجرای اصلی، stratum همان key_id است. به این ترتیب اثر
    nibble کلید ناشناخته درون هر گروه ثابت می‌ماند.
    """

    traces = np.asarray(traces, dtype=np.float64)
    labels = np.asarray(labels)
    strata = np.asarray(strata)

    profile = np.zeros(traces.shape[1], dtype=np.float64)
    total_weight = 0.0

    for stratum in np.unique(strata):
        indices = np.where(strata == stratum)[0]

        if indices.size < 8:
            continue

        block = traces[indices]
        block_labels = labels[indices]

        grand_mean = np.mean(block, axis=0)
        total_sum_squares = np.sum(
            (block - grand_mean[None, :]) ** 2,
            axis=0,
        )

        between_sum_squares = np.zeros(
            traces.shape[1],
            dtype=np.float64,
        )

        valid_class_count = 0

        for class_value in np.unique(block_labels):
            class_mask = block_labels == class_value
            class_count = int(np.sum(class_mask))

            if class_count < minimum_class_count:
                continue

            valid_class_count += 1
            class_mean = np.mean(
                block[class_mask],
                axis=0,
            )

            between_sum_squares += (
                class_count
                * (class_mean - grand_mean) ** 2
            )

        if valid_class_count < 2:
            continue

        eta_squared = np.divide(
            between_sum_squares,
            total_sum_squares,
            out=np.zeros_like(between_sum_squares),
            where=total_sum_squares > 1e-14,
        )

        weight = float(indices.size)
        profile += weight * eta_squared
        total_weight += weight

    if total_weight <= 0:
        raise RuntimeError(
            "No valid stratum was available for eta-squared"
        )

    return profile / total_weight


def shuffle_labels_within_key_session(
    labels: np.ndarray,
    key_ids: np.ndarray,
    session_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    permutation فقط داخل هر زوج key/session انجام می‌شود تا
    ساختار session و تعداد نمونه هر کلید حفظ شود.
    """

    shuffled = np.asarray(labels).copy()

    pair_ids = (
        key_ids.astype(np.int64) * 1_000_000
        + session_ids.astype(np.int64)
    )

    for pair_id in np.unique(pair_ids):
        indices = np.where(pair_ids == pair_id)[0]
        shuffled[indices] = rng.permutation(
            shuffled[indices]
        )

    return shuffled


def triangular_smooth(
    values: np.ndarray,
    radius: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if radius <= 0:
        return values.copy()

    ascending = np.arange(
        1,
        radius + 2,
        dtype=np.float64,
    )
    kernel = np.concatenate([
        ascending,
        ascending[-2::-1],
    ])
    kernel /= np.sum(kernel)

    padded = np.pad(
        values,
        (radius, radius),
        mode="edge",
    )

    result = np.convolve(
        padded,
        kernel,
        mode="valid",
    )

    if result.shape != values.shape:
        raise RuntimeError(
            "Triangular smoothing changed profile length"
        )

    return result


def robust_standardize_profile(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    median = float(np.median(values))
    mad = float(
        1.4826 * np.median(np.abs(values - median))
    )

    scale = mad if mad > 1e-10 else float(np.std(values))

    if scale <= 1e-10:
        scale = 1.0

    return (values - median) / scale


def build_dependency_profiles(
    traces: np.ndarray,
    labels: np.ndarray,
    key_ids: np.ndarray,
    session_ids: np.ndarray,
    config: Stage04Config,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    تولید پروفایل‌های نهایی وابستگی داده برای S0 تا S7.
    """

    balanced, session_diagnostics = (
        residualize_and_balance_sessions(
            traces,
            session_ids,
        )
    )

    number_of_sboxes = labels.shape[1]
    number_of_samples = traces.shape[1]

    observed_eta = np.empty(
        (number_of_sboxes, number_of_samples),
        dtype=np.float64,
    )

    for sbox_index in range(number_of_sboxes):
        observed_eta[sbox_index] = (
            eta_squared_profile_stratified(
                balanced,
                labels[:, sbox_index],
                key_ids,
                config.minimum_class_count,
            )
        )

    rng = np.random.default_rng(config.random_seed)

    permutation_mean = np.zeros_like(observed_eta)
    permutation_m2 = np.zeros_like(observed_eta)

    for permutation_index in range(
        config.permutation_count
    ):
        for sbox_index in range(number_of_sboxes):
            shuffled = shuffle_labels_within_key_session(
                labels[:, sbox_index],
                key_ids,
                session_ids,
                rng,
            )

            current = eta_squared_profile_stratified(
                balanced,
                shuffled,
                key_ids,
                config.minimum_class_count,
            )

            count = permutation_index + 1
            delta = current - permutation_mean[sbox_index]
            permutation_mean[sbox_index] += delta / count
            permutation_m2[sbox_index] += (
                delta
                * (
                    current
                    - permutation_mean[sbox_index]
                )
            )

    if config.permutation_count > 1:
        permutation_std = np.sqrt(
            np.maximum(
                permutation_m2
                / (config.permutation_count - 1),
                1e-16,
            )
        )
    else:
        permutation_std = np.ones_like(
            permutation_mean
        )

    permutation_z = (
        observed_eta - permutation_mean
    ) / (permutation_std + 1e-12)

    smoothed_z = np.empty_like(permutation_z)
    standardized = np.empty_like(permutation_z)

    for sbox_index in range(number_of_sboxes):
        smoothed_z[sbox_index] = triangular_smooth(
            permutation_z[sbox_index],
            config.smoothing_radius_samples,
        )
        standardized[sbox_index] = (
            robust_standardize_profile(
                smoothed_z[sbox_index]
            )
        )

    specificity = np.empty_like(standardized)

    for sbox_index in range(number_of_sboxes):
        other_profiles = np.delete(
            standardized,
            sbox_index,
            axis=0,
        )

        specificity[sbox_index] = (
            standardized[sbox_index]
            - config.specificity_weight
            * np.median(
                other_profiles,
                axis=0,
            )
        )

    arrays = {
        "balanced_traces": balanced,
        "observed_eta_squared": observed_eta,
        "permutation_mean_eta_squared": (
            permutation_mean
        ),
        "permutation_std_eta_squared": (
            permutation_std
        ),
        "permutation_z": permutation_z,
        "smoothed_permutation_z": smoothed_z,
        "standardized_dependency": standardized,
        "specificity_profiles": specificity,
    }

    diagnostics = {
        "session_balancing": session_diagnostics,
        "permutation_count": (
            config.permutation_count
        ),
        "profile_shape": list(
            specificity.shape
        ),
        "maximum_permutation_z_per_target": {
            f"S{sbox_index}": float(
                np.max(permutation_z[sbox_index])
            )
            for sbox_index in range(number_of_sboxes)
        },
    }

    return arrays, diagnostics


# ============================================================
# 5. جست‌وجوی comb منظم هشت‌رویدادی
# ============================================================

def sample_index_for_absolute_sample(
    absolute_sample_indices: np.ndarray,
    target_sample: int,
) -> int:
    position = int(np.searchsorted(
        absolute_sample_indices,
        target_sample,
    ))

    candidates = []

    if 0 <= position < absolute_sample_indices.size:
        candidates.append(position)

    if 0 <= position - 1 < absolute_sample_indices.size:
        candidates.append(position - 1)

    if not candidates:
        raise ValueError(
            "Target sample is outside ROI"
        )

    return min(
        candidates,
        key=lambda index: abs(
            int(absolute_sample_indices[index])
            - int(target_sample)
        ),
    )


def local_prominence(
    profile: np.ndarray,
    center_index: int,
    spacing_samples: int,
) -> float:
    """
    اختلاف مقدار center با background نزدیک، بدون استفاده از
    نمونه‌های خیلی نزدیک به رویداد.
    """

    inner = max(
        1,
        int(round(0.22 * spacing_samples)),
    )
    outer = max(
        inner + 1,
        int(round(0.65 * spacing_samples)),
    )

    background_indices: List[int] = []

    for offset in range(inner + 1, outer + 1):
        left = center_index - offset
        right = center_index + offset

        if 0 <= left < profile.size:
            background_indices.append(left)

        if 0 <= right < profile.size:
            background_indices.append(right)

    if not background_indices:
        return 0.0

    return float(
        profile[center_index]
        - np.mean(profile[background_indices])
    )


def search_regular_sbox_comb(
    specificity_profiles: np.ndarray,
    absolute_sample_indices: np.ndarray,
    round_start_sample: int,
    round_period_samples: int,
    config: Stage04Config,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    یک comb با هشت center تقریباً هم‌فاصله جست‌وجو می‌شود.

    ترتیب زمانی S0 تا S7 از ترتیب اجرای حلقه S-box در پیاده‌سازی
    شناخته‌شده LBlock گرفته می‌شود، نه از ground truth شبیه‌ساز.
    """

    minimum_spacing = max(
        3,
        int(round(
            config.minimum_spacing_fraction_of_round
            * round_period_samples
        )),
    )

    maximum_spacing = max(
        minimum_spacing + 1,
        int(round(
            config.maximum_spacing_fraction_of_round
            * round_period_samples
        )),
    )

    roi_first = int(absolute_sample_indices[0])
    roi_last = int(absolute_sample_indices[-1])

    candidates: List[Dict[str, Any]] = []

    for spacing in range(
        minimum_spacing,
        maximum_spacing + 1,
    ):
        earliest_first = max(
            roi_first + 2,
            int(round(
                round_start_sample
                + config.earliest_first_center_fraction_of_round
                * round_period_samples
            )),
        )

        latest_by_first_fraction = int(round(
            round_start_sample
            + config.latest_first_center_fraction_of_round
            * round_period_samples
        ))

        latest_by_last_fraction = int(math.floor(
            round_start_sample
            + config.latest_last_center_fraction_of_round
            * round_period_samples
            - (config.number_of_sboxes - 1)
            * spacing
        ))

        latest_by_roi = (
            roi_last - 2
            - (config.number_of_sboxes - 1)
            * spacing
        )

        latest_first = min(
            latest_by_first_fraction,
            latest_by_last_fraction,
            latest_by_roi,
        )

        if latest_first < earliest_first:
            continue

        for first_center in range(
            earliest_first,
            latest_first + 1,
        ):
            centers = np.asarray([
                first_center + sbox_index * spacing
                for sbox_index in range(
                    config.number_of_sboxes
                )
            ], dtype=np.int32)

            center_scores = []
            prominences = []

            for sbox_index, center in enumerate(centers):
                center_index = (
                    sample_index_for_absolute_sample(
                        absolute_sample_indices,
                        int(center),
                    )
                )

                center_scores.append(float(
                    specificity_profiles[
                        sbox_index,
                        center_index,
                    ]
                ))

                prominences.append(
                    local_prominence(
                        specificity_profiles[
                            sbox_index
                        ],
                        center_index,
                        spacing,
                    )
                )

            center_scores_array = np.asarray(
                center_scores,
                dtype=np.float64,
            )
            prominence_array = np.asarray(
                prominences,
                dtype=np.float64,
            )

            score = (
                float(np.mean(center_scores_array))
                + config.prominence_weight
                * float(np.mean(prominence_array))
                - config.profile_balance_penalty
                * float(np.std(center_scores_array))
            )

            candidates.append({
                "score": float(score),
                "first_center_sample": int(
                    first_center
                ),
                "spacing_samples": int(spacing),
                "centers": centers.astype(int).tolist(),
                "mean_center_specificity": float(
                    np.mean(center_scores_array)
                ),
                "minimum_center_specificity": float(
                    np.min(center_scores_array)
                ),
                "std_center_specificity": float(
                    np.std(center_scores_array)
                ),
                "mean_prominence": float(
                    np.mean(prominence_array)
                ),
                "center_specificities": (
                    center_scores_array.tolist()
                ),
                "prominences": (
                    prominence_array.tolist()
                ),
            })

    if not candidates:
        raise RuntimeError(
            "No valid S-box comb candidate was generated"
        )

    candidates.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    best = candidates[0]

    structurally_distinct = None

    for candidate in candidates[1:]:
        if (
            candidate["spacing_samples"]
            != best["spacing_samples"]
            or abs(
                candidate["first_center_sample"]
                - best["first_center_sample"]
            ) >= 2
        ):
            structurally_distinct = candidate
            break

    if structurally_distinct is None:
        structurally_distinct = candidates[
            min(1, len(candidates) - 1)
        ]

    best = dict(best)
    best["second_distinct_score"] = float(
        structurally_distinct["score"]
    )
    best["absolute_score_gap"] = float(
        best["score"]
        - structurally_distinct["score"]
    )
    best["relative_score_gap"] = float(
        (
            best["score"]
            - structurally_distinct["score"]
        )
        / max(abs(best["score"]), 1e-12)
    )
    best["candidate_count"] = len(candidates)
    best["spacing_search_range"] = [
        int(minimum_spacing),
        int(maximum_spacing),
    ]

    return best, candidates


# ============================================================
# 6. Bootstrap و نقشه‌های زیرمجموعه‌ای
# ============================================================

def build_specificity_without_new_permutations(
    traces: np.ndarray,
    labels: np.ndarray,
    key_ids: np.ndarray,
    session_ids: np.ndarray,
    permutation_mean: np.ndarray,
    permutation_std: np.ndarray,
    config: Stage04Config,
) -> np.ndarray:
    balanced, _ = residualize_and_balance_sessions(
        traces,
        session_ids,
    )

    observed = np.empty_like(permutation_mean)

    for sbox_index in range(labels.shape[1]):
        observed[sbox_index] = (
            eta_squared_profile_stratified(
                balanced,
                labels[:, sbox_index],
                key_ids,
                config.minimum_class_count,
            )
        )

    normalized = (
        observed - permutation_mean
    ) / (permutation_std + 1e-12)

    standardized = np.empty_like(normalized)

    for sbox_index in range(labels.shape[1]):
        smoothed = triangular_smooth(
            normalized[sbox_index],
            config.smoothing_radius_samples,
        )
        standardized[sbox_index] = (
            robust_standardize_profile(smoothed)
        )

    specificity = np.empty_like(standardized)

    for sbox_index in range(labels.shape[1]):
        specificity[sbox_index] = (
            standardized[sbox_index]
            - config.specificity_weight
            * np.median(
                np.delete(
                    standardized,
                    sbox_index,
                    axis=0,
                ),
                axis=0,
            )
        )

    return specificity


def run_stratified_bootstrap(
    traces: np.ndarray,
    labels: np.ndarray,
    key_ids: np.ndarray,
    session_ids: np.ndarray,
    absolute_sample_indices: np.ndarray,
    round_start_sample: int,
    round_period_samples: int,
    permutation_mean: np.ndarray,
    permutation_std: np.ndarray,
    config: Stage04Config,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rng = np.random.default_rng(
        config.random_seed + 404
    )

    combined_strata = (
        key_ids.astype(np.int64) * 1_000_000
        + session_ids.astype(np.int64)
    )

    unique_strata = np.unique(combined_strata)
    bootstrap_centers = np.empty(
        (
            config.bootstrap_iterations,
            config.number_of_sboxes,
        ),
        dtype=np.int32,
    )

    rows: List[Dict[str, Any]] = []

    for bootstrap_index in range(
        config.bootstrap_iterations
    ):
        selected_indices: List[int] = []

        for stratum in unique_strata:
            indices = np.where(
                combined_strata == stratum
            )[0]

            selected_indices.extend(
                rng.choice(
                    indices,
                    size=indices.size,
                    replace=True,
                ).tolist()
            )

        selected = np.asarray(
            selected_indices,
            dtype=np.int32,
        )

        specificity = (
            build_specificity_without_new_permutations(
                traces[selected],
                labels[selected],
                key_ids[selected],
                session_ids[selected],
                permutation_mean,
                permutation_std,
                config,
            )
        )

        best, _ = search_regular_sbox_comb(
            specificity,
            absolute_sample_indices,
            round_start_sample,
            round_period_samples,
            config,
        )

        centers = np.asarray(
            best["centers"],
            dtype=np.int32,
        )

        bootstrap_centers[
            bootstrap_index
        ] = centers

        row: Dict[str, Any] = {
            "bootstrap_index": bootstrap_index,
            "first_center_sample": int(
                best["first_center_sample"]
            ),
            "spacing_samples": int(
                best["spacing_samples"]
            ),
            "comb_score": float(best["score"]),
        }

        for sbox_index, center in enumerate(centers):
            row[f"S{sbox_index}_center"] = int(center)

        rows.append(row)

    return bootstrap_centers, rows


def estimate_map_for_subset(
    subset_indices: np.ndarray,
    traces: np.ndarray,
    labels: np.ndarray,
    key_ids: np.ndarray,
    session_ids: np.ndarray,
    absolute_sample_indices: np.ndarray,
    round_start_sample: int,
    round_period_samples: int,
    config: Stage04Config,
) -> Optional[Dict[str, Any]]:
    if subset_indices.size < config.minimum_subset_traces:
        return None

    subset_traces = traces[subset_indices]
    subset_labels = labels[subset_indices]
    subset_keys = key_ids[subset_indices]
    subset_sessions = session_ids[subset_indices]

    balanced, _ = residualize_and_balance_sessions(
        subset_traces,
        subset_sessions,
    )

    observed = np.empty(
        (
            config.number_of_sboxes,
            traces.shape[1],
        ),
        dtype=np.float64,
    )

    for sbox_index in range(config.number_of_sboxes):
        observed[sbox_index] = (
            eta_squared_profile_stratified(
                balanced,
                subset_labels[:, sbox_index],
                subset_keys,
                config.minimum_class_count,
            )
        )

    standardized = np.empty_like(observed)

    for sbox_index in range(config.number_of_sboxes):
        smoothed = triangular_smooth(
            observed[sbox_index],
            config.smoothing_radius_samples,
        )
        standardized[sbox_index] = (
            robust_standardize_profile(smoothed)
        )

    specificity = np.empty_like(standardized)

    for sbox_index in range(config.number_of_sboxes):
        specificity[sbox_index] = (
            standardized[sbox_index]
            - config.specificity_weight
            * np.median(
                np.delete(
                    standardized,
                    sbox_index,
                    axis=0,
                ),
                axis=0,
            )
        )

    best, _ = search_regular_sbox_comb(
        specificity,
        absolute_sample_indices,
        round_start_sample,
        round_period_samples,
        config,
    )

    return best


def build_subset_maps(
    traces: np.ndarray,
    labels: np.ndarray,
    key_ids: np.ndarray,
    session_ids: np.ndarray,
    absolute_sample_indices: np.ndarray,
    round_start_sample: int,
    round_period_samples: int,
    config: Stage04Config,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    session_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []

    for session_id in np.unique(session_ids):
        indices = np.where(
            session_ids == session_id
        )[0]

        estimate = estimate_map_for_subset(
            indices,
            traces,
            labels,
            key_ids,
            session_ids,
            absolute_sample_indices,
            round_start_sample,
            round_period_samples,
            config,
        )

        if estimate is None:
            continue

        row: Dict[str, Any] = {
            "subset_type": "session",
            "session_id": int(session_id),
            "trace_count": int(indices.size),
            "first_center_sample": int(
                estimate["first_center_sample"]
            ),
            "spacing_samples": int(
                estimate["spacing_samples"]
            ),
            "comb_score": float(
                estimate["score"]
            ),
        }

        for sbox_index, center in enumerate(
            estimate["centers"]
        ):
            row[f"S{sbox_index}_center"] = int(center)

        session_rows.append(row)

    for key_id in np.unique(key_ids):
        indices = np.where(key_ids == key_id)[0]

        estimate = estimate_map_for_subset(
            indices,
            traces,
            labels,
            key_ids,
            session_ids,
            absolute_sample_indices,
            round_start_sample,
            round_period_samples,
            config,
        )

        if estimate is None:
            continue

        row = {
            "subset_type": "key",
            "key_id": int(key_id),
            "trace_count": int(indices.size),
            "first_center_sample": int(
                estimate["first_center_sample"]
            ),
            "spacing_samples": int(
                estimate["spacing_samples"]
            ),
            "comb_score": float(
                estimate["score"]
            ),
        }

        for sbox_index, center in enumerate(
            estimate["centers"]
        ):
            row[f"S{sbox_index}_center"] = int(center)

        key_rows.append(row)

    return session_rows, key_rows


# ============================================================
# 7. ساخت پنجره‌ها و timing map نهایی
# ============================================================

def percentile_interval(
    values: np.ndarray,
    confidence_percent: float,
) -> Tuple[float, float]:
    alpha = (
        100.0 - confidence_percent
    ) / 2.0

    return (
        float(np.percentile(values, alpha)),
        float(np.percentile(
            values,
            100.0 - alpha,
        )),
    )


def build_timing_map(
    best_comb: Mapping[str, Any],
    bootstrap_centers: np.ndarray,
    session_rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
    specificity_profiles: np.ndarray,
    absolute_sample_indices: np.ndarray,
    round_start_sample: int,
    round_period_samples: int,
    sampling_rate_hz: float,
    roi_start_sample: int,
    roi_end_sample: int,
    config: Stage04Config,
) -> Dict[str, Any]:
    centers = np.asarray(
        best_comb["centers"],
        dtype=np.int32,
    )
    spacing = int(best_comb["spacing_samples"])

    left_half = spacing // 2
    right_half = spacing - left_half

    bootstrap_std = np.std(
        bootstrap_centers,
        axis=0,
        ddof=1,
    )

    maximum_bootstrap_std = float(
        np.max(bootstrap_std)
    )

    exploration_half_width = max(
        int(math.ceil(
            config.exploration_half_width_fraction_of_spacing
            * spacing
        )),
        int(math.ceil(
            2.0 * maximum_bootstrap_std
        )),
        2,
    )

    seconds_per_sample = 1.0 / sampling_rate_hz

    session_center_matrix = (
        np.asarray([
            [
                int(row[f"S{sbox_index}_center"])
                for sbox_index in range(
                    config.number_of_sboxes
                )
            ]
            for row in session_rows
        ], dtype=np.float64)
        if session_rows
        else np.empty(
            (0, config.number_of_sboxes),
            dtype=np.float64,
        )
    )

    key_center_matrix = (
        np.asarray([
            [
                int(row[f"S{sbox_index}_center"])
                for sbox_index in range(
                    config.number_of_sboxes
                )
            ]
            for row in key_rows
        ], dtype=np.float64)
        if key_rows
        else np.empty(
            (0, config.number_of_sboxes),
            dtype=np.float64,
        )
    )

    sbox_entries = []

    for sbox_index, center in enumerate(centers):
        center_index = sample_index_for_absolute_sample(
            absolute_sample_indices,
            int(center),
        )

        bootstrap_low, bootstrap_high = (
            percentile_interval(
                bootstrap_centers[:, sbox_index],
                config.bootstrap_confidence_percent,
            )
        )

        local_radius = max(1, spacing // 2)
        local_start = max(
            0,
            center_index - local_radius,
        )
        local_end = min(
            absolute_sample_indices.size,
            center_index + local_radius + 1,
        )

        independent_peak_index = (
            local_start
            + int(np.argmax(
                specificity_profiles[
                    sbox_index,
                    local_start:local_end,
                ]
            ))
        )

        independent_peak_sample = int(
            absolute_sample_indices[
                independent_peak_index
            ]
        )

        core_start = max(
            roi_start_sample,
            int(center) - left_half,
        )
        core_end = min(
            roi_end_sample,
            int(center) + right_half,
        )

        exploration_start = max(
            roi_start_sample,
            int(center) - exploration_half_width,
        )
        exploration_end = min(
            roi_end_sample,
            int(center) + exploration_half_width + 1,
        )

        if session_center_matrix.shape[0]:
            session_min = float(np.min(
                session_center_matrix[:, sbox_index]
            ))
            session_max = float(np.max(
                session_center_matrix[:, sbox_index]
            ))
            session_std = float(np.std(
                session_center_matrix[:, sbox_index],
                ddof=0,
            ))
        else:
            session_min = None
            session_max = None
            session_std = None

        if key_center_matrix.shape[0]:
            key_min = float(np.min(
                key_center_matrix[:, sbox_index]
            ))
            key_max = float(np.max(
                key_center_matrix[:, sbox_index]
            ))
            key_std = float(np.std(
                key_center_matrix[:, sbox_index],
                ddof=0,
            ))
        else:
            key_min = None
            key_max = None
            key_std = None

        sbox_entries.append({
            "sbox": f"S{sbox_index}",
            "sbox_index": sbox_index,
            "center_sample": int(center),
            "center_offset_from_round_32_start_samples": int(
                center - round_start_sample
            ),
            "center_time_from_trace_start_us": float(
                center * seconds_per_sample * 1e6
            ),
            "center_time_from_round_32_start_ns": float(
                (
                    center - round_start_sample
                )
                * seconds_per_sample
                * 1e9
            ),
            "core_window_start_sample_inclusive": int(
                core_start
            ),
            "core_window_end_sample_exclusive": int(
                core_end
            ),
            "exploration_window_start_sample_inclusive": int(
                exploration_start
            ),
            "exploration_window_end_sample_exclusive": int(
                exploration_end
            ),
            "specificity_at_center": float(
                specificity_profiles[
                    sbox_index,
                    center_index,
                ]
            ),
            "local_prominence": float(
                local_prominence(
                    specificity_profiles[sbox_index],
                    center_index,
                    spacing,
                )
            ),
            "independent_local_peak_sample": int(
                independent_peak_sample
            ),
            "independent_peak_minus_regularized_center": int(
                independent_peak_sample - center
            ),
            "bootstrap_center_median": float(
                np.median(
                    bootstrap_centers[:, sbox_index]
                )
            ),
            "bootstrap_center_std": float(
                bootstrap_std[sbox_index]
            ),
            "bootstrap_confidence_interval_samples": [
                bootstrap_low,
                bootstrap_high,
            ],
            "session_center_minimum": session_min,
            "session_center_maximum": session_max,
            "session_center_std": session_std,
            "key_center_minimum": key_min,
            "key_center_maximum": key_max,
            "key_center_std": key_std,
        })

    first_values = bootstrap_centers[:, 0]
    spacing_values = np.diff(
        bootstrap_centers,
        axis=1,
    )

    full_map_matches = np.all(
        bootstrap_centers
        == centers[None, :],
        axis=1,
    )

    timing_map = {
        "algorithm": "LBlock-64/80",
        "stage": 4,
        "estimation_source": (
            "Stage 03 public aligned traces, public ciphertexts, "
            "key_id and session_id only"
        ),
        "mapping_method": (
            "Key-stratified observable-X32 eta-squared profiles, "
            "within-key/session permutation normalization, and "
            "regular eight-event comb search"
        ),
        "known_implementation_assumption": (
            "The implementation evaluates S0 through S7 "
            "sequentially in increasing index order."
        ),
        "ground_truth_used_for_estimation": False,
        "round_32_start_sample": int(
            round_start_sample
        ),
        "round_period_samples": int(
            round_period_samples
        ),
        "estimated_first_sbox_center_sample": int(
            centers[0]
        ),
        "estimated_sbox_spacing_samples": spacing,
        "estimated_sbox_spacing_ns": float(
            spacing * seconds_per_sample * 1e9
        ),
        "roi_start_sample_inclusive": int(
            roi_start_sample
        ),
        "roi_end_sample_exclusive": int(
            roi_end_sample
        ),
        "core_window_policy": (
            "Adjacent non-overlapping windows split at "
            "regular-comb midpoints."
        ),
        "exploration_window_half_width_samples": int(
            exploration_half_width
        ),
        "bootstrap": {
            "iterations": int(
                config.bootstrap_iterations
            ),
            "confidence_percent": float(
                config.bootstrap_confidence_percent
            ),
            "exact_full_map_match_rate": float(
                np.mean(full_map_matches)
            ),
            "first_center_median": float(
                np.median(first_values)
            ),
            "first_center_std": float(
                np.std(first_values, ddof=1)
            ),
            "spacing_median": float(
                np.median(spacing_values)
            ),
            "spacing_std": float(
                np.std(spacing_values, ddof=1)
            ),
        },
        "comb_search": {
            "best_score": float(
                best_comb["score"]
            ),
            "second_distinct_score": float(
                best_comb["second_distinct_score"]
            ),
            "absolute_score_gap": float(
                best_comb["absolute_score_gap"]
            ),
            "relative_score_gap": float(
                best_comb["relative_score_gap"]
            ),
            "candidate_count": int(
                best_comb["candidate_count"]
            ),
            "spacing_search_range": (
                best_comb["spacing_search_range"]
            ),
        },
        "sboxes": sbox_entries,
        "next_stage_usage": {
            "nominal_target_time": (
                "Use center_sample for the selected S-box."
            ),
            "robustness_scan": (
                "Sweep timing offsets inside the corresponding "
                "exploration window in Stage 05."
            ),
            "do_not_use": (
                "Private validation timings must never be used "
                "to configure the fault campaign."
            ),
        },
    }

    return timing_map


# ============================================================
# 8. کنترل‌های عمومی
# ============================================================

def validate_public_timing_map(
    timing_map: Mapping[str, Any],
    profile_arrays: Mapping[str, np.ndarray],
    class_coverage: Mapping[str, Any],
    bootstrap_centers: np.ndarray,
    config: Stage04Config,
) -> Dict[str, Any]:
    centers = np.asarray([
        int(entry["center_sample"])
        for entry in timing_map["sboxes"]
    ], dtype=np.int32)

    core_starts = np.asarray([
        int(entry[
            "core_window_start_sample_inclusive"
        ])
        for entry in timing_map["sboxes"]
    ], dtype=np.int32)

    core_ends = np.asarray([
        int(entry[
            "core_window_end_sample_exclusive"
        ])
        for entry in timing_map["sboxes"]
    ], dtype=np.int32)

    round_start = int(
        timing_map["round_32_start_sample"]
    )
    round_end = (
        round_start
        + int(timing_map["round_period_samples"])
    )

    roi_start = int(
        timing_map["roi_start_sample_inclusive"]
    )
    roi_end = int(
        timing_map["roi_end_sample_exclusive"]
    )

    checks: Dict[str, Any] = {}

    checks["eight_centers_generated"] = {
        "passed": bool(
            centers.size == config.number_of_sboxes
        ),
        "center_count": int(centers.size),
    }

    checks["centers_strictly_increasing"] = {
        "passed": bool(
            np.all(np.diff(centers) > 0)
        ),
        "centers": centers.astype(int).tolist(),
    }

    checks["regular_spacing"] = {
        "passed": bool(
            np.all(
                np.diff(centers)
                == int(
                    timing_map[
                        "estimated_sbox_spacing_samples"
                    ]
                )
            )
        ),
        "observed_spacings": (
            np.diff(centers).astype(int).tolist()
        ),
    }

    checks["centers_inside_roi"] = {
        "passed": bool(
            np.all(centers >= roi_start)
            and np.all(centers < roi_end)
        ),
    }

    checks["centers_inside_round_32"] = {
        "passed": bool(
            np.all(centers >= round_start)
            and np.all(centers < round_end)
        ),
        "round_interval": [
            round_start,
            round_end,
        ],
    }

    checks["core_windows_valid"] = {
        "passed": bool(
            np.all(core_starts < centers)
            and np.all(centers < core_ends)
            and np.all(core_starts >= roi_start)
            and np.all(core_ends <= roi_end)
        ),
    }

    checks["core_windows_non_overlapping"] = {
        "passed": bool(
            np.all(core_ends[:-1] <= core_starts[1:])
        ),
    }

    finite_profiles = all(
        np.all(np.isfinite(array))
        for array in profile_arrays.values()
        if isinstance(array, np.ndarray)
    )

    checks["all_profiles_finite"] = {
        "passed": bool(finite_profiles),
    }

    minimum_observed_classes = min(
        int(
            class_coverage["targets"][
                f"S{sbox_index}"
            ][
                "minimum_observed_classes_per_key"
            ]
        )
        for sbox_index in range(
            config.number_of_sboxes
        )
    )

    checks["observable_class_coverage"] = {
        "passed": bool(
            minimum_observed_classes
            >= config.number_of_classes - 1
        ),
        "minimum_observed_classes_per_key": int(
            minimum_observed_classes
        ),
    }

    bootstrap_spacing = np.diff(
        bootstrap_centers,
        axis=1,
    )

    dominant_spacing = int(
        np.rint(np.median(bootstrap_spacing))
    )

    spacing_stability_rate = float(
        np.mean(
            bootstrap_spacing
            == dominant_spacing
        )
    )

    checks["bootstrap_spacing_stability"] = {
        "passed": bool(
            spacing_stability_rate >= 0.75
        ),
        "dominant_spacing": dominant_spacing,
        "per_gap_stability_rate": (
            np.mean(
                bootstrap_spacing
                == dominant_spacing,
                axis=0,
            ).tolist()
        ),
        "overall_stability_rate": (
            spacing_stability_rate
        ),
    }

    checks["public_only_estimation_contract"] = {
        "passed": bool(
            not timing_map[
                "ground_truth_used_for_estimation"
            ]
        ),
        "statement": (
            "No private timing, key value, round key or "
            "internal state was used by the estimator."
        ),
    }

    all_passed = all(
        bool(check["passed"])
        for check in checks.values()
    )

    return {
        "all_public_checks_passed": all_passed,
        "checks": checks,
    }


# ============================================================
# 9. ارزیابی private پس از freeze
# ============================================================

def transformed_hidden_centers(
    original_centers: np.ndarray,
    global_shifts: np.ndarray,
    affine_intercepts: np.ndarray,
    affine_slopes: np.ndarray,
    global_trigger_sample: float,
) -> np.ndarray:
    """
    همان تبدیل محور زمانی Stage 03:

      original = aligned + global_shift
                 + intercept
                 + slope*(aligned-trigger)
    """

    denominator = 1.0 + affine_slopes[:, None]

    return (
        original_centers
        - global_shifts[:, None]
        - affine_intercepts[:, None]
        + affine_slopes[:, None]
        * global_trigger_sample
    ) / denominator


def evaluate_frozen_map_against_private(
    stage3_run_directory: Path,
    stage2_run_directory: Path,
    timing_map: Mapping[str, Any],
    validation_directory: Path,
    freeze_sha256: str,
) -> Dict[str, Any]:
    hidden_path = (
        stage2_run_directory
        / "private_ground_truth"
        / "hidden_timing_and_crypto_ground_truth.npz"
    )

    aligned_path = (
        stage3_run_directory
        / "public"
        / "aligned_healthy_traces.npz"
    )

    if not hidden_path.is_file():
        return {
            "available": False,
            "reason": f"Missing private file: {hidden_path}",
            "estimation_freeze_sha256": freeze_sha256,
        }

    if not aligned_path.is_file():
        return {
            "available": False,
            "reason": f"Missing alignment file: {aligned_path}",
            "estimation_freeze_sha256": freeze_sha256,
        }

    with np.load(
        hidden_path,
        allow_pickle=False,
    ) as hidden:
        original_centers = np.asarray(
            hidden["sbox_centers"][:, -1, :],
            dtype=np.float64,
        )

    with np.load(
        aligned_path,
        allow_pickle=False,
    ) as aligned:
        global_shifts = np.asarray(
            aligned["global_shifts"],
            dtype=np.float64,
        )
        affine_intercepts = np.asarray(
            aligned["affine_intercepts"],
            dtype=np.float64,
        )
        affine_slopes = np.asarray(
            aligned["affine_slopes"],
            dtype=np.float64,
        )
        global_trigger_sample = float(
            np.asarray(
                aligned["global_trigger_sample"]
            ).item()
        )

    if (
        original_centers.shape[0]
        != global_shifts.shape[0]
    ):
        raise ValueError(
            "Private/public trace count mismatch"
        )

    aligned_hidden = transformed_hidden_centers(
        original_centers,
        global_shifts,
        affine_intercepts,
        affine_slopes,
        global_trigger_sample,
    )

    estimated = np.asarray([
        int(entry["center_sample"])
        for entry in timing_map["sboxes"]
    ], dtype=np.float64)

    hidden_medians = np.median(
        aligned_hidden,
        axis=0,
    )
    hidden_means = np.mean(
        aligned_hidden,
        axis=0,
    )
    hidden_stds = np.std(
        aligned_hidden,
        axis=0,
    )

    signed_errors = estimated - hidden_medians
    absolute_errors = np.abs(signed_errors)

    core_coverage_per_sbox = []
    exploration_coverage_per_sbox = []

    for sbox_index, entry in enumerate(
        timing_map["sboxes"]
    ):
        core_start = float(
            entry[
                "core_window_start_sample_inclusive"
            ]
        )
        core_end = float(
            entry[
                "core_window_end_sample_exclusive"
            ]
        )

        exploration_start = float(
            entry[
                "exploration_window_start_sample_inclusive"
            ]
        )
        exploration_end = float(
            entry[
                "exploration_window_end_sample_exclusive"
            ]
        )

        core_coverage_per_sbox.append(float(
            np.mean(
                (
                    aligned_hidden[:, sbox_index]
                    >= core_start
                )
                & (
                    aligned_hidden[:, sbox_index]
                    < core_end
                )
            )
        ))

        exploration_coverage_per_sbox.append(float(
            np.mean(
                (
                    aligned_hidden[:, sbox_index]
                    >= exploration_start
                )
                & (
                    aligned_hidden[:, sbox_index]
                    < exploration_end
                )
            )
        ))

    hidden_spacing = np.diff(
        hidden_medians
    )
    estimated_spacing = float(
        timing_map["estimated_sbox_spacing_samples"]
    )

    nearest_estimated_index = np.argmin(
        np.abs(
            hidden_medians[:, None]
            - estimated[None, :]
        ),
        axis=1,
    )

    assignment_correct = bool(
        np.array_equal(
            nearest_estimated_index,
            np.arange(estimated.size),
        )
    )

    evaluation = {
        "available": True,
        "warning": (
            "Validation only. Private timings were opened "
            "after the public timing map had been frozen."
        ),
        "estimation_freeze_sha256": freeze_sha256,
        "private_file_sha256": sha256_file(
            hidden_path
        ),
        "estimated_centers": (
            estimated.tolist()
        ),
        "hidden_median_centers": (
            hidden_medians.tolist()
        ),
        "hidden_mean_centers": (
            hidden_means.tolist()
        ),
        "hidden_center_std_per_sbox": (
            hidden_stds.tolist()
        ),
        "signed_center_errors_samples": (
            signed_errors.tolist()
        ),
        "absolute_center_errors_samples": (
            absolute_errors.tolist()
        ),
        "center_mean_absolute_error_samples": float(
            np.mean(absolute_errors)
        ),
        "center_maximum_absolute_error_samples": float(
            np.max(absolute_errors)
        ),
        "hidden_median_spacing_samples": float(
            np.median(hidden_spacing)
        ),
        "estimated_spacing_samples": (
            estimated_spacing
        ),
        "spacing_absolute_error_samples": float(
            abs(
                estimated_spacing
                - np.median(hidden_spacing)
            )
        ),
        "nearest_center_assignment_correct": (
            assignment_correct
        ),
        "core_window_coverage_per_sbox": (
            core_coverage_per_sbox
        ),
        "core_window_overall_coverage": float(
            np.mean(core_coverage_per_sbox)
        ),
        "exploration_window_coverage_per_sbox": (
            exploration_coverage_per_sbox
        ),
        "exploration_window_overall_coverage": float(
            np.mean(
                exploration_coverage_per_sbox
            )
        ),
        "hidden_centers_strictly_ordered": bool(
            np.all(
                np.diff(hidden_medians) > 0
            )
        ),
    }

    validation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        validation_directory
        / "private_sbox_timing_evaluation.json",
        evaluation,
    )

    if plt is not None:
        fig = plt.figure(figsize=(11, 5.5))
        axis = fig.add_subplot(1, 1, 1)

        for sbox_index in range(estimated.size):
            axis.scatter(
                np.full(
                    aligned_hidden.shape[0],
                    sbox_index,
                    dtype=np.float64,
                ),
                aligned_hidden[:, sbox_index],
                s=5,
                alpha=0.15,
            )

            axis.plot(
                [sbox_index - 0.28, sbox_index + 0.28],
                [estimated[sbox_index], estimated[sbox_index]],
                linewidth=2.0,
            )

        axis.set_title(
            "Validation only — hidden aligned centers "
            "and public estimates"
        )
        axis.set_xlabel("S-box index")
        axis.set_ylabel("Absolute sample")
        axis.set_xticks(range(estimated.size))
        axis.grid(alpha=0.2)
        fig.tight_layout()

        fig.savefig(
            validation_directory
            / "private_sbox_center_validation.png",
            dpi=180,
        )
        plt.close(fig)

    return evaluation


# ============================================================
# 10. شکل‌ها و جداول عمومی
# ============================================================

def save_public_plots(
    public_directory: Path,
    absolute_sample_indices: np.ndarray,
    profile_arrays: Mapping[str, np.ndarray],
    timing_map: Mapping[str, Any],
    bootstrap_centers: np.ndarray,
    session_rows: Sequence[Mapping[str, Any]],
    config: Stage04Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []

    centers = np.asarray([
        int(entry["center_sample"])
        for entry in timing_map["sboxes"]
    ], dtype=np.int32)

    # --------------------------------------------------------
    # 1) Heatmap وابستگی هر nibble
    # --------------------------------------------------------
    heatmap = profile_arrays[
        "specificity_profiles"
    ]

    fig = plt.figure(figsize=(13, 6))
    axis = fig.add_subplot(1, 1, 1)

    image = axis.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[
            int(absolute_sample_indices[0]),
            int(absolute_sample_indices[-1]),
            -0.5,
            7.5,
        ],
    )

    for sbox_index, center in enumerate(centers):
        axis.scatter(
            [center],
            [sbox_index],
            marker="x",
            s=70,
        )

    axis.set_title(
        "Observable-X32 dependency profiles and "
        "regularized S-box centers"
    )
    axis.set_xlabel("Absolute sample")
    axis.set_ylabel("Target nibble / S-box")
    axis.set_yticks(range(8))
    axis.set_yticklabels([
        f"S{i}" for i in range(8)
    ])
    fig.colorbar(
        image,
        ax=axis,
        label="Specificity score",
    )
    fig.tight_layout()

    path = (
        public_directory
        / "sbox_dependency_heatmap.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # --------------------------------------------------------
    # 2) پروفایل‌های خطی
    # --------------------------------------------------------
    fig = plt.figure(figsize=(13, 10))
    axis = fig.add_subplot(1, 1, 1)

    for sbox_index in range(8):
        profile = heatmap[sbox_index]
        offset = 4.0 * sbox_index

        axis.plot(
            absolute_sample_indices,
            profile + offset,
            linewidth=0.9,
            label=f"S{sbox_index}",
        )

        axis.axvline(
            centers[sbox_index],
            linestyle="--",
            linewidth=0.65,
        )

    axis.set_title(
        "Stacked S-box timing dependency profiles"
    )
    axis.set_xlabel("Absolute sample")
    axis.set_ylabel("Profile score with display offset")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4)
    fig.tight_layout()

    path = (
        public_directory
        / "sbox_dependency_profiles.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # --------------------------------------------------------
    # 3) timing map و پنجره‌ها
    # --------------------------------------------------------
    fig = plt.figure(figsize=(13, 6))
    axis = fig.add_subplot(1, 1, 1)

    for entry in timing_map["sboxes"]:
        index = int(entry["sbox_index"])

        exploration_start = int(
            entry[
                "exploration_window_start_sample_inclusive"
            ]
        )
        exploration_end = int(
            entry[
                "exploration_window_end_sample_exclusive"
            ]
        )
        core_start = int(
            entry[
                "core_window_start_sample_inclusive"
            ]
        )
        core_end = int(
            entry[
                "core_window_end_sample_exclusive"
            ]
        )
        center = int(entry["center_sample"])

        axis.plot(
            [exploration_start, exploration_end],
            [index, index],
            linewidth=7,
            alpha=0.25,
        )
        axis.plot(
            [core_start, core_end],
            [index, index],
            linewidth=7,
            alpha=0.75,
        )
        axis.scatter(
            [center],
            [index],
            marker="x",
            s=65,
        )

    axis.set_title(
        "Stage 04 public S-box timing map"
    )
    axis.set_xlabel("Absolute sample")
    axis.set_ylabel("S-box")
    axis.set_yticks(range(8))
    axis.set_yticklabels([
        f"S{i}" for i in range(8)
    ])
    axis.grid(alpha=0.2)
    fig.tight_layout()

    path = (
        public_directory
        / "sbox_timing_map_windows.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # --------------------------------------------------------
    # 4) Bootstrap
    # --------------------------------------------------------
    fig = plt.figure(figsize=(11, 5.5))
    axis = fig.add_subplot(1, 1, 1)

    for sbox_index in range(8):
        axis.scatter(
            np.full(
                bootstrap_centers.shape[0],
                sbox_index,
                dtype=np.float64,
            ),
            bootstrap_centers[:, sbox_index],
            s=10,
            alpha=0.25,
        )
        axis.scatter(
            [sbox_index],
            [centers[sbox_index]],
            marker="x",
            s=80,
        )

    axis.set_title(
        "Bootstrap stability of S-box centers"
    )
    axis.set_xlabel("S-box")
    axis.set_ylabel("Estimated absolute sample")
    axis.set_xticks(range(8))
    axis.set_xticklabels([
        f"S{i}" for i in range(8)
    ])
    axis.grid(alpha=0.2)
    fig.tight_layout()

    path = (
        public_directory
        / "bootstrap_sbox_center_stability.png"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # --------------------------------------------------------
    # 5) Session maps
    # --------------------------------------------------------
    if session_rows:
        fig = plt.figure(figsize=(11, 5.5))
        axis = fig.add_subplot(1, 1, 1)

        for row in session_rows:
            values = [
                int(row[f"S{i}_center"])
                for i in range(8)
            ]
            axis.plot(
                range(8),
                values,
                marker="o",
                linewidth=0.9,
                label=f"session {row['session_id']}",
            )

        axis.plot(
            range(8),
            centers,
            marker="x",
            linewidth=1.6,
            label="global map",
        )

        axis.set_title(
            "Timing-map consistency across sessions"
        )
        axis.set_xlabel("S-box")
        axis.set_ylabel("Absolute sample")
        axis.set_xticks(range(8))
        axis.set_xticklabels([
            f"S{i}" for i in range(8)
        ])
        axis.grid(alpha=0.2)
        axis.legend()
        fig.tight_layout()

        path = (
            public_directory
            / "session_timing_map_consistency.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(path.name)

    return generated


# ============================================================
# 11. اجرای کامل Stage 04
# ============================================================

def run_stage_04(
    config: Stage04Config,
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    stage3_run_directory = Path(
        config.input_stage3_run_directory
    ).expanduser().resolve()

    stage3_summary_path = (
        stage3_run_directory
        / "stage_03_summary.json"
    )

    if not stage3_summary_path.is_file():
        raise FileNotFoundError(
            f"Stage 03 summary not found: "
            f"{stage3_summary_path}"
        )

    stage3_summary = read_json(
        stage3_summary_path
    )

    if not stage3_summary.get(
        "all_checks_passed",
        False,
    ):
        raise RuntimeError(
            "Stage 03 did not pass all checks"
        )

    public_input_directory = (
        stage3_run_directory / "public"
    )

    roi_traces_path = (
        public_input_directory
        / "final_round_roi_traces.npz"
    )
    roi_json_path = (
        public_input_directory
        / "final_round_roi.json"
    )
    metadata_path = (
        public_input_directory
        / "aligned_trace_metadata.csv"
    )
    aligned_full_path = (
        public_input_directory
        / "aligned_healthy_traces.npz"
    )

    required_inputs = [
        roi_traces_path,
        roi_json_path,
        metadata_path,
        aligned_full_path,
    ]

    missing = [
        str(path)
        for path in required_inputs
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Stage 03 public files:\n  - "
            + "\n  - ".join(missing)
        )

    roi_definition = read_json(
        roi_json_path
    )

    with np.load(
        roi_traces_path,
        allow_pickle=False,
    ) as data:
        traces = np.asarray(
            data["traces"],
            dtype=np.float32,
        )
        alignment_signal = np.asarray(
            data["alignment_signal"],
            dtype=np.float32,
        )
        trace_ids = np.asarray(
            data["trace_ids"],
            dtype=np.int32,
        )
        absolute_sample_indices = np.asarray(
            data["absolute_sample_indices"],
            dtype=np.int32,
        )
        sample_axis_seconds = np.asarray(
            data["sample_axis_seconds"],
            dtype=np.float64,
        )
        roi_start_sample = int(
            np.asarray(
                data["roi_start_sample"]
            ).item()
        )
        roi_end_sample = int(
            np.asarray(
                data["roi_end_sample"]
            ).item()
        )

    metadata_rows = read_csv_rows(
        metadata_path
    )
    ordered_metadata = (
        reorder_metadata_by_trace_id(
            metadata_rows,
            trace_ids,
        )
    )

    (
        metadata_trace_ids,
        key_ids,
        session_ids,
        observable_labels,
    ) = extract_public_observable_labels(
        ordered_metadata,
        config.number_of_sboxes,
    )

    if not np.array_equal(
        metadata_trace_ids,
        trace_ids,
    ):
        raise ValueError(
            "Trace ID ordering mismatch"
        )

    if traces.shape != alignment_signal.shape:
        raise ValueError(
            "Amplitude and alignment ROI shapes differ"
        )

    if traces.shape[0] != trace_ids.size:
        raise ValueError(
            "Trace count mismatch"
        )

    if traces.shape[1] != absolute_sample_indices.size:
        raise ValueError(
            "Sample-axis length mismatch"
        )

    if not np.all(
        np.diff(absolute_sample_indices) == 1
    ):
        raise ValueError(
            "ROI absolute sample axis must be contiguous"
        )

    round_start_sample = int(
        roi_definition[
            "estimated_round_32_start_sample"
        ]
    )
    round_period_samples = int(
        roi_definition[
            "estimated_round_period_samples"
        ]
    )

    sampling_rate_hz = float(
        stage3_summary["sampling_rate_hz"]
    )

    # --------------------------------------------------------
    # ساخت پوشه run
    # --------------------------------------------------------
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_id = (
        f"stage04_{timestamp}_seed"
        f"{config.random_seed}"
    )

    run_directory = (
        Path(config.output_root)
        .expanduser()
        .resolve()
        / run_id
    )

    public_output_directory = (
        run_directory / "public"
    )
    validation_directory = (
        run_directory / "validation_only"
    )

    public_output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # audit برچسب‌های عمومی
    # --------------------------------------------------------
    class_coverage = (
        build_class_coverage_report(
            observable_labels,
            key_ids,
            config.number_of_classes,
        )
    )

    label_audit = {
        "allowed_fields_used": [
            "trace_id",
            "ciphertext_hex",
            "key_id",
            "session_id",
        ],
        "observable_relation": (
            "X32 is the left 32-bit half of ciphertext. "
            "For target i, class = nibble_i(X32)."
        ),
        "why_key_value_is_not_required": (
            "Within each key_id, XOR with the unknown key nibble "
            "is a fixed permutation of the 16 observable classes. "
            "Non-parametric class dependence is preserved."
        ),
        "forbidden_fields_used": [],
        "master_key_value_used": False,
        "round_key_value_used": False,
        "internal_state_used": False,
        "private_timing_used": False,
        "class_coverage": class_coverage,
    }

    write_json(
        public_output_directory
        / "observable_label_audit.json",
        label_audit,
    )

    # --------------------------------------------------------
    # پروفایل‌های وابستگی
    # --------------------------------------------------------
    (
        profile_arrays,
        profile_diagnostics,
    ) = build_dependency_profiles(
        traces,
        observable_labels,
        key_ids,
        session_ids,
        config,
    )

    # --------------------------------------------------------
    # comb اصلی
    # --------------------------------------------------------
    best_comb, comb_candidates = (
        search_regular_sbox_comb(
            profile_arrays[
                "specificity_profiles"
            ],
            absolute_sample_indices,
            round_start_sample,
            round_period_samples,
            config,
        )
    )

    # --------------------------------------------------------
    # bootstrap
    # --------------------------------------------------------
    (
        bootstrap_centers,
        bootstrap_rows,
    ) = run_stratified_bootstrap(
        traces,
        observable_labels,
        key_ids,
        session_ids,
        absolute_sample_indices,
        round_start_sample,
        round_period_samples,
        profile_arrays[
            "permutation_mean_eta_squared"
        ],
        profile_arrays[
            "permutation_std_eta_squared"
        ],
        config,
    )

    # --------------------------------------------------------
    # session/key maps
    # --------------------------------------------------------
    (
        session_rows,
        key_rows,
    ) = build_subset_maps(
        traces,
        observable_labels,
        key_ids,
        session_ids,
        absolute_sample_indices,
        round_start_sample,
        round_period_samples,
        config,
    )

    # --------------------------------------------------------
    # timing map
    # --------------------------------------------------------
    timing_map = build_timing_map(
        best_comb,
        bootstrap_centers,
        session_rows,
        key_rows,
        profile_arrays[
            "specificity_profiles"
        ],
        absolute_sample_indices,
        round_start_sample,
        round_period_samples,
        sampling_rate_hz,
        roi_start_sample,
        roi_end_sample,
        config,
    )

    # --------------------------------------------------------
    # ذخیره خروجی‌های عمومی
    # --------------------------------------------------------
    write_json(
        public_output_directory
        / "stage_04_config.json",
        asdict(config),
    )

    write_json(
        public_output_directory
        / "sbox_timing_map.json",
        timing_map,
    )

    # نام کوتاه و ثابت برای مصرف مراحل بعد
    write_json(
        public_output_directory
        / "lblock_final_round_timing_map.json",
        timing_map,
    )

    write_json(
        public_output_directory
        / "dependency_profile_diagnostics.json",
        profile_diagnostics,
    )

    np.savez_compressed(
        public_output_directory
        / "sbox_dependency_profiles.npz",
        trace_ids=trace_ids,
        absolute_sample_indices=(
            absolute_sample_indices
        ),
        sample_axis_seconds=sample_axis_seconds,
        observable_labels=observable_labels,
        observed_eta_squared=profile_arrays[
            "observed_eta_squared"
        ].astype(np.float64),
        permutation_mean_eta_squared=profile_arrays[
            "permutation_mean_eta_squared"
        ].astype(np.float64),
        permutation_std_eta_squared=profile_arrays[
            "permutation_std_eta_squared"
        ].astype(np.float64),
        permutation_z=profile_arrays[
            "permutation_z"
        ].astype(np.float64),
        smoothed_permutation_z=profile_arrays[
            "smoothed_permutation_z"
        ].astype(np.float64),
        standardized_dependency=profile_arrays[
            "standardized_dependency"
        ].astype(np.float64),
        specificity_profiles=profile_arrays[
            "specificity_profiles"
        ].astype(np.float64),
        bootstrap_centers=bootstrap_centers,
    )

    profile_rows: List[Dict[str, Any]] = []

    for sample_index, absolute_sample in enumerate(
        absolute_sample_indices
    ):
        row: Dict[str, Any] = {
            "absolute_sample": int(
                absolute_sample
            ),
            "roi_relative_sample": int(
                sample_index
            ),
        }

        for sbox_index in range(
            config.number_of_sboxes
        ):
            row[
                f"S{sbox_index}_eta_squared"
            ] = float(
                profile_arrays[
                    "observed_eta_squared"
                ][sbox_index, sample_index]
            )
            row[
                f"S{sbox_index}_permutation_z"
            ] = float(
                profile_arrays[
                    "smoothed_permutation_z"
                ][sbox_index, sample_index]
            )
            row[
                f"S{sbox_index}_specificity"
            ] = float(
                profile_arrays[
                    "specificity_profiles"
                ][sbox_index, sample_index]
            )

        profile_rows.append(row)

    write_csv_rows(
        public_output_directory
        / "sbox_timing_profiles.csv",
        profile_rows,
    )

    write_csv_rows(
        public_output_directory
        / "bootstrap_timing_stability.csv",
        bootstrap_rows,
    )

    if session_rows:
        write_csv_rows(
            public_output_directory
            / "session_timing_maps.csv",
            session_rows,
        )

    if key_rows:
        write_csv_rows(
            public_output_directory
            / "key_timing_maps.csv",
            key_rows,
        )

    candidate_rows = []

    for rank, candidate in enumerate(
        comb_candidates[:250],
        start=1,
    ):
        candidate_rows.append({
            "rank": rank,
            "score": float(candidate["score"]),
            "first_center_sample": int(
                candidate["first_center_sample"]
            ),
            "spacing_samples": int(
                candidate["spacing_samples"]
            ),
            "mean_center_specificity": float(
                candidate[
                    "mean_center_specificity"
                ]
            ),
            "minimum_center_specificity": float(
                candidate[
                    "minimum_center_specificity"
                ]
            ),
            "mean_prominence": float(
                candidate["mean_prominence"]
            ),
        })

    write_csv_rows(
        public_output_directory
        / "top_comb_candidates.csv",
        candidate_rows,
    )

    data_access_manifest = {
        "estimation_mode": "public-only",
        "stage_03_run_directory": str(
            stage3_run_directory
        ),
        "files_opened_by_estimator": [
            str(stage3_summary_path),
            str(roi_traces_path),
            str(roi_json_path),
            str(metadata_path),
        ],
        "stage_02_files_opened_by_estimator": [],
        "private_files_opened_by_estimator": [],
        "explicitly_not_read": [
            "Stage 02 trace_simulation_config.json timing fields",
            "hidden_nominal_timing_map.json",
            "hidden_timing_and_crypto_ground_truth.npz",
            "private_key_manifest.json",
        ],
    }

    write_json(
        public_output_directory
        / "data_access_manifest.json",
        data_access_manifest,
    )

    generated_plots = save_public_plots(
        public_output_directory,
        absolute_sample_indices,
        profile_arrays,
        timing_map,
        bootstrap_centers,
        session_rows,
        config,
    )

    # --------------------------------------------------------
    # کنترل‌های عمومی
    # --------------------------------------------------------
    public_validation = (
        validate_public_timing_map(
            timing_map,
            {
                key: value
                for key, value in profile_arrays.items()
                if key != "balanced_traces"
            },
            class_coverage,
            bootstrap_centers,
            config,
        )
    )

    write_json(
        run_directory
        / "stage_04_public_validation_checks.json",
        public_validation,
    )

    # --------------------------------------------------------
    # freeze پیش از private evaluation
    # --------------------------------------------------------
    freeze_files = [
        public_output_directory
        / "sbox_timing_map.json",
        public_output_directory
        / "sbox_dependency_profiles.npz",
        public_output_directory
        / "observable_label_audit.json",
        public_output_directory
        / "bootstrap_timing_stability.csv",
        public_output_directory
        / "data_access_manifest.json",
    ]

    freeze_manifest = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "statement": (
            "The public Stage 04 timing map was frozen "
            "before private timing validation."
        ),
        "files": {
            path.name: sha256_file(path)
            for path in freeze_files
        },
    }

    freeze_source = json.dumps(
        freeze_manifest,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    freeze_sha256 = hashlib.sha256(
        freeze_source
    ).hexdigest()

    freeze_manifest["freeze_sha256"] = (
        freeze_sha256
    )

    write_json(
        run_directory
        / "estimation_freeze_manifest.json",
        freeze_manifest,
    )

    # --------------------------------------------------------
    # private evaluation
    # --------------------------------------------------------
    stage2_run_directory = Path(
        stage3_summary[
            "input_stage_02_run_directory"
        ]
    ).expanduser().resolve()

    if config.enable_private_evaluation:
        private_evaluation = (
            evaluate_frozen_map_against_private(
                stage3_run_directory,
                stage2_run_directory,
                timing_map,
                validation_directory,
                freeze_sha256,
            )
        )
    else:
        private_evaluation = {
            "available": False,
            "reason": (
                "Private evaluation disabled by config"
            ),
            "estimation_freeze_sha256": (
                freeze_sha256
            ),
        }

    private_validation_passed = True

    if private_evaluation.get(
        "available",
        False,
    ):
        private_validation_passed = bool(
            private_evaluation[
                "center_mean_absolute_error_samples"
            ] <= 2.0
            and private_evaluation[
                "center_maximum_absolute_error_samples"
            ] <= 3.0
            and private_evaluation[
                "spacing_absolute_error_samples"
            ] <= 1.0
            and private_evaluation[
                "nearest_center_assignment_correct"
            ]
            and private_evaluation[
                "exploration_window_overall_coverage"
            ] >= 0.85
        )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    all_checks_passed = bool(
        public_validation[
            "all_public_checks_passed"
        ]
        and private_validation_passed
    )

    estimated_centers = {
        entry["sbox"]: int(
            entry["center_sample"]
        )
        for entry in timing_map["sboxes"]
    }

    estimated_offsets = {
        entry["sbox"]: int(
            entry[
                "center_offset_from_round_32_start_samples"
            ]
        )
        for entry in timing_map["sboxes"]
    }

    public_files = sorted(
        path.name
        for path in public_output_directory.iterdir()
        if path.is_file()
    )

    validation_files = (
        sorted(
            path.name
            for path in validation_directory.iterdir()
            if path.is_file()
        )
        if validation_directory.is_dir()
        else []
    )

    summary = {
        "stage": 4,
        "run_id": run_id,
        "run_directory": str(
            run_directory.resolve()
        ),
        "input_stage_03_run_directory": str(
            stage3_run_directory
        ),
        "input_stage_02_run_directory_for_validation_only": str(
            stage2_run_directory
        ),
        "public_directory": str(
            public_output_directory.resolve()
        ),
        "validation_only_directory": str(
            validation_directory.resolve()
        ),
        "all_checks_passed": (
            all_checks_passed
        ),
        "all_public_checks_passed": bool(
            public_validation[
                "all_public_checks_passed"
            ]
        ),
        "private_validation_available": bool(
            private_evaluation.get(
                "available",
                False,
            )
        ),
        "private_validation_passed": bool(
            private_validation_passed
        ),
        "number_of_traces": int(
            traces.shape[0]
        ),
        "roi_width_samples": int(
            traces.shape[1]
        ),
        "round_32_start_sample": int(
            round_start_sample
        ),
        "estimated_sbox_spacing_samples": int(
            timing_map[
                "estimated_sbox_spacing_samples"
            ]
        ),
        "estimated_sbox_centers": (
            estimated_centers
        ),
        "estimated_sbox_offsets_from_round_32_start": (
            estimated_offsets
        ),
        "bootstrap_exact_full_map_match_rate": float(
            timing_map["bootstrap"][
                "exact_full_map_match_rate"
            ]
        ),
        "comb_relative_score_gap": float(
            timing_map["comb_search"][
                "relative_score_gap"
            ]
        ),
        "private_center_mean_absolute_error_samples": (
            float(
                private_evaluation[
                    "center_mean_absolute_error_samples"
                ]
            )
            if private_evaluation.get(
                "available",
                False,
            )
            else None
        ),
        "private_center_maximum_absolute_error_samples": (
            float(
                private_evaluation[
                    "center_maximum_absolute_error_samples"
                ]
            )
            if private_evaluation.get(
                "available",
                False,
            )
            else None
        ),
        "private_exploration_window_coverage": (
            float(
                private_evaluation[
                    "exploration_window_overall_coverage"
                ]
            )
            if private_evaluation.get(
                "available",
                False,
            )
            else None
        ),
        "estimation_freeze_sha256": (
            freeze_sha256
        ),
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "public_files": public_files,
        "validation_only_files": (
            validation_files
        ),
        "generated_plots": generated_plots,
    }

    write_json(
        run_directory / "stage_04_summary.json",
        summary,
    )

    write_json(
        run_directory / "run_manifest.json",
        {
            "stage": 4,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": asdict(config),
            "input_file_sha256": {
                "stage_03_summary.json": sha256_file(
                    stage3_summary_path
                ),
                "final_round_roi_traces.npz": sha256_file(
                    roi_traces_path
                ),
                "final_round_roi.json": sha256_file(
                    roi_json_path
                ),
                "aligned_trace_metadata.csv": sha256_file(
                    metadata_path
                ),
            },
        },
    )

    print("\n" + "=" * 78)
    print(
        "Stage 04 complete: public-only final-round "
        "S-box timing map"
    )
    print("=" * 78)
    print(
        "Run directory                    :",
        summary["run_directory"],
    )
    print(
        "All checks passed                :",
        summary["all_checks_passed"],
    )
    print(
        "Public checks passed             :",
        summary["all_public_checks_passed"],
    )
    print(
        "Private validation available     :",
        summary["private_validation_available"],
    )
    print(
        "Private validation passed        :",
        summary["private_validation_passed"],
    )
    print(
        "Trace count                      :",
        summary["number_of_traces"],
    )
    print(
        "Round-32 start                   :",
        summary["round_32_start_sample"],
    )
    print(
        "Estimated S-box spacing          :",
        summary["estimated_sbox_spacing_samples"],
    )
    print(
        "Estimated S-box centers          :",
        summary["estimated_sbox_centers"],
    )
    print(
        "Bootstrap exact-map rate         :",
        f"{summary['bootstrap_exact_full_map_match_rate']:.6f}",
    )
    print(
        "Private center MAE               :",
        summary[
            "private_center_mean_absolute_error_samples"
        ],
    )
    print(
        "Private maximum center error     :",
        summary[
            "private_center_maximum_absolute_error_samples"
        ],
    )
    print(
        "Private exploration coverage     :",
        summary[
            "private_exploration_window_coverage"
        ],
    )
    print(
        "Elapsed seconds                  :",
        f"{summary['elapsed_seconds']:.3f}",
    )
    print("=" * 78)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 04 failed validation. Inspect "
            "stage_04_public_validation_checks.json and "
            "validation_only/private_sbox_timing_evaluation.json"
        )

    return summary


def load_stage_04_config(
    config_path: str | Path,
) -> Stage04Config:
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return Stage04Config(**raw)


if __name__ == "__main__":
    default_config = Stage04Config(
        input_stage3_run_directory=(
            './runs/stage_03'
            '/stage03_20260718_170153_519460_seed20260718'
        ),
        output_root=(
            './runs/stage_04'
        ),
    )

    run_stage_04(default_config)

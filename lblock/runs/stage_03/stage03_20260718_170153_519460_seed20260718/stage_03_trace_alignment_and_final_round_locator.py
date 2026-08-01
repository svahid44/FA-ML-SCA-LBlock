# ============================================================
# Stage 03 — Public-only trace alignment and final-round locator
#
# هدف این مرحله:
#   1) بارگذاری فقط داده‌های عمومی Stage 02
#   2) حذف baseline و drift بدون ازبین‌بردن دامنه نشتی
#   3) هم‌ترازی سراسری traceها
#   4) تصحیح drift زمانی خطی در طول trace
#   5) تخمین دوره هر round مستقیماً از traceها
#   6) تعیین ROI دور 32 بدون استفاده از زمان مخفی S-boxها
#   7) ارزیابی جداگانه و فقط پس از freeze شدن نتیجه عمومی
#
# نکته علمی:
# pipeline اصلی هیچ فایل private_ground_truth را دریافت نمی‌کند.
# ارزیابی private در تابعی مستقل و فقط بعد از ذخیره و hash کردن
# خروجی public انجام می‌شود.
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


# Matplotlib فقط برای شکل‌های گزارش استفاده می‌شود.
# در صورت نبودن آن، محاسبات و فایل‌های عددی همچنان تولید می‌شوند.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass(frozen=True)
class Stage03Config:
    """تنظیمات هم‌ترازی و پیدا کردن ROI دور آخر."""

    input_stage2_run_directory: str
    output_root: str = "runs/stage_03"
    random_seed: int = 20260718

    # LBlock دارای 32 دور است؛ این عدد ویژگی الگوریتم است، نه timing مخفی.
    number_of_rounds: int = 32

    # preprocessing
    highpass_window_samples: int = 51
    edge_baseline_samples: int = 64

    # coarse global alignment
    global_alignment_iterations: int = 2
    maximum_global_shift_samples: int = 12

    # local affine timing correction
    number_of_local_anchors: int = 10
    local_anchor_window_samples: int = 288
    maximum_local_shift_samples: int = 8
    minimum_anchor_correlation: float = 0.05

    # period and final-round ROI locator
    period_search_lower_factor: float = 0.65
    period_search_upper_factor: float = 1.35
    round_start_search_radius_samples: int = 18
    final_round_refinement_radius_samples: int = 10
    final_round_guard_samples: int = 10

    # خروجی و ارزیابی
    save_aligned_full_traces: bool = True
    save_plots: bool = True
    enable_private_evaluation: bool = True


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("CSV rows cannot be empty")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    """میانگین متحرک با padding لبه‌ای و طول خروجی ثابت."""

    values = np.asarray(values, dtype=np.float64)

    if window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    cumulative = np.cumsum(
        np.concatenate(([0.0], padded)),
        dtype=np.float64,
    )
    result = (cumulative[window:] - cumulative[:-window]) / window

    if result.shape[0] != values.shape[0]:
        raise RuntimeError("moving-average length mismatch")

    return result


def moving_average_matrix(
    matrix: np.ndarray,
    window: int,
) -> np.ndarray:
    """اعمال میانگین متحرک روی هر trace بدون وابستگی به SciPy."""

    matrix = np.asarray(matrix, dtype=np.float64)

    if window <= 1:
        return matrix.copy()

    output = np.empty_like(matrix, dtype=np.float64)

    for row_index in range(matrix.shape[0]):
        output[row_index] = moving_average_1d(
            matrix[row_index],
            window,
        )

    return output


def normalized_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """ضریب همبستگی نرمال‌شده با کنترل بردارهای تقریباً ثابت."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    if left.shape != right.shape or left.size < 4:
        return float("-inf")

    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)

    denominator = (
        np.linalg.norm(left_centered)
        * np.linalg.norm(right_centered)
    )

    if denominator <= 1e-15:
        return float("-inf")

    return float(
        np.dot(left_centered, right_centered) / denominator
    )


def sample_trace_at_coordinates(
    trace: np.ndarray,
    source_coordinates: np.ndarray,
) -> np.ndarray:
    """
    resample کردن trace.

    برای هر sample خروجی y، مقدار ورودی در source_coordinates[y]
    خوانده می‌شود. این قرارداد باعث می‌شود shift مثبت، رویداد دیررس
    را به سمت چپ و روی reference منتقل کند.
    """

    trace = np.asarray(trace, dtype=np.float64)
    axis = np.arange(trace.shape[0], dtype=np.float64)

    return np.interp(
        source_coordinates,
        axis,
        trace,
        left=float(trace[0]),
        right=float(trace[-1]),
    )


def apply_integer_shift(
    trace: np.ndarray,
    shift_samples: int,
) -> np.ndarray:
    axis = np.arange(trace.shape[0], dtype=np.float64)

    return sample_trace_at_coordinates(
        trace,
        axis + int(shift_samples),
    )


def apply_affine_residual_warp(
    trace: np.ndarray,
    intercept_samples: float,
    slope_samples_per_sample: float,
    anchor_sample: float,
) -> np.ndarray:
    """
    تصحیح residual shift که در طول trace به‌شکل خطی تغییر می‌کند.

    source(y) = y + a + b * (y - anchor)
    """

    axis = np.arange(trace.shape[0], dtype=np.float64)
    source_coordinates = (
        axis
        + intercept_samples
        + slope_samples_per_sample * (axis - anchor_sample)
    )

    return sample_trace_at_coordinates(
        trace,
        source_coordinates,
    )


def robust_scale_rows(
    matrix: np.ndarray,
    active_start: int,
    active_end: int,
) -> np.ndarray:
    """
    نسخه نرمال‌شده فقط برای alignment.

    این نرمال‌سازی روی فایل trace دامنه‌محور نهایی اعمال نمی‌شود؛
    بنابراین اطلاعات دامنه برای مراحل SCA بعدی حفظ می‌شود.
    """

    active = matrix[:, active_start:active_end]
    medians = np.median(active, axis=1)
    deviations = np.median(
        np.abs(active - medians[:, None]),
        axis=1,
    )
    scales = 1.4826 * deviations
    scales = np.where(scales < 1e-8, 1.0, scales)

    return (
        matrix - medians[:, None]
    ) / scales[:, None]


def preprocess_traces(
    traces: np.ndarray,
    global_trigger_sample: int,
    config: Stage03Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    تولید دو نمایش:

    amplitude_preserving:
        baseline و روند خطی حذف می‌شود ولی gain هر trace حفظ می‌شود.

    alignment_signal:
        high-pass و robust normalization برای correlation زمانی.
    """

    traces = np.asarray(traces, dtype=np.float64)

    if traces.ndim != 2:
        raise ValueError("traces must be a 2-D array")

    number_of_traces, number_of_samples = traces.shape

    if number_of_traces < 8:
        raise ValueError("at least 8 traces are required")

    pre_end = max(8, min(
        global_trigger_sample - 4,
        number_of_samples // 8,
    ))

    if pre_end <= 4:
        raise ValueError("pretrigger region is too short")

    baseline = np.median(traces[:, :pre_end], axis=1)
    centered = traces - baseline[:, None]

    edge = min(
        config.edge_baseline_samples,
        max(8, number_of_samples // 16),
    )

    left_level = np.mean(centered[:, :edge], axis=1)
    right_level = np.mean(centered[:, -edge:], axis=1)

    normalized_axis = np.linspace(
        0.0,
        1.0,
        number_of_samples,
        dtype=np.float64,
    )

    linear_trend = (
        left_level[:, None]
        + (right_level - left_level)[:, None]
        * normalized_axis[None, :]
    )

    amplitude_preserving = centered - linear_trend

    local_mean = moving_average_matrix(
        amplitude_preserving,
        config.highpass_window_samples,
    )

    highpass = amplitude_preserving - local_mean

    active_start = max(0, global_trigger_sample - 12)
    active_end = max(
        active_start + 32,
        number_of_samples - edge,
    )

    alignment_signal = robust_scale_rows(
        highpass,
        active_start,
        active_end,
    )

    stats = {
        "number_of_traces": number_of_traces,
        "number_of_samples": number_of_samples,
        "pretrigger_end_sample": pre_end,
        "edge_baseline_samples": edge,
        "highpass_window_samples": (
            config.highpass_window_samples
        ),
        "amplitude_preserving_global_mean": float(
            np.mean(amplitude_preserving)
        ),
        "amplitude_preserving_global_std": float(
            np.std(amplitude_preserving)
        ),
        "alignment_signal_global_mean": float(
            np.mean(alignment_signal)
        ),
        "alignment_signal_global_std": float(
            np.std(alignment_signal)
        ),
    }

    return (
        amplitude_preserving.astype(np.float32),
        alignment_signal.astype(np.float32),
        stats,
    )


def best_shift_against_reference(
    trace: np.ndarray,
    reference: np.ndarray,
    region_start: int,
    region_end: int,
    maximum_shift: int,
) -> Tuple[int, float]:
    """جست‌وجوی integer shift با normalized cross-correlation."""

    best_shift = 0
    best_score = float("-inf")

    number_of_samples = trace.shape[0]

    for shift in range(-maximum_shift, maximum_shift + 1):
        source_start = region_start + shift
        source_end = region_end + shift

        if source_start < 0 or source_end > number_of_samples:
            continue

        score = normalized_correlation(
            reference[region_start:region_end],
            trace[source_start:source_end],
        )

        if score > best_score:
            best_score = score
            best_shift = shift

    return best_shift, best_score


def correlation_to_reference(
    matrix: np.ndarray,
    reference: np.ndarray,
    region_start: int,
    region_end: int,
) -> np.ndarray:
    scores = np.empty(matrix.shape[0], dtype=np.float64)

    for trace_index in range(matrix.shape[0]):
        scores[trace_index] = normalized_correlation(
            matrix[trace_index, region_start:region_end],
            reference[region_start:region_end],
        )

    return scores


def global_align(
    alignment_signal: np.ndarray,
    global_trigger_sample: int,
    config: Stage03Config,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    هم‌ترازی coarse در چند iteration.

    reference از median خود traceهای عمومی ساخته می‌شود.
    """

    current = np.asarray(
        alignment_signal,
        dtype=np.float64,
    ).copy()

    number_of_traces, number_of_samples = current.shape

    region_start = max(0, global_trigger_sample - 12)
    region_end = number_of_samples - max(
        config.edge_baseline_samples,
        24,
    )

    cumulative_shifts = np.zeros(
        number_of_traces,
        dtype=np.int32,
    )
    iteration_rows: List[Dict[str, Any]] = []

    initial_reference = np.median(current, axis=0)
    initial_correlations = correlation_to_reference(
        current,
        initial_reference,
        region_start,
        region_end,
    )

    for iteration_index in range(
        config.global_alignment_iterations
    ):
        reference = np.median(current, axis=0)
        residual_shifts = np.zeros(
            number_of_traces,
            dtype=np.int32,
        )
        residual_scores = np.zeros(
            number_of_traces,
            dtype=np.float64,
        )

        for trace_index in range(number_of_traces):
            shift, score = best_shift_against_reference(
                current[trace_index],
                reference,
                region_start,
                region_end,
                config.maximum_global_shift_samples,
            )
            residual_shifts[trace_index] = shift
            residual_scores[trace_index] = score

        for trace_index in range(number_of_traces):
            current[trace_index] = apply_integer_shift(
                current[trace_index],
                int(residual_shifts[trace_index]),
            )

        cumulative_shifts += residual_shifts

        iteration_rows.append({
            "iteration": iteration_index + 1,
            "median_residual_shift": float(
                np.median(residual_shifts)
            ),
            "maximum_absolute_residual_shift": int(
                np.max(np.abs(residual_shifts))
            ),
            "median_alignment_correlation": float(
                np.median(residual_scores)
            ),
            "minimum_alignment_correlation": float(
                np.min(residual_scores)
            ),
        })

    final_reference = np.median(current, axis=0)
    final_correlations = correlation_to_reference(
        current,
        final_reference,
        region_start,
        region_end,
    )

    diagnostics = {
        "region_start": region_start,
        "region_end": region_end,
        "iterations": iteration_rows,
        "initial_median_correlation": float(
            np.median(initial_correlations)
        ),
        "final_median_correlation": float(
            np.median(final_correlations)
        ),
        "initial_minimum_correlation": float(
            np.min(initial_correlations)
        ),
        "final_minimum_correlation": float(
            np.min(final_correlations)
        ),
        "global_shift_minimum": int(
            np.min(cumulative_shifts)
        ),
        "global_shift_maximum": int(
            np.max(cumulative_shifts)
        ),
        "global_shift_median": float(
            np.median(cumulative_shifts)
        ),
    }

    return (
        current.astype(np.float32),
        cumulative_shifts,
        diagnostics,
    )


def choose_anchor_centers(
    number_of_samples: int,
    global_trigger_sample: int,
    window_samples: int,
    anchor_count: int,
    edge_guard: int,
) -> np.ndarray:
    half_window = window_samples // 2

    first = max(
        global_trigger_sample + half_window + 8,
        half_window + edge_guard,
    )
    last = min(
        number_of_samples - half_window - edge_guard - 1,
        number_of_samples - half_window - 1,
    )

    if last <= first:
        raise ValueError("trace is too short for local anchors")

    return np.linspace(
        first,
        last,
        anchor_count,
        dtype=np.int32,
    )


def estimate_local_shift(
    trace: np.ndarray,
    reference: np.ndarray,
    center: int,
    window_samples: int,
    maximum_shift: int,
) -> Tuple[int, float]:
    half_window = window_samples // 2
    reference_start = center - half_window
    reference_end = center + half_window

    best_shift = 0
    best_score = float("-inf")

    for shift in range(-maximum_shift, maximum_shift + 1):
        source_start = reference_start + shift
        source_end = reference_end + shift

        if source_start < 0 or source_end > trace.shape[0]:
            continue

        score = normalized_correlation(
            reference[reference_start:reference_end],
            trace[source_start:source_end],
        )

        if score > best_score:
            best_score = score
            best_shift = shift

    return best_shift, best_score


def robust_weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    fit خطی y = intercept + slope*x با یک مرحله حذف outlier.

    خروجی سوم mask نقاط پذیرفته‌شده است.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(weights)
        & (weights > 0.0)
    )

    if np.sum(valid) < 2:
        return 0.0, 0.0, valid

    def solve(mask: np.ndarray) -> Tuple[float, float]:
        design = np.column_stack([
            np.ones(np.sum(mask), dtype=np.float64),
            x[mask],
        ])
        sqrt_w = np.sqrt(weights[mask])[:, None]
        weighted_design = design * sqrt_w
        weighted_target = y[mask] * sqrt_w[:, 0]

        coefficients, _, _, _ = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )

        return float(coefficients[0]), float(coefficients[1])

    intercept, slope = solve(valid)
    residuals = y - (intercept + slope * x)

    valid_residuals = residuals[valid]
    residual_median = np.median(valid_residuals)
    residual_mad = 1.4826 * np.median(
        np.abs(valid_residuals - residual_median)
    )

    if residual_mad > 1e-8:
        refined = valid & (
            np.abs(residuals - residual_median)
            <= max(2.5 * residual_mad, 1.25)
        )

        if np.sum(refined) >= 2:
            intercept, slope = solve(refined)
            valid = refined

    return intercept, slope, valid


def local_affine_align(
    globally_aligned_signal: np.ndarray,
    globally_aligned_amplitude: np.ndarray,
    global_trigger_sample: int,
    config: Stage03Config,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    """
    تخمین residual shift در چند نقطه و fit یک affine warp.

    این مرحله clock-scale drift کوچک و drift تجمعی را جبران می‌کند.
    """

    signal = np.asarray(
        globally_aligned_signal,
        dtype=np.float64,
    )
    amplitude = np.asarray(
        globally_aligned_amplitude,
        dtype=np.float64,
    )

    number_of_traces, number_of_samples = signal.shape
    reference = np.median(signal, axis=0)

    anchors = choose_anchor_centers(
        number_of_samples,
        global_trigger_sample,
        config.local_anchor_window_samples,
        config.number_of_local_anchors,
        config.edge_baseline_samples,
    )

    local_shifts = np.zeros(
        (number_of_traces, anchors.shape[0]),
        dtype=np.float64,
    )
    local_scores = np.zeros_like(local_shifts)

    intercepts = np.zeros(number_of_traces, dtype=np.float64)
    slopes = np.zeros(number_of_traces, dtype=np.float64)
    accepted_anchor_counts = np.zeros(
        number_of_traces,
        dtype=np.int32,
    )

    aligned_signal = np.empty_like(signal)
    aligned_amplitude = np.empty_like(amplitude)

    centered_anchor_positions = (
        anchors.astype(np.float64)
        - float(global_trigger_sample)
    )

    for trace_index in range(number_of_traces):
        for anchor_index, anchor in enumerate(anchors):
            shift, score = estimate_local_shift(
                signal[trace_index],
                reference,
                int(anchor),
                config.local_anchor_window_samples,
                config.maximum_local_shift_samples,
            )

            local_shifts[trace_index, anchor_index] = shift
            local_scores[trace_index, anchor_index] = score

        weights = np.clip(
            local_scores[trace_index]
            - config.minimum_anchor_correlation,
            0.0,
            None,
        ) ** 2

        # برای جلوگیری از fit تهی در traceهای سخت، یک وزن بسیار کوچک
        # به همه anchorها داده می‌شود.
        weights += 1e-4

        intercept, slope, accepted = robust_weighted_linear_fit(
            centered_anchor_positions,
            local_shifts[trace_index],
            weights,
        )

        # محدودیت محافظه‌کارانه برای جلوگیری از warp غیرواقعی.
        intercept = float(np.clip(
            intercept,
            -config.maximum_local_shift_samples,
            config.maximum_local_shift_samples,
        ))

        maximum_slope = (
            2.0
            * config.maximum_local_shift_samples
            / max(number_of_samples, 1)
        )
        slope = float(np.clip(
            slope,
            -maximum_slope,
            maximum_slope,
        ))

        intercepts[trace_index] = intercept
        slopes[trace_index] = slope
        accepted_anchor_counts[trace_index] = int(
            np.sum(accepted)
        )

        aligned_signal[trace_index] = (
            apply_affine_residual_warp(
                signal[trace_index],
                intercept,
                slope,
                float(global_trigger_sample),
            )
        )

        aligned_amplitude[trace_index] = (
            apply_affine_residual_warp(
                amplitude[trace_index],
                intercept,
                slope,
                float(global_trigger_sample),
            )
        )

    final_reference = np.median(aligned_signal, axis=0)

    active_start = max(0, global_trigger_sample - 12)
    active_end = number_of_samples - max(
        config.edge_baseline_samples,
        24,
    )

    before_correlations = correlation_to_reference(
        signal,
        reference,
        active_start,
        active_end,
    )
    after_correlations = correlation_to_reference(
        aligned_signal,
        final_reference,
        active_start,
        active_end,
    )

    # residual shift بعد از warp برای کنترل علمی alignment.
    residual_after = np.zeros_like(local_shifts)

    for trace_index in range(number_of_traces):
        for anchor_index, anchor in enumerate(anchors):
            shift, _ = estimate_local_shift(
                aligned_signal[trace_index],
                final_reference,
                int(anchor),
                config.local_anchor_window_samples,
                config.maximum_local_shift_samples,
            )
            residual_after[trace_index, anchor_index] = shift

    diagnostics = {
        "anchor_centers": anchors.tolist(),
        "median_absolute_local_shift_before": float(
            np.median(np.abs(local_shifts))
        ),
        "median_absolute_local_shift_after": float(
            np.median(np.abs(residual_after))
        ),
        "maximum_absolute_local_shift_before": float(
            np.max(np.abs(local_shifts))
        ),
        "maximum_absolute_local_shift_after": float(
            np.max(np.abs(residual_after))
        ),
        "median_trace_correlation_before_affine": float(
            np.median(before_correlations)
        ),
        "median_trace_correlation_after_affine": float(
            np.median(after_correlations)
        ),
        "minimum_trace_correlation_after_affine": float(
            np.min(after_correlations)
        ),
        "affine_intercept_minimum": float(
            np.min(intercepts)
        ),
        "affine_intercept_maximum": float(
            np.max(intercepts)
        ),
        "affine_slope_minimum": float(
            np.min(slopes)
        ),
        "affine_slope_maximum": float(
            np.max(slopes)
        ),
        "affine_slope_ppm_median": float(
            np.median(slopes) * 1e6
        ),
        "accepted_anchor_count_minimum": int(
            np.min(accepted_anchor_counts)
        ),
        "accepted_anchor_count_median": float(
            np.median(accepted_anchor_counts)
        ),
    }

    return (
        aligned_signal.astype(np.float32),
        aligned_amplitude.astype(np.float32),
        intercepts,
        slopes,
        diagnostics,
    )


def compute_activity_profile(
    aligned_signal: np.ndarray,
) -> np.ndarray:
    """
    profile فعالیت زمانی از ترکیب:
      - MAD مشتق traceها
      - standard deviation دامنه در بین traceها

    استفاده از چند trace باعث می‌شود data-dependent leakage برجسته شود.
    """

    signal = np.asarray(aligned_signal, dtype=np.float64)
    derivative = np.diff(
        signal,
        axis=1,
        prepend=signal[:, :1],
    )

    derivative_median = np.median(
        derivative,
        axis=0,
    )
    derivative_mad = (
        1.4826
        * np.median(
            np.abs(
                derivative - derivative_median[None, :]
            ),
            axis=0,
        )
    )

    across_trace_std = np.std(signal, axis=0)

    derivative_scale = np.median(derivative_mad)
    std_scale = np.median(across_trace_std)

    if derivative_scale <= 1e-12:
        derivative_scale = 1.0
    if std_scale <= 1e-12:
        std_scale = 1.0

    activity = (
        0.65 * derivative_mad / derivative_scale
        + 0.35 * across_trace_std / std_scale
    )

    activity = moving_average_1d(activity, 5)
    activity -= np.min(activity)

    maximum = np.max(activity)

    if maximum > 0:
        activity /= maximum

    return activity.astype(np.float64)


def estimate_round_period(
    activity: np.ndarray,
    global_trigger_sample: int,
    number_of_rounds: int,
    config: Stage03Config,
) -> Tuple[int, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    تخمین period دور با autocorrelation profile فعالیت.

    محدوده جست‌وجو از طول trace و تعداد دورها ساخته می‌شود؛
    مقدار timing پنهان simulator استفاده نمی‌شود.
    """

    number_of_samples = activity.shape[0]

    usable_length = (
        number_of_samples - global_trigger_sample
    )
    expected_period = usable_length / number_of_rounds

    lower = max(
        24,
        int(math.floor(
            expected_period
            * config.period_search_lower_factor
        )),
    )
    upper = min(
        number_of_samples // 3,
        int(math.ceil(
            expected_period
            * config.period_search_upper_factor
        )),
    )

    if upper <= lower:
        raise RuntimeError("invalid round-period search range")

    search_start = max(0, global_trigger_sample - 8)
    search_end = number_of_samples - 16
    active = activity[search_start:search_end]
    active = active - np.mean(active)

    lags = np.arange(lower, upper + 1, dtype=np.int32)
    scores = np.empty(lags.shape[0], dtype=np.float64)

    for index, lag in enumerate(lags):
        scores[index] = normalized_correlation(
            active[:-lag],
            active[lag:],
        )

    best_index = int(np.argmax(scores))
    estimated_period = int(lags[best_index])

    diagnostics = {
        "expected_period_from_trace_length": float(
            expected_period
        ),
        "search_lower_sample": int(lower),
        "search_upper_sample": int(upper),
        "estimated_period_samples": estimated_period,
        "best_autocorrelation": float(
            scores[best_index]
        ),
        "second_best_autocorrelation": float(
            np.partition(scores, -2)[-2]
            if scores.shape[0] >= 2
            else scores[best_index]
        ),
    }

    return estimated_period, lags, scores, diagnostics


def round_segments(
    activity: np.ndarray,
    round_start: int,
    period: int,
    number_of_rounds: int,
) -> Optional[np.ndarray]:
    segments = []

    for round_index in range(number_of_rounds):
        start = round_start + round_index * period
        end = start + period

        if start < 0 or end > activity.shape[0]:
            return None

        segments.append(activity[start:end])

    return np.stack(segments, axis=0)


def estimate_round_start(
    activity: np.ndarray,
    global_trigger_sample: int,
    period: int,
    number_of_rounds: int,
    config: Stage03Config,
) -> Tuple[int, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    تعیین phase دورها.

    score ترکیبی:
      - شباهت شکل 32 دور
      - کم بودن فعالیت در مرز بین دورها

    global trigger تنها مرجع عمومی زمانی است.
    """

    candidates = np.arange(
        max(
            0,
            global_trigger_sample
            - config.round_start_search_radius_samples,
        ),
        min(
            activity.shape[0] - number_of_rounds * period,
            global_trigger_sample
            + config.round_start_search_radius_samples,
        ) + 1,
        dtype=np.int32,
    )

    if candidates.size == 0:
        raise RuntimeError("no valid round-start candidate")

    scores = np.full(
        candidates.shape[0],
        -np.inf,
        dtype=np.float64,
    )

    consistency_values = np.zeros_like(scores)
    boundary_penalties = np.zeros_like(scores)

    activity_mean = max(float(np.mean(activity)), 1e-12)

    for candidate_index, candidate in enumerate(candidates):
        segments = round_segments(
            activity,
            int(candidate),
            period,
            number_of_rounds,
        )

        if segments is None:
            continue

        template = np.median(segments, axis=0)
        correlations = np.asarray([
            normalized_correlation(segment, template)
            for segment in segments
        ])

        finite_correlations = correlations[
            np.isfinite(correlations)
        ]

        if finite_correlations.size == 0:
            continue

        consistency = float(
            np.mean(finite_correlations)
        )

        boundary_samples = []

        for boundary_index in range(number_of_rounds + 1):
            center = int(candidate) + boundary_index * period
            left = max(0, center - 2)
            right = min(activity.shape[0], center + 3)

            if right > left:
                boundary_samples.append(
                    float(np.mean(activity[left:right]))
                )

        boundary_penalty = (
            float(np.mean(boundary_samples))
            / activity_mean
        )

        score = consistency - 0.08 * boundary_penalty

        scores[candidate_index] = score
        consistency_values[candidate_index] = consistency
        boundary_penalties[candidate_index] = boundary_penalty

    best_index = int(np.argmax(scores))
    estimated_start = int(candidates[best_index])

    diagnostics = {
        "estimated_round_1_start_sample": estimated_start,
        "best_phase_score": float(scores[best_index]),
        "best_consistency": float(
            consistency_values[best_index]
        ),
        "best_boundary_penalty": float(
            boundary_penalties[best_index]
        ),
        "candidate_minimum": int(candidates[0]),
        "candidate_maximum": int(candidates[-1]),
    }

    return estimated_start, candidates, scores, diagnostics


def refine_final_round_start(
    activity: np.ndarray,
    estimated_round_start: int,
    period: int,
    number_of_rounds: int,
    config: Stage03Config,
) -> Tuple[int, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    refinement مستقل دور آخر با template دورهای 1 تا 31.
    """

    previous_segments = []

    for round_index in range(number_of_rounds - 1):
        start = estimated_round_start + round_index * period
        end = start + period

        if start < 0 or end > activity.shape[0]:
            continue

        previous_segments.append(activity[start:end])

    if not previous_segments:
        raise RuntimeError(
            "could not build previous-round template"
        )

    template = np.median(
        np.stack(previous_segments, axis=0),
        axis=0,
    )

    coarse_start = (
        estimated_round_start
        + (number_of_rounds - 1) * period
    )

    candidates = np.arange(
        coarse_start
        - config.final_round_refinement_radius_samples,
        coarse_start
        + config.final_round_refinement_radius_samples
        + 1,
        dtype=np.int32,
    )

    scores = np.full(
        candidates.shape[0],
        -np.inf,
        dtype=np.float64,
    )

    for candidate_index, candidate in enumerate(candidates):
        end = int(candidate) + period

        if candidate < 0 or end > activity.shape[0]:
            continue

        scores[candidate_index] = normalized_correlation(
            activity[int(candidate):end],
            template,
        )

    best_index = int(np.argmax(scores))
    refined_start = int(candidates[best_index])

    diagnostics = {
        "coarse_final_round_start_sample": int(
            coarse_start
        ),
        "refined_final_round_start_sample": refined_start,
        "refinement_offset_samples": int(
            refined_start - coarse_start
        ),
        "best_template_correlation": float(
            scores[best_index]
        ),
    }

    return refined_start, candidates, scores, diagnostics


def locate_final_round_roi(
    aligned_signal: np.ndarray,
    global_trigger_sample: int,
    sampling_rate_hz: float,
    config: Stage03Config,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    activity = compute_activity_profile(aligned_signal)

    (
        period,
        period_lags,
        period_scores,
        period_diagnostics,
    ) = estimate_round_period(
        activity,
        global_trigger_sample,
        config.number_of_rounds,
        config,
    )

    (
        round_start,
        phase_candidates,
        phase_scores,
        phase_diagnostics,
    ) = estimate_round_start(
        activity,
        global_trigger_sample,
        period,
        config.number_of_rounds,
        config,
    )

    (
        refined_final_start,
        refinement_candidates,
        refinement_scores,
        refinement_diagnostics,
    ) = refine_final_round_start(
        activity,
        round_start,
        period,
        config.number_of_rounds,
        config,
    )

    number_of_samples = aligned_signal.shape[1]

    roi_start = max(
        0,
        refined_final_start
        - config.final_round_guard_samples,
    )
    roi_end = min(
        number_of_samples,
        refined_final_start
        + period
        + config.final_round_guard_samples,
    )

    if roi_end <= roi_start:
        raise RuntimeError("invalid final-round ROI")

    sample_period_seconds = 1.0 / sampling_rate_hz

    roi = {
        "algorithm": "LBlock-64/80",
        "locator_stage": 3,
        "estimation_source": (
            "public healthy traces only"
        ),
        "number_of_rounds": config.number_of_rounds,
        "global_trigger_sample": int(
            global_trigger_sample
        ),
        "estimated_round_period_samples": int(period),
        "estimated_round_period_ns": float(
            period * sample_period_seconds * 1e9
        ),
        "estimated_round_1_start_sample": int(
            round_start
        ),
        "estimated_round_32_start_sample": int(
            refined_final_start
        ),
        "roi_start_sample_inclusive": int(roi_start),
        "roi_end_sample_exclusive": int(roi_end),
        "roi_width_samples": int(roi_end - roi_start),
        "roi_start_time_us": float(
            roi_start * sample_period_seconds * 1e6
        ),
        "roi_end_time_us": float(
            roi_end * sample_period_seconds * 1e6
        ),
        "guard_samples": int(
            config.final_round_guard_samples
        ),
        "warning": (
            "No S-box-specific timing has been estimated in Stage 03. "
            "Stage 04 must map S0..S7 inside this ROI."
        ),
    }

    arrays = {
        "activity_profile": activity,
        "period_lags": period_lags,
        "period_scores": period_scores,
        "phase_candidates": phase_candidates,
        "phase_scores": phase_scores,
        "refinement_candidates": refinement_candidates,
        "refinement_scores": refinement_scores,
    }

    diagnostics = {
        "period": period_diagnostics,
        "round_phase": phase_diagnostics,
        "final_round_refinement": (
            refinement_diagnostics
        ),
    }

    return roi, arrays, diagnostics


def build_alignment_metadata(
    public_metadata_rows: Sequence[Mapping[str, str]],
    global_shifts: np.ndarray,
    affine_intercepts: np.ndarray,
    affine_slopes: np.ndarray,
    pre_correlations: np.ndarray,
    post_correlations: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for index, source in enumerate(public_metadata_rows):
        row = dict(source)
        row.update({
            "global_alignment_shift_samples": int(
                global_shifts[index]
            ),
            "affine_residual_intercept_samples": float(
                affine_intercepts[index]
            ),
            "affine_residual_slope_samples_per_sample": float(
                affine_slopes[index]
            ),
            "affine_residual_slope_ppm": float(
                affine_slopes[index] * 1e6
            ),
            "reference_correlation_before": float(
                pre_correlations[index]
            ),
            "reference_correlation_after": float(
                post_correlations[index]
            ),
        })
        rows.append(row)

    return rows


def save_public_plots(
    public_directory: Path,
    original_signal: np.ndarray,
    aligned_signal: np.ndarray,
    activity_arrays: Mapping[str, np.ndarray],
    final_round_roi: Mapping[str, Any],
    config: Stage03Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []
    number_of_samples = aligned_signal.shape[1]
    axis = np.arange(number_of_samples)

    # --------------------------------------------------------
    # overlay before/after
    # --------------------------------------------------------
    selected_count = min(24, aligned_signal.shape[0])
    selected_indices = np.linspace(
        0,
        aligned_signal.shape[0] - 1,
        selected_count,
        dtype=np.int32,
    )

    fig = plt.figure(figsize=(13, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    for index in selected_indices:
        ax1.plot(
            axis,
            original_signal[index],
            linewidth=0.45,
            alpha=0.28,
        )
    ax1.set_title(
        "Before alignment — public traces"
    )
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Normalized alignment signal")
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(2, 1, 2)
    for index in selected_indices:
        ax2.plot(
            axis,
            aligned_signal[index],
            linewidth=0.45,
            alpha=0.28,
        )
    ax2.set_title(
        "After global + affine alignment"
    )
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Normalized alignment signal")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    path = public_directory / "alignment_before_after.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # --------------------------------------------------------
    # activity and final ROI
    # --------------------------------------------------------
    activity = activity_arrays["activity_profile"]
    roi_start = int(
        final_round_roi["roi_start_sample_inclusive"]
    )
    roi_end = int(
        final_round_roi["roi_end_sample_exclusive"]
    )
    final_start = int(
        final_round_roi[
            "estimated_round_32_start_sample"
        ]
    )

    fig = plt.figure(figsize=(13, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(axis, activity, linewidth=1.0)
    ax.axvline(
        final_start,
        linestyle="--",
        linewidth=1.0,
        label="Estimated round-32 start",
    )
    ax.axvspan(
        roi_start,
        roi_end,
        alpha=0.2,
        label="Final-round ROI",
    )
    ax.set_title(
        "Public-only activity profile and final-round ROI"
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Normalized activity")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    path = public_directory / "final_round_roi_locator.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    # --------------------------------------------------------
    # period autocorrelation
    # --------------------------------------------------------
    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        activity_arrays["period_lags"],
        activity_arrays["period_scores"],
        linewidth=1.1,
    )
    estimated_period = int(
        final_round_roi["estimated_round_period_samples"]
    )
    ax.axvline(
        estimated_period,
        linestyle="--",
        linewidth=1.0,
        label=f"Selected period = {estimated_period}",
    )
    ax.set_title(
        "Round-period estimation from activity autocorrelation"
    )
    ax.set_xlabel("Lag [samples]")
    ax.set_ylabel("Normalized correlation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    path = public_directory / "round_period_autocorrelation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    return generated


def transformed_hidden_center(
    original_center: np.ndarray,
    global_shift: np.ndarray,
    intercept: np.ndarray,
    slope: np.ndarray,
    anchor_sample: float,
) -> np.ndarray:
    """
    تبدیل center مخفی از محور trace اصلی به محور trace aligned.

    pipeline:
      1) global alignment با source(y)=y+g
      2) affine residual با source(z)=z+a+b(z-anchor)

    بنابراین:
      original = aligned + g + a + b(aligned-anchor)
    """

    denominator = 1.0 + slope

    return (
        original_center
        - global_shift
        - intercept
        + slope * anchor_sample
    ) / denominator


def evaluate_frozen_estimate_against_private(
    stage2_run_directory: Path,
    validation_directory: Path,
    final_round_roi: Mapping[str, Any],
    global_shifts: np.ndarray,
    affine_intercepts: np.ndarray,
    affine_slopes: np.ndarray,
    global_trigger_sample: int,
    estimation_freeze_sha256: str,
) -> Dict[str, Any]:
    """
    ارزیابی validation-only.

    این تابع بعد از freeze شدن خروجی public فراخوانی می‌شود.
    هیچ مقدار private به locator بازگردانده نمی‌شود.
    """

    private_file = (
        stage2_run_directory
        / "private_ground_truth"
        / "hidden_timing_and_crypto_ground_truth.npz"
    )

    if not private_file.is_file():
        return {
            "available": False,
            "reason": (
                "hidden_timing_and_crypto_ground_truth.npz "
                "was not found"
            ),
            "estimation_freeze_sha256": (
                estimation_freeze_sha256
            ),
        }

    with np.load(private_file, allow_pickle=False) as hidden:
        sbox_centers = np.asarray(
            hidden["sbox_centers"],
            dtype=np.float64,
        )

    if sbox_centers.ndim != 3:
        raise ValueError(
            "private sbox_centers must have shape "
            "(trace, round, sbox)"
        )

    if sbox_centers.shape[0] != global_shifts.shape[0]:
        raise ValueError(
            "private/public trace count mismatch"
        )

    final_centers_original = sbox_centers[:, -1, :]

    transformed_centers = transformed_hidden_center(
        final_centers_original,
        global_shifts[:, None],
        affine_intercepts[:, None],
        affine_slopes[:, None],
        float(global_trigger_sample),
    )

    roi_start = float(
        final_round_roi["roi_start_sample_inclusive"]
    )
    roi_end = float(
        final_round_roi["roi_end_sample_exclusive"]
    )

    inside = (
        (transformed_centers >= roi_start)
        & (transformed_centers < roi_end)
    )

    per_trace_full_coverage = np.all(inside, axis=1)

    original_std_per_sbox = np.std(
        final_centers_original,
        axis=0,
    )
    aligned_std_per_sbox = np.std(
        transformed_centers,
        axis=0,
    )

    # دوره واقعی از فاصله center یک S-box در دورهای متوالی،
    # بدون استفاده در estimate عمومی.
    hidden_periods = np.diff(
        sbox_centers,
        axis=1,
    )
    hidden_period_median = float(
        np.median(hidden_periods)
    )
    estimated_period = float(
        final_round_roi["estimated_round_period_samples"]
    )

    evaluation = {
        "available": True,
        "warning": (
            "Validation only. These values were computed "
            "after the public estimate was frozen."
        ),
        "estimation_freeze_sha256": estimation_freeze_sha256,
        "private_file_sha256": sha256_file(private_file),
        "all_sbox_center_coverage_rate": float(
            np.mean(inside)
        ),
        "full_final_round_coverage_rate_per_trace": float(
            np.mean(per_trace_full_coverage)
        ),
        "minimum_margin_to_roi_start_samples": float(
            np.min(transformed_centers - roi_start)
        ),
        "minimum_margin_to_roi_end_samples": float(
            np.min(roi_end - transformed_centers)
        ),
        "hidden_median_round_period_samples": (
            hidden_period_median
        ),
        "absolute_round_period_error_samples": float(
            abs(estimated_period - hidden_period_median)
        ),
        "original_final_center_std_per_sbox": (
            original_std_per_sbox.tolist()
        ),
        "aligned_final_center_std_per_sbox": (
            aligned_std_per_sbox.tolist()
        ),
        "median_original_final_center_std": float(
            np.median(original_std_per_sbox)
        ),
        "median_aligned_final_center_std": float(
            np.median(aligned_std_per_sbox)
        ),
        "median_center_std_reduction_fraction": float(
            1.0
            - np.median(aligned_std_per_sbox)
            / max(
                np.median(original_std_per_sbox),
                1e-12,
            )
        ),
    }

    validation_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        validation_directory
        / "private_timing_evaluation.json",
        evaluation,
    )

    if plt is not None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)

        for sbox_index in range(8):
            ax.scatter(
                np.full(
                    transformed_centers.shape[0],
                    sbox_index,
                    dtype=np.float64,
                ),
                transformed_centers[:, sbox_index],
                s=6,
                alpha=0.22,
            )

        ax.axhspan(
            roi_start,
            roi_end,
            alpha=0.15,
            label="Estimated final-round ROI",
        )
        ax.set_title(
            "Validation only — aligned hidden final-round centers"
        )
        ax.set_xlabel("S-box index")
        ax.set_ylabel("Aligned sample")
        ax.set_xticks(range(8))
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            validation_directory
            / "private_final_center_validation.png",
            dpi=180,
        )
        plt.close(fig)

    return evaluation


def validate_stage_03_public_outputs(
    original_traces: np.ndarray,
    aligned_traces: np.ndarray,
    trace_ids: np.ndarray,
    metadata_rows: Sequence[Mapping[str, str]],
    global_shifts: np.ndarray,
    affine_intercepts: np.ndarray,
    affine_slopes: np.ndarray,
    final_round_roi: Mapping[str, Any],
    global_diagnostics: Mapping[str, Any],
    affine_diagnostics: Mapping[str, Any],
    config: Stage03Config,
) -> Dict[str, Any]:
    number_of_traces, number_of_samples = original_traces.shape

    checks: Dict[str, Any] = {}

    checks["shape_preserved"] = {
        "passed": (
            aligned_traces.shape
            == original_traces.shape
        ),
        "original_shape": list(original_traces.shape),
        "aligned_shape": list(aligned_traces.shape),
    }

    checks["finite_values"] = {
        "passed": bool(
            np.all(np.isfinite(aligned_traces))
        ),
    }

    checks["trace_count_consistency"] = {
        "passed": bool(
            trace_ids.shape[0] == number_of_traces
            and len(metadata_rows) == number_of_traces
            and global_shifts.shape[0] == number_of_traces
            and affine_intercepts.shape[0] == number_of_traces
            and affine_slopes.shape[0] == number_of_traces
        ),
    }

    roi_start = int(
        final_round_roi["roi_start_sample_inclusive"]
    )
    roi_end = int(
        final_round_roi["roi_end_sample_exclusive"]
    )
    period = int(
        final_round_roi["estimated_round_period_samples"]
    )

    checks["final_round_roi_bounds"] = {
        "passed": bool(
            0 <= roi_start < roi_end <= number_of_samples
        ),
        "roi_start": roi_start,
        "roi_end": roi_end,
    }

    checks["roi_width_plausibility"] = {
        "passed": bool(
            period
            <= (roi_end - roi_start)
            <= period + 2 * config.final_round_guard_samples + 2
        ),
        "period": period,
        "roi_width": roi_end - roi_start,
    }

    expected_period = (
        number_of_samples
        - int(final_round_roi["global_trigger_sample"])
    ) / config.number_of_rounds

    checks["round_period_plausibility"] = {
        "passed": bool(
            config.period_search_lower_factor
            * expected_period
            <= period
            <= config.period_search_upper_factor
            * expected_period
        ),
        "estimated_period": period,
        "expected_from_length": expected_period,
    }

    checks["global_shift_bounds"] = {
        "passed": bool(
            np.max(np.abs(global_shifts))
            <= (
                config.maximum_global_shift_samples
                * config.global_alignment_iterations
            )
        ),
        "minimum": int(np.min(global_shifts)),
        "maximum": int(np.max(global_shifts)),
    }

    checks["alignment_correlation_not_degraded"] = {
        "passed": bool(
            affine_diagnostics[
                "median_trace_correlation_after_affine"
            ]
            + 1e-6
            >= global_diagnostics[
                "initial_median_correlation"
            ]
        ),
        "before": float(
            global_diagnostics[
                "initial_median_correlation"
            ]
        ),
        "after": float(
            affine_diagnostics[
                "median_trace_correlation_after_affine"
            ]
        ),
    }

    checks["local_residual_not_increased"] = {
        "passed": bool(
            affine_diagnostics[
                "median_absolute_local_shift_after"
            ]
            <= affine_diagnostics[
                "median_absolute_local_shift_before"
            ]
            + 0.25
        ),
        "before": float(
            affine_diagnostics[
                "median_absolute_local_shift_before"
            ]
        ),
        "after": float(
            affine_diagnostics[
                "median_absolute_local_shift_after"
            ]
        ),
    }

    checks["public_only_input_contract"] = {
        "passed": True,
        "statement": (
            "The estimation pipeline received only Stage 02 "
            "public files. Private evaluation is a separate "
            "post-freeze function."
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


def run_stage_03(
    config: Stage03Config,
) -> Dict[str, Any]:
    start_time = time.perf_counter()

    stage2_run_directory = Path(
        config.input_stage2_run_directory
    ).expanduser().resolve()

    stage2_summary_path = (
        stage2_run_directory / "stage_02_summary.json"
    )
    public_input_directory = (
        stage2_run_directory / "public"
    )

    if not stage2_summary_path.is_file():
        raise FileNotFoundError(
            f"Stage 02 summary not found: "
            f"{stage2_summary_path}"
        )

    stage2_summary = read_json(stage2_summary_path)

    if not stage2_summary.get(
        "all_checks_passed",
        False,
    ):
        raise RuntimeError(
            "Stage 02 did not pass all checks"
        )

    traces_path = (
        public_input_directory / "healthy_traces.npz"
    )
    metadata_path = (
        public_input_directory
        / "healthy_trace_metadata.csv"
    )
    simulation_config_path = (
        public_input_directory
        / "trace_simulation_config.json"
    )

    required_public_inputs = [
        traces_path,
        metadata_path,
        simulation_config_path,
    ]

    missing_inputs = [
        str(path)
        for path in required_public_inputs
        if not path.is_file()
    ]

    if missing_inputs:
        raise FileNotFoundError(
            "Missing Stage 02 public files:\n  - "
            + "\n  - ".join(missing_inputs)
        )

    # ساخت run directory مستقل
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    run_id = (
        f"stage03_{timestamp}_seed{config.random_seed}"
    )
    run_directory = (
        Path(config.output_root).expanduser().resolve()
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
    # بارگذاری فقط public inputs
    # --------------------------------------------------------
    with np.load(traces_path, allow_pickle=False) as data:
        original_traces = np.asarray(
            data["traces"],
            dtype=np.float32,
        )
        trace_ids = np.asarray(
            data["trace_ids"],
            dtype=np.int32,
        )
        sample_axis_seconds = np.asarray(
            data["sample_axis_seconds"],
            dtype=np.float64,
        )
        global_trigger_sample = int(
            np.asarray(
                data["global_trigger_sample"]
            ).item()
        )
        sampling_rate_hz = float(
            np.asarray(
                data["sampling_rate_hz"]
            ).item()
        )

    metadata_rows = read_csv(metadata_path)
    simulation_config = read_json(
        simulation_config_path
    )

    if original_traces.ndim != 2:
        raise ValueError(
            "healthy traces must be a 2-D matrix"
        )

    if trace_ids.shape[0] != original_traces.shape[0]:
        raise ValueError(
            "trace_ids length does not match traces"
        )

    if len(metadata_rows) != original_traces.shape[0]:
        raise ValueError(
            "metadata row count does not match traces"
        )

    # --------------------------------------------------------
    # preprocessing
    # --------------------------------------------------------
    (
        amplitude_preserving,
        alignment_signal,
        preprocessing_stats,
    ) = preprocess_traces(
        original_traces,
        global_trigger_sample,
        config,
    )

    # reference correlation قبل از alignment
    active_start = max(
        0,
        global_trigger_sample - 12,
    )
    active_end = original_traces.shape[1] - max(
        config.edge_baseline_samples,
        24,
    )
    initial_reference = np.median(
        alignment_signal,
        axis=0,
    )
    pre_correlations = correlation_to_reference(
        alignment_signal,
        initial_reference,
        active_start,
        active_end,
    )

    # --------------------------------------------------------
    # global alignment
    # --------------------------------------------------------
    (
        globally_aligned_signal,
        global_shifts,
        global_diagnostics,
    ) = global_align(
        alignment_signal,
        global_trigger_sample,
        config,
    )

    globally_aligned_amplitude = np.empty_like(
        amplitude_preserving,
        dtype=np.float32,
    )

    for trace_index in range(
        amplitude_preserving.shape[0]
    ):
        globally_aligned_amplitude[trace_index] = (
            apply_integer_shift(
                amplitude_preserving[trace_index],
                int(global_shifts[trace_index]),
            )
        )

    # --------------------------------------------------------
    # local affine correction
    # --------------------------------------------------------
    (
        aligned_signal,
        aligned_amplitude,
        affine_intercepts,
        affine_slopes,
        affine_diagnostics,
    ) = local_affine_align(
        globally_aligned_signal,
        globally_aligned_amplitude,
        global_trigger_sample,
        config,
    )

    final_reference = np.median(
        aligned_signal,
        axis=0,
    )
    post_correlations = correlation_to_reference(
        aligned_signal,
        final_reference,
        active_start,
        active_end,
    )

    # --------------------------------------------------------
    # final-round locator — public only
    # --------------------------------------------------------
    (
        final_round_roi,
        activity_arrays,
        locator_diagnostics,
    ) = locate_final_round_roi(
        aligned_signal,
        global_trigger_sample,
        sampling_rate_hz,
        config,
    )

    # --------------------------------------------------------
    # ذخیره خروجی‌های public
    # --------------------------------------------------------
    write_json(
        public_output_directory
        / "stage_03_config.json",
        asdict(config),
    )

    access_manifest = {
        "estimation_mode": "public-only",
        "stage_02_run_directory": str(
            stage2_run_directory
        ),
        "files_opened_by_estimator": [
            str(traces_path),
            str(metadata_path),
            str(simulation_config_path),
            str(stage2_summary_path),
        ],
        "private_files_opened_by_estimator": [],
        "private_evaluation_policy": (
            "Allowed only after public estimate files "
            "have been written and hashed."
        ),
    }

    write_json(
        public_output_directory
        / "data_access_manifest.json",
        access_manifest,
    )

    write_json(
        public_output_directory
        / "preprocessing_statistics.json",
        preprocessing_stats,
    )

    write_json(
        public_output_directory
        / "alignment_diagnostics.json",
        {
            "global_alignment": global_diagnostics,
            "local_affine_alignment": affine_diagnostics,
        },
    )

    write_json(
        public_output_directory
        / "final_round_roi.json",
        final_round_roi,
    )

    write_json(
        public_output_directory
        / "final_round_locator_diagnostics.json",
        locator_diagnostics,
    )

    np.savez_compressed(
        public_output_directory
        / "alignment_reference_and_activity.npz",
        reference_trace=final_reference.astype(
            np.float32
        ),
        activity_profile=activity_arrays[
            "activity_profile"
        ].astype(np.float32),
        period_lags=activity_arrays[
            "period_lags"
        ].astype(np.int32),
        period_scores=activity_arrays[
            "period_scores"
        ].astype(np.float64),
        phase_candidates=activity_arrays[
            "phase_candidates"
        ].astype(np.int32),
        phase_scores=activity_arrays[
            "phase_scores"
        ].astype(np.float64),
        refinement_candidates=activity_arrays[
            "refinement_candidates"
        ].astype(np.int32),
        refinement_scores=activity_arrays[
            "refinement_scores"
        ].astype(np.float64),
    )

    if config.save_aligned_full_traces:
        np.savez_compressed(
            public_output_directory
            / "aligned_healthy_traces.npz",
            traces=aligned_amplitude.astype(np.float32),
            alignment_signal=aligned_signal.astype(
                np.float32
            ),
            trace_ids=trace_ids,
            sample_axis_seconds=sample_axis_seconds,
            global_trigger_sample=np.asarray(
                global_trigger_sample,
                dtype=np.int32,
            ),
            sampling_rate_hz=np.asarray(
                sampling_rate_hz,
                dtype=np.float64,
            ),
            global_shifts=global_shifts.astype(
                np.int32
            ),
            affine_intercepts=affine_intercepts.astype(
                np.float64
            ),
            affine_slopes=affine_slopes.astype(
                np.float64
            ),
        )

    roi_start = int(
        final_round_roi["roi_start_sample_inclusive"]
    )
    roi_end = int(
        final_round_roi["roi_end_sample_exclusive"]
    )

    np.savez_compressed(
        public_output_directory
        / "final_round_roi_traces.npz",
        traces=aligned_amplitude[
            :, roi_start:roi_end
        ].astype(np.float32),
        alignment_signal=aligned_signal[
            :, roi_start:roi_end
        ].astype(np.float32),
        trace_ids=trace_ids,
        absolute_sample_indices=np.arange(
            roi_start,
            roi_end,
            dtype=np.int32,
        ),
        sample_axis_seconds=sample_axis_seconds[
            roi_start:roi_end
        ],
        roi_start_sample=np.asarray(
            roi_start,
            dtype=np.int32,
        ),
        roi_end_sample=np.asarray(
            roi_end,
            dtype=np.int32,
        ),
    )

    alignment_metadata = build_alignment_metadata(
        metadata_rows,
        global_shifts,
        affine_intercepts,
        affine_slopes,
        pre_correlations,
        post_correlations,
    )

    write_csv(
        public_output_directory
        / "aligned_trace_metadata.csv",
        alignment_metadata,
    )

    generated_plots = save_public_plots(
        public_output_directory,
        alignment_signal,
        aligned_signal,
        activity_arrays,
        final_round_roi,
        config,
    )

    # --------------------------------------------------------
    # public validation
    # --------------------------------------------------------
    public_validation = (
        validate_stage_03_public_outputs(
            original_traces,
            aligned_amplitude,
            trace_ids,
            metadata_rows,
            global_shifts,
            affine_intercepts,
            affine_slopes,
            final_round_roi,
            global_diagnostics,
            affine_diagnostics,
            config,
        )
    )

    write_json(
        run_directory
        / "stage_03_public_validation_checks.json",
        public_validation,
    )

    # --------------------------------------------------------
    # freeze public estimate before private evaluation
    # --------------------------------------------------------
    freeze_files = [
        public_output_directory
        / "final_round_roi.json",
        public_output_directory
        / "alignment_diagnostics.json",
        public_output_directory
        / "alignment_reference_and_activity.npz",
        public_output_directory
        / "final_round_roi_traces.npz",
    ]

    freeze_manifest = {
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "statement": (
            "Public Stage 03 estimate frozen before "
            "private timing evaluation."
        ),
        "files": {
            path.name: sha256_file(path)
            for path in freeze_files
        },
    }

    freeze_digest_source = json.dumps(
        freeze_manifest,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    freeze_sha256 = hashlib.sha256(
        freeze_digest_source
    ).hexdigest()
    freeze_manifest["freeze_sha256"] = freeze_sha256

    write_json(
        run_directory
        / "estimation_freeze_manifest.json",
        freeze_manifest,
    )

    # --------------------------------------------------------
    # private evaluation — validation only, after freeze
    # --------------------------------------------------------
    if config.enable_private_evaluation:
        private_evaluation = (
            evaluate_frozen_estimate_against_private(
                stage2_run_directory,
                validation_directory,
                final_round_roi,
                global_shifts.astype(np.float64),
                affine_intercepts,
                affine_slopes,
                global_trigger_sample,
                freeze_sha256,
            )
        )
    else:
        private_evaluation = {
            "available": False,
            "reason": (
                "private evaluation disabled by config"
            ),
            "estimation_freeze_sha256": freeze_sha256,
        }

    # آستانه‌های validation-only محافظه‌کارانه‌اند.
    private_validation_passed = True

    if private_evaluation.get("available", False):
        private_validation_passed = bool(
            private_evaluation[
                "all_sbox_center_coverage_rate"
            ] >= 0.99
            and private_evaluation[
                "full_final_round_coverage_rate_per_trace"
            ] >= 0.98
            and private_evaluation[
                "absolute_round_period_error_samples"
            ] <= 3.0
        )

    elapsed_seconds = time.perf_counter() - start_time

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

    all_checks_passed = bool(
        public_validation[
            "all_public_checks_passed"
        ]
        and private_validation_passed
    )

    summary = {
        "stage": 3,
        "run_id": run_id,
        "run_directory": str(
            run_directory.resolve()
        ),
        "input_stage_02_run_directory": str(
            stage2_run_directory
        ),
        "public_directory": str(
            public_output_directory.resolve()
        ),
        "validation_only_directory": str(
            validation_directory.resolve()
        ),
        "all_checks_passed": all_checks_passed,
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
            original_traces.shape[0]
        ),
        "number_of_samples_per_trace": int(
            original_traces.shape[1]
        ),
        "sampling_rate_hz": sampling_rate_hz,
        "global_trigger_sample": (
            global_trigger_sample
        ),
        "estimated_round_period_samples": int(
            final_round_roi[
                "estimated_round_period_samples"
            ]
        ),
        "estimated_round_32_start_sample": int(
            final_round_roi[
                "estimated_round_32_start_sample"
            ]
        ),
        "final_round_roi_start_sample": roi_start,
        "final_round_roi_end_sample": roi_end,
        "final_round_roi_width_samples": (
            roi_end - roi_start
        ),
        "median_reference_correlation_before": float(
            np.median(pre_correlations)
        ),
        "median_reference_correlation_after": float(
            np.median(post_correlations)
        ),
        "median_absolute_local_shift_before": float(
            affine_diagnostics[
                "median_absolute_local_shift_before"
            ]
        ),
        "median_absolute_local_shift_after": float(
            affine_diagnostics[
                "median_absolute_local_shift_after"
            ]
        ),
        "private_all_sbox_center_coverage_rate": (
            float(
                private_evaluation[
                    "all_sbox_center_coverage_rate"
                ]
            )
            if private_evaluation.get(
                "available",
                False,
            )
            else None
        ),
        "private_round_period_absolute_error_samples": (
            float(
                private_evaluation[
                    "absolute_round_period_error_samples"
                ]
            )
            if private_evaluation.get(
                "available",
                False,
            )
            else None
        ),
        "estimation_freeze_sha256": freeze_sha256,
        "elapsed_seconds": float(elapsed_seconds),
        "public_files": public_files,
        "validation_only_files": validation_files,
        "generated_plots": generated_plots,
    }

    write_json(
        run_directory / "stage_03_summary.json",
        summary,
    )

    write_json(
        run_directory / "run_manifest.json",
        {
            "stage": 3,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": asdict(config),
            "input_file_sha256": {
                "stage_02_summary.json": sha256_file(
                    stage2_summary_path
                ),
                "healthy_traces.npz": sha256_file(
                    traces_path
                ),
                "healthy_trace_metadata.csv": sha256_file(
                    metadata_path
                ),
                "trace_simulation_config.json": sha256_file(
                    simulation_config_path
                ),
            },
            "stage_02_simulation_config_snapshot": (
                simulation_config
            ),
        },
    )

    print("\n" + "=" * 76)
    print(
        "Stage 03 complete: alignment and "
        "public-only final-round locator"
    )
    print("=" * 76)
    print("Run directory                  :", summary["run_directory"])
    print("All checks passed              :", summary["all_checks_passed"])
    print("Public checks passed           :", summary["all_public_checks_passed"])
    print("Private validation available   :", summary["private_validation_available"])
    print("Private validation passed      :", summary["private_validation_passed"])
    print("Trace shape                    :", (
        summary["number_of_traces"],
        summary["number_of_samples_per_trace"],
    ))
    print("Estimated round period         :", summary["estimated_round_period_samples"])
    print("Estimated round-32 start       :", summary["estimated_round_32_start_sample"])
    print("Final-round ROI                :", (
        summary["final_round_roi_start_sample"],
        summary["final_round_roi_end_sample"],
    ))
    print("Median correlation before      :", (
        f"{summary['median_reference_correlation_before']:.6f}"
    ))
    print("Median correlation after       :", (
        f"{summary['median_reference_correlation_after']:.6f}"
    ))
    print("Private S-box center coverage  :", (
        summary["private_all_sbox_center_coverage_rate"]
    ))
    print("Private period absolute error  :", (
        summary["private_round_period_absolute_error_samples"]
    ))
    print("Elapsed seconds                :", (
        f"{summary['elapsed_seconds']:.3f}"
    ))
    print("=" * 76)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 03 failed validation. Inspect "
            "stage_03_public_validation_checks.json and "
            "validation_only/private_timing_evaluation.json"
        )

    return summary


def load_stage_03_config(
    config_path: str | Path,
) -> Stage03Config:
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    return Stage03Config(**raw)


if __name__ == "__main__":
    default_config = Stage03Config(
        input_stage2_run_directory=(
            r"C:\Users\SADRA\Desktop\LBlock\runs\stage_02"
            r"\stage02_20260718_165358_459976_seed20260718"
        ),
        output_root=(
            r"C:\Users\SADRA\Desktop\LBlock\runs\stage_03"
        ),
    )

    run_stage_03(default_config)

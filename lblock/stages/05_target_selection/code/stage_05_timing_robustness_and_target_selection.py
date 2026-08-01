from __future__ import annotations

"""Stage 05: public-only timing robustness and two-nibble target selection.

Inputs
------
A successful Stage 04 run directory. The script reads only Stage 03/04 public
artifacts during target selection. Private simulator timing is opened only
*after* the public result has been saved and hashed.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
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
class Stage05Config:
    input_stage4_run_directory: str
    output_root: str = "runs/stage_05"
    random_seed: int = 20260718

    number_of_sboxes: int = 8
    number_of_classes: int = 16
    minimum_class_count: int = 2
    smoothing_radius_samples: int = 2
    specificity_weight: float = 0.35

    local_first_center_radius_samples: int = 7
    local_spacing_radius_samples: int = 3

    subset_sizes: Tuple[int, ...] = (128, 256, 512, 768)
    subset_repetitions: int = 20

    stress_repetitions: int = 8
    # (residual jitter sigma in samples, additive-noise multiplier)
    stress_scenarios: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (1.5, 0.0),
        (0.0, 0.25),
        (0.0, 0.50),
        (1.0, 0.25),
        (1.5, 0.50),
    )

    safe_response_fraction: float = 0.60
    minimum_isolation_margin: float = 0.0

    # Single-target ranking weights; must sum to 1.
    weight_signal: float = 0.15
    weight_prominence: float = 0.15
    weight_isolation: float = 0.18
    weight_safe_width: float = 0.10
    weight_subset: float = 0.16
    weight_stress: float = 0.14
    weight_group: float = 0.07
    weight_bootstrap: float = 0.03
    weight_peak: float = 0.02

    # Pair ranking weights; must sum to 1.
    pair_weight_mean: float = 0.60
    pair_weight_worst: float = 0.20
    pair_weight_separation: float = 0.15
    pair_weight_nonoverlap: float = 0.05

    save_plots: bool = True
    enable_private_evaluation: bool = True


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: List[str] = []
    seen = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def minmax(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        return np.full(array.shape, 0.5)
    clean = array.copy()
    clean[~finite] = np.median(clean[finite])
    low, high = float(clean.min()), float(clean.max())
    result = np.full(clean.shape, 0.5) if high - low <= 1e-12 else (clean - low) / (high - low)
    return result if higher_is_better else 1.0 - result


def reorder_metadata(rows: Sequence[Mapping[str, str]], trace_ids: np.ndarray) -> List[Dict[str, str]]:
    by_id = {int(row["trace_id"]): dict(row) for row in rows}
    missing = [int(value) for value in trace_ids if int(value) not in by_id]
    if missing:
        raise ValueError(f"Metadata missing trace IDs: {missing[:10]}")
    return [by_id[int(value)] for value in trace_ids]


def residualize_sessions(traces: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    output = np.empty_like(np.asarray(traces, dtype=np.float64))
    for session in np.unique(sessions):
        indices = np.where(sessions == session)[0]
        block = np.asarray(traces[indices], dtype=np.float64)
        residual = block - block.mean(axis=0, keepdims=True)
        median = float(np.median(residual))
        scale = float(1.4826 * np.median(np.abs(residual - median)))
        if scale <= 1e-10:
            scale = float(np.std(residual)) or 1.0
        output[indices] = residual / scale
    return output


def eta_squared_stratified(
    traces: np.ndarray,
    labels: np.ndarray,
    key_ids: np.ndarray,
    minimum_class_count: int,
) -> np.ndarray:
    result = np.zeros(traces.shape[1], dtype=np.float64)
    total_weight = 0.0
    for key_id in np.unique(key_ids):
        indices = np.where(key_ids == key_id)[0]
        if indices.size < 8:
            continue
        block = traces[indices]
        block_labels = labels[indices]
        mean = block.mean(axis=0)
        total_ss = np.sum((block - mean) ** 2, axis=0)
        between_ss = np.zeros(block.shape[1], dtype=np.float64)
        valid_classes = 0
        for class_value in np.unique(block_labels):
            mask = block_labels == class_value
            count = int(mask.sum())
            if count < minimum_class_count:
                continue
            valid_classes += 1
            class_mean = block[mask].mean(axis=0)
            between_ss += count * (class_mean - mean) ** 2
        if valid_classes < 2:
            continue
        eta = np.divide(between_ss, total_ss, out=np.zeros_like(between_ss), where=total_ss > 1e-14)
        result += indices.size * eta
        total_weight += indices.size
    if total_weight <= 0:
        raise RuntimeError("No valid key stratum for eta-squared")
    return result / total_weight


def smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(values, dtype=np.float64).copy()
    ascending = np.arange(1, radius + 2, dtype=np.float64)
    kernel = np.concatenate([ascending, ascending[-2::-1]])
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if scale <= 1e-10:
        scale = float(np.std(values)) or 1.0
    return (values - median) / scale


def build_specificity(
    traces: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
    sessions: np.ndarray,
    config: Stage05Config,
) -> np.ndarray:
    balanced = residualize_sessions(traces, sessions)
    standardized = np.empty((config.number_of_sboxes, traces.shape[1]), dtype=np.float64)
    for index in range(config.number_of_sboxes):
        eta = eta_squared_stratified(balanced, labels[:, index], keys, config.minimum_class_count)
        standardized[index] = robust_z(smooth(eta, config.smoothing_radius_samples))
    specificity = np.empty_like(standardized)
    for index in range(config.number_of_sboxes):
        common = np.median(np.delete(standardized, index, axis=0), axis=0)
        specificity[index] = standardized[index] - config.specificity_weight * common
    return specificity


def axis_index(axis: np.ndarray, sample: int) -> int:
    position = int(np.searchsorted(axis, sample))
    candidates = [index for index in (position - 1, position) if 0 <= index < axis.size]
    if not candidates:
        raise ValueError("Sample outside ROI")
    return min(candidates, key=lambda index: abs(int(axis[index]) - sample))


def prominence(profile: np.ndarray, center_index: int, spacing: int) -> float:
    inner = max(1, round(0.22 * spacing))
    outer = max(inner + 1, round(0.65 * spacing))
    background: List[float] = []
    for offset in range(inner + 1, outer + 1):
        for candidate in (center_index - offset, center_index + offset):
            if 0 <= candidate < profile.size:
                background.append(float(profile[candidate]))
    return float(profile[center_index] - np.mean(background)) if background else 0.0


def search_local_comb(
    specificity: np.ndarray,
    axis: np.ndarray,
    nominal_first: int,
    nominal_spacing: int,
    config: Stage05Config,
) -> Dict[str, Any]:
    best: Dict[str, Any] | None = None
    for spacing in range(max(2, nominal_spacing - config.local_spacing_radius_samples),
                         nominal_spacing + config.local_spacing_radius_samples + 1):
        for first in range(nominal_first - config.local_first_center_radius_samples,
                           nominal_first + config.local_first_center_radius_samples + 1):
            centers = np.asarray([first + index * spacing for index in range(config.number_of_sboxes)])
            if centers[0] < axis[0] or centers[-1] > axis[-1]:
                continue
            values, local_prominences = [], []
            for index, center in enumerate(centers):
                position = axis_index(axis, int(center))
                values.append(float(specificity[index, position]))
                local_prominences.append(prominence(specificity[index], position, spacing))
            values_array = np.asarray(values)
            score = float(values_array.mean() + 0.15 * np.mean(local_prominences) - 0.05 * values_array.std())
            candidate = {
                "score": score,
                "first_center_sample": int(first),
                "spacing_samples": int(spacing),
                "centers": centers.astype(int).tolist(),
            }
            if best is None or score > best["score"]:
                best = candidate
    if best is None:
        raise RuntimeError("No local comb candidate")
    return best


def stratified_subset(keys: np.ndarray, sessions: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    pair = keys.astype(np.int64) * 1_000_000 + sessions.astype(np.int64)
    groups = rng.permutation(np.unique(pair))
    base, remainder = divmod(count, len(groups))
    selected: List[int] = []
    for order, group in enumerate(groups):
        candidates = np.where(pair == group)[0]
        take = min(candidates.size, base + (order < remainder))
        if take:
            selected.extend(rng.choice(candidates, take, replace=False).tolist())
    if len(selected) < count:
        unused = np.setdiff1d(np.arange(pair.size), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(unused, count - len(selected), replace=False).tolist())
    selected_array = np.asarray(selected, dtype=np.int32)
    rng.shuffle(selected_array)
    return selected_array


def summarize_center_runs(rows: Sequence[Mapping[str, Any]], nominal: np.ndarray, sbox_count: int) -> Dict[str, Any]:
    matrix = np.asarray([[int(row[f"S{i}_center"]) for i in range(sbox_count)] for row in rows], dtype=np.int32)
    errors = matrix - nominal
    per_sbox = {}
    for index in range(sbox_count):
        absolute = np.abs(errors[:, index])
        per_sbox[f"S{index}"] = {
            "exact_center_rate": float(np.mean(absolute == 0)),
            "within_one_sample_rate": float(np.mean(absolute <= 1)),
            "mean_absolute_error_samples": float(np.mean(absolute)),
            "p95_absolute_error_samples": float(np.percentile(absolute, 95)),
            "maximum_absolute_error_samples": int(np.max(absolute)),
        }
    return {
        "run_count": len(rows),
        "exact_full_map_rate": float(np.mean(np.all(matrix == nominal, axis=1))),
        "all_centers_within_one_sample_rate": float(np.mean(np.all(np.abs(errors) <= 1, axis=1))),
        "center_mae_samples": float(np.mean(np.abs(errors))),
        "maximum_absolute_error_samples": int(np.max(np.abs(errors))),
        "per_sbox": per_sbox,
    }


def run_subset_tests(
    traces: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
    sessions: np.ndarray,
    axis: np.ndarray,
    nominal: np.ndarray,
    spacing: int,
    config: Stage05Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed + 501)
    rows: List[Dict[str, Any]] = []
    by_size: Dict[str, Any] = {}
    for size in config.subset_sizes:
        if size > traces.shape[0]:
            continue
        local_rows = []
        for repetition in range(config.subset_repetitions):
            selected = stratified_subset(keys, sessions, size, rng)
            estimate = search_local_comb(
                build_specificity(traces[selected], labels[selected], keys[selected], sessions[selected], config),
                axis, int(nominal[0]), spacing, config,
            )
            centers = np.asarray(estimate["centers"], dtype=np.int32)
            row: Dict[str, Any] = {
                "subset_size": int(size),
                "repetition": repetition,
                "spacing_samples": int(estimate["spacing_samples"]),
                "comb_score": float(estimate["score"]),
            }
            for index in range(config.number_of_sboxes):
                row[f"S{index}_center"] = int(centers[index])
                row[f"S{index}_error"] = int(centers[index] - nominal[index])
            rows.append(row)
            local_rows.append(row)
        by_size[str(size)] = summarize_center_runs(local_rows, nominal, config.number_of_sboxes)
    overall = summarize_center_runs(rows, nominal, config.number_of_sboxes)
    overall["by_subset_size"] = by_size
    return rows, overall


def run_group_holdouts(
    traces: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
    sessions: np.ndarray,
    axis: np.ndarray,
    nominal: np.ndarray,
    spacing: int,
    config: Stage05Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_type, values in (("session", sessions), ("key", keys)):
        for held_out in np.unique(values):
            selected = np.where(values != held_out)[0]
            estimate = search_local_comb(
                build_specificity(traces[selected], labels[selected], keys[selected], sessions[selected], config),
                axis, int(nominal[0]), spacing, config,
            )
            centers = np.asarray(estimate["centers"], dtype=np.int32)
            row: Dict[str, Any] = {
                "group_type": group_type,
                "held_out_value": int(held_out),
                "remaining_trace_count": int(selected.size),
                "spacing_samples": int(estimate["spacing_samples"]),
            }
            for index in range(config.number_of_sboxes):
                row[f"S{index}_center"] = int(centers[index])
                row[f"S{index}_error"] = int(centers[index] - nominal[index])
            rows.append(row)
    return rows, summarize_center_runs(rows, nominal, config.number_of_sboxes)


def shift_trace(trace: np.ndarray, shift: float) -> np.ndarray:
    axis = np.arange(trace.size, dtype=np.float64)
    return np.interp(axis - shift, axis, trace, left=float(trace[0]), right=float(trace[-1]))


def stress_traces(traces: np.ndarray, jitter: float, noise: float, rng: np.random.Generator) -> np.ndarray:
    stressed = np.asarray(traces, dtype=np.float64).copy()
    if jitter > 0:
        for index, shift in enumerate(rng.normal(0.0, jitter, size=stressed.shape[0])):
            stressed[index] = shift_trace(stressed[index], float(shift))
    if noise > 0:
        scale = float(np.median(np.std(stressed, axis=1))) or 1.0
        stressed += rng.normal(0.0, noise * scale, size=stressed.shape)
    return stressed


def run_stress_tests(
    traces: np.ndarray,
    labels: np.ndarray,
    keys: np.ndarray,
    sessions: np.ndarray,
    axis: np.ndarray,
    nominal: np.ndarray,
    spacing: int,
    config: Stage05Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(config.random_seed + 502)
    rows: List[Dict[str, Any]] = []
    scenario_summary: List[Dict[str, Any]] = []
    for jitter, noise in config.stress_scenarios:
        local_rows = []
        for repetition in range(config.stress_repetitions):
            stressed = stress_traces(traces, jitter, noise, rng)
            estimate = search_local_comb(
                build_specificity(stressed, labels, keys, sessions, config),
                axis, int(nominal[0]), spacing, config,
            )
            centers = np.asarray(estimate["centers"], dtype=np.int32)
            row: Dict[str, Any] = {
                "jitter_sigma_samples": float(jitter),
                "noise_multiplier": float(noise),
                "repetition": repetition,
                "spacing_samples": int(estimate["spacing_samples"]),
            }
            for index in range(config.number_of_sboxes):
                row[f"S{index}_center"] = int(centers[index])
                row[f"S{index}_error"] = int(centers[index] - nominal[index])
            rows.append(row)
            local_rows.append(row)
        local_summary = summarize_center_runs(local_rows, nominal, config.number_of_sboxes)
        scenario_summary.append({
            "jitter_sigma_samples": float(jitter),
            "noise_multiplier": float(noise),
            **{key: value for key, value in local_summary.items() if key != "per_sbox"},
        })
    overall = summarize_center_runs(rows, nominal, config.number_of_sboxes)
    overall["scenarios"] = scenario_summary
    return rows, overall


def contiguous_interval(mask: np.ndarray, center: int) -> Tuple[int, int]:
    if not bool(mask[center]):
        return center, center + 1
    start, end = center, center + 1
    while start > 0 and mask[start - 1]:
        start -= 1
    while end < mask.size and mask[end]:
        end += 1
    return start, end


def offset_analysis(
    profiles: np.ndarray,
    axis: np.ndarray,
    timing_map: Mapping[str, Any],
    config: Stage05Config,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for entry in timing_map["sboxes"]:
        index = int(entry["sbox_index"])
        name = f"S{index}"
        center = int(entry["center_sample"])
        start = axis_index(axis, int(entry["exploration_window_start_sample_inclusive"]))
        end = axis_index(axis, int(entry["exploration_window_end_sample_exclusive"]) - 1) + 1
        local_axis = axis[start:end]
        own = profiles[index, start:end]
        other = np.max(np.delete(profiles[:, start:end], index, axis=0), axis=0)
        margin = own - other
        denominator = float(own.max() - own.min())
        response = np.zeros_like(own) if denominator <= 1e-12 else (own - own.min()) / denominator
        safe = (response >= config.safe_response_fraction) & (margin > config.minimum_isolation_margin)
        center_local = int(np.argmin(np.abs(local_axis - center)))
        safe_start, safe_end = contiguous_interval(safe, center_local)
        start_sample = int(local_axis[safe_start])
        end_sample = int(local_axis[safe_end - 1] + 1)
        peak_sample = int(local_axis[int(np.argmax(own))])
        summary[name] = {
            "center_sample": center,
            "local_peak_sample": peak_sample,
            "peak_minus_center_samples": peak_sample - center,
            "safe_interval_start_sample_inclusive": start_sample,
            "safe_interval_end_sample_exclusive": end_sample,
            "safe_interval_width_samples": end_sample - start_sample,
            "symmetric_tolerance_samples": min(center - start_sample, end_sample - 1 - center),
            "specificity_at_center": float(own[center_local]),
            "isolation_margin_at_center": float(margin[center_local]),
        }
        for position, sample in enumerate(local_axis):
            rows.append({
                "sbox": name,
                "sbox_index": index,
                "absolute_sample": int(sample),
                "offset_from_center_samples": int(sample - center),
                "own_specificity": float(own[position]),
                "strongest_other_specificity": float(other[position]),
                "isolation_margin": float(margin[position]),
                "normalized_response": float(response[position]),
                "safe_sample": bool(safe[position]),
            })
    return rows, summary


def rank_targets(
    timing_map: Mapping[str, Any],
    offsets: Mapping[str, Any],
    subsets: Mapping[str, Any],
    groups: Mapping[str, Any],
    stresses: Mapping[str, Any],
    config: Stage05Config,
) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    for entry in timing_map["sboxes"]:
        index, name = int(entry["sbox_index"]), entry["sbox"]
        raw.append({
            "sbox": name,
            "sbox_index": index,
            "center_sample": int(entry["center_sample"]),
            "specificity_at_center": float(entry["specificity_at_center"]),
            "local_prominence": float(entry["local_prominence"]),
            "isolation_margin_at_center": float(offsets[name]["isolation_margin_at_center"]),
            "safe_interval_width_samples": int(offsets[name]["safe_interval_width_samples"]),
            "symmetric_tolerance_samples": int(offsets[name]["symmetric_tolerance_samples"]),
            "subset_within_one_sample_rate": float(subsets["per_sbox"][name]["within_one_sample_rate"]),
            "stress_within_one_sample_rate": float(stresses["per_sbox"][name]["within_one_sample_rate"]),
            "group_maximum_absolute_error_samples": float(groups["per_sbox"][name]["maximum_absolute_error_samples"]),
            "bootstrap_center_std": float(entry["bootstrap_center_std"]),
            "independent_peak_distance_samples": abs(float(entry["independent_peak_minus_regularized_center"])),
        })
    normalized = {
        "signal": minmax([row["specificity_at_center"] for row in raw]),
        "prominence": minmax([row["local_prominence"] for row in raw]),
        "isolation": minmax([row["isolation_margin_at_center"] for row in raw]),
        "safe_width": minmax([row["safe_interval_width_samples"] for row in raw]),
        "subset": minmax([row["subset_within_one_sample_rate"] for row in raw]),
        "stress": minmax([row["stress_within_one_sample_rate"] for row in raw]),
        "group": minmax([row["group_maximum_absolute_error_samples"] for row in raw], False),
        "bootstrap": minmax([row["bootstrap_center_std"] for row in raw], False),
        "peak": minmax([row["independent_peak_distance_samples"] for row in raw], False),
    }
    weights = {
        "signal": config.weight_signal,
        "prominence": config.weight_prominence,
        "isolation": config.weight_isolation,
        "safe_width": config.weight_safe_width,
        "subset": config.weight_subset,
        "stress": config.weight_stress,
        "group": config.weight_group,
        "bootstrap": config.weight_bootstrap,
        "peak": config.weight_peak,
    }
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Single-target weights must sum to one")
    for position, row in enumerate(raw):
        for metric in normalized:
            row[f"normalized_{metric}"] = float(normalized[metric][position])
        row["target_quality_score"] = float(sum(weights[metric] * normalized[metric][position] for metric in weights))
    raw.sort(key=lambda row: (row["target_quality_score"], row["specificity_at_center"]), reverse=True)
    for rank, row in enumerate(raw, 1):
        row["rank"] = rank
    return raw


def overlap_fraction(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    first_start, first_end = int(first["exploration_window_start_sample_inclusive"]), int(first["exploration_window_end_sample_exclusive"])
    second_start, second_end = int(second["exploration_window_start_sample_inclusive"]), int(second["exploration_window_end_sample_exclusive"])
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start))
    union = max(first_end, second_end) - min(first_start, second_start)
    return float(overlap / union) if union else 0.0


def rank_pairs(ranking: Sequence[Mapping[str, Any]], timing_map: Mapping[str, Any], config: Stage05Config) -> List[Dict[str, Any]]:
    if abs(config.pair_weight_mean + config.pair_weight_worst + config.pair_weight_separation + config.pair_weight_nonoverlap - 1.0) > 1e-9:
        raise ValueError("Pair weights must sum to one")
    quality = {int(row["sbox_index"]): float(row["target_quality_score"]) for row in ranking}
    entries = {int(row["sbox_index"]): row for row in timing_map["sboxes"]}
    spacing = float(timing_map["estimated_sbox_spacing_samples"])
    rows: List[Dict[str, Any]] = []
    for first, second in combinations(range(config.number_of_sboxes), 2):
        q1, q2 = quality[first], quality[second]
        gap = abs(int(entries[second]["center_sample"]) - int(entries[first]["center_sample"]))
        separation = min(gap / max(3.0 * spacing, 1.0), 1.0)
        overlap = overlap_fraction(entries[first], entries[second])
        score = (
            config.pair_weight_mean * ((q1 + q2) / 2.0)
            + config.pair_weight_worst * min(q1, q2)
            + config.pair_weight_separation * separation
            + config.pair_weight_nonoverlap * (1.0 - overlap)
        )
        rows.append({
            "first_sbox": f"S{first}",
            "second_sbox": f"S{second}",
            "first_sbox_index": first,
            "second_sbox_index": second,
            "first_quality_score": q1,
            "second_quality_score": q2,
            "mean_quality_score": (q1 + q2) / 2.0,
            "worst_quality_score": min(q1, q2),
            "center_gap_samples": gap,
            "sbox_index_gap": second - first,
            "separation_score": separation,
            "exploration_window_overlap_fraction": overlap,
            "pair_score": score,
        })
    rows.sort(key=lambda row: (row["pair_score"], row["worst_quality_score"], row["center_gap_samples"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def make_target_contract(
    pair_ranking: Sequence[Mapping[str, Any]],
    target_ranking: Sequence[Mapping[str, Any]],
    timing_map: Mapping[str, Any],
    offsets: Mapping[str, Any],
) -> Dict[str, Any]:
    pair = pair_ranking[0]
    names = [pair["first_sbox"], pair["second_sbox"]]
    rank_by_name = {row["sbox"]: row for row in target_ranking}
    map_by_name = {row["sbox"]: row for row in timing_map["sboxes"]}
    targets = []
    for name in names:
        rank, entry, offset = rank_by_name[name], map_by_name[name], offsets[name]
        index = int(rank["sbox_index"])
        targets.append({
            "sbox": name,
            "sbox_index": index,
            "target_key_nibble": f"K32[{index}]",
            "target_key_bits": [4 * index + bit for bit in range(4)],
            "center_sample": int(entry["center_sample"]),
            "center_offset_from_round_32_start_samples": int(entry["center_offset_from_round_32_start_samples"]),
            "core_window_start_sample_inclusive": int(entry["core_window_start_sample_inclusive"]),
            "core_window_end_sample_exclusive": int(entry["core_window_end_sample_exclusive"]),
            "exploration_window_start_sample_inclusive": int(entry["exploration_window_start_sample_inclusive"]),
            "exploration_window_end_sample_exclusive": int(entry["exploration_window_end_sample_exclusive"]),
            "public_safe_interval_start_sample_inclusive": int(offset["safe_interval_start_sample_inclusive"]),
            "public_safe_interval_end_sample_exclusive": int(offset["safe_interval_end_sample_exclusive"]),
            "target_quality_score": float(rank["target_quality_score"]),
            "single_target_rank": int(rank["rank"]),
        })
    bits = sorted(bit for target in targets for bit in target["target_key_bits"])
    return {
        "algorithm": "LBlock-64/80",
        "stage": 5,
        "selection_source": "Public Stage 03/04 data and public robustness tests only",
        "ground_truth_used_for_selection": False,
        "selected_pair_rank": int(pair["rank"]),
        "selected_pair_score": float(pair["pair_score"]),
        "selected_sboxes": names,
        "selected_sbox_indices": [int(pair["first_sbox_index"]), int(pair["second_sbox_index"])],
        "selected_last_round_key_nibbles": [target["target_key_nibble"] for target in targets],
        "selected_last_round_key_bit_indices": bits,
        "total_target_bits": len(bits),
        "pair_center_gap_samples": int(pair["center_gap_samples"]),
        "pair_exploration_window_overlap_fraction": float(pair["exploration_window_overlap_fraction"]),
        "targets": targets,
        "next_stage_contract": {
            "fault_engine_targets": names,
            "nominal_times": {target["sbox"]: target["center_sample"] for target in targets},
            "timing_scan_policy": "Sweep only public exploration windows; never configure from private timing.",
            "minimum_key_recovery_goal": "Recover both selected 4-bit nibbles (8 bits of K32).",
        },
    }


def validate_public(
    timing_map: Mapping[str, Any],
    contract: Mapping[str, Any],
    ranking: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    subsets: Mapping[str, Any],
    groups: Mapping[str, Any],
    stresses: Mapping[str, Any],
    config: Stage05Config,
) -> Dict[str, Any]:
    selected = contract["selected_sbox_indices"]
    smallest = str(min(int(value) for value in subsets["by_subset_size"]))
    clean = next(row for row in stresses["scenarios"] if row["jitter_sigma_samples"] == 0 and row["noise_multiplier"] == 0)
    checks = {
        "two_distinct_targets": {"passed": len(selected) == 2 and len(set(selected)) == 2, "selected": selected},
        "eight_target_bits": {"passed": contract["total_target_bits"] == 8 and len(set(contract["selected_last_round_key_bit_indices"])) == 8},
        "ranking_complete": {"passed": len(ranking) == 8 and len(pairs) == 28},
        "selected_centers_match_stage4": {
            "passed": all(
                contract["targets"][position]["center_sample"] == timing_map["sboxes"][index]["center_sample"]
                for position, index in enumerate(selected)
            )
        },
        "smallest_subset_usable": {
            "passed": subsets["by_subset_size"][smallest]["all_centers_within_one_sample_rate"] >= 0.70,
            "subset_size": int(smallest),
            "rate": subsets["by_subset_size"][smallest]["all_centers_within_one_sample_rate"],
        },
        "group_holdout_usable": {"passed": groups["all_centers_within_one_sample_rate"] >= 0.75, "rate": groups["all_centers_within_one_sample_rate"]},
        "clean_reestimation_consistent": {"passed": clean["all_centers_within_one_sample_rate"] >= 0.95, "rate": clean["all_centers_within_one_sample_rate"]},
        "public_only_contract": {"passed": not contract["ground_truth_used_for_selection"]},
    }
    return {"all_public_checks_passed": all(item["passed"] for item in checks.values()), "checks": checks}


def transformed_centers(original: np.ndarray, shifts: np.ndarray, intercepts: np.ndarray, slopes: np.ndarray, trigger: float) -> np.ndarray:
    return (original - shifts[:, None] - intercepts[:, None] + slopes[:, None] * trigger) / (1.0 + slopes[:, None])


def private_evaluation(
    stage3: Path,
    stage2: Path,
    contract: Mapping[str, Any],
    output: Path,
    freeze_hash: str,
) -> Dict[str, Any]:
    hidden_path = stage2 / "private_ground_truth" / "hidden_timing_and_crypto_ground_truth.npz"
    aligned_path = stage3 / "public" / "aligned_healthy_traces.npz"
    if not hidden_path.is_file() or not aligned_path.is_file():
        return {"available": False, "reason": "Private timing or alignment file missing", "estimation_freeze_sha256": freeze_hash}
    with np.load(hidden_path, allow_pickle=False) as hidden:
        original = np.asarray(hidden["sbox_centers"][:, -1, :], dtype=np.float64)
    with np.load(aligned_path, allow_pickle=False) as aligned:
        aligned_hidden = transformed_centers(
            original,
            np.asarray(aligned["global_shifts"], dtype=np.float64),
            np.asarray(aligned["affine_intercepts"], dtype=np.float64),
            np.asarray(aligned["affine_slopes"], dtype=np.float64),
            float(np.asarray(aligned["global_trigger_sample"]).item()),
        )
    rows = []
    for target in contract["targets"]:
        index = int(target["sbox_index"])
        values = aligned_hidden[:, index]
        center = float(target["center_sample"])
        exploration = np.mean((values >= target["exploration_window_start_sample_inclusive"]) & (values < target["exploration_window_end_sample_exclusive"]))
        safe = np.mean((values >= target["public_safe_interval_start_sample_inclusive"]) & (values < target["public_safe_interval_end_sample_exclusive"]))
        rows.append({
            "sbox": target["sbox"],
            "estimated_center": center,
            "hidden_median_center": float(np.median(values)),
            "absolute_center_error_samples": abs(center - float(np.median(values))),
            "exploration_window_coverage": float(exploration),
            "public_safe_interval_coverage": float(safe),
            "hidden_center_std_samples": float(np.std(values)),
        })
    result = {
        "available": True,
        "warning": "Validation only; opened after public selection freeze.",
        "estimation_freeze_sha256": freeze_hash,
        "selected_targets": rows,
        "mean_absolute_center_error_samples": float(np.mean([row["absolute_center_error_samples"] for row in rows])),
        "maximum_absolute_center_error_samples": float(np.max([row["absolute_center_error_samples"] for row in rows])),
        "mean_exploration_window_coverage": float(np.mean([row["exploration_window_coverage"] for row in rows])),
        "mean_public_safe_interval_coverage": float(np.mean([row["public_safe_interval_coverage"] for row in rows])),
    }
    write_json(output / "private_selected_target_evaluation.json", result)
    return result


def save_plots(
    output: Path,
    ranking: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    offset_rows: Sequence[Mapping[str, Any]],
    subset_rows: Sequence[Mapping[str, Any]],
    stress_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    config: Stage05Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []
    names: List[str] = []
    ordered = sorted(ranking, key=lambda row: row["sbox_index"])
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar([row["sbox"] for row in ordered], [row["target_quality_score"] for row in ordered])
    axis.set_title("Public robustness score for each final-round S-box")
    axis.set_ylabel("Target quality score")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output / "sbox_target_quality_ranking.png"; fig.savefig(path, dpi=180); plt.close(fig); names.append(path.name)

    top = pairs[:12]
    fig, axis = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(top))
    axis.bar(positions, [row["pair_score"] for row in top])
    axis.set_xticks(positions, [row["first_sbox"] + "+" + row["second_sbox"] for row in top], rotation=45, ha="right")
    axis.set_title("Top two-nibble target pairs")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = output / "two_nibble_pair_ranking.png"; fig.savefig(path, dpi=180); plt.close(fig); names.append(path.name)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    for target in contract["selected_sboxes"]:
        local = [row for row in offset_rows if row["sbox"] == target]
        axis.plot([row["offset_from_center_samples"] for row in local], [row["normalized_response"] for row in local], marker="o", label=target)
    axis.axvline(0, linestyle=":")
    axis.set_title("Offset sensitivity of selected targets")
    axis.set_xlabel("Offset from nominal center [samples]")
    axis.set_ylabel("Normalized public response")
    axis.grid(alpha=0.2); axis.legend(); fig.tight_layout()
    path = output / "selected_target_offset_sensitivity.png"; fig.savefig(path, dpi=180); plt.close(fig); names.append(path.name)

    sizes = sorted(set(int(row["subset_size"]) for row in subset_rows))
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for target in contract["selected_sboxes"]:
        index = int(target[1:])
        means = [np.mean([abs(int(row[f"S{index}_error"])) for row in subset_rows if int(row["subset_size"]) == size]) for size in sizes]
        axis.plot(sizes, means, marker="o", label=target)
    axis.set_title("Timing robustness versus trace count")
    axis.set_xlabel("Number of traces"); axis.set_ylabel("Mean absolute center error [samples]")
    axis.grid(alpha=0.2); axis.legend(); fig.tight_layout()
    path = output / "trace_count_robustness.png"; fig.savefig(path, dpi=180); plt.close(fig); names.append(path.name)

    scenarios = sorted(set((float(row["jitter_sigma_samples"]), float(row["noise_multiplier"])) for row in stress_rows))
    matrix = np.zeros((2, len(scenarios)))
    for target_position, target in enumerate(contract["selected_sboxes"]):
        index = int(target[1:])
        for scenario_position, (jitter, noise) in enumerate(scenarios):
            matrix[target_position, scenario_position] = np.mean([
                abs(int(row[f"S{index}_error"])) for row in stress_rows
                if float(row["jitter_sigma_samples"]) == jitter and float(row["noise_multiplier"]) == noise
            ])
    fig, axis = plt.subplots(figsize=(12, 4.5))
    image = axis.imshow(matrix, aspect="auto", origin="lower")
    axis.set_yticks(range(2), contract["selected_sboxes"])
    axis.set_xticks(range(len(scenarios)), [f"j={j}, n={n}" for j, n in scenarios], rotation=45, ha="right")
    axis.set_title("Selected-target robustness under controlled stress")
    fig.colorbar(image, ax=axis, label="Mean absolute error [samples]")
    fig.tight_layout()
    path = output / "selected_target_stress_robustness.png"; fig.savefig(path, dpi=180); plt.close(fig); names.append(path.name)
    return names


def run_stage_05(config: Stage05Config) -> Dict[str, Any]:
    started = time.perf_counter()
    stage4 = Path(config.input_stage4_run_directory).expanduser().resolve()
    summary4_path = stage4 / "stage_04_summary.json"
    map_path = stage4 / "public" / "lblock_final_round_timing_map.json"
    profiles_path = stage4 / "public" / "sbox_dependency_profiles.npz"
    for path in (summary4_path, map_path, profiles_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary4, timing_map = read_json(summary4_path), read_json(map_path)
    if not summary4.get("all_checks_passed", False):
        raise RuntimeError("Stage 04 did not pass all checks")
    stage3 = Path(summary4["input_stage_03_run_directory"]).expanduser().resolve()
    stage2 = Path(summary4["input_stage_02_run_directory_for_validation_only"]).expanduser().resolve()
    roi_path = stage3 / "public" / "final_round_roi_traces.npz"
    metadata_path = stage3 / "public" / "aligned_trace_metadata.csv"
    for path in (roi_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(profiles_path, allow_pickle=False) as data:
        profile_trace_ids = np.asarray(data["trace_ids"], dtype=np.int32)
        axis = np.asarray(data["absolute_sample_indices"], dtype=np.int32)
        labels = np.asarray(data["observable_labels"], dtype=np.uint8)
        public_profiles = np.asarray(data["specificity_profiles"], dtype=np.float64)
    with np.load(roi_path, allow_pickle=False) as data:
        traces = np.asarray(data["traces"], dtype=np.float32)
        roi_trace_ids = np.asarray(data["trace_ids"], dtype=np.int32)
        roi_axis = np.asarray(data["absolute_sample_indices"], dtype=np.int32)
    if not np.array_equal(profile_trace_ids, roi_trace_ids) or not np.array_equal(axis, roi_axis):
        raise ValueError("Stage 03/04 trace or sample axes differ")
    metadata = reorder_metadata(read_csv(metadata_path), roi_trace_ids)
    keys = np.asarray([int(row["key_id"]) for row in metadata], dtype=np.int32)
    sessions = np.asarray([int(row["session_id"]) for row in metadata], dtype=np.int32)
    nominal = np.asarray([int(entry["center_sample"]) for entry in timing_map["sboxes"]], dtype=np.int32)
    spacing = int(timing_map["estimated_sbox_spacing_samples"])

    run_id = f"stage05_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_seed{config.random_seed}"
    run_dir = Path(config.output_root).expanduser().resolve() / run_id
    public = run_dir / "public"; validation = run_dir / "validation_only"
    public.mkdir(parents=True, exist_ok=False)

    offset_rows, offset_summary = offset_analysis(public_profiles, axis, timing_map, config)
    subset_rows, subset_summary = run_subset_tests(traces, labels, keys, sessions, axis, nominal, spacing, config)
    group_rows, group_summary = run_group_holdouts(traces, labels, keys, sessions, axis, nominal, spacing, config)
    stress_rows, stress_summary = run_stress_tests(traces, labels, keys, sessions, axis, nominal, spacing, config)
    ranking = rank_targets(timing_map, offset_summary, subset_summary, group_summary, stress_summary, config)
    pairs = rank_pairs(ranking, timing_map, config)
    contract = make_target_contract(pairs, ranking, timing_map, offset_summary)

    serializable_config = asdict(config)
    serializable_config["subset_sizes"] = list(config.subset_sizes)
    serializable_config["stress_scenarios"] = [list(value) for value in config.stress_scenarios]
    write_json(public / "stage_05_config.json", serializable_config)
    write_json(public / "timing_robustness_summary.json", {
        "offset_analysis": offset_summary,
        "subset_size_robustness": subset_summary,
        "group_holdout_robustness": group_summary,
        "stress_robustness": stress_summary,
    })
    write_json(public / "selected_two_nibble_targets.json", contract)
    write_json(public / "lblock_8bit_target_contract.json", contract)
    write_csv(public / "sbox_robustness_ranking.csv", ranking)
    write_csv(public / "two_nibble_pair_ranking.csv", pairs)
    write_csv(public / "offset_sensitivity_profiles.csv", offset_rows)
    write_csv(public / "subset_size_robustness_runs.csv", subset_rows)
    write_csv(public / "group_holdout_robustness.csv", group_rows)
    write_csv(public / "stress_robustness_runs.csv", stress_rows)
    write_json(public / "data_access_manifest.json", {
        "estimation_mode": "public-only",
        "files_opened_before_freeze": [str(summary4_path), str(map_path), str(profiles_path), str(roi_path), str(metadata_path)],
        "private_files_opened_before_freeze": [],
        "master_key_used": False,
        "round_key_used": False,
        "internal_state_used": False,
        "hidden_timing_used": False,
    })
    generated_plots = save_plots(public, ranking, pairs, offset_rows, subset_rows, stress_rows, contract, config)

    public_checks = validate_public(timing_map, contract, ranking, pairs, subset_summary, group_summary, stress_summary, config)
    write_json(run_dir / "stage_05_public_validation_checks.json", public_checks)

    freeze_files = [
        public / "selected_two_nibble_targets.json",
        public / "sbox_robustness_ranking.csv",
        public / "two_nibble_pair_ranking.csv",
        public / "timing_robustness_summary.json",
        public / "data_access_manifest.json",
    ]
    freeze_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": "Public Stage 05 result frozen before private validation.",
        "files": {path.name: sha256_file(path) for path in freeze_files},
    }
    freeze_hash = hashlib.sha256(json.dumps(freeze_manifest, sort_keys=True).encode()).hexdigest()
    freeze_manifest["freeze_sha256"] = freeze_hash
    write_json(run_dir / "estimation_freeze_manifest.json", freeze_manifest)

    if config.enable_private_evaluation:
        private = private_evaluation(stage3, stage2, contract, validation, freeze_hash)
    else:
        private = {"available": False, "reason": "Disabled", "estimation_freeze_sha256": freeze_hash}
    private_passed = True
    if private.get("available"):
        private_passed = bool(
            private["maximum_absolute_center_error_samples"] <= 1.0
            and private["mean_exploration_window_coverage"] >= 0.85
        )
    all_passed = bool(public_checks["all_public_checks_passed"] and private_passed)
    smallest = str(min(int(value) for value in subset_summary["by_subset_size"]))
    quality = {row["sbox"]: row["target_quality_score"] for row in ranking}

    summary = {
        "stage": 5,
        "run_id": run_id,
        "run_directory": str(run_dir),
        "input_stage_04_run_directory": str(stage4),
        "input_stage_03_run_directory": str(stage3),
        "public_directory": str(public),
        "validation_only_directory": str(validation),
        "all_checks_passed": all_passed,
        "all_public_checks_passed": bool(public_checks["all_public_checks_passed"]),
        "private_validation_available": bool(private.get("available", False)),
        "private_validation_passed": private_passed,
        "number_of_traces": int(traces.shape[0]),
        "roi_width_samples": int(traces.shape[1]),
        "selected_sboxes": contract["selected_sboxes"],
        "selected_sbox_indices": contract["selected_sbox_indices"],
        "selected_last_round_key_nibbles": contract["selected_last_round_key_nibbles"],
        "selected_last_round_key_bit_indices": contract["selected_last_round_key_bit_indices"],
        "total_target_bits": contract["total_target_bits"],
        "selected_pair_score": contract["selected_pair_score"],
        "selected_pair_center_gap_samples": contract["pair_center_gap_samples"],
        "selected_target_quality_scores": {name: quality[name] for name in contract["selected_sboxes"]},
        "smallest_subset_size": int(smallest),
        "smallest_subset_all_centers_within_one_sample_rate": subset_summary["by_subset_size"][smallest]["all_centers_within_one_sample_rate"],
        "group_holdout_all_centers_within_one_sample_rate": group_summary["all_centers_within_one_sample_rate"],
        "private_selected_target_mean_center_error_samples": private.get("mean_absolute_center_error_samples"),
        "private_selected_target_maximum_center_error_samples": private.get("maximum_absolute_center_error_samples"),
        "private_selected_target_exploration_coverage": private.get("mean_exploration_window_coverage"),
        "estimation_freeze_sha256": freeze_hash,
        "elapsed_seconds": float(time.perf_counter() - started),
        "public_files": sorted(path.name for path in public.iterdir() if path.is_file()),
        "validation_only_files": sorted(path.name for path in validation.iterdir() if path.is_file()) if validation.is_dir() else [],
        "generated_plots": generated_plots,
    }
    write_json(run_dir / "stage_05_summary.json", summary)
    write_json(run_dir / "run_manifest.json", {
        "stage": 5,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": serializable_config,
        "input_file_sha256": {
            "stage_04_summary.json": sha256_file(summary4_path),
            "lblock_final_round_timing_map.json": sha256_file(map_path),
            "sbox_dependency_profiles.npz": sha256_file(profiles_path),
            "final_round_roi_traces.npz": sha256_file(roi_path),
            "aligned_trace_metadata.csv": sha256_file(metadata_path),
        },
    })

    print("\n" + "=" * 78)
    print("Stage 05 complete: timing robustness and two-nibble selection")
    print("=" * 78)
    print("Run directory                    :", summary["run_directory"])
    print("All checks passed                :", summary["all_checks_passed"])
    print("Selected S-boxes                 :", summary["selected_sboxes"])
    print("Selected K32 nibbles             :", summary["selected_last_round_key_nibbles"])
    print("Target bits                      :", summary["selected_last_round_key_bit_indices"])
    print("Selected pair score              :", f"{summary['selected_pair_score']:.6f}")
    print("Smallest-subset within-one rate  :", summary["smallest_subset_all_centers_within_one_sample_rate"])
    print("Group-holdout within-one rate    :", summary["group_holdout_all_centers_within_one_sample_rate"])
    print("Private selected-target max error:", summary["private_selected_target_maximum_center_error_samples"])
    print("Elapsed seconds                  :", f"{summary['elapsed_seconds']:.3f}")
    print("=" * 78)

    if not all_passed:
        raise AssertionError(
            "Stage 05 failed validation; inspect stage_05_public_validation_checks.json "
            "and validation_only/private_selected_target_evaluation.json"
        )
    return summary


def load_stage_05_config(path: str | Path) -> Stage05Config:
    raw = read_json(Path(path))
    raw["subset_sizes"] = tuple(raw["subset_sizes"])
    raw["stress_scenarios"] = tuple(tuple(value) for value in raw["stress_scenarios"])
    return Stage05Config(**raw)


if __name__ == "__main__":
    run_stage_05(Stage05Config(
        input_stage4_run_directory=(
            './runs/stage_04'
            '/stage04_20260718_171931_573552_seed20260718'
        ),
        output_root='./runs/stage_05',
    ))

"""
Stage 09 — Leakage-safe ML dataset construction for the LBlock fault campaign.

This stage does not train a classifier.  It converts the successful Stage 08
campaign into an auditable machine-learning contract while preserving the
scientific separation between:

  * public observations and user-controlled glitch parameters,
  * private simulator labels and internal state,
  * key-disjoint train/validation/test/final-attack partitions.

The public ML package is generated and frozen before the Stage 08 private
Ground Truth file is opened.  The final-attack labels are exported into a
separate locked directory and are not part of the normal Stage 10 training
contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
except Exception:  # pragma: no cover - plots are optional
    plt = None


# ============================================================
# 1. Configuration and fixed scientific contracts
# ============================================================


@dataclass(frozen=True)
class Stage09Config:
    input_stage8_run_directory: str
    output_root: str = "runs/stage_09"
    random_seed: int = 20260718

    target_window_radius_samples: int = 24
    pulse_window_radius_samples: int = 24
    highpass_moving_average_width: int = 9
    trace_standard_deviation_floor: float = 1.0e-6

    duplicate_trace_round_decimals: int = 5
    association_quantile_bins: int = 16

    strict_source_train_fraction: float = 0.50
    strict_source_validation_fraction: float = 0.125
    strict_source_test_fraction: float = 0.125
    strict_source_attack_fraction: float = 0.25
    minimum_strict_source_rows_train: int = 1000
    minimum_strict_source_rows_validation: int = 100
    minimum_strict_source_rows_test: int = 100
    minimum_strict_source_rows_attack: int = 250

    maximum_primary_feature_duplicate_fraction: float = 0.001
    maximum_cross_partition_trace_duplicate_count: int = 0
    save_plots: bool = True


PARTITION_ORDER: Tuple[str, ...] = (
    "train",
    "validation",
    "test",
    "attack",
)

CATEGORY_ORDER: Tuple[str, ...] = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)

CATEGORY_TO_ID: Dict[str, int] = {
    name: index for index, name in enumerate(CATEGORY_ORDER)
}

# Fields that are allowed to exist as metadata but must never enter X.
GROUP_ONLY_PUBLIC_COLUMNS = {
    "experiment_id",
    "campaign_partition",
    "key_id",
    "session_id",
    "source_healthy_trace_id",
}

# Public fields preserved for later cryptanalytic stages, but not used by the
# fault-quality classifier.  Raw cryptographic values can create unintended
# key/state shortcuts and are unnecessary for quality estimation.
ATTACK_PAYLOAD_ONLY_COLUMNS = {
    "plaintext_hex",
    "healthy_ciphertext_hex",
    "faulty_ciphertext_hex",
}

# The assigned simulation fault model is useful for stratified reporting, but
# a physical attacker does not observe the actual internal fault mechanism.
# Therefore it is excluded from the primary model features.
STRATIFICATION_ONLY_COLUMNS = {
    "fault_model",
}

# Exact private columns and semantic prefixes that are forbidden in public X.
FORBIDDEN_EXACT_FEATURES = {
    "category",
    "category_id",
    "design_regime",
    "master_key_hex",
    "round_key_32_hex",
    "x31_hex",
    "x32_hex",
    "target_original_input",
    "target_faulted_input",
    "impacted_sboxes",
    "impacted_sbox_count",
    "target_impacted",
    "off_target_impacted",
    "changed_sbox_input_count",
    "fault_effective",
    "invalid_subtype",
    "invalid_probability",
    "global_jitter_samples",
    "injection_jitter_samples",
    "model_details_json",
    "actual_centers",
    "hit_scores",
    "activation_probabilities",
    "impacted_mask",
}

FORBIDDEN_FEATURE_PREFIXES: Tuple[str, ...] = (
    "master_key",
    "round_key",
    "true_key",
    "x31",
    "x32",
    "target_original",
    "target_faulted",
    "impacted_",
    "actual_",
    "hidden_",
    "oracle_",
    "ground_truth",
)

# Public scalar features used by the primary Stage 10 quality classifier.
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

TRACE_VIEW_NAMES: Tuple[str, ...] = (
    "full_trace_highpass",
    "target_aligned_zscore",
    "pulse_aligned_zscore",
)


# ============================================================
# 2. Generic I/O, hashing, and validation helpers
# ============================================================


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_bool_series(series: pd.Series, allow_missing: bool = False) -> pd.Series:
    """Convert common CSV boolean encodings to pandas' nullable Boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "": pd.NA,
        "nan": pd.NA,
        "none": pd.NA,
        "<na>": pd.NA,
    }
    converted = normalized.map(mapping).astype("boolean")

    if not allow_missing and converted.isna().any():
        bad = normalized[converted.isna()].dropna().unique().tolist()
        raise ValueError(f"Cannot convert boolean values: {bad[:10]}")
    return converted


def validate_config(config: Stage09Config) -> Dict[str, Any]:
    fractions = {
        "train": config.strict_source_train_fraction,
        "validation": config.strict_source_validation_fraction,
        "test": config.strict_source_test_fraction,
        "attack": config.strict_source_attack_fraction,
    }
    errors: List[str] = []

    if not np.isclose(sum(fractions.values()), 1.0):
        errors.append("Strict source-disjoint fractions must sum to one")
    if any(value <= 0.0 for value in fractions.values()):
        errors.append("Strict source-disjoint fractions must be positive")
    if config.target_window_radius_samples < 1:
        errors.append("target_window_radius_samples must be positive")
    if config.pulse_window_radius_samples < 1:
        errors.append("pulse_window_radius_samples must be positive")
    if config.highpass_moving_average_width < 3:
        errors.append("highpass_moving_average_width must be >= 3")
    if config.highpass_moving_average_width % 2 == 0:
        errors.append("highpass_moving_average_width must be odd")
    if config.association_quantile_bins < 2:
        errors.append("association_quantile_bins must be >= 2")

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "passed": True,
        "strict_source_partition_fractions": fractions,
    }


def verify_stage08_public_freeze(
    run_directory: Path,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest_path = run_directory / "public_freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = read_json(manifest_path)
    stored_freeze = str(manifest.get("freeze_sha256", ""))
    payload = dict(manifest)
    payload.pop("freeze_sha256", None)

    # Stage 08 used pretty-independent sorted JSON with default separators for
    # the freeze payload.  Reproduce that exact serialization first.
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    recomputed_freeze = hashlib.sha256(serialized).hexdigest()

    file_results: Dict[str, Any] = {}
    public_directory = run_directory / "public"
    for name, expected_hash in manifest.get("files", {}).items():
        path = public_directory / name
        observed = sha256_file(path) if path.is_file() else None
        file_results[name] = {
            "exists": path.is_file(),
            "expected_sha256": expected_hash,
            "observed_sha256": observed,
            "passed": bool(path.is_file() and observed == expected_hash),
        }

    summary_freeze = str(summary.get("public_freeze_sha256", ""))
    passed = bool(
        stored_freeze
        and stored_freeze == recomputed_freeze
        and stored_freeze == summary_freeze
        and all(item["passed"] for item in file_results.values())
    )

    return {
        "passed": passed,
        "manifest_path": str(manifest_path),
        "stored_freeze_sha256": stored_freeze,
        "recomputed_freeze_sha256": recomputed_freeze,
        "summary_freeze_sha256": summary_freeze,
        "files": file_results,
    }


def counts_from_fractions(total: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    names = list(fractions.keys())
    raw = np.asarray([total * fractions[name] for name in names], dtype=np.float64)
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(np.sum(counts))
    order = np.argsort(-(raw - counts))
    for index in order[:remainder]:
        counts[index] += 1
    return {name: int(counts[index]) for index, name in enumerate(names)}


# ============================================================
# 3. Public-only dataset loading and transformation
# ============================================================


def load_stage08_public(run_directory: Path) -> Dict[str, Any]:
    summary_path = run_directory / "stage_08_summary.json"
    public_csv_path = run_directory / "public" / "large_fault_campaign_public.csv"
    trace_path = run_directory / "public" / "large_fault_response_traces.npz"
    split_path = run_directory / "public" / "campaign_partition_manifest.json"
    feature_path = run_directory / "public" / "trace_feature_definitions.json"

    for path in (summary_path, public_csv_path, trace_path, split_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    if not bool(summary.get("all_checks_passed", False)):
        raise RuntimeError("Stage 08 did not pass all checks")

    public_df = pd.read_csv(public_csv_path, low_memory=False)
    public_df["response_received"] = normalize_bool_series(
        public_df["response_received"], allow_missing=False
    )
    public_df["ciphertext_equal"] = normalize_bool_series(
        public_df["ciphertext_equal"], allow_missing=True
    )

    with np.load(trace_path, allow_pickle=False) as archive:
        traces = np.asarray(archive["traces"], dtype=np.float32)
        trace_ids = np.asarray(archive["experiment_ids"], dtype=np.int64)
        absolute_samples = np.asarray(
            archive["absolute_sample_indices"], dtype=np.int64
        )
        sample_axis_seconds = np.asarray(
            archive["sample_axis_seconds"], dtype=np.float64
        )

    split_manifest = read_json(split_path)
    source_feature_definitions = read_json(feature_path)

    return {
        "summary": summary,
        "summary_path": summary_path,
        "public_df": public_df,
        "public_csv_path": public_csv_path,
        "traces": traces,
        "trace_ids": trace_ids,
        "trace_path": trace_path,
        "absolute_samples": absolute_samples,
        "sample_axis_seconds": sample_axis_seconds,
        "split_manifest": split_manifest,
        "split_path": split_path,
        "source_feature_definitions": source_feature_definitions,
        "source_feature_definitions_path": feature_path,
    }


def validate_public_alignment(public: Mapping[str, Any]) -> Dict[str, Any]:
    frame: pd.DataFrame = public["public_df"]
    traces: np.ndarray = public["traces"]
    trace_ids: np.ndarray = public["trace_ids"]
    absolute_samples: np.ndarray = public["absolute_samples"]
    summary: Mapping[str, Any] = public["summary"]

    required_columns = {
        "experiment_id",
        "campaign_partition",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
        "source_healthy_trace_id",
        "fault_model",
        "nominal_target_center_sample",
        "timing_offset_samples",
        "first_pulse_nominal_sample",
        "width_samples",
        "strength",
        "repeat",
        "repeat_spacing_samples",
        "plaintext_hex",
        "healthy_ciphertext_hex",
        "response_received",
        "faulty_ciphertext_hex",
        "ciphertext_equal",
        "ciphertext_hamming_distance",
        "trace_mean",
        "trace_std",
        "trace_peak_to_peak",
        "trace_max_absolute",
        "trace_high_frequency_energy",
        "target_window_energy",
        "pulse_window_energy",
        "saturation_fraction",
    }
    missing = sorted(required_columns - set(frame.columns))

    ids = frame["experiment_id"].to_numpy(dtype=np.int64)
    observed_partitions = set(frame["campaign_partition"].astype(str).unique())

    checks = {
        "required_public_columns": {
            "passed": not missing,
            "missing": missing,
        },
        "row_count_matches_summary": {
            "passed": len(frame) == int(summary["number_of_experiments"]),
            "observed": len(frame),
            "expected": int(summary["number_of_experiments"]),
        },
        "trace_shape_matches_rows": {
            "passed": bool(
                traces.ndim == 2
                and traces.shape[0] == len(frame)
                and traces.shape[1] == len(absolute_samples)
            ),
            "trace_shape": list(traces.shape),
            "absolute_sample_count": int(len(absolute_samples)),
        },
        "experiment_ids_unique": {
            "passed": len(np.unique(ids)) == len(ids),
        },
        "trace_ids_unique": {
            "passed": len(np.unique(trace_ids)) == len(trace_ids),
        },
        "trace_row_order_matches_csv": {
            "passed": bool(np.array_equal(ids, trace_ids)),
        },
        "expected_partitions_present": {
            "passed": observed_partitions == set(PARTITION_ORDER),
            "observed": sorted(observed_partitions),
            "expected": list(PARTITION_ORDER),
        },
        "finite_input_traces": {
            "passed": bool(np.all(np.isfinite(traces))),
        },
        "strictly_increasing_sample_axis": {
            "passed": bool(np.all(np.diff(absolute_samples) > 0)),
        },
    }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def moving_average_same(values: np.ndarray, width: int) -> np.ndarray:
    """Row-wise centered moving average with edge padding."""
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
    """Extract fixed windows, zero-padding samples outside the Stage 03 ROI."""
    width = 2 * radius + 1
    output = np.zeros((traces.shape[0], width), dtype=np.float32)
    valid_fraction = np.zeros(traces.shape[0], dtype=np.float32)
    first_sample = int(absolute_samples[0])
    last_sample = int(absolute_samples[-1])

    for row_index in range(traces.shape[0]):
        center_sample = int(round(float(centers[row_index])))
        requested_start = center_sample - radius
        requested_end = center_sample + radius

        valid_start = max(requested_start, first_sample)
        valid_end = min(requested_end, last_sample)
        if valid_end < valid_start:
            continue

        source_start = valid_start - first_sample
        source_end = valid_end - first_sample + 1
        destination_start = valid_start - requested_start
        destination_end = destination_start + (source_end - source_start)
        output[row_index, destination_start:destination_end] = traces[
            row_index, source_start:source_end
        ]
        valid_fraction[row_index] = float(source_end - source_start) / float(width)

    return output, valid_fraction


def build_public_feature_views(
    frame: pd.DataFrame,
    traces: np.ndarray,
    absolute_samples: np.ndarray,
    config: Stage09Config,
) -> Dict[str, Any]:
    trace_mean = np.mean(traces, axis=1, keepdims=True, dtype=np.float64)
    trace_std = np.std(traces, axis=1, keepdims=True, dtype=np.float64)
    trace_std = np.maximum(trace_std, config.trace_standard_deviation_floor)
    zscore = ((traces - trace_mean) / trace_std).astype(np.float32)
    low_frequency = moving_average_same(
        zscore, config.highpass_moving_average_width
    )
    highpass = (zscore - low_frequency).astype(np.float32)

    target_centers = frame["nominal_target_center_sample"].to_numpy(
        dtype=np.float64
    )
    pulse_centers = frame["first_pulse_nominal_sample"].to_numpy(
        dtype=np.float64
    )
    target_window, target_valid = extract_aligned_windows(
        zscore,
        absolute_samples,
        target_centers,
        config.target_window_radius_samples,
    )
    pulse_window, pulse_valid = extract_aligned_windows(
        zscore,
        absolute_samples,
        pulse_centers,
        config.pulse_window_radius_samples,
    )

    response_received = frame["response_received"].astype("boolean")
    ciphertext_equal = frame["ciphertext_equal"].astype("boolean")
    ciphertext_hd = pd.to_numeric(
        frame["ciphertext_hamming_distance"], errors="coerce"
    )

    width = pd.to_numeric(frame["width_samples"], errors="raise").to_numpy(
        dtype=np.float64
    )
    strength = pd.to_numeric(frame["strength"], errors="raise").to_numpy(
        dtype=np.float64
    )
    repeat = pd.to_numeric(frame["repeat"], errors="raise").to_numpy(
        dtype=np.float64
    )
    repeat_spacing = pd.to_numeric(
        frame["repeat_spacing_samples"], errors="raise"
    ).to_numpy(dtype=np.float64)
    timing_offset = pd.to_numeric(
        frame["timing_offset_samples"], errors="raise"
    ).to_numpy(dtype=np.float64)

    target_energy = pd.to_numeric(
        frame["target_window_energy"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    pulse_energy = pd.to_numeric(
        frame["pulse_window_energy"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    trace_std_feature = pd.to_numeric(
        frame["trace_std"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    hf_energy = pd.to_numeric(
        frame["trace_high_frequency_energy"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    equal_missing = ciphertext_equal.isna().to_numpy(dtype=np.float64)
    equal_observed = ciphertext_equal.fillna(False).astype(int).to_numpy(
        dtype=np.float64
    )
    hd_missing = ciphertext_hd.isna().to_numpy(dtype=np.float64)
    hd_observed = ciphertext_hd.fillna(-1.0).to_numpy(dtype=np.float64)

    pulse_span = width + np.maximum(repeat - 1.0, 0.0) * repeat_spacing
    glitch_energy = width * strength * repeat

    tabular = pd.DataFrame({
        "experiment_id": frame["experiment_id"].to_numpy(dtype=np.int64),
        "campaign_partition": frame["campaign_partition"].astype(str),
        "target_sbox": frame["target_sbox"].astype(str),
        "target_sbox_index": frame["target_sbox_index"].to_numpy(dtype=np.int64),
        "key_id": frame["key_id"].to_numpy(dtype=np.int64),
        "session_id": frame["session_id"].to_numpy(dtype=np.int64),
        "source_healthy_trace_id": frame["source_healthy_trace_id"].to_numpy(
            dtype=np.int64
        ),
        "fault_model": frame["fault_model"].astype(str),
        "target_is_s5": (
            frame["target_sbox_index"].to_numpy(dtype=np.int64) == 5
        ).astype(np.float64),
        "timing_offset_samples": timing_offset,
        "absolute_timing_offset_samples": np.abs(timing_offset),
        "width_samples": width,
        "strength": strength,
        "repeat": repeat,
        "repeat_spacing_samples": repeat_spacing,
        "pulse_span_samples": pulse_span,
        "glitch_energy_proxy": glitch_energy,
        "response_received_numeric": response_received.astype(int).to_numpy(
            dtype=np.float64
        ),
        "ciphertext_equal_observed": equal_observed,
        "ciphertext_equal_missing": equal_missing,
        "ciphertext_hamming_distance_observed": hd_observed,
        "ciphertext_hamming_distance_missing": hd_missing,
        "trace_mean": pd.to_numeric(frame["trace_mean"], errors="raise"),
        "trace_std": pd.to_numeric(frame["trace_std"], errors="raise"),
        "trace_peak_to_peak": pd.to_numeric(
            frame["trace_peak_to_peak"], errors="raise"
        ),
        "trace_max_absolute": pd.to_numeric(
            frame["trace_max_absolute"], errors="raise"
        ),
        "trace_high_frequency_energy": hf_energy,
        "target_window_energy": target_energy,
        "pulse_window_energy": pulse_energy,
        "saturation_fraction": pd.to_numeric(
            frame["saturation_fraction"], errors="raise"
        ),
        "pulse_to_target_energy_ratio": (
            pulse_energy / np.maximum(target_energy, 1.0e-9)
        ),
        "high_frequency_to_variance_ratio": (
            hf_energy / np.maximum(trace_std_feature ** 2, 1.0e-9)
        ),
        "target_window_valid_fraction": target_valid,
        "pulse_window_valid_fraction": pulse_valid,
    })

    primary_matrix = tabular.loc[:, list(PRIMARY_TABULAR_FEATURES)].to_numpy(
        dtype=np.float32
    )

    return {
        "tabular": tabular,
        "primary_matrix": primary_matrix,
        "full_trace_highpass": highpass,
        "target_aligned_zscore": target_window,
        "pulse_aligned_zscore": pulse_window,
        "absolute_samples": absolute_samples,
    }


# ============================================================
# 4. Public-only split, overlap, and duplicate audits
# ============================================================


def assign_strict_source_partitions(
    source_ids: Sequence[int],
    config: Stage09Config,
) -> Dict[int, str]:
    unique_ids = np.asarray(sorted(set(map(int, source_ids))), dtype=np.int64)
    rng = np.random.default_rng(config.random_seed + 9009)
    rng.shuffle(unique_ids)

    fractions = {
        "train": config.strict_source_train_fraction,
        "validation": config.strict_source_validation_fraction,
        "test": config.strict_source_test_fraction,
        "attack": config.strict_source_attack_fraction,
    }
    counts = counts_from_fractions(len(unique_ids), fractions)

    result: Dict[int, str] = {}
    cursor = 0
    for partition in PARTITION_ORDER:
        next_cursor = cursor + counts[partition]
        for source_id in unique_ids[cursor:next_cursor]:
            result[int(source_id)] = partition
        cursor = next_cursor

    if len(result) != len(unique_ids):
        raise AssertionError("Strict source assignment is incomplete")
    return result


def build_strict_source_membership(
    tabular: pd.DataFrame,
    config: Stage09Config,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    assignment = assign_strict_source_partitions(
        tabular["source_healthy_trace_id"].to_numpy(dtype=np.int64),
        config,
    )
    assigned = tabular["source_healthy_trace_id"].map(assignment).astype(str)
    included = assigned == tabular["campaign_partition"].astype(str)

    membership = pd.DataFrame({
        "experiment_id": tabular["experiment_id"].to_numpy(dtype=np.int64),
        "campaign_partition": tabular["campaign_partition"].astype(str),
        "source_healthy_trace_id": tabular[
            "source_healthy_trace_id"
        ].to_numpy(dtype=np.int64),
        "assigned_source_partition": assigned,
        "included_in_strict_source_disjoint_subset": included.astype(bool),
    })

    included_frame = membership.loc[included].copy()
    row_counts = {
        partition: int(
            np.sum(
                included_frame["campaign_partition"].to_numpy(dtype=str)
                == partition
            )
        )
        for partition in PARTITION_ORDER
    }
    source_sets = {
        partition: set(
            included_frame.loc[
                included_frame["campaign_partition"] == partition,
                "source_healthy_trace_id",
            ].astype(int)
        )
        for partition in PARTITION_ORDER
    }
    overlaps: Dict[str, int] = {}
    for left_index, left in enumerate(PARTITION_ORDER):
        for right in PARTITION_ORDER[left_index + 1:]:
            overlaps[f"{left}__{right}"] = len(
                source_sets[left] & source_sets[right]
            )

    minimums = {
        "train": config.minimum_strict_source_rows_train,
        "validation": config.minimum_strict_source_rows_validation,
        "test": config.minimum_strict_source_rows_test,
        "attack": config.minimum_strict_source_rows_attack,
    }
    passed = bool(
        all(row_counts[name] >= minimums[name] for name in PARTITION_ORDER)
        and all(value == 0 for value in overlaps.values())
    )

    audit = {
        "passed": passed,
        "unique_source_trace_count": len(assignment),
        "included_row_count": int(included.sum()),
        "included_fraction": float(included.mean()),
        "included_rows_by_partition": row_counts,
        "minimum_rows_by_partition": minimums,
        "cross_partition_source_overlap_counts": overlaps,
        "statement": (
            "This robustness subset is simultaneously key-disjoint and "
            "healthy-source-trace-disjoint. It is intended for secondary "
            "evaluation, not for replacing the primary Stage 08 partitions."
        ),
    }
    return membership, audit


def pairwise_set_overlaps(
    frame: pd.DataFrame,
    column: str,
) -> Dict[str, int]:
    sets = {
        partition: set(
            frame.loc[
                frame["campaign_partition"] == partition,
                column,
            ].dropna().astype(str)
        )
        for partition in PARTITION_ORDER
    }
    result: Dict[str, int] = {}
    for left_index, left in enumerate(PARTITION_ORDER):
        for right in PARTITION_ORDER[left_index + 1:]:
            result[f"{left}__{right}"] = len(sets[left] & sets[right])
    return result


def row_hashes(matrix: np.ndarray, decimals: int) -> np.ndarray:
    rounded = np.round(matrix.astype(np.float64), decimals=decimals)
    hashes = np.empty(rounded.shape[0], dtype="U16")
    for index, row in enumerate(rounded):
        hashes[index] = hashlib.blake2b(
            np.ascontiguousarray(row).tobytes(), digest_size=8
        ).hexdigest()
    return hashes


def cross_partition_duplicate_count(
    hashes: Sequence[str],
    partitions: Sequence[str],
) -> Dict[str, Any]:
    frame = pd.DataFrame({
        "hash": np.asarray(hashes, dtype=str),
        "partition": np.asarray(partitions, dtype=str),
    })
    grouped = frame.groupby("hash", sort=False)["partition"].nunique()
    cross_hashes = set(grouped[grouped > 1].index.astype(str))
    row_count = int(frame["hash"].isin(cross_hashes).sum())
    return {
        "cross_partition_duplicate_hash_count": int(len(cross_hashes)),
        "rows_in_cross_partition_duplicate_groups": row_count,
        "row_fraction": float(row_count / max(len(frame), 1)),
    }


def public_leakage_and_overlap_audit(
    original_public: pd.DataFrame,
    tabular: pd.DataFrame,
    primary_matrix: np.ndarray,
    trace_view: np.ndarray,
    strict_audit: Mapping[str, Any],
    config: Stage09Config,
) -> Dict[str, Any]:
    feature_names = set(PRIMARY_TABULAR_FEATURES)
    forbidden_exact_present = sorted(feature_names & FORBIDDEN_EXACT_FEATURES)
    forbidden_prefix_present = sorted(
        name
        for name in feature_names
        if any(name.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
    )
    group_columns_present = sorted(feature_names & GROUP_ONLY_PUBLIC_COLUMNS)
    payload_columns_present = sorted(feature_names & ATTACK_PAYLOAD_ONLY_COLUMNS)
    stratification_columns_present = sorted(
        feature_names & STRATIFICATION_ONLY_COLUMNS
    )

    primary_hashes = row_hashes(primary_matrix, decimals=8)
    primary_duplicates = cross_partition_duplicate_count(
        primary_hashes,
        tabular["campaign_partition"].astype(str).to_numpy(),
    )
    trace_hash_values = row_hashes(
        trace_view,
        decimals=config.duplicate_trace_round_decimals,
    )
    trace_duplicates = cross_partition_duplicate_count(
        trace_hash_values,
        tabular["campaign_partition"].astype(str).to_numpy(),
    )

    source_overlap = pairwise_set_overlaps(
        original_public,
        "source_healthy_trace_id",
    )
    plaintext_overlap = pairwise_set_overlaps(
        original_public,
        "plaintext_hex",
    )
    ciphertext_overlap = pairwise_set_overlaps(
        original_public,
        "healthy_ciphertext_hex",
    )

    key_sets = {
        partition: set(
            original_public.loc[
                original_public["campaign_partition"] == partition,
                "key_id",
            ].astype(int)
        )
        for partition in PARTITION_ORDER
    }
    key_overlaps: Dict[str, int] = {}
    for left_index, left in enumerate(PARTITION_ORDER):
        for right in PARTITION_ORDER[left_index + 1:]:
            key_overlaps[f"{left}__{right}"] = len(
                key_sets[left] & key_sets[right]
            )

    passed = bool(
        not forbidden_exact_present
        and not forbidden_prefix_present
        and not group_columns_present
        and not payload_columns_present
        and not stratification_columns_present
        and all(value == 0 for value in key_overlaps.values())
        and primary_duplicates["row_fraction"]
            <= config.maximum_primary_feature_duplicate_fraction
        and trace_duplicates["cross_partition_duplicate_hash_count"]
            <= config.maximum_cross_partition_trace_duplicate_count
        and bool(strict_audit["passed"])
        and np.all(np.isfinite(primary_matrix))
        and np.all(np.isfinite(trace_view))
    )

    return {
        "passed": passed,
        "primary_feature_names": list(PRIMARY_TABULAR_FEATURES),
        "forbidden_exact_features_present": forbidden_exact_present,
        "forbidden_prefix_features_present": forbidden_prefix_present,
        "group_only_columns_selected_as_features": group_columns_present,
        "attack_payload_columns_selected_as_features": payload_columns_present,
        "stratification_only_columns_selected_as_features": (
            stratification_columns_present
        ),
        "key_overlap_counts": key_overlaps,
        "plaintext_overlap_counts": plaintext_overlap,
        "healthy_ciphertext_overlap_counts": ciphertext_overlap,
        "healthy_source_trace_overlap_counts": source_overlap,
        "source_overlap_interpretation": (
            "Stage 08 intentionally reuses the finite healthy ROI source pool. "
            "source_healthy_trace_id is excluded from X, traces are normalized, "
            "and a stricter source-disjoint robustness subset is exported."
        ),
        "primary_tabular_duplicate_audit": primary_duplicates,
        "processed_trace_duplicate_audit": trace_duplicates,
        "strict_source_disjoint_subset": dict(strict_audit),
        "all_primary_features_finite": bool(np.all(np.isfinite(primary_matrix))),
        "all_processed_traces_finite": bool(np.all(np.isfinite(trace_view))),
    }


# ============================================================
# 5. Private label loading, joining, and consistency checks
# ============================================================


def load_private_labels_after_public_freeze(run_directory: Path) -> pd.DataFrame:
    path = (
        run_directory
        / "private_ground_truth"
        / "large_fault_ground_truth.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def build_label_table(private_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "experiment_id",
        "campaign_partition",
        "category",
        "category_id",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
    }
    missing = sorted(required - set(private_df.columns))
    if missing:
        raise KeyError(f"Missing private label columns: {missing}")

    category = private_df["category"].astype(str)
    output = pd.DataFrame({
        "experiment_id": private_df["experiment_id"].to_numpy(dtype=np.int64),
        "campaign_partition": private_df["campaign_partition"].astype(str),
        "category": category,
        "category_id": private_df["category_id"].to_numpy(dtype=np.int64),
        "target_sbox": private_df["target_sbox"].astype(str),
        "target_sbox_index": private_df["target_sbox_index"].to_numpy(
            dtype=np.int64
        ),
        "key_id": private_df["key_id"].to_numpy(dtype=np.int64),
        "session_id": private_df["session_id"].to_numpy(dtype=np.int64),
        "is_valid_response": (category != "invalid_reset").astype(np.int8),
        "is_clean_target": category.isin({
            "clean_target_ineffective",
            "clean_target_effective",
        }).astype(np.int8),
        "is_clean_target_ineffective": (
            category == "clean_target_ineffective"
        ).astype(np.int8),
        "is_clean_target_effective": (
            category == "clean_target_effective"
        ).astype(np.int8),
        "is_missed": (category == "missed").astype(np.int8),
        "is_off_target": (category == "off_target").astype(np.int8),
        "is_multi_hit": (category == "multi_hit").astype(np.int8),
        "is_invalid_reset": (category == "invalid_reset").astype(np.int8),
        "sifa_oracle_usable": (
            category == "clean_target_ineffective"
        ).astype(np.int8),
        "sefa_oracle_usable": (
            category == "clean_target_effective"
        ).astype(np.int8),
        "shfa_oracle_branch": np.select(
            [
                category == "clean_target_ineffective",
                category == "clean_target_effective",
            ],
            [1, 2],
            default=0,
        ).astype(np.int8),
    })
    return output


def validate_label_join(
    public_df: pd.DataFrame,
    labels: pd.DataFrame,
) -> Dict[str, Any]:
    public_ids = public_df["experiment_id"].to_numpy(dtype=np.int64)
    label_ids = labels["experiment_id"].to_numpy(dtype=np.int64)

    joined = public_df.loc[:, [
        "experiment_id",
        "campaign_partition",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
        "response_received",
        "ciphertext_equal",
    ]].merge(
        labels,
        on="experiment_id",
        how="outer",
        suffixes=("_public", "_private"),
        indicator=True,
        validate="one_to_one",
    )

    category_expected_ids = labels["category"].map(CATEGORY_TO_ID)
    category_id_consistent = bool(
        category_expected_ids.notna().all()
        and np.array_equal(
            category_expected_ids.to_numpy(dtype=np.int64),
            labels["category_id"].to_numpy(dtype=np.int64),
        )
    )

    metadata_columns = (
        "campaign_partition",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
    )
    metadata_consistency = {
        column: bool(
            (
                joined[f"{column}_public"].astype(str)
                == joined[f"{column}_private"].astype(str)
            ).all()
        )
        for column in metadata_columns
    }

    response_received = joined["response_received"].astype("boolean")
    category = joined["category"].astype(str)
    response_consistency = bool(
        (
            response_received.fillna(False).to_numpy(dtype=bool)
            == (category != "invalid_reset").to_numpy(dtype=bool)
        ).all()
    )

    # Clean ineffective rows must have an observed equal ciphertext; clean
    # effective rows must have an observed different ciphertext.
    equal = joined["ciphertext_equal"].astype("boolean")
    ineffective_mask = category == "clean_target_ineffective"
    effective_mask = category == "clean_target_effective"
    clean_observable_consistency = bool(
        equal.loc[ineffective_mask].fillna(False).all()
        and (~equal.loc[effective_mask].fillna(True)).all()
    )

    checks = {
        "public_ids_unique": len(np.unique(public_ids)) == len(public_ids),
        "private_ids_unique": len(np.unique(label_ids)) == len(label_ids),
        "id_sets_equal": set(public_ids.tolist()) == set(label_ids.tolist()),
        "one_to_one_join_complete": bool((joined["_merge"] == "both").all()),
        "category_id_mapping_consistent": category_id_consistent,
        "metadata_consistent": all(metadata_consistency.values()),
        "response_status_consistent": response_consistency,
        "clean_ciphertext_observables_consistent": clean_observable_consistency,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metadata_column_consistency": metadata_consistency,
    }


# ============================================================
# 6. Label-aware diagnostics performed only after public freeze
# ============================================================


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def discrete_mutual_information(x: Sequence[Any], y: Sequence[Any]) -> float:
    frame = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)})
    contingency = pd.crosstab(frame["x"], frame["y"], dropna=False).to_numpy(
        dtype=np.float64
    )
    total = float(np.sum(contingency))
    if total <= 0:
        return 0.0
    row = np.sum(contingency, axis=1, keepdims=True)
    column = np.sum(contingency, axis=0, keepdims=True)
    expected = row @ column / total
    mask = contingency > 0
    return float(
        np.sum(
            (contingency[mask] / total)
            * np.log2(contingency[mask] / expected[mask])
        )
    )


def discretize_for_association(series: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return series.astype(str).fillna("<missing>")

    unique_count = int(numeric.nunique(dropna=True))
    if unique_count <= bins:
        return numeric.fillna(-9.87654321e99).astype(str)

    ranked = numeric.rank(method="first")
    binned = pd.qcut(
        ranked,
        q=min(bins, unique_count),
        duplicates="drop",
    )
    return binned.astype(str).where(numeric.notna(), "<missing>")


def feature_label_association(
    tabular: pd.DataFrame,
    labels: pd.DataFrame,
    config: Stage09Config,
) -> pd.DataFrame:
    merged = tabular.merge(
        labels.loc[:, ["experiment_id", "category"]],
        on="experiment_id",
        how="inner",
        validate="one_to_one",
    )
    train = merged.loc[merged["campaign_partition"] == "train"].copy()
    y = train["category"].astype(str)
    label_entropy = entropy_from_counts(y.value_counts().to_numpy())

    rows: List[Dict[str, Any]] = []
    for feature in PRIMARY_TABULAR_FEATURES:
        discretized = discretize_for_association(
            train[feature], config.association_quantile_bins
        )
        mutual_information = discrete_mutual_information(discretized, y)
        normalized = mutual_information / max(label_entropy, 1.0e-12)

        contingency = pd.crosstab(discretized, y, dropna=False)
        conditional_purity = float(
            contingency.max(axis=1).sum() / max(contingency.to_numpy().sum(), 1)
        )
        rows.append({
            "feature": feature,
            "train_unique_value_count": int(train[feature].nunique(dropna=False)),
            "discretized_value_count": int(discretized.nunique(dropna=False)),
            "mutual_information_bits": float(mutual_information),
            "normalized_mutual_information": float(normalized),
            "single_feature_majority_purity": conditional_purity,
            "interpretation": (
                "strong_observable_shortcut"
                if normalized >= 0.50
                else "moderate_association"
                if normalized >= 0.15
                else "limited_association"
            ),
        })

    return pd.DataFrame(rows).sort_values(
        "normalized_mutual_information", ascending=False
    ).reset_index(drop=True)


def partition_class_statistics(labels: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for partition in PARTITION_ORDER:
        selected = labels.loc[labels["campaign_partition"] == partition]
        total = len(selected)
        for category in CATEGORY_ORDER:
            count = int(np.sum(selected["category"].astype(str) == category))
            rows.append({
                "campaign_partition": partition,
                "category": category,
                "count": count,
                "fraction": float(count / max(total, 1)),
            })
    return pd.DataFrame(rows)


def partition_target_model_statistics(
    public_df: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    merged = public_df.loc[:, [
        "experiment_id",
        "campaign_partition",
        "target_sbox",
        "fault_model",
    ]].merge(
        labels.loc[:, ["experiment_id", "category"]],
        on="experiment_id",
        how="inner",
        validate="one_to_one",
    )
    grouped = (
        merged.groupby(
            ["campaign_partition", "target_sbox", "fault_model", "category"],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    group_totals = grouped.groupby(
        ["campaign_partition", "target_sbox", "fault_model"]
    )["count"].transform("sum")
    grouped["within_group_fraction"] = grouped["count"] / group_totals
    return grouped


# ============================================================
# 7. Plotting helpers
# ============================================================


def save_public_plots(
    output_directory: Path,
    tabular: pd.DataFrame,
    strict_membership: pd.DataFrame,
    config: Stage09Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []

    partition_counts = [
        int(np.sum(tabular["campaign_partition"] == partition))
        for partition in PARTITION_ORDER
    ]
    strict_counts = [
        int(np.sum(
            (strict_membership["campaign_partition"] == partition)
            & strict_membership[
                "included_in_strict_source_disjoint_subset"
            ].astype(bool)
        ))
        for partition in PARTITION_ORDER
    ]

    figure = plt.figure(figsize=(9.5, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    positions = np.arange(len(PARTITION_ORDER))
    axis.bar(positions - 0.18, partition_counts, width=0.36, label="Primary")
    axis.bar(positions + 0.18, strict_counts, width=0.36, label="Strict source-disjoint")
    axis.set_xticks(positions)
    axis.set_xticklabels(PARTITION_ORDER)
    axis.set_ylabel("Row count")
    axis.set_title("Stage 09 public ML partition sizes")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "public_partition_and_strict_subset_sizes.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    figure = plt.figure(figsize=(10, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    for target in ("S0", "S5"):
        selected = tabular.loc[tabular["target_sbox"] == target]
        axis.hist(
            selected["timing_offset_samples"].to_numpy(dtype=float),
            bins=50,
            alpha=0.45,
            label=target,
        )
    axis.set_xlabel("Timing offset [samples]")
    axis.set_ylabel("Count")
    axis.set_title("Public timing-offset coverage retained for ML")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_directory / "public_timing_offset_coverage.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    return generated


def save_validation_plots(
    output_directory: Path,
    class_stats: pd.DataFrame,
    association: pd.DataFrame,
    config: Stage09Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []

    generated: List[str] = []

    matrix = np.zeros((len(PARTITION_ORDER), len(CATEGORY_ORDER)), dtype=np.int64)
    for partition_index, partition in enumerate(PARTITION_ORDER):
        for category_index, category in enumerate(CATEGORY_ORDER):
            match = class_stats.loc[
                (class_stats["campaign_partition"] == partition)
                & (class_stats["category"] == category),
                "count",
            ]
            matrix[partition_index, category_index] = int(match.iloc[0])

    figure = plt.figure(figsize=(12, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_yticks(range(len(PARTITION_ORDER)))
    axis.set_yticklabels(PARTITION_ORDER)
    axis.set_xticks(range(len(CATEGORY_ORDER)))
    axis.set_xticklabels(CATEGORY_ORDER, rotation=35, ha="right")
    axis.set_title("Private six-class distribution by fixed partition")
    figure.colorbar(image, ax=axis, label="Count")
    figure.tight_layout()
    path = output_directory / "private_partition_class_distribution.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    top = association.head(12).iloc[::-1]
    figure = plt.figure(figsize=(10.5, 6.5))
    axis = figure.add_subplot(1, 1, 1)
    axis.barh(
        top["feature"].astype(str),
        top["normalized_mutual_information"].to_numpy(dtype=float),
    )
    axis.set_xlabel("Normalized mutual information with six-class label")
    axis.set_title("Train-only single-feature association audit")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    path = output_directory / "train_feature_label_association.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    generated.append(path.name)

    return generated


# ============================================================
# 8. Complete Stage 09 runner
# ============================================================


def run_stage_09(config: Stage09Config) -> Dict[str, Any]:
    start_time = time.perf_counter()
    configuration_validation = validate_config(config)

    stage8_run_directory = Path(
        config.input_stage8_run_directory
    ).expanduser().resolve()
    public = load_stage08_public(stage8_run_directory)
    stage8_summary = public["summary"]

    stage8_freeze_validation = verify_stage08_public_freeze(
        stage8_run_directory,
        stage8_summary,
    )
    if not stage8_freeze_validation["passed"]:
        raise RuntimeError("Stage 08 public freeze verification failed")

    public_alignment = validate_public_alignment(public)
    if not public_alignment["passed"]:
        raise RuntimeError("Stage 08 public row/trace alignment failed")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage09_{timestamp}_seed{config.random_seed}"
    run_directory = Path(config.output_root).expanduser().resolve() / run_id
    public_ml_directory = run_directory / "public_ml"
    private_label_directory = run_directory / "private_labels"
    locked_attack_directory = run_directory / "locked_attack_labels"
    validation_directory = run_directory / "validation_only"
    public_ml_directory.mkdir(parents=True, exist_ok=False)

    public_df: pd.DataFrame = public["public_df"]
    feature_views = build_public_feature_views(
        public_df,
        public["traces"],
        public["absolute_samples"],
        config,
    )
    tabular: pd.DataFrame = feature_views["tabular"]

    strict_membership, strict_audit = build_strict_source_membership(
        tabular,
        config,
    )

    leakage_audit = public_leakage_and_overlap_audit(
        public_df,
        tabular,
        feature_views["primary_matrix"],
        feature_views["full_trace_highpass"],
        strict_audit,
        config,
    )
    if not leakage_audit["passed"]:
        raise RuntimeError("Stage 09 public leakage audit failed")

    # ------------------------- public ML outputs ----------------------
    tabular.to_csv(
        public_ml_directory / "ml_tabular_features_public.csv",
        index=False,
    )

    np.savez_compressed(
        public_ml_directory / "ml_trace_views_public.npz",
        experiment_ids=tabular["experiment_id"].to_numpy(dtype=np.int64),
        full_trace_highpass=feature_views["full_trace_highpass"],
        target_aligned_zscore=feature_views["target_aligned_zscore"],
        pulse_aligned_zscore=feature_views["pulse_aligned_zscore"],
        absolute_sample_indices=public["absolute_samples"],
        sample_axis_seconds=public["sample_axis_seconds"],
    )

    strict_membership.to_csv(
        public_ml_directory / "strict_source_disjoint_membership.csv",
        index=False,
    )
    strict_ids = strict_membership.loc[
        strict_membership[
            "included_in_strict_source_disjoint_subset"
        ].astype(bool),
        "experiment_id",
    ].to_numpy(dtype=np.int64)
    np.savez_compressed(
        public_ml_directory / "strict_source_disjoint_experiment_ids.npz",
        experiment_ids=strict_ids,
    )

    # Preserve only the public cryptanalytic payload for the final attack
    # stages.  This file is not an ML feature matrix.
    attack_payload_columns = [
        "experiment_id",
        "campaign_partition",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
        "fault_model",
        "timing_offset_samples",
        "width_samples",
        "strength",
        "repeat",
        "repeat_spacing_samples",
        "plaintext_hex",
        "healthy_ciphertext_hex",
        "response_received",
        "faulty_ciphertext_hex",
        "ciphertext_equal",
        "ciphertext_hamming_distance",
    ]
    public_df.loc[
        public_df["campaign_partition"] == "attack",
        attack_payload_columns,
    ].to_csv(
        public_ml_directory / "final_attack_payload_public.csv",
        index=False,
    )

    feature_contract = {
        "stage": 9,
        "primary_quality_classifier_feature_names": list(
            PRIMARY_TABULAR_FEATURES
        ),
        "primary_trace_view": "target_aligned_zscore",
        "secondary_trace_views": [
            "pulse_aligned_zscore",
            "full_trace_highpass",
        ],
        "trace_view_shapes": {
            "full_trace_highpass": list(
                feature_views["full_trace_highpass"].shape
            ),
            "target_aligned_zscore": list(
                feature_views["target_aligned_zscore"].shape
            ),
            "pulse_aligned_zscore": list(
                feature_views["pulse_aligned_zscore"].shape
            ),
        },
        "metadata_only_columns": sorted(GROUP_ONLY_PUBLIC_COLUMNS),
        "attack_payload_only_columns": sorted(ATTACK_PAYLOAD_ONLY_COLUMNS),
        "stratification_only_columns": sorted(STRATIFICATION_ONLY_COLUMNS),
        "forbidden_exact_feature_names": sorted(FORBIDDEN_EXACT_FEATURES),
        "forbidden_feature_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "preprocessing_rule": (
            "Fit imputation/scaling/model parameters on Train only. "
            "Use Validation for model selection, Test once for final classifier "
            "evaluation, and never use Attack labels during Stages 10-14."
        ),
        "fault_model_feature_policy": (
            "fault_model is excluded from the primary model because the true "
            "physical fault mechanism is not directly observable."
        ),
        "cryptographic_value_policy": (
            "Raw plaintext and ciphertext values are retained only in the "
            "final attack payload and are excluded from quality-classifier X."
        ),
    }
    write_json(
        public_ml_directory / "ml_feature_contract.json",
        feature_contract,
    )

    split_contract = {
        "stage": 9,
        "split_unit": "key_id",
        "partition_order": list(PARTITION_ORDER),
        "partition_counts": stage8_summary["partition_counts"],
        "partition_key_ids": stage8_summary["partition_key_ids"],
        "allowed_uses": {
            "train": "fit preprocessing and model parameters",
            "validation": "select hyperparameters and thresholds",
            "test": "one-time held-out model evaluation",
            "attack": "apply frozen model and perform final cryptanalysis",
        },
        "attack_labels_locked": True,
        "strict_source_disjoint_subset_available": True,
    }
    write_json(
        public_ml_directory / "ml_split_contract.json",
        split_contract,
    )

    dataset_card = {
        "stage": 9,
        "name": "LBlock Stage 08 fault-quality ML dataset",
        "row_count": int(len(tabular)),
        "target_sboxes": stage8_summary["selected_sboxes"],
        "target_key_nibbles": stage8_summary[
            "selected_last_round_key_nibbles"
        ],
        "target_bit_indices": stage8_summary[
            "selected_last_round_key_bit_indices"
        ],
        "label_space": list(CATEGORY_ORDER),
        "public_observations_only_in_X": True,
        "known_limitations": [
            "The campaign is simulated rather than measured on hardware.",
            "Healthy source traces are reused across primary partitions.",
            "A key-and-source-disjoint robustness subset is provided.",
            "Observable ciphertext equality is a legitimate but strong shortcut.",
            "Attack labels must remain unopened during model development.",
        ],
    }
    write_json(public_ml_directory / "dataset_card.json", dataset_card)

    write_json(
        public_ml_directory / "public_leakage_audit.json",
        leakage_audit,
    )
    write_json(
        public_ml_directory / "strict_source_disjoint_audit.json",
        strict_audit,
    )

    public_plots = save_public_plots(
        public_ml_directory,
        tabular,
        strict_membership,
        config,
    )

    # Freeze every public Stage 09 output before opening Stage 08 private data.
    public_files = sorted(
        path for path in public_ml_directory.iterdir() if path.is_file()
    )
    freeze_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": (
            "Stage 09 public ML features, trace views, split contract, and "
            "leakage audit were frozen before Stage 08 private labels were opened."
        ),
        "source_stage_08_public_freeze_sha256": stage8_summary[
            "public_freeze_sha256"
        ],
        "files": {
            path.name: sha256_file(path)
            for path in public_files
        },
    }
    freeze_sha256 = stable_json_sha256(freeze_manifest)
    freeze_manifest["freeze_sha256"] = freeze_sha256
    write_json(
        run_directory / "public_ml_freeze_manifest.json",
        freeze_manifest,
    )

    # ---------------------- private labels after freeze --------------
    private_df = load_private_labels_after_public_freeze(stage8_run_directory)
    labels = build_label_table(private_df)
    label_join_validation = validate_label_join(public_df, labels)
    if not label_join_validation["passed"]:
        raise RuntimeError("Stage 09 public/private label join failed")

    private_label_directory.mkdir(parents=True, exist_ok=True)
    locked_attack_directory.mkdir(parents=True, exist_ok=True)
    development_labels = labels.loc[
        labels["campaign_partition"].isin({"train", "validation", "test"})
    ].copy()
    attack_labels = labels.loc[
        labels["campaign_partition"] == "attack"
    ].copy()

    development_labels.to_csv(
        private_label_directory / "ml_quality_labels_train_validation_test.csv",
        index=False,
    )
    np.savez_compressed(
        private_label_directory / "ml_quality_labels_train_validation_test.npz",
        experiment_ids=development_labels["experiment_id"].to_numpy(
            dtype=np.int64
        ),
        category_ids=development_labels["category_id"].to_numpy(dtype=np.int8),
        is_clean_target=development_labels["is_clean_target"].to_numpy(
            dtype=np.int8
        ),
        is_clean_target_ineffective=development_labels[
            "is_clean_target_ineffective"
        ].to_numpy(dtype=np.int8),
        is_clean_target_effective=development_labels[
            "is_clean_target_effective"
        ].to_numpy(dtype=np.int8),
    )

    attack_labels.to_csv(
        locked_attack_directory / "attack_quality_labels_LOCKED.csv",
        index=False,
    )
    attack_label_hash = sha256_file(
        locked_attack_directory / "attack_quality_labels_LOCKED.csv"
    )
    write_json(
        locked_attack_directory / "DO_NOT_OPEN_BEFORE_FINAL_ATTACK.json",
        {
            "stage": 9,
            "warning": (
                "Do not read these labels during Stage 10 model training, "
                "Stage 11 optimization, Stage 12 closed-loop tuning, or the "
                "construction of Stages 13-14 attack scores."
            ),
            "allowed_first_use": (
                "Final post-freeze benchmark and attack-quality evaluation"
            ),
            "row_count": int(len(attack_labels)),
            "sha256": attack_label_hash,
        },
    )

    # ------------------------- validation-only outputs ----------------
    validation_directory.mkdir(parents=True, exist_ok=True)
    class_stats = partition_class_statistics(labels)
    target_model_stats = partition_target_model_statistics(public_df, labels)
    association = feature_label_association(tabular, labels, config)

    class_stats.to_csv(
        validation_directory / "partition_class_distribution.csv",
        index=False,
    )
    target_model_stats.to_csv(
        validation_directory / "partition_target_model_class_distribution.csv",
        index=False,
    )
    association.to_csv(
        validation_directory / "train_feature_label_association.csv",
        index=False,
    )

    label_counts = {
        partition: {
            category: int(np.sum(
                (labels["campaign_partition"] == partition)
                & (labels["category"] == category)
            ))
            for category in CATEGORY_ORDER
        }
        for partition in PARTITION_ORDER
    }

    private_use_manifest = {
        "private_data_opened_after_public_ml_freeze": True,
        "public_ml_freeze_sha256": freeze_sha256,
        "private_columns_used_as_labels": [
            "category",
            "category_id",
        ],
        "private_columns_used_for_join_validation_only": [
            "campaign_partition",
            "target_sbox",
            "target_sbox_index",
            "key_id",
            "session_id",
        ],
        "private_columns_never_selected_as_features": sorted(
            FORBIDDEN_EXACT_FEATURES
        ),
        "attack_label_file_sha256": attack_label_hash,
    }
    write_json(
        validation_directory / "private_data_use_manifest.json",
        private_use_manifest,
    )

    validation_plots = save_validation_plots(
        validation_directory,
        class_stats,
        association,
        config,
    )

    overall_checks = {
        "configuration_validation": configuration_validation,
        "stage_08_all_checks_passed": {
            "passed": bool(stage8_summary["all_checks_passed"]),
        },
        "stage_08_public_freeze_verified": stage8_freeze_validation,
        "public_row_trace_alignment": public_alignment,
        "public_leakage_audit": leakage_audit,
        "public_private_label_join": label_join_validation,
        "fixed_key_partitions_preserved": {
            "passed": all(
                int(np.sum(tabular["campaign_partition"] == partition))
                == int(stage8_summary["partition_counts"][partition])
                for partition in PARTITION_ORDER
            ),
        },
        "attack_rows_excluded_from_development_labels": {
            "passed": bool(
                "attack" not in set(
                    development_labels["campaign_partition"].astype(str)
                )
                and set(attack_labels["campaign_partition"].astype(str))
                    == {"attack"}
            ),
        },
        "all_six_classes_present_in_train": {
            "passed": set(
                labels.loc[
                    labels["campaign_partition"] == "train",
                    "category",
                ].astype(str)
            ) == set(CATEGORY_ORDER),
        },
        "all_six_classes_present_in_validation": {
            "passed": set(
                labels.loc[
                    labels["campaign_partition"] == "validation",
                    "category",
                ].astype(str)
            ) == set(CATEGORY_ORDER),
        },
        "all_six_classes_present_in_test": {
            "passed": set(
                labels.loc[
                    labels["campaign_partition"] == "test",
                    "category",
                ].astype(str)
            ) == set(CATEGORY_ORDER),
        },
        "public_ml_freeze_created": {
            "passed": bool(freeze_sha256),
            "freeze_sha256": freeze_sha256,
        },
    }
    all_checks_passed = all(
        bool(value["passed"])
        for value in overall_checks.values()
    )
    write_json(
        validation_directory / "stage_09_dataset_validation.json",
        {
            "all_checks_passed": all_checks_passed,
            "checks": overall_checks,
            "label_counts_by_partition": label_counts,
        },
    )

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "stage": 9,
        "run_id": run_id,
        "run_directory": str(run_directory.resolve()),
        "input_stage_08_run_directory": str(stage8_run_directory),
        "public_ml_directory": str(public_ml_directory.resolve()),
        "private_label_directory": str(private_label_directory.resolve()),
        "locked_attack_label_directory": str(locked_attack_directory.resolve()),
        "validation_only_directory": str(validation_directory.resolve()),
        "all_checks_passed": bool(all_checks_passed),
        "stage_08_public_freeze_verified": bool(
            stage8_freeze_validation["passed"]
        ),
        "public_leakage_audit_passed": bool(leakage_audit["passed"]),
        "public_private_label_join_passed": bool(
            label_join_validation["passed"]
        ),
        "number_of_rows": int(len(tabular)),
        "number_of_primary_tabular_features": int(
            len(PRIMARY_TABULAR_FEATURES)
        ),
        "primary_tabular_features": list(PRIMARY_TABULAR_FEATURES),
        "trace_view_shapes": {
            name: list(feature_views[name].shape)
            for name in TRACE_VIEW_NAMES
        },
        "partition_counts": {
            partition: int(np.sum(tabular["campaign_partition"] == partition))
            for partition in PARTITION_ORDER
        },
        "partition_key_ids": stage8_summary["partition_key_ids"],
        "development_label_row_count": int(len(development_labels)),
        "locked_attack_label_row_count": int(len(attack_labels)),
        "label_counts_by_partition": label_counts,
        "strict_source_disjoint_subset": strict_audit,
        "public_source_trace_overlap_counts": leakage_audit[
            "healthy_source_trace_overlap_counts"
        ],
        "cross_partition_processed_trace_duplicates": leakage_audit[
            "processed_trace_duplicate_audit"
        ],
        "top_train_feature_label_associations": association.head(8).to_dict(
            orient="records"
        ),
        "public_ml_freeze_sha256": freeze_sha256,
        "attack_label_sha256": attack_label_hash,
        "elapsed_seconds": float(elapsed_seconds),
        "public_ml_files": sorted(
            path.name for path in public_ml_directory.iterdir() if path.is_file()
        ),
        "private_label_files": sorted(
            path.name for path in private_label_directory.iterdir() if path.is_file()
        ),
        "locked_attack_files": sorted(
            path.name for path in locked_attack_directory.iterdir() if path.is_file()
        ),
        "validation_files": sorted(
            path.name for path in validation_directory.iterdir() if path.is_file()
        ),
        "generated_plots": {
            "public": public_plots,
            "validation_only": validation_plots,
        },
    }
    write_json(run_directory / "stage_09_summary.json", summary)

    write_json(
        run_directory / "run_manifest.json",
        {
            "stage": 9,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": asdict(config),
            "input_sha256": {
                "stage_08_summary.json": sha256_file(public["summary_path"]),
                "large_fault_campaign_public.csv": sha256_file(
                    public["public_csv_path"]
                ),
                "large_fault_response_traces.npz": sha256_file(
                    public["trace_path"]
                ),
                "campaign_partition_manifest.json": sha256_file(
                    public["split_path"]
                ),
            },
        },
    )

    print("\n" + "=" * 82)
    print("Stage 09 complete: leakage-safe ML dataset and audit")
    print("=" * 82)
    print("Run directory                    :", summary["run_directory"])
    print("All checks passed                :", summary["all_checks_passed"])
    print("Rows                             :", summary["number_of_rows"])
    print("Primary tabular feature count    :", summary["number_of_primary_tabular_features"])
    print("Partitions                       :", summary["partition_counts"])
    print("Development / locked attack rows :", summary["development_label_row_count"], "/", summary["locked_attack_label_row_count"])
    print("Strict source-disjoint rows       :", summary["strict_source_disjoint_subset"]["included_row_count"])
    print("Public leakage audit              :", summary["public_leakage_audit_passed"])
    print("Public/private join               :", summary["public_private_label_join_passed"])
    print("Public ML freeze SHA-256          :", summary["public_ml_freeze_sha256"])
    print("Elapsed seconds                   :", f"{summary['elapsed_seconds']:.3f}")
    print("=" * 82)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 09 failed validation. Inspect "
            "validation_only/stage_09_dataset_validation.json"
        )
    return summary


def load_stage_09_config(path: str | Path) -> Stage09Config:
    return Stage09Config(**read_json(Path(path)))


if __name__ == "__main__":
    default_config = Stage09Config(
        input_stage8_run_directory=(
            './runs/stage_08'
            '/stage08_20260718_182609_471595_seed20260718'
        ),
        output_root=(
            './runs/stage_09'
        ),
    )
    run_stage_09(default_config)

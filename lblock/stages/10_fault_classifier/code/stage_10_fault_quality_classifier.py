from __future__ import annotations

"""
Stage 10 — Leakage-safe six-class fault-quality classifier for LBlock-64/80.

Scientific contract
-------------------
1. Consume the frozen public ML dataset and development labels from Stage 09.
2. Never open the locked Stage-09 attack labels.
3. Preserve the key-disjoint train/validation/test/attack partitions.
4. Compare four pre-declared feature views on validation data only:
      parameter_only, trace_only, parameter_trace, observable_full.
5. Compare a weighted multinomial logistic model and an Extra-Trees model.
6. Select one model using validation macro-F1, validation log-loss, and branch AP.
7. Fit one scalar temperature on validation probabilities.
8. Evaluate the selected model once on the held-out test keys.
9. Retrain the selected architecture on train+validation and emit public,
   unlabeled probabilities for the locked final-attack partition.
10. Perform a secondary source-trace-disjoint robustness experiment.

This stage produces simulated-data model results. It does not claim physical
ChipWhisperer performance.
"""

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Avoid uncontrolled native-thread oversubscription on Windows/Jupyter.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CLASS_NAMES: Tuple[str, ...] = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)
CLASS_IDS = np.arange(len(CLASS_NAMES), dtype=np.int64)
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

PARTITION_NAMES: Tuple[str, ...] = (
    "train",
    "validation",
    "test",
    "attack",
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

PARAMETER_FEATURES: Tuple[str, ...] = (
    "target_is_s5",
    "timing_offset_samples",
    "absolute_timing_offset_samples",
    "width_samples",
    "strength",
    "repeat",
    "repeat_spacing_samples",
    "pulse_span_samples",
    "glitch_energy_proxy",
    "target_window_valid_fraction",
    "pulse_window_valid_fraction",
)

TRACE_ARRAY_NAMES: Tuple[str, ...] = (
    "full_trace_highpass",
    "target_aligned_zscore",
    "pulse_aligned_zscore",
)

FORBIDDEN_FEATURE_TOKENS: Tuple[str, ...] = (
    "category",
    "label",
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
    "design_regime",
    "activation_probability",
    "hit_score",
)


@dataclass(frozen=True)
class Stage10Config:
    input_stage9_run_directory: str
    output_root: str = "runs/stage_10"
    random_seed: int = 20260718

    # Candidate families are fixed before test evaluation.
    candidate_families: Tuple[str, ...] = ("logistic", "extra_trees")
    logistic_c: float = 1.0
    logistic_max_iter: int = 700
    extra_trees_estimators: int = 180
    extra_trees_max_depth: Optional[int] = 26
    extra_trees_min_samples_leaf: int = 5
    extra_trees_max_features: str = "sqrt"
    extra_trees_n_jobs: int = 4

    # Calibration and branch-threshold settings.
    minimum_temperature: float = 0.20
    maximum_temperature: float = 5.00
    calibration_bins: int = 15
    branch_threshold_minimum: float = 0.05
    branch_threshold_maximum: float = 0.95
    branch_threshold_steps: int = 181

    # Robustness and artifact settings.
    run_strict_source_disjoint_experiment: bool = True
    save_plots: bool = True
    probability_clip: float = 1e-7
    model_reload_check_rows: int = 512


@dataclass
class FeatureStore:
    frame: pd.DataFrame
    experiment_ids: np.ndarray
    tabular: np.ndarray
    full_trace: np.ndarray
    target_trace: np.ndarray
    pulse_trace: np.ndarray
    row_index_by_experiment_id: Dict[int, int]


@dataclass
class CandidateResult:
    view_name: str
    family: str
    estimator: Any
    temperature: float
    validation_probabilities: np.ndarray
    validation_metrics: Dict[str, Any]
    feature_count: int
    feature_names: List[str]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def apply_temperature(
    probabilities: np.ndarray,
    temperature: float,
    clip: float,
) -> np.ndarray:
    safe = np.clip(np.asarray(probabilities, dtype=np.float64), clip, 1.0)
    logits = np.log(safe)
    return stable_softmax(logits / float(temperature))


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    config: Stage10Config,
) -> Tuple[float, Dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    before = float(log_loss(labels, probabilities, labels=CLASS_IDS))

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        calibrated = apply_temperature(
            probabilities,
            temperature,
            config.probability_clip,
        )
        return float(log_loss(labels, calibrated, labels=CLASS_IDS))

    result = minimize_scalar(
        objective,
        bounds=(
            math.log(config.minimum_temperature),
            math.log(config.maximum_temperature),
        ),
        method="bounded",
        options={"xatol": 1e-5},
    )
    temperature = float(np.exp(result.x))
    after = float(result.fun)

    # Never accept a calibration temperature that worsens validation log-loss.
    if not result.success or after > before + 1e-10:
        temperature = 1.0
        after = before

    return temperature, {
        "optimization_success": bool(result.success),
        "temperature": temperature,
        "validation_log_loss_before": before,
        "validation_log_loss_after": after,
        "improvement": before - after,
    }


def multiclass_brier_score(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int,
) -> Tuple[float, pd.DataFrame]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    rows: List[Dict[str, Any]] = []
    ece = 0.0

    for bin_index in range(number_of_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index == number_of_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(np.sum(mask))
        if count == 0:
            accuracy = float("nan")
            confidence = float("nan")
            contribution = 0.0
        else:
            accuracy = float(np.mean(correct[mask]))
            confidence = float(np.mean(confidences[mask]))
            contribution = count / len(labels) * abs(accuracy - confidence)
            ece += contribution

        rows.append({
            "bin_index": bin_index,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": count,
            "mean_confidence": confidence,
            "empirical_accuracy": accuracy,
            "ece_contribution": contribution,
        })

    return float(ece), pd.DataFrame(rows)


def binary_branch_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    positive_class_id: int,
    threshold: float,
) -> Dict[str, Any]:
    truth = (np.asarray(labels) == positive_class_id).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = (scores >= threshold).astype(np.int64)

    result: Dict[str, Any] = {
        "positive_class": CLASS_NAMES[positive_class_id],
        "threshold": float(threshold),
        "support": int(np.sum(truth)),
        "prevalence": float(np.mean(truth)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(truth, scores)),
    }
    if len(np.unique(truth)) == 2:
        result["roc_auc"] = float(roc_auc_score(truth, scores))
    else:
        result["roc_auc"] = float("nan")
    return result


def choose_branch_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    positive_class_id: int,
    config: Stage10Config,
) -> Tuple[float, Dict[str, Any]]:
    truth = (np.asarray(labels) == positive_class_id).astype(np.int64)
    thresholds = np.linspace(
        config.branch_threshold_minimum,
        config.branch_threshold_maximum,
        config.branch_threshold_steps,
    )

    best_key: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    best_threshold = 0.5
    best_metrics: Dict[str, Any] = {}

    for threshold in thresholds:
        predicted = (scores >= threshold).astype(np.int64)
        precision = float(precision_score(truth, predicted, zero_division=0))
        recall = float(recall_score(truth, predicted, zero_division=0))
        f1 = float(f1_score(truth, predicted, zero_division=0))
        # Prefer F1, then precision, then lower threshold for greater recall.
        key = (f1, precision, -float(threshold))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "threshold": best_threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    best_metrics["average_precision"] = float(
        average_precision_score(truth, scores)
    )
    if len(np.unique(truth)) == 2:
        best_metrics["roc_auc"] = float(roc_auc_score(truth, scores))
    else:
        best_metrics["roc_auc"] = float("nan")
    return best_threshold, best_metrics


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    calibration_bins: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.argmax(probabilities, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=CLASS_IDS,
        zero_division=0,
    )
    per_class = pd.DataFrame({
        "class_id": CLASS_IDS,
        "class_name": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64),
    })

    ece, reliability = expected_calibration_error(
        labels,
        probabilities,
        calibration_bins,
    )

    metrics = {
        "number_of_rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
        "log_loss": float(log_loss(labels, probabilities, labels=CLASS_IDS)),
        "multiclass_brier_score": multiclass_brier_score(labels, probabilities),
        "expected_calibration_error": ece,
    }
    matrix = confusion_matrix(labels, predictions, labels=CLASS_IDS)
    return metrics, per_class, reliability, matrix


def dataframe_digest(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Stage-09 contract loading and auditing
# ---------------------------------------------------------------------------

def verify_stage09_public_freeze(stage9_run: Path, summary: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = stage9_run / "public_ml_freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)

    public_directory = stage9_run / "public_ml"
    mismatches: List[Dict[str, str]] = []
    for file_name, expected_digest in manifest["files"].items():
        path = public_directory / file_name
        if not path.is_file():
            mismatches.append({
                "file": file_name,
                "issue": "missing",
                "expected": str(expected_digest),
                "observed": "",
            })
            continue
        observed = sha256_file(path)
        if observed != expected_digest:
            mismatches.append({
                "file": file_name,
                "issue": "sha256_mismatch",
                "expected": str(expected_digest),
                "observed": observed,
            })

    manifest_digest_matches_summary = (
        manifest.get("freeze_sha256") == summary.get("public_ml_freeze_sha256")
    )
    return {
        "passed": not mismatches and manifest_digest_matches_summary,
        "manifest_path": str(manifest_path),
        "manifest_freeze_sha256": manifest.get("freeze_sha256"),
        "summary_freeze_sha256": summary.get("public_ml_freeze_sha256"),
        "manifest_digest_matches_summary": manifest_digest_matches_summary,
        "file_mismatches": mismatches,
    }


def load_stage09_contract(config: Stage10Config) -> Dict[str, Any]:
    stage9_run = Path(config.input_stage9_run_directory).expanduser().resolve()
    summary_path = stage9_run / "stage_09_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = read_json(summary_path)
    if not summary.get("all_checks_passed", False):
        raise RuntimeError("Stage 09 did not pass all checks")
    if not summary.get("public_leakage_audit_passed", False):
        raise RuntimeError("Stage 09 public leakage audit failed")
    if not summary.get("public_private_label_join_passed", False):
        raise RuntimeError("Stage 09 public/private join failed")

    freeze_verification = verify_stage09_public_freeze(stage9_run, summary)
    if not freeze_verification["passed"]:
        raise RuntimeError("Stage 09 public freeze verification failed")

    public_directory = stage9_run / "public_ml"
    private_label_directory = stage9_run / "private_labels"
    locked_attack_directory = stage9_run / "locked_attack_labels"

    paths = {
        "stage9_run": stage9_run,
        "summary_path": summary_path,
        "summary": summary,
        "freeze_verification": freeze_verification,
        "feature_csv": public_directory / "ml_tabular_features_public.csv",
        "trace_npz": public_directory / "ml_trace_views_public.npz",
        "feature_contract": public_directory / "ml_feature_contract.json",
        "split_contract": public_directory / "ml_split_contract.json",
        "strict_ids": public_directory / "strict_source_disjoint_experiment_ids.npz",
        "strict_membership": public_directory / "strict_source_disjoint_membership.csv",
        "attack_payload": public_directory / "final_attack_payload_public.csv",
        "development_labels": private_label_directory / "ml_quality_labels_train_validation_test.csv",
        "locked_attack_labels": locked_attack_directory / "attack_quality_labels_LOCKED.csv",
        "locked_attack_warning": locked_attack_directory / "DO_NOT_OPEN_BEFORE_FINAL_ATTACK.json",
    }
    for name, path in paths.items():
        if name in {"stage9_run", "summary", "freeze_verification"}:
            continue
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    return paths


def audit_feature_names(feature_names: Iterable[str]) -> Dict[str, Any]:
    names = [str(name) for name in feature_names]
    violations: List[str] = []
    for name in names:
        lowered = name.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            violations.append(name)
    return {
        "passed": not violations,
        "feature_count": len(names),
        "violations": sorted(set(violations)),
    }


def load_feature_store(paths: Mapping[str, Any]) -> FeatureStore:
    frame = pd.read_csv(paths["feature_csv"])
    missing = sorted(set(PRIMARY_TABULAR_FEATURES) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing primary tabular features: {missing}")

    with np.load(paths["trace_npz"], allow_pickle=False) as archive:
        required = {"experiment_ids", *TRACE_ARRAY_NAMES}
        missing_arrays = sorted(required - set(archive.files))
        if missing_arrays:
            raise RuntimeError(f"Missing trace arrays: {missing_arrays}")
        experiment_ids = archive["experiment_ids"].astype(np.int64)
        full_trace = archive["full_trace_highpass"].astype(np.float32)
        target_trace = archive["target_aligned_zscore"].astype(np.float32)
        pulse_trace = archive["pulse_aligned_zscore"].astype(np.float32)

    frame_ids = frame["experiment_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_ids, experiment_ids):
        raise RuntimeError("Feature CSV and trace NPZ experiment IDs are misaligned")
    if len(np.unique(experiment_ids)) != len(experiment_ids):
        raise RuntimeError("Duplicate experiment IDs in public ML features")

    arrays = [full_trace, target_trace, pulse_trace]
    if any(array.shape[0] != len(frame) for array in arrays):
        raise RuntimeError("Trace row count does not match public feature rows")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise RuntimeError("Trace arrays contain NaN or Inf")

    tabular = frame.loc[:, PRIMARY_TABULAR_FEATURES].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(tabular)):
        raise RuntimeError("Primary tabular features contain NaN or Inf")

    row_index_by_experiment_id = {
        int(experiment_id): int(index)
        for index, experiment_id in enumerate(experiment_ids)
    }

    return FeatureStore(
        frame=frame,
        experiment_ids=experiment_ids,
        tabular=tabular,
        full_trace=full_trace,
        target_trace=target_trace,
        pulse_trace=pulse_trace,
        row_index_by_experiment_id=row_index_by_experiment_id,
    )


def load_development_labels(paths: Mapping[str, Any]) -> pd.DataFrame:
    labels = pd.read_csv(paths["development_labels"])
    required = {
        "experiment_id",
        "campaign_partition",
        "category",
        "category_id",
        "key_id",
        "session_id",
        "target_sbox",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RuntimeError(f"Missing development-label columns: {missing}")
    if labels["experiment_id"].duplicated().any():
        raise RuntimeError("Duplicate experiment IDs in development labels")
    if set(labels["campaign_partition"].unique()) != {"train", "validation", "test"}:
        raise RuntimeError("Development labels contain an unexpected partition")
    if not set(labels["category_id"].astype(int).unique()).issubset(set(CLASS_IDS.tolist())):
        raise RuntimeError("Unexpected category ID in development labels")
    return labels


def verify_partition_contract(
    store: FeatureStore,
    labels: pd.DataFrame,
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_counts = {
        partition: int(np.sum(store.frame["campaign_partition"] == partition))
        for partition in PARTITION_NAMES
    }
    expected_counts = {
        key: int(value)
        for key, value in summary["partition_counts"].items()
    }

    observed_keys = {
        partition: sorted(
            store.frame.loc[
                store.frame["campaign_partition"] == partition,
                "key_id",
            ].astype(int).unique().tolist()
        )
        for partition in PARTITION_NAMES
    }
    expected_keys = {
        key: list(map(int, values))
        for key, values in summary["partition_key_ids"].items()
    }

    development_ids = set(labels["experiment_id"].astype(int))
    attack_ids = set(
        store.frame.loc[
            store.frame["campaign_partition"] == "attack",
            "experiment_id",
        ].astype(int)
    )

    return {
        "passed": (
            observed_counts == expected_counts
            and observed_keys == expected_keys
            and development_ids.isdisjoint(attack_ids)
            and len(development_ids) == int(summary["development_label_row_count"])
        ),
        "observed_partition_counts": observed_counts,
        "expected_partition_counts": expected_counts,
        "observed_partition_key_ids": observed_keys,
        "expected_partition_key_ids": expected_keys,
        "development_attack_id_overlap": len(development_ids & attack_ids),
    }


# ---------------------------------------------------------------------------
# Feature views and estimators
# ---------------------------------------------------------------------------

def row_indices_for_ids(store: FeatureStore, experiment_ids: Sequence[int]) -> np.ndarray:
    try:
        return np.asarray(
            [store.row_index_by_experiment_id[int(value)] for value in experiment_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError(f"Experiment ID absent from public feature store: {exc}") from exc


def feature_view(
    store: FeatureStore,
    row_indices: np.ndarray,
    view_name: str,
) -> Tuple[np.ndarray, List[str]]:
    row_indices = np.asarray(row_indices, dtype=np.int64)

    tabular_name_to_index = {
        name: index for index, name in enumerate(PRIMARY_TABULAR_FEATURES)
    }
    parameter_indices = np.asarray(
        [tabular_name_to_index[name] for name in PARAMETER_FEATURES],
        dtype=np.int64,
    )
    parameter_values = store.tabular[row_indices][:, parameter_indices]

    trace_values = np.concatenate(
        [
            store.full_trace[row_indices],
            store.target_trace[row_indices],
            store.pulse_trace[row_indices],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    trace_names = (
        [f"full_trace_highpass_{index:03d}" for index in range(store.full_trace.shape[1])]
        + [f"target_aligned_zscore_{index:03d}" for index in range(store.target_trace.shape[1])]
        + [f"pulse_aligned_zscore_{index:03d}" for index in range(store.pulse_trace.shape[1])]
    )

    if view_name == "parameter_only":
        values = parameter_values
        names = list(PARAMETER_FEATURES)
    elif view_name == "trace_only":
        values = trace_values
        names = trace_names
    elif view_name == "parameter_trace":
        values = np.concatenate([parameter_values, trace_values], axis=1)
        names = list(PARAMETER_FEATURES) + trace_names
    elif view_name == "observable_full":
        values = np.concatenate([store.tabular[row_indices], trace_values], axis=1)
        names = list(PRIMARY_TABULAR_FEATURES) + trace_names
    else:
        raise ValueError(f"Unknown feature view: {view_name}")

    audit = audit_feature_names(names)
    if not audit["passed"]:
        raise RuntimeError(f"Forbidden feature in {view_name}: {audit['violations']}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Non-finite values in view {view_name}")
    return values.astype(np.float32, copy=False), names


def make_estimator(
    family: str,
    config: Stage10Config,
    random_state: int,
) -> Any:
    if family == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=config.logistic_c,
                    max_iter=config.logistic_max_iter,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ])

    if family == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=config.extra_trees_estimators,
                    max_depth=config.extra_trees_max_depth,
                    min_samples_leaf=config.extra_trees_min_samples_leaf,
                    max_features=config.extra_trees_max_features,
                    class_weight="balanced",
                    bootstrap=False,
                    n_jobs=config.extra_trees_n_jobs,
                    random_state=random_state,
                ),
            ),
        ])

    raise ValueError(f"Unknown candidate family: {family}")


def aligned_predict_proba(estimator: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if np.array_equal(classes, CLASS_IDS):
        return probabilities

    aligned = np.zeros((len(features), len(CLASS_NAMES)), dtype=np.float64)
    for source_index, class_id in enumerate(classes):
        aligned[:, int(class_id)] = probabilities[:, source_index]
    return aligned


def fit_candidate(
    view_name: str,
    family: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    feature_names: List[str],
    config: Stage10Config,
    candidate_index: int,
) -> CandidateResult:
    estimator = make_estimator(
        family,
        config,
        config.random_seed + 1000 + candidate_index,
    )
    with threadpool_limits(limits=max(1, config.extra_trees_n_jobs)):
        estimator.fit(train_features, train_labels)

    raw_probabilities = aligned_predict_proba(estimator, validation_features)
    temperature, calibration = fit_temperature(
        raw_probabilities,
        validation_labels,
        config,
    )
    probabilities = apply_temperature(
        raw_probabilities,
        temperature,
        config.probability_clip,
    )
    metrics, _, _, _ = classification_metrics(
        validation_labels,
        probabilities,
        config.calibration_bins,
    )
    metrics["calibration"] = calibration
    metrics["clean_ineffective_average_precision"] = float(
        average_precision_score(
            (validation_labels == CLASS_TO_ID["clean_target_ineffective"]).astype(int),
            probabilities[:, CLASS_TO_ID["clean_target_ineffective"]],
        )
    )
    metrics["clean_effective_average_precision"] = float(
        average_precision_score(
            (validation_labels == CLASS_TO_ID["clean_target_effective"]).astype(int),
            probabilities[:, CLASS_TO_ID["clean_target_effective"]],
        )
    )
    metrics["mean_attack_branch_average_precision"] = float(
        0.5
        * (
            metrics["clean_ineffective_average_precision"]
            + metrics["clean_effective_average_precision"]
        )
    )

    return CandidateResult(
        view_name=view_name,
        family=family,
        estimator=estimator,
        temperature=temperature,
        validation_probabilities=probabilities,
        validation_metrics=metrics,
        feature_count=train_features.shape[1],
        feature_names=feature_names,
    )


def candidate_selection_key(candidate: CandidateResult) -> Tuple[float, float, float, float]:
    metrics = candidate.validation_metrics
    return (
        float(metrics["macro_f1"]),
        float(metrics["mean_attack_branch_average_precision"]),
        -float(metrics["log_loss"]),
        -float(metrics["expected_calibration_error"]),
    )


def candidate_table(candidates: Sequence[CandidateResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate.validation_metrics
        rows.append({
            "view_name": candidate.view_name,
            "family": candidate.family,
            "feature_count": candidate.feature_count,
            "temperature": candidate.temperature,
            "validation_accuracy": metrics["accuracy"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_weighted_f1": metrics["weighted_f1"],
            "validation_log_loss": metrics["log_loss"],
            "validation_brier": metrics["multiclass_brier_score"],
            "validation_ece": metrics["expected_calibration_error"],
            "validation_clean_ineffective_ap": metrics["clean_ineffective_average_precision"],
            "validation_clean_effective_ap": metrics["clean_effective_average_precision"],
            "validation_mean_branch_ap": metrics["mean_attack_branch_average_precision"],
        })
    return pd.DataFrame(rows).sort_values(
        [
            "validation_macro_f1",
            "validation_mean_branch_ap",
            "validation_log_loss",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prediction tables, plots, and strict-source experiment
# ---------------------------------------------------------------------------

def probability_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    temperature: float,
    selected_view: str,
    selected_family: str,
) -> pd.DataFrame:
    output = metadata.loc[:, [
        "experiment_id",
        "campaign_partition",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
        "fault_model",
    ]].copy()

    predicted_ids = np.argmax(probabilities, axis=1)
    output["predicted_category_id"] = predicted_ids
    output["predicted_category"] = [CLASS_NAMES[int(value)] for value in predicted_ids]
    output["prediction_confidence"] = np.max(probabilities, axis=1)
    output["selected_view"] = selected_view
    output["selected_family"] = selected_family
    output["temperature"] = temperature

    for class_id, class_name in enumerate(CLASS_NAMES):
        output[f"p_{class_name}"] = probabilities[:, class_id]

    output["p_clean_target"] = (
        output["p_clean_target_ineffective"]
        + output["p_clean_target_effective"]
    )
    output["p_attack_unusable"] = (
        output["p_missed"]
        + output["p_off_target"]
        + output["p_multi_hit"]
        + output["p_invalid_reset"]
    )
    return output


def plot_confusion_matrix(
    matrix: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(9.5, 8.0))
    axis = fig.add_subplot(1, 1, 1)
    image = axis.imshow(matrix, aspect="auto")
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_validation_comparison(frame: pd.DataFrame, path: Path) -> None:
    labels = [f"{row.view_name}\n{row.family}" for row in frame.itertuples()]
    values = frame["validation_macro_f1"].to_numpy()
    fig = plt.figure(figsize=(12.0, 6.0))
    axis = fig.add_subplot(1, 1, 1)
    positions = np.arange(len(frame))
    axis.bar(positions, values)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Stage 10 candidate comparison on validation keys only")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_class_metrics(frame: pd.DataFrame, path: Path, title: str) -> None:
    positions = np.arange(len(frame))
    width = 0.25
    fig = plt.figure(figsize=(12.0, 6.0))
    axis = fig.add_subplot(1, 1, 1)
    axis.bar(positions - width, frame["precision"], width=width, label="Precision")
    axis.bar(positions, frame["recall"], width=width, label="Recall")
    axis.bar(positions + width, frame["f1"], width=width, label="F1")
    axis.set_xticks(positions)
    axis.set_xticklabels(frame["class_name"], rotation=35, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reliability(frame: pd.DataFrame, path: Path, title: str) -> None:
    valid = frame[frame["count"] > 0].copy()
    fig = plt.figure(figsize=(7.0, 6.0))
    axis = fig.add_subplot(1, 1, 1)
    axis.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    axis.plot(valid["mean_confidence"], valid["empirical_accuracy"], marker="o", label="Selected model")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_strict_source_experiment(
    selected: CandidateResult,
    store: FeatureStore,
    labels: pd.DataFrame,
    paths: Mapping[str, Any],
    config: Stage10Config,
) -> Dict[str, Any]:
    with np.load(paths["strict_ids"], allow_pickle=False) as archive:
        available_names = set(archive.files)
        if "experiment_ids" in available_names:
            strict_ids = set(archive["experiment_ids"].astype(int).tolist())
        else:
            # Stage 09 may save separate arrays per partition. Combine all ID arrays.
            id_arrays = [
                archive[name].astype(int)
                for name in archive.files
                if "experiment" in name.lower() and "id" in name.lower()
            ]
            if not id_arrays:
                raise RuntimeError("No strict-source experiment ID array was found")
            strict_ids = set(np.concatenate(id_arrays).tolist())

    strict_labels = labels[labels["experiment_id"].astype(int).isin(strict_ids)].copy()
    split = {
        partition: strict_labels[strict_labels["campaign_partition"] == partition].copy()
        for partition in ("train", "validation", "test")
    }
    if any(len(frame) == 0 for frame in split.values()):
        raise RuntimeError("A strict-source development partition is empty")

    indices = {
        partition: row_indices_for_ids(store, frame["experiment_id"].astype(int).tolist())
        for partition, frame in split.items()
    }
    features = {}
    feature_names: Optional[List[str]] = None
    for partition in ("train", "validation", "test"):
        matrix, names = feature_view(store, indices[partition], selected.view_name)
        features[partition] = matrix
        if feature_names is None:
            feature_names = names

    y_train = split["train"]["category_id"].to_numpy(dtype=np.int64)
    y_validation = split["validation"]["category_id"].to_numpy(dtype=np.int64)
    y_test = split["test"]["category_id"].to_numpy(dtype=np.int64)

    estimator = make_estimator(
        selected.family,
        config,
        config.random_seed + 9090,
    )
    with threadpool_limits(limits=max(1, config.extra_trees_n_jobs)):
        estimator.fit(features["train"], y_train)

    raw_validation = aligned_predict_proba(estimator, features["validation"])
    temperature, calibration = fit_temperature(raw_validation, y_validation, config)
    raw_test = aligned_predict_proba(estimator, features["test"])
    test_probabilities = apply_temperature(raw_test, temperature, config.probability_clip)
    metrics, per_class, reliability, matrix = classification_metrics(
        y_test,
        test_probabilities,
        config.calibration_bins,
    )

    return {
        "view_name": selected.view_name,
        "family": selected.family,
        "feature_count": int(features["train"].shape[1]),
        "temperature": temperature,
        "calibration": calibration,
        "partition_counts": {key: len(value) for key, value in split.items()},
        "test_metrics": metrics,
        "per_class": per_class,
        "reliability": reliability,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# Complete Stage-10 runner
# ---------------------------------------------------------------------------

def run_stage_10(config: Stage10Config) -> Dict[str, Any]:
    start_time = time.perf_counter()
    paths = load_stage09_contract(config)
    summary9 = paths["summary"]
    store = load_feature_store(paths)
    labels = load_development_labels(paths)

    feature_audit = audit_feature_names(PRIMARY_TABULAR_FEATURES)
    partition_audit = verify_partition_contract(store, labels, summary9)
    if not feature_audit["passed"]:
        raise RuntimeError(f"Feature audit failed: {feature_audit['violations']}")
    if not partition_audit["passed"]:
        raise RuntimeError("Stage-09 partition contract verification failed")

    # Important: the locked attack label file is deliberately not read here.
    attack_label_access_manifest = {
        "attack_labels_accessed": False,
        "locked_attack_label_path": str(paths["locked_attack_labels"]),
        "locked_attack_label_file_exists": Path(paths["locked_attack_labels"]).is_file(),
        "expected_attack_label_sha256_from_stage09_summary": summary9["attack_label_sha256"],
        "statement": (
            "Stage 10 never opens the locked attack-label CSV. Attack predictions "
            "are produced from public features only."
        ),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage10_{timestamp}_seed{config.random_seed}"
    run_directory = Path(config.output_root).expanduser().resolve() / run_id
    model_directory = run_directory / "models"
    prediction_directory = run_directory / "predictions"
    validation_directory = run_directory / "validation_only"
    model_directory.mkdir(parents=True, exist_ok=False)
    prediction_directory.mkdir(parents=True, exist_ok=True)
    validation_directory.mkdir(parents=True, exist_ok=True)

    split_labels = {
        partition: labels[labels["campaign_partition"] == partition].copy()
        for partition in ("train", "validation", "test")
    }
    split_indices = {
        partition: row_indices_for_ids(
            store,
            split_labels[partition]["experiment_id"].astype(int).tolist(),
        )
        for partition in ("train", "validation", "test")
    }
    attack_metadata = store.frame[store.frame["campaign_partition"] == "attack"].copy()
    attack_indices = row_indices_for_ids(
        store,
        attack_metadata["experiment_id"].astype(int).tolist(),
    )

    y_train = split_labels["train"]["category_id"].to_numpy(dtype=np.int64)
    y_validation = split_labels["validation"]["category_id"].to_numpy(dtype=np.int64)
    y_test = split_labels["test"]["category_id"].to_numpy(dtype=np.int64)

    feature_views = (
        "parameter_only",
        "trace_only",
        "parameter_trace",
        "observable_full",
    )
    candidates: List[CandidateResult] = []
    candidate_index = 0

    print("Training Stage 10 candidate models...")
    for view_name in feature_views:
        x_train, feature_names = feature_view(store, split_indices["train"], view_name)
        x_validation, _ = feature_view(store, split_indices["validation"], view_name)

        for family in config.candidate_families:
            candidate_index += 1
            print(f"  [{candidate_index:02d}] {view_name} / {family}")
            candidate = fit_candidate(
                view_name,
                family,
                x_train,
                y_train,
                x_validation,
                y_validation,
                feature_names,
                config,
                candidate_index,
            )
            candidates.append(candidate)

    table = candidate_table(candidates)
    write_csv(validation_directory / "candidate_validation_results.csv", table)
    if config.save_plots:
        plot_validation_comparison(
            table,
            validation_directory / "candidate_validation_macro_f1.png",
        )

    selected = max(candidates, key=candidate_selection_key)
    selected_validation_metadata = store.frame.iloc[split_indices["validation"]].reset_index(drop=True)
    validation_prediction_frame = probability_frame(
        selected_validation_metadata,
        selected.validation_probabilities,
        selected.temperature,
        selected.view_name,
        selected.family,
    )
    validation_prediction_frame["true_category_id"] = y_validation
    validation_prediction_frame["true_category"] = [CLASS_NAMES[int(value)] for value in y_validation]
    write_csv(
        validation_directory / "selected_validation_predictions_with_labels.csv",
        validation_prediction_frame,
    )

    # Select branch thresholds only on validation data.
    ineffective_id = CLASS_TO_ID["clean_target_ineffective"]
    effective_id = CLASS_TO_ID["clean_target_effective"]
    ineffective_threshold, validation_ineffective = choose_branch_threshold(
        y_validation,
        selected.validation_probabilities[:, ineffective_id],
        ineffective_id,
        config,
    )
    effective_threshold, validation_effective = choose_branch_threshold(
        y_validation,
        selected.validation_probabilities[:, effective_id],
        effective_id,
        config,
    )

    threshold_contract = {
        "selection_partition": "validation",
        "clean_target_ineffective": validation_ineffective,
        "clean_target_effective": validation_effective,
    }
    write_json(model_directory / "branch_threshold_contract.json", threshold_contract)

    # One and only one held-out test evaluation after model/view/family selection.
    x_test, _ = feature_view(store, split_indices["test"], selected.view_name)
    raw_test_probabilities = aligned_predict_proba(selected.estimator, x_test)
    test_probabilities = apply_temperature(
        raw_test_probabilities,
        selected.temperature,
        config.probability_clip,
    )
    test_metrics, test_per_class, test_reliability, test_matrix = classification_metrics(
        y_test,
        test_probabilities,
        config.calibration_bins,
    )
    test_branch_metrics = {
        "clean_target_ineffective": binary_branch_metrics(
            y_test,
            test_probabilities[:, ineffective_id],
            ineffective_id,
            ineffective_threshold,
        ),
        "clean_target_effective": binary_branch_metrics(
            y_test,
            test_probabilities[:, effective_id],
            effective_id,
            effective_threshold,
        ),
    }

    write_json(validation_directory / "selected_test_metrics.json", test_metrics)
    write_json(validation_directory / "selected_test_branch_metrics.json", test_branch_metrics)
    write_csv(validation_directory / "selected_test_per_class_metrics.csv", test_per_class)
    write_csv(validation_directory / "selected_test_reliability.csv", test_reliability)

    test_metadata = store.frame.iloc[split_indices["test"]].reset_index(drop=True)
    test_prediction_frame = probability_frame(
        test_metadata,
        test_probabilities,
        selected.temperature,
        selected.view_name,
        selected.family,
    )
    test_prediction_frame["true_category_id"] = y_test
    test_prediction_frame["true_category"] = [CLASS_NAMES[int(value)] for value in y_test]
    write_csv(validation_directory / "selected_test_predictions_with_labels.csv", test_prediction_frame)

    if config.save_plots:
        plot_confusion_matrix(
            test_matrix,
            validation_directory / "selected_test_confusion_matrix.png",
            "Selected Stage 10 model — held-out test keys",
        )
        plot_per_class_metrics(
            test_per_class,
            validation_directory / "selected_test_per_class_metrics.png",
            "Selected Stage 10 model — per-class test metrics",
        )
        plot_reliability(
            test_reliability,
            validation_directory / "selected_test_reliability.png",
            "Selected Stage 10 model — test reliability",
        )

    # Save the selection estimator before deployment refit.
    selection_bundle = {
        "stage": 10,
        "purpose": "validation-selected estimator used for the single held-out test evaluation",
        "selected_view": selected.view_name,
        "selected_family": selected.family,
        "feature_names": selected.feature_names,
        "class_names": CLASS_NAMES,
        "temperature": selected.temperature,
        "branch_thresholds": {
            "clean_target_ineffective": ineffective_threshold,
            "clean_target_effective": effective_threshold,
        },
        "estimator": selected.estimator,
    }
    selection_model_path = model_directory / "selected_test_evaluation_model.joblib"
    joblib.dump(selection_bundle, selection_model_path, compress=3)

    # Refit same architecture on train+validation for unlabeled deployment.
    development_labels = pd.concat(
        [split_labels["train"], split_labels["validation"]],
        ignore_index=True,
    )
    development_indices = row_indices_for_ids(
        store,
        development_labels["experiment_id"].astype(int).tolist(),
    )
    x_development, deployment_feature_names = feature_view(
        store,
        development_indices,
        selected.view_name,
    )
    y_development = development_labels["category_id"].to_numpy(dtype=np.int64)

    deployment_estimator = make_estimator(
        selected.family,
        config,
        config.random_seed + 50000,
    )
    with threadpool_limits(limits=max(1, config.extra_trees_n_jobs)):
        deployment_estimator.fit(x_development, y_development)

    # Public predictions for all development rows (without labels) and locked attack rows.
    raw_development = aligned_predict_proba(deployment_estimator, x_development)
    development_probabilities = apply_temperature(
        raw_development,
        selected.temperature,
        config.probability_clip,
    )
    development_metadata = store.frame.iloc[development_indices].reset_index(drop=True)
    development_output = probability_frame(
        development_metadata,
        development_probabilities,
        selected.temperature,
        selected.view_name,
        selected.family,
    )
    write_csv(
        prediction_directory / "development_quality_probabilities_public.csv",
        development_output,
    )

    x_attack, _ = feature_view(store, attack_indices, selected.view_name)
    raw_attack = aligned_predict_proba(deployment_estimator, x_attack)
    attack_probabilities = apply_temperature(
        raw_attack,
        selected.temperature,
        config.probability_clip,
    )
    attack_output = probability_frame(
        attack_metadata.reset_index(drop=True),
        attack_probabilities,
        selected.temperature,
        selected.view_name,
        selected.family,
    )
    write_csv(
        prediction_directory / "attack_quality_probabilities_public.csv",
        attack_output,
    )

    # Reload check establishes portable deterministic inference.
    deployment_bundle = {
        "stage": 10,
        "purpose": "frozen deployment model for Stage 11/12 and final attack scoring",
        "selected_view": selected.view_name,
        "selected_family": selected.family,
        "feature_names": deployment_feature_names,
        "class_names": CLASS_NAMES,
        "temperature": selected.temperature,
        "branch_thresholds": {
            "clean_target_ineffective": ineffective_threshold,
            "clean_target_effective": effective_threshold,
        },
        "estimator": deployment_estimator,
        "training_partitions": ["train", "validation"],
        "attack_labels_used": False,
    }
    deployment_model_path = model_directory / "fault_quality_deployment_model.joblib"
    joblib.dump(deployment_bundle, deployment_model_path, compress=3)

    reloaded = joblib.load(deployment_model_path)
    check_count = min(config.model_reload_check_rows, len(x_attack))
    reload_raw = aligned_predict_proba(reloaded["estimator"], x_attack[:check_count])
    reload_probabilities = apply_temperature(
        reload_raw,
        float(reloaded["temperature"]),
        config.probability_clip,
    )
    reload_passed = bool(
        np.allclose(reload_probabilities, attack_probabilities[:check_count], atol=1e-12, rtol=1e-10)
    )

    # Secondary source-disjoint robustness experiment.
    strict_result: Optional[Dict[str, Any]] = None
    if config.run_strict_source_disjoint_experiment:
        print("Running strict source-disjoint robustness experiment...")
        strict_result = run_strict_source_experiment(selected, store, labels, paths, config)
        strict_serializable = {
            key: value
            for key, value in strict_result.items()
            if key not in {"per_class", "reliability", "confusion_matrix"}
        }
        write_json(validation_directory / "strict_source_disjoint_metrics.json", strict_serializable)
        write_csv(
            validation_directory / "strict_source_disjoint_per_class_metrics.csv",
            strict_result["per_class"],
        )
        write_csv(
            validation_directory / "strict_source_disjoint_reliability.csv",
            strict_result["reliability"],
        )
        if config.save_plots:
            plot_confusion_matrix(
                strict_result["confusion_matrix"],
                validation_directory / "strict_source_disjoint_confusion_matrix.png",
                "Strict source-disjoint robustness test",
            )

    selected_model_card = {
        "stage": 10,
        "model_role": "six-class fault-quality classifier",
        "class_names": list(CLASS_NAMES),
        "selected_view": selected.view_name,
        "selected_family": selected.family,
        "feature_count": selected.feature_count,
        "feature_names": selected.feature_names,
        "selection_rule": (
            "Maximum validation macro-F1; ties broken by mean SIFA/SEFA branch AP, "
            "lower validation log-loss, and lower validation ECE."
        ),
        "temperature": selected.temperature,
        "validation_metrics": selected.validation_metrics,
        "held_out_test_metrics": test_metrics,
        "held_out_test_branch_metrics": test_branch_metrics,
        "strict_source_disjoint_test_metrics": (
            strict_result["test_metrics"] if strict_result is not None else None
        ),
        "attack_labels_used": False,
        "limitations": [
            "All traces and fault labels are simulated.",
            "Observable-full includes legitimate public ciphertext shortcuts.",
            "Strict source-disjoint evaluation is secondary and smaller.",
            "The test set is used once after validation-only selection.",
        ],
    }
    write_json(model_directory / "selected_model_card.json", selected_model_card)
    write_json(model_directory / "attack_label_access_manifest.json", attack_label_access_manifest)

    feature_view_contract = {
        "parameter_only": list(PARAMETER_FEATURES),
        "trace_only": {
            "arrays": list(TRACE_ARRAY_NAMES),
            "dimension": int(
                store.full_trace.shape[1]
                + store.target_trace.shape[1]
                + store.pulse_trace.shape[1]
            ),
        },
        "parameter_trace": {
            "parameter_features": list(PARAMETER_FEATURES),
            "trace_arrays": list(TRACE_ARRAY_NAMES),
        },
        "observable_full": {
            "tabular_features": list(PRIMARY_TABULAR_FEATURES),
            "trace_arrays": list(TRACE_ARRAY_NAMES),
        },
        "forbidden_feature_tokens": list(FORBIDDEN_FEATURE_TOKENS),
    }
    write_json(model_directory / "feature_view_contract.json", feature_view_contract)

    probability_checks = {
        "validation_finite": bool(np.all(np.isfinite(selected.validation_probabilities))),
        "test_finite": bool(np.all(np.isfinite(test_probabilities))),
        "attack_finite": bool(np.all(np.isfinite(attack_probabilities))),
        "validation_row_sums": bool(np.allclose(np.sum(selected.validation_probabilities, axis=1), 1.0)),
        "test_row_sums": bool(np.allclose(np.sum(test_probabilities, axis=1), 1.0)),
        "attack_row_sums": bool(np.allclose(np.sum(attack_probabilities, axis=1), 1.0)),
    }

    validation_checks = {
        "stage09_all_checks_passed": bool(summary9["all_checks_passed"]),
        "stage09_public_freeze_verified": bool(paths["freeze_verification"]["passed"]),
        "stage09_public_leakage_audit_passed": bool(summary9["public_leakage_audit_passed"]),
        "feature_name_audit_passed": bool(feature_audit["passed"]),
        "partition_contract_passed": bool(partition_audit["passed"]),
        "locked_attack_labels_accessed": False,
        "candidate_count_correct": len(candidates) == len(feature_views) * len(config.candidate_families),
        "test_evaluation_count": 1,
        "probability_checks": probability_checks,
        "deployment_model_reload_passed": reload_passed,
        "attack_prediction_row_count_correct": len(attack_output) == int(summary9["partition_counts"]["attack"]),
        "selected_model_has_all_classes": set(selected.estimator.classes_.astype(int).tolist()) == set(CLASS_IDS.tolist()),
    }
    all_checks_passed = bool(
        validation_checks["stage09_all_checks_passed"]
        and validation_checks["stage09_public_freeze_verified"]
        and validation_checks["stage09_public_leakage_audit_passed"]
        and validation_checks["feature_name_audit_passed"]
        and validation_checks["partition_contract_passed"]
        and not validation_checks["locked_attack_labels_accessed"]
        and validation_checks["candidate_count_correct"]
        and validation_checks["test_evaluation_count"] == 1
        and all(probability_checks.values())
        and validation_checks["deployment_model_reload_passed"]
        and validation_checks["attack_prediction_row_count_correct"]
        and validation_checks["selected_model_has_all_classes"]
    )
    write_json(validation_directory / "stage_10_validation_checks.json", {
        "all_checks_passed": all_checks_passed,
        "checks": validation_checks,
        "stage09_freeze_verification": paths["freeze_verification"],
        "partition_audit": partition_audit,
        "feature_audit": feature_audit,
    })

    # Freeze all model and unlabeled-prediction outputs. No attack labels were opened.
    freeze_paths = sorted(
        [path for path in model_directory.iterdir() if path.is_file()]
        + [path for path in prediction_directory.iterdir() if path.is_file()]
    )
    freeze_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": (
            "The selected and deployment models, thresholds, and public probability "
            "outputs were frozen without reading Stage-09 locked attack labels."
        ),
        "attack_labels_accessed": False,
        "files": {
            str(path.relative_to(run_directory)): sha256_file(path)
            for path in freeze_paths
        },
    }
    freeze_payload = json.dumps(freeze_manifest, sort_keys=True).encode("utf-8")
    freeze_sha256 = hashlib.sha256(freeze_payload).hexdigest()
    freeze_manifest["freeze_sha256"] = freeze_sha256
    write_json(run_directory / "model_freeze_manifest.json", freeze_manifest)

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "stage": 10,
        "run_id": run_id,
        "run_directory": str(run_directory),
        "input_stage_09_run_directory": str(paths["stage9_run"]),
        "all_checks_passed": all_checks_passed,
        "stage_09_public_freeze_verified": bool(paths["freeze_verification"]["passed"]),
        "public_feature_leakage_audit_passed": bool(feature_audit["passed"]),
        "attack_labels_accessed": False,
        "number_of_candidates": len(candidates),
        "candidate_views": list(feature_views),
        "candidate_families": list(config.candidate_families),
        "selected_view": selected.view_name,
        "selected_family": selected.family,
        "selected_feature_count": selected.feature_count,
        "temperature": selected.temperature,
        "validation_metrics": selected.validation_metrics,
        "test_metrics": test_metrics,
        "test_branch_metrics": test_branch_metrics,
        "branch_thresholds": {
            "clean_target_ineffective": ineffective_threshold,
            "clean_target_effective": effective_threshold,
        },
        "strict_source_disjoint_test_metrics": (
            strict_result["test_metrics"] if strict_result is not None else None
        ),
        "partition_counts": summary9["partition_counts"],
        "development_prediction_row_count": int(len(development_output)),
        "attack_prediction_row_count": int(len(attack_output)),
        "attack_prediction_class_counts": {
            class_name: int(np.sum(np.argmax(attack_probabilities, axis=1) == class_id))
            for class_id, class_name in enumerate(CLASS_NAMES)
        },
        "attack_mean_class_probabilities": {
            class_name: float(np.mean(attack_probabilities[:, class_id]))
            for class_id, class_name in enumerate(CLASS_NAMES)
        },
        "model_reload_check_passed": reload_passed,
        "model_freeze_sha256": freeze_sha256,
        "stage_09_public_ml_freeze_sha256": summary9["public_ml_freeze_sha256"],
        "locked_attack_label_expected_sha256": summary9["attack_label_sha256"],
        "elapsed_seconds": float(elapsed_seconds),
        "model_files": sorted(path.name for path in model_directory.iterdir() if path.is_file()),
        "prediction_files": sorted(path.name for path in prediction_directory.iterdir() if path.is_file()),
        "validation_files": sorted(path.name for path in validation_directory.iterdir() if path.is_file()),
    }
    write_json(run_directory / "stage_10_summary.json", summary)

    write_json(run_directory / "run_manifest.json", {
        "stage": 10,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "config": {
            **asdict(config),
            "candidate_families": list(config.candidate_families),
        },
        "input_sha256": {
            "stage_09_summary.json": sha256_file(paths["summary_path"]),
            "ml_tabular_features_public.csv": sha256_file(paths["feature_csv"]),
            "ml_trace_views_public.npz": sha256_file(paths["trace_npz"]),
            "ml_quality_labels_train_validation_test.csv": sha256_file(paths["development_labels"]),
        },
        "locked_attack_labels_opened": False,
    })

    print("\n" + "=" * 86)
    print("Stage 10 complete: six-class LBlock fault-quality classifier")
    print("=" * 86)
    print("Run directory                 :", summary["run_directory"])
    print("All checks passed             :", summary["all_checks_passed"])
    print("Selected view / family        :", summary["selected_view"], "/", summary["selected_family"])
    print("Validation macro-F1           :", f"{summary['validation_metrics']['macro_f1']:.6f}")
    print("Held-out test macro-F1        :", f"{summary['test_metrics']['macro_f1']:.6f}")
    print("Held-out test balanced acc.   :", f"{summary['test_metrics']['balanced_accuracy']:.6f}")
    print("Held-out test log-loss        :", f"{summary['test_metrics']['log_loss']:.6f}")
    print("Attack labels accessed        :", summary["attack_labels_accessed"])
    print("Attack prediction rows        :", summary["attack_prediction_row_count"])
    print("Model freeze SHA-256          :", summary["model_freeze_sha256"])
    print("Elapsed seconds               :", f"{summary['elapsed_seconds']:.3f}")
    print("=" * 86)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 10 validation failed. Inspect validation_only/stage_10_validation_checks.json"
        )
    return summary


def load_stage_10_config(path: str | Path) -> Stage10Config:
    raw = read_json(Path(path))
    if "candidate_families" in raw:
        raw["candidate_families"] = tuple(raw["candidate_families"])
    return Stage10Config(**raw)


if __name__ == "__main__":
    default_config = Stage10Config(
        input_stage9_run_directory=(
            './runs/stage_09'
            '/stage09_20260718_184149_756498_seed20260718'
        ),
        output_root='./runs/stage_10',
    )
    run_stage_10(default_config)

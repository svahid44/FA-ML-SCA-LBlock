from __future__ import annotations

"""Stage 11 — Leakage-safe pre-injection glitch optimizer for LBlock-64/80.

The Stage-10 classifier is a post-injection quality estimator because it uses
response traces and ciphertext observables.  This stage distils its public,
calibrated probability outputs into a second model whose inputs are available
*before* a glitch is applied.  The resulting surrogate can therefore rank and
recommend glitch parameters for Stage 12 without using private fault labels,
internal cipher states, Test rows, or locked Attack labels.
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
except Exception:  # pragma: no cover - plots are optional
    plt = None

from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES: Tuple[str, ...] = (
    "missed",
    "clean_target_ineffective",
    "clean_target_effective",
    "off_target",
    "multi_hit",
    "invalid_reset",
)
PROBABILITY_COLUMNS: Tuple[str, ...] = tuple(f"p_{name}" for name in CLASS_NAMES)

# Every feature below is known or derived before injection.
PRE_INJECTION_FEATURES: Tuple[str, ...] = (
    "target_is_s5",
    "timing_offset_samples",
    "absolute_timing_offset_samples",
    "width_samples",
    "strength",
    "repeat",
    "repeat_spacing_samples",
    "pulse_span_samples",
    "glitch_energy_proxy",
)

PRIMITIVE_PARAMETER_COLUMNS: Tuple[str, ...] = (
    "timing_offset_samples",
    "width_samples",
    "strength",
    "repeat",
    "repeat_spacing_samples",
)

TARGET_SBOXES: Tuple[str, ...] = ("S0", "S5")
OBJECTIVES: Tuple[str, ...] = ("SIFA", "SEFA", "SHFA")
THEORETICAL_INEFFECTIVE_RATE = 81.0 / 256.0
THEORETICAL_EFFECTIVE_RATE = 1.0 - THEORETICAL_INEFFECTIVE_RATE
EPSILON = 1.0e-12


@dataclass(frozen=True)
class Stage11Config:
    input_stage10_run_directory: str
    output_root: str = "runs/stage_11"
    random_seed: int = 20260718

    candidate_families: Tuple[str, ...] = ("ridge", "extra_trees")
    ridge_alpha: float = 3.0
    extra_trees_estimators: int = 220
    extra_trees_max_depth: int = 24
    extra_trees_min_samples_leaf: int = 8
    extra_trees_max_features: float = 0.85
    extra_trees_n_jobs: int = 4

    candidate_count_per_target: int = 40000
    local_candidate_fraction: float = 0.40
    empirical_candidate_fraction: float = 0.15
    robust_lower_quantile: float = 0.005
    robust_upper_quantile: float = 0.995

    isolation_forest_estimators: int = 120
    isolation_forest_max_samples: int = 4096
    support_exploit_quantile: float = 0.05
    support_explore_quantile: float = 0.01

    offset_jitter_sigma_samples: float = 0.35
    width_relative_jitter: float = 0.06
    strength_relative_jitter: float = 0.06
    repeat_spacing_relative_jitter: float = 0.08
    robustness_scenarios: int = 9
    robustness_risk_penalty: float = 0.60
    exploration_disagreement_weight: float = 0.35

    exploit_recommendations_per_target_objective: int = 24
    explore_recommendations_per_target_objective: int = 12
    diversity_minimum_distance: float = 0.12

    minimum_branch_spearman: float = 0.20
    minimum_top_decile_uplift: float = 1.08
    probability_reload_check_rows: int = 512
    save_plots: bool = True

    stage12_recommended_experiments: int = 24000
    stage12_exploitation_fraction: float = 0.70
    stage12_exploration_fraction: float = 0.20
    stage12_control_fraction: float = 0.10


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinity")


def serialize_config(config: Stage11Config) -> Dict[str, Any]:
    result = asdict(config)
    result["candidate_families"] = list(config.candidate_families)
    return result


def normalize_probability_matrix(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != len(CLASS_NAMES):
        raise ValueError("Probability matrix must have six columns")
    result = np.clip(result, 1.0e-9, None)
    result /= np.sum(result, axis=1, keepdims=True)
    return result


def add_derived_pre_injection_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["target_is_s5"] = (
        result["target_sbox"].astype(str).eq("S5").astype(np.float64)
    )
    result["absolute_timing_offset_samples"] = np.abs(
        pd.to_numeric(result["timing_offset_samples"], errors="raise")
    )
    result["pulse_span_samples"] = (
        pd.to_numeric(result["width_samples"], errors="raise")
        + (
            pd.to_numeric(result["repeat"], errors="raise") - 1.0
        )
        * pd.to_numeric(result["repeat_spacing_samples"], errors="raise")
    )
    result["glitch_energy_proxy"] = (
        pd.to_numeric(result["width_samples"], errors="raise")
        * pd.to_numeric(result["strength"], errors="raise")
        * pd.to_numeric(result["repeat"], errors="raise")
    )
    return result


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [name for name in PRE_INJECTION_FEATURES if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing pre-injection features: {missing}")
    values = frame.loc[:, PRE_INJECTION_FEATURES].to_numpy(dtype=np.float64)
    ensure_finite("pre-injection feature matrix", values)
    return values


def target_probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = [name for name in PROBABILITY_COLUMNS if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing Stage-10 probability columns: {missing}")
    values = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    ensure_finite("Stage-10 soft targets", values)
    return normalize_probability_matrix(values)


# ---------------------------------------------------------------------------
# Input verification and public-only data assembly
# ---------------------------------------------------------------------------


def verify_stage10_freeze(stage10_directory: Path) -> Dict[str, Any]:
    summary_path = stage10_directory / "stage_10_summary.json"
    manifest_path = stage10_directory / "model_freeze_manifest.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)

    if not bool(summary.get("all_checks_passed", False)):
        raise RuntimeError("Stage 10 did not pass all checks")
    if bool(summary.get("attack_labels_accessed", True)):
        raise RuntimeError("Stage 10 reports that Attack labels were accessed")
    if bool(manifest.get("attack_labels_accessed", True)):
        raise RuntimeError("Stage 10 freeze manifest is not Attack-label safe")

    file_results: Dict[str, Any] = {}
    all_files_match = True
    for relative_name, expected_hash in manifest.get("files", {}).items():
        path = stage10_directory / relative_name
        exists = path.is_file()
        observed_hash = sha256_file(path) if exists else None
        matches = exists and observed_hash == expected_hash
        all_files_match = all_files_match and matches
        file_results[relative_name] = {
            "exists": exists,
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "matches": matches,
        }

    freeze_matches_summary = (
        str(manifest.get("freeze_sha256"))
        == str(summary.get("model_freeze_sha256"))
    )
    passed = bool(all_files_match and freeze_matches_summary)
    if not passed:
        raise RuntimeError("Stage 10 model freeze verification failed")

    return {
        "passed": passed,
        "summary": summary,
        "summary_path": summary_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "file_results": file_results,
        "freeze_matches_summary": freeze_matches_summary,
    }


def verify_stage9_public_freeze(stage9_directory: Path) -> Dict[str, Any]:
    summary_path = stage9_directory / "stage_09_summary.json"
    manifest_path = stage9_directory / "public_ml_freeze_manifest.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if not bool(summary.get("all_checks_passed", False)):
        raise RuntimeError("Stage 09 did not pass all checks")

    all_files_match = True
    file_results: Dict[str, Any] = {}
    for relative_name, expected_hash in manifest.get("files", {}).items():
        path = stage9_directory / "public_ml" / relative_name
        if not path.is_file():
            # Older Stage-09 manifests may store paths relative to run root.
            path = stage9_directory / relative_name
        exists = path.is_file()
        observed_hash = sha256_file(path) if exists else None
        matches = exists and observed_hash == expected_hash
        all_files_match = all_files_match and matches
        file_results[relative_name] = {
            "exists": exists,
            "observed_sha256": observed_hash,
            "expected_sha256": expected_hash,
            "matches": matches,
        }

    freeze_hash = str(
        manifest.get("freeze_sha256", manifest.get("public_ml_freeze_sha256", ""))
    )
    summary_hash = str(summary.get("public_ml_freeze_sha256", ""))
    freeze_matches_summary = bool(freeze_hash and freeze_hash == summary_hash)
    passed = bool(all_files_match and freeze_matches_summary)
    if not passed:
        raise RuntimeError("Stage 09 public ML freeze verification failed")

    return {
        "passed": passed,
        "summary": summary,
        "summary_path": summary_path,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "file_results": file_results,
        "freeze_matches_summary": freeze_matches_summary,
    }


def load_public_development_dataset(
    stage10_directory: Path,
    stage9_directory: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    probabilities_path = (
        stage10_directory
        / "predictions"
        / "development_quality_probabilities_public.csv"
    )
    features_path = stage9_directory / "public_ml" / "ml_tabular_features_public.csv"
    if not probabilities_path.is_file():
        raise FileNotFoundError(probabilities_path)
    if not features_path.is_file():
        raise FileNotFoundError(features_path)

    probabilities = pd.read_csv(probabilities_path)
    feature_columns = [
        "experiment_id",
        "campaign_partition",
        "target_sbox",
        "target_sbox_index",
        "key_id",
        "session_id",
        "source_healthy_trace_id",
        "fault_model",
        "timing_offset_samples",
        "width_samples",
        "strength",
        "repeat",
        "repeat_spacing_samples",
    ]
    features = pd.read_csv(features_path, usecols=feature_columns)

    if probabilities["experiment_id"].duplicated().any():
        raise RuntimeError("Stage-10 development probabilities contain duplicate IDs")
    if features["experiment_id"].duplicated().any():
        raise RuntimeError("Stage-09 public features contain duplicate IDs")

    probability_payload = probabilities[
        ["experiment_id", *PROBABILITY_COLUMNS]
    ].copy()
    merged = features.merge(
        probability_payload,
        on="experiment_id",
        how="inner",
        validate="one_to_one",
    )
    merged = add_derived_pre_injection_features(merged)

    expected_ids = set(probabilities["experiment_id"].astype(int).tolist())
    observed_ids = set(merged["experiment_id"].astype(int).tolist())
    if expected_ids != observed_ids:
        raise RuntimeError("Public feature/probability join is incomplete")

    partitions = set(merged["campaign_partition"].astype(str).unique())
    if partitions != {"train", "validation"}:
        raise RuntimeError(
            "Stage 11 may train only on Stage-10 Train and Validation predictions"
        )

    probability_values = target_probability_matrix(merged)
    row_sum_error = float(np.max(np.abs(np.sum(probability_values, axis=1) - 1.0)))
    audit = {
        "probabilities_path": str(probabilities_path),
        "features_path": str(features_path),
        "number_of_rows": int(len(merged)),
        "partition_counts": {
            key: int(value)
            for key, value in merged["campaign_partition"].value_counts().to_dict().items()
        },
        "target_counts": {
            key: int(value)
            for key, value in merged["target_sbox"].value_counts().to_dict().items()
        },
        "maximum_probability_row_sum_error": row_sum_error,
        "test_rows_used": 0,
        "attack_rows_used": 0,
        "private_labels_used": False,
    }
    return merged.sort_values("experiment_id").reset_index(drop=True), audit


def pre_injection_feature_leakage_audit() -> Dict[str, Any]:
    forbidden_tokens = (
        "response",
        "ciphertext",
        "trace",
        "category",
        "label",
        "master_key",
        "round_key",
        "true_key",
        "x31",
        "x32",
        "target_original",
        "target_faulted",
        "impacted",
        "actual_center",
        "hit_score",
        "activation",
        "oracle",
        "ground_truth",
        "fault_model",
        "session_id",
        "key_id",
        "source_healthy",
    )
    violations = [
        name
        for name in PRE_INJECTION_FEATURES
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    primitive_missing = [
        name for name in PRIMITIVE_PARAMETER_COLUMNS if name not in PRE_INJECTION_FEATURES
    ]
    return {
        "passed": not violations and not primitive_missing,
        "feature_names": list(PRE_INJECTION_FEATURES),
        "forbidden_token_violations": violations,
        "primitive_parameter_features_missing": primitive_missing,
        "statement": (
            "All optimizer inputs are selected or derived before fault injection. "
            "No response, ciphertext, trace, key, session, fault-model, or private-label "
            "field is present."
        ),
    }


# ---------------------------------------------------------------------------
# Surrogate modelling
# ---------------------------------------------------------------------------


def build_surrogate(family: str, config: Stage11Config) -> Any:
    if family == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=config.ridge_alpha)),
        ])
    if family == "extra_trees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=config.extra_trees_estimators,
                    max_depth=config.extra_trees_max_depth,
                    min_samples_leaf=config.extra_trees_min_samples_leaf,
                    max_features=config.extra_trees_max_features,
                    random_state=config.random_seed,
                    n_jobs=config.extra_trees_n_jobs,
                ),
            ),
        ])
    raise ValueError(f"Unknown surrogate family: {family}")


def surrogate_predict(model: Any, x: np.ndarray) -> np.ndarray:
    return normalize_probability_matrix(np.asarray(model.predict(x), dtype=np.float64))


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correlation = spearmanr(y_true, y_pred).correlation
    if correlation is None or not np.isfinite(correlation):
        return 0.0
    return float(correlation)


def top_fraction_uplift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraction: float = 0.10,
) -> float:
    count = max(1, int(math.ceil(len(y_pred) * fraction)))
    order = np.argsort(-y_pred)[:count]
    baseline = float(np.mean(y_true))
    selected = float(np.mean(y_true[order]))
    return float(selected / max(baseline, EPSILON))


def evaluate_surrogate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    class_metrics: Dict[str, Any] = {}
    for index, class_name in enumerate(CLASS_NAMES):
        true_values = y_true[:, index]
        predicted_values = y_pred[:, index]
        class_metrics[class_name] = {
            "mae": float(mean_absolute_error(true_values, predicted_values)),
            "rmse": float(math.sqrt(mean_squared_error(true_values, predicted_values))),
            "r2": float(r2_score(true_values, predicted_values)),
            "spearman": safe_spearman(true_values, predicted_values),
            "top_decile_uplift": top_fraction_uplift(true_values, predicted_values),
            "true_mean": float(np.mean(true_values)),
            "predicted_mean": float(np.mean(predicted_values)),
        }

    ineffective = class_metrics["clean_target_ineffective"]
    effective = class_metrics["clean_target_effective"]
    return {
        "all_class_mae": float(mean_absolute_error(y_true, y_pred)),
        "all_class_rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "maximum_probability_row_sum_error": float(
            np.max(np.abs(np.sum(y_pred, axis=1) - 1.0))
        ),
        "mean_branch_spearman": float(
            0.5 * (ineffective["spearman"] + effective["spearman"])
        ),
        "mean_branch_rmse": float(
            0.5 * (ineffective["rmse"] + effective["rmse"])
        ),
        "mean_branch_top_decile_uplift": float(
            0.5
            * (
                ineffective["top_decile_uplift"]
                + effective["top_decile_uplift"]
            )
        ),
        "class_metrics": class_metrics,
    }


def train_and_select_surrogate(
    frame: pd.DataFrame,
    config: Stage11Config,
) -> Tuple[str, Any, List[Dict[str, Any]], np.ndarray, np.ndarray]:
    train = frame[frame["campaign_partition"] == "train"].copy()
    validation = frame[frame["campaign_partition"] == "validation"].copy()
    x_train = feature_matrix(train)
    y_train = target_probability_matrix(train)
    x_validation = feature_matrix(validation)
    y_validation = target_probability_matrix(validation)

    candidate_results: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}
    validation_predictions: Dict[str, np.ndarray] = {}

    for family in config.candidate_families:
        model = build_surrogate(family, config)
        model.fit(x_train, y_train)
        prediction = surrogate_predict(model, x_validation)
        metrics = evaluate_surrogate(y_validation, prediction)
        fitted_models[family] = model
        validation_predictions[family] = prediction
        candidate_results.append({
            "family": family,
            "training_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            **{key: value for key, value in metrics.items() if key != "class_metrics"},
            "sifa_spearman": metrics["class_metrics"]["clean_target_ineffective"]["spearman"],
            "sifa_rmse": metrics["class_metrics"]["clean_target_ineffective"]["rmse"],
            "sifa_top_decile_uplift": metrics["class_metrics"]["clean_target_ineffective"]["top_decile_uplift"],
            "sefa_spearman": metrics["class_metrics"]["clean_target_effective"]["spearman"],
            "sefa_rmse": metrics["class_metrics"]["clean_target_effective"]["rmse"],
            "sefa_top_decile_uplift": metrics["class_metrics"]["clean_target_effective"]["top_decile_uplift"],
            "full_metrics": metrics,
        })

    ordered = sorted(
        candidate_results,
        key=lambda row: (
            -float(row["mean_branch_spearman"]),
            float(row["mean_branch_rmse"]),
            float(row["all_class_mae"]),
            str(row["family"]),
        ),
    )
    selected_family = str(ordered[0]["family"])
    selected_model = fitted_models[selected_family]
    selected_prediction = validation_predictions[selected_family]
    return (
        selected_family,
        selected_model,
        candidate_results,
        y_validation,
        selected_prediction,
    )


# ---------------------------------------------------------------------------
# Candidate generation and robust utility calculation
# ---------------------------------------------------------------------------


def target_parameter_bounds(
    frame: pd.DataFrame,
    target_sbox: str,
    config: Stage11Config,
) -> Dict[str, Any]:
    subset = frame[frame["target_sbox"] == target_sbox]
    if subset.empty:
        raise RuntimeError(f"No development rows for {target_sbox}")

    bounds: Dict[str, Any] = {}
    for name in (
        "timing_offset_samples",
        "width_samples",
        "strength",
        "repeat_spacing_samples",
    ):
        values = pd.to_numeric(subset[name], errors="raise").to_numpy(dtype=np.float64)
        lower = float(np.quantile(values, config.robust_lower_quantile))
        upper = float(np.quantile(values, config.robust_upper_quantile))
        if not upper > lower:
            lower = float(np.min(values))
            upper = float(np.max(values) + 1.0e-6)
        bounds[name] = {"lower": lower, "upper": upper}

    repeats = sorted(set(int(value) for value in subset["repeat"].tolist()))
    bounds["repeat"] = {"allowed": repeats}
    return bounds


def shfa_utility(probabilities: np.ndarray) -> np.ndarray:
    p_i = probabilities[:, CLASS_NAMES.index("clean_target_ineffective")]
    p_e = probabilities[:, CLASS_NAMES.index("clean_target_effective")]
    normalized_i = p_i / THEORETICAL_INEFFECTIVE_RATE
    normalized_e = p_e / THEORETICAL_EFFECTIVE_RATE
    return (
        2.0 * normalized_i * normalized_e
        / np.maximum(normalized_i + normalized_e, EPSILON)
    )


def observed_teacher_utility(frame: pd.DataFrame) -> np.ndarray:
    probabilities = target_probability_matrix(frame)
    p_i = probabilities[:, CLASS_NAMES.index("clean_target_ineffective")]
    p_e = probabilities[:, CLASS_NAMES.index("clean_target_effective")]
    p_h = shfa_utility(probabilities)
    # The union of the three objectives prevents local candidate generation
    # from collapsing around only the easier effective branch.
    return np.maximum.reduce([p_i, p_e, p_h])


def sample_uniform_candidates(
    count: int,
    bounds: Mapping[str, Any],
    target_sbox: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    repeat_values = np.asarray(bounds["repeat"]["allowed"], dtype=np.int32)
    result = pd.DataFrame({
        "target_sbox": [target_sbox] * count,
        "timing_offset_samples": rng.uniform(
            bounds["timing_offset_samples"]["lower"],
            bounds["timing_offset_samples"]["upper"],
            size=count,
        ),
        "width_samples": rng.uniform(
            bounds["width_samples"]["lower"],
            bounds["width_samples"]["upper"],
            size=count,
        ),
        "strength": rng.uniform(
            bounds["strength"]["lower"],
            bounds["strength"]["upper"],
            size=count,
        ),
        "repeat": rng.choice(repeat_values, size=count, replace=True),
        "repeat_spacing_samples": rng.uniform(
            bounds["repeat_spacing_samples"]["lower"],
            bounds["repeat_spacing_samples"]["upper"],
            size=count,
        ),
    })
    return add_derived_pre_injection_features(result)


def sample_local_candidates(
    count: int,
    target_frame: pd.DataFrame,
    bounds: Mapping[str, Any],
    target_sbox: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if count <= 0:
        return sample_uniform_candidates(0, bounds, target_sbox, rng)

    utilities = observed_teacher_utility(target_frame)
    threshold = float(np.quantile(utilities, 0.80))
    elite = target_frame.loc[utilities >= threshold, list(PRIMITIVE_PARAMETER_COLUMNS)]
    if elite.empty:
        elite = target_frame.loc[:, PRIMITIVE_PARAMETER_COLUMNS]
    base_indices = rng.integers(0, len(elite), size=count)
    base = elite.iloc[base_indices].reset_index(drop=True).copy()

    ranges = {
        name: float(bounds[name]["upper"] - bounds[name]["lower"])
        for name in (
            "timing_offset_samples",
            "width_samples",
            "strength",
            "repeat_spacing_samples",
        )
    }
    base["timing_offset_samples"] += rng.normal(0.0, 0.08 * ranges["timing_offset_samples"], size=count)
    base["width_samples"] += rng.normal(0.0, 0.06 * ranges["width_samples"], size=count)
    base["strength"] += rng.normal(0.0, 0.06 * ranges["strength"], size=count)
    base["repeat_spacing_samples"] += rng.normal(
        0.0,
        0.06 * ranges["repeat_spacing_samples"],
        size=count,
    )

    for name in (
        "timing_offset_samples",
        "width_samples",
        "strength",
        "repeat_spacing_samples",
    ):
        base[name] = np.clip(
            base[name].to_numpy(dtype=np.float64),
            bounds[name]["lower"],
            bounds[name]["upper"],
        )
    allowed = np.asarray(bounds["repeat"]["allowed"], dtype=np.int32)
    base["repeat"] = np.asarray([
        int(allowed[np.argmin(np.abs(allowed - int(round(value))))])
        for value in base["repeat"].tolist()
    ])
    base.insert(0, "target_sbox", target_sbox)
    return add_derived_pre_injection_features(base)


def sample_empirical_candidates(
    count: int,
    target_frame: pd.DataFrame,
    target_sbox: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if count <= 0:
        return target_frame.iloc[:0].copy()
    indices = rng.integers(0, len(target_frame), size=count)
    result = target_frame.iloc[indices][list(PRIMITIVE_PARAMETER_COLUMNS)].reset_index(drop=True)
    result.insert(0, "target_sbox", target_sbox)
    return add_derived_pre_injection_features(result)


def build_candidate_pool(
    frame: pd.DataFrame,
    target_sbox: str,
    config: Stage11Config,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    bounds = target_parameter_bounds(frame, target_sbox, config)
    target_frame = frame[frame["target_sbox"] == target_sbox].copy()
    total = int(config.candidate_count_per_target)
    local_count = int(round(total * config.local_candidate_fraction))
    empirical_count = int(round(total * config.empirical_candidate_fraction))
    uniform_count = total - local_count - empirical_count

    parts = [
        sample_uniform_candidates(uniform_count, bounds, target_sbox, rng),
        sample_local_candidates(local_count, target_frame, bounds, target_sbox, rng),
        sample_empirical_candidates(empirical_count, target_frame, target_sbox, rng),
    ]
    candidates = pd.concat(parts, ignore_index=True)
    candidates["repeat"] = candidates["repeat"].astype(np.int32)
    candidates = candidates.drop_duplicates(subset=list(PRE_INJECTION_FEATURES)).reset_index(drop=True)
    candidates.insert(0, "candidate_id", np.arange(len(candidates), dtype=np.int64))
    return candidates, bounds


def fit_support_model(
    frame: pd.DataFrame,
    config: Stage11Config,
) -> Tuple[IsolationForest, float, float, Dict[str, Any]]:
    x = feature_matrix(frame)
    model = IsolationForest(
        n_estimators=config.isolation_forest_estimators,
        max_samples=min(config.isolation_forest_max_samples, len(frame)),
        contamination="auto",
        random_state=config.random_seed + 11001,
        n_jobs=config.extra_trees_n_jobs,
    )
    model.fit(x)
    scores = model.decision_function(x)
    exploit_threshold = float(np.quantile(scores, config.support_exploit_quantile))
    explore_threshold = float(np.quantile(scores, config.support_explore_quantile))
    return model, exploit_threshold, explore_threshold, {
        "development_score_mean": float(np.mean(scores)),
        "development_score_std": float(np.std(scores)),
        "exploit_threshold": exploit_threshold,
        "explore_threshold": explore_threshold,
    }


def jittered_candidate_frame(
    candidates: pd.DataFrame,
    scenario_index: int,
    config: Stage11Config,
    bounds: Mapping[str, Any],
) -> pd.DataFrame:
    result = candidates.copy()
    if scenario_index == 0:
        return result

    rng = np.random.default_rng([
        config.random_seed,
        11100,
        scenario_index,
        0 if str(candidates.iloc[0]["target_sbox"]) == "S0" else 5,
    ])
    count = len(result)
    result["timing_offset_samples"] += rng.normal(
        0.0,
        config.offset_jitter_sigma_samples,
        size=count,
    )
    result["width_samples"] *= 1.0 + rng.normal(
        0.0,
        config.width_relative_jitter,
        size=count,
    )
    result["strength"] *= 1.0 + rng.normal(
        0.0,
        config.strength_relative_jitter,
        size=count,
    )
    result["repeat_spacing_samples"] *= 1.0 + rng.normal(
        0.0,
        config.repeat_spacing_relative_jitter,
        size=count,
    )

    for name in (
        "timing_offset_samples",
        "width_samples",
        "strength",
        "repeat_spacing_samples",
    ):
        result[name] = np.clip(
            result[name].to_numpy(dtype=np.float64),
            bounds[name]["lower"],
            bounds[name]["upper"],
        )
    return add_derived_pre_injection_features(result)


def score_candidate_pool(
    candidates: pd.DataFrame,
    selected_model: Any,
    all_models: Mapping[str, Any],
    support_model: IsolationForest,
    bounds: Mapping[str, Any],
    config: Stage11Config,
) -> pd.DataFrame:
    scenario_predictions: List[np.ndarray] = []
    for scenario_index in range(config.robustness_scenarios):
        jittered = jittered_candidate_frame(candidates, scenario_index, config, bounds)
        scenario_predictions.append(
            surrogate_predict(selected_model, feature_matrix(jittered))
        )
    prediction_cube = np.stack(scenario_predictions, axis=0)
    mean_probabilities = np.mean(prediction_cube, axis=0)
    std_probabilities = np.std(prediction_cube, axis=0)

    base_model_predictions = np.stack([
        surrogate_predict(model, feature_matrix(candidates))
        for model in all_models.values()
    ], axis=0)
    disagreement = np.mean(
        np.std(
            base_model_predictions[:, :, [
                CLASS_NAMES.index("clean_target_ineffective"),
                CLASS_NAMES.index("clean_target_effective"),
            ]],
            axis=0,
        ),
        axis=1,
    )

    support_scores = support_model.decision_function(feature_matrix(candidates))
    p_i_index = CLASS_NAMES.index("clean_target_ineffective")
    p_e_index = CLASS_NAMES.index("clean_target_effective")
    p_i_mean = mean_probabilities[:, p_i_index]
    p_e_mean = mean_probabilities[:, p_e_index]
    p_i_std = std_probabilities[:, p_i_index]
    p_e_std = std_probabilities[:, p_e_index]

    scenario_shfa = np.stack(
        [shfa_utility(prediction_cube[index]) for index in range(prediction_cube.shape[0])],
        axis=0,
    )
    shfa_mean = np.mean(scenario_shfa, axis=0)
    shfa_std = np.std(scenario_shfa, axis=0)

    result = candidates.copy()
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"predicted_mean_{class_name}"] = mean_probabilities[:, index]
        result[f"predicted_std_{class_name}"] = std_probabilities[:, index]
    result["predicted_mean_clean_target"] = p_i_mean + p_e_mean
    result["model_disagreement"] = disagreement
    result["support_score"] = support_scores
    result["robust_utility_SIFA"] = (
        p_i_mean - config.robustness_risk_penalty * p_i_std
    )
    result["robust_utility_SEFA"] = (
        p_e_mean - config.robustness_risk_penalty * p_e_std
    )
    result["robust_utility_SHFA"] = (
        shfa_mean - config.robustness_risk_penalty * shfa_std
    )
    result["exploration_utility_SIFA"] = (
        result["robust_utility_SIFA"]
        + config.exploration_disagreement_weight * disagreement
    )
    result["exploration_utility_SEFA"] = (
        result["robust_utility_SEFA"]
        + config.exploration_disagreement_weight * disagreement
    )
    result["exploration_utility_SHFA"] = (
        result["robust_utility_SHFA"]
        + config.exploration_disagreement_weight * disagreement
    )
    ensure_finite(
        "candidate scoring",
        result.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64),
    )
    return result


def standardized_parameter_matrix(frame: pd.DataFrame, bounds: Mapping[str, Any]) -> np.ndarray:
    columns = []
    for name in (
        "timing_offset_samples",
        "width_samples",
        "strength",
        "repeat_spacing_samples",
    ):
        lower = float(bounds[name]["lower"])
        upper = float(bounds[name]["upper"])
        columns.append(
            (
                frame[name].to_numpy(dtype=np.float64) - lower
            )
            / max(upper - lower, EPSILON)
        )
    repeats = np.asarray(bounds["repeat"]["allowed"], dtype=np.float64)
    repeat_min = float(np.min(repeats))
    repeat_max = float(np.max(repeats))
    columns.append(
        (frame["repeat"].to_numpy(dtype=np.float64) - repeat_min)
        / max(repeat_max - repeat_min, 1.0)
    )
    return np.column_stack(columns)


def select_diverse_rows(
    frame: pd.DataFrame,
    score_column: str,
    count: int,
    bounds: Mapping[str, Any],
    minimum_distance: float,
    exclude_candidate_ids: Iterable[int] = (),
) -> pd.DataFrame:
    excluded = set(int(value) for value in exclude_candidate_ids)
    ordered = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    ordered = ordered[~ordered["candidate_id"].astype(int).isin(excluded)].reset_index(drop=True)
    vectors = standardized_parameter_matrix(ordered, bounds)

    selected_indices: List[int] = []
    threshold = float(minimum_distance)
    while len(selected_indices) < count and threshold >= 0.0:
        selected_indices = []
        for index in range(len(ordered)):
            if not selected_indices:
                selected_indices.append(index)
            else:
                distance = np.min(
                    np.linalg.norm(vectors[index] - vectors[selected_indices], axis=1)
                )
                if distance >= threshold:
                    selected_indices.append(index)
            if len(selected_indices) >= count:
                break
        if len(selected_indices) < count:
            threshold = max(0.0, threshold - 0.02)
            if threshold == 0.0:
                selected_indices = list(range(min(count, len(ordered))))
                break

    return ordered.iloc[selected_indices[:count]].copy().reset_index(drop=True)


def build_recommendations(
    scored_by_target: Mapping[str, pd.DataFrame],
    bounds_by_target: Mapping[str, Mapping[str, Any]],
    exploit_support_threshold: float,
    explore_support_threshold: float,
    config: Stage11Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    exploit_parts: List[pd.DataFrame] = []
    explore_parts: List[pd.DataFrame] = []

    for target_sbox in TARGET_SBOXES:
        frame = scored_by_target[target_sbox]
        bounds = bounds_by_target[target_sbox]
        exploit_eligible = frame[frame["support_score"] >= exploit_support_threshold]
        explore_eligible = frame[frame["support_score"] >= explore_support_threshold]
        if exploit_eligible.empty or explore_eligible.empty:
            raise RuntimeError(f"No support-safe optimizer candidates for {target_sbox}")

        for objective in OBJECTIVES:
            robust_column = f"robust_utility_{objective}"
            exploration_column = f"exploration_utility_{objective}"
            exploit = select_diverse_rows(
                exploit_eligible,
                robust_column,
                config.exploit_recommendations_per_target_objective,
                bounds,
                config.diversity_minimum_distance,
            )
            exploit.insert(1, "objective", objective)
            exploit.insert(2, "recommendation_mode", "exploit")
            exploit.insert(3, "rank", np.arange(1, len(exploit) + 1, dtype=np.int32))
            exploit_parts.append(exploit)

            minimum_acceptable = float(frame[robust_column].quantile(0.65))
            exploration_pool = explore_eligible[
                explore_eligible[robust_column] >= minimum_acceptable
            ]
            explore = select_diverse_rows(
                exploration_pool,
                exploration_column,
                config.explore_recommendations_per_target_objective,
                bounds,
                config.diversity_minimum_distance,
                exclude_candidate_ids=exploit["candidate_id"].astype(int).tolist(),
            )
            explore.insert(1, "objective", objective)
            explore.insert(2, "recommendation_mode", "explore")
            explore.insert(3, "rank", np.arange(1, len(explore) + 1, dtype=np.int32))
            explore_parts.append(explore)

    exploit_frame = pd.concat(exploit_parts, ignore_index=True)
    explore_frame = pd.concat(explore_parts, ignore_index=True)
    return exploit_frame, explore_frame


# ---------------------------------------------------------------------------
# Plots and reports
# ---------------------------------------------------------------------------


def save_validation_plots(
    directory: Path,
    y_validation: np.ndarray,
    prediction: np.ndarray,
    config: Stage11Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    for class_name in ("clean_target_ineffective", "clean_target_effective"):
        index = CLASS_NAMES.index(class_name)
        figure = plt.figure(figsize=(6.5, 5.5))
        axis = figure.add_subplot(1, 1, 1)
        axis.hexbin(
            y_validation[:, index],
            prediction[:, index],
            gridsize=38,
            mincnt=1,
        )
        lower = min(float(np.min(y_validation[:, index])), float(np.min(prediction[:, index])))
        upper = max(float(np.max(y_validation[:, index])), float(np.max(prediction[:, index])))
        axis.plot([lower, upper], [lower, upper], linestyle="--")
        axis.set_xlabel("Stage-10 teacher probability")
        axis.set_ylabel("Stage-11 pre-injection surrogate probability")
        axis.set_title(f"Validation distillation: {class_name}")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        path = directory / f"validation_teacher_vs_surrogate_{class_name}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        generated.append(path.name)
    return generated


def save_recommendation_plots(
    directory: Path,
    scored_by_target: Mapping[str, pd.DataFrame],
    config: Stage11Config,
) -> List[str]:
    if plt is None or not config.save_plots:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    for target_sbox in TARGET_SBOXES:
        frame = scored_by_target[target_sbox]
        display = frame.iloc[
            np.linspace(0, len(frame) - 1, min(12000, len(frame)), dtype=np.int64)
        ]
        for objective in OBJECTIVES:
            figure = plt.figure(figsize=(7.5, 5.8))
            axis = figure.add_subplot(1, 1, 1)
            scatter = axis.scatter(
                display["timing_offset_samples"],
                display["width_samples"],
                c=display[f"robust_utility_{objective}"],
                s=7,
                alpha=0.35,
            )
            axis.set_xlabel("Timing offset [samples]")
            axis.set_ylabel("Width [samples]")
            axis.set_title(f"{target_sbox} robust {objective} surrogate utility")
            axis.grid(alpha=0.2)
            figure.colorbar(scatter, ax=axis, label="Robust predicted utility")
            figure.tight_layout()
            path = directory / f"{target_sbox}_{objective}_response_surface.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            generated.append(path.name)
    return generated


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_stage_11(config: Stage11Config) -> Dict[str, Any]:
    start_time = time.perf_counter()
    stage10_directory = Path(config.input_stage10_run_directory).expanduser().resolve()
    stage10_verification = verify_stage10_freeze(stage10_directory)
    stage10_summary = stage10_verification["summary"]

    stage9_directory = Path(stage10_summary["input_stage_09_run_directory"]).expanduser().resolve()
    stage9_verification = verify_stage9_public_freeze(stage9_directory)
    development, data_audit = load_public_development_dataset(
        stage10_directory,
        stage9_directory,
    )
    leakage_audit = pre_injection_feature_leakage_audit()
    if not leakage_audit["passed"]:
        raise RuntimeError("Pre-injection feature leakage audit failed")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"stage11_{timestamp}_seed{config.random_seed}"
    run_directory = Path(config.output_root).expanduser().resolve() / run_id
    model_directory = run_directory / "models"
    recommendation_directory = run_directory / "recommendations"
    validation_directory = run_directory / "validation_only"
    model_directory.mkdir(parents=True, exist_ok=False)
    recommendation_directory.mkdir(parents=True, exist_ok=True)
    validation_directory.mkdir(parents=True, exist_ok=True)

    (
        selected_family,
        selected_train_model,
        candidate_results,
        y_validation,
        selected_validation_prediction,
    ) = train_and_select_surrogate(development, config)

    selected_candidate = next(
        row for row in candidate_results if row["family"] == selected_family
    )
    selected_validation_metrics = selected_candidate["full_metrics"]

    # Save a flat candidate table for human review.
    flat_candidate_results = [
        {key: value for key, value in row.items() if key != "full_metrics"}
        for row in candidate_results
    ]
    pd.DataFrame(flat_candidate_results).to_csv(
        validation_directory / "candidate_surrogate_validation_results.csv",
        index=False,
    )
    write_json(
        validation_directory / "selected_surrogate_validation_metrics.json",
        selected_validation_metrics,
    )

    validation_mask = development["campaign_partition"].eq("validation").to_numpy()
    validation_rows = development.loc[
        validation_mask,
        ["experiment_id", "target_sbox", "key_id", "session_id"],
    ].reset_index(drop=True)
    for index, class_name in enumerate(CLASS_NAMES):
        validation_rows[f"teacher_p_{class_name}"] = y_validation[:, index]
        validation_rows[f"surrogate_p_{class_name}"] = selected_validation_prediction[:, index]
    validation_rows.to_csv(
        validation_directory / "validation_teacher_surrogate_predictions.csv",
        index=False,
    )

    validation_plots = save_validation_plots(
        validation_directory,
        y_validation,
        selected_validation_prediction,
        config,
    )

    # Refit every candidate family on Train+Validation.  The selected model is
    # used for exploitation; cross-family disagreement supports exploration.
    x_development = feature_matrix(development)
    y_development = target_probability_matrix(development)
    deployment_models: Dict[str, Any] = {}
    for family in config.candidate_families:
        model = build_surrogate(family, config)
        model.fit(x_development, y_development)
        deployment_models[family] = model
        joblib.dump(model, model_directory / f"glitch_surrogate_{family}.joblib")

    selected_deployment_model = deployment_models[selected_family]
    selected_model_path = model_directory / "pre_injection_glitch_optimizer.joblib"
    joblib.dump(selected_deployment_model, selected_model_path)

    reload_count = min(config.probability_reload_check_rows, len(development))
    reloaded_model = joblib.load(selected_model_path)
    original_reload_prediction = surrogate_predict(
        selected_deployment_model,
        x_development[:reload_count],
    )
    reloaded_prediction = surrogate_predict(
        reloaded_model,
        x_development[:reload_count],
    )
    reload_check_passed = bool(
        np.allclose(original_reload_prediction, reloaded_prediction, atol=1.0e-12)
    )

    support_model, exploit_support_threshold, explore_support_threshold, support_stats = (
        fit_support_model(development, config)
    )
    support_model_path = model_directory / "pre_injection_support_model.joblib"
    joblib.dump(support_model, support_model_path)

    rng = np.random.default_rng(config.random_seed + 11011)
    scored_by_target: Dict[str, pd.DataFrame] = {}
    bounds_by_target: Dict[str, Dict[str, Any]] = {}
    candidate_pool_summary: Dict[str, Any] = {}

    for target_sbox in TARGET_SBOXES:
        candidates, bounds = build_candidate_pool(
            development,
            target_sbox,
            config,
            rng,
        )
        scored = score_candidate_pool(
            candidates,
            selected_deployment_model,
            deployment_models,
            support_model,
            bounds,
            config,
        )
        scored_by_target[target_sbox] = scored
        bounds_by_target[target_sbox] = bounds
        candidate_pool_summary[target_sbox] = {
            "candidate_count": int(len(scored)),
            "exploit_support_safe_count": int(
                np.sum(scored["support_score"] >= exploit_support_threshold)
            ),
            "explore_support_safe_count": int(
                np.sum(scored["support_score"] >= explore_support_threshold)
            ),
            "parameter_bounds": bounds,
            "maximum_robust_utilities": {
                objective: float(scored[f"robust_utility_{objective}"].max())
                for objective in OBJECTIVES
            },
        }

    exploit_recommendations, explore_recommendations = build_recommendations(
        scored_by_target,
        bounds_by_target,
        exploit_support_threshold,
        explore_support_threshold,
        config,
    )
    exploit_path = recommendation_directory / "exploit_parameter_recommendations.csv"
    explore_path = recommendation_directory / "explore_parameter_recommendations.csv"
    exploit_recommendations.to_csv(exploit_path, index=False)
    explore_recommendations.to_csv(explore_path, index=False)
    combined_recommendations = pd.concat(
        [exploit_recommendations, explore_recommendations],
        ignore_index=True,
    )
    combined_path = recommendation_directory / "all_parameter_recommendations.csv"
    combined_recommendations.to_csv(combined_path, index=False)

    objective_contract = {
        "stage": 11,
        "teacher": "Stage-10 calibrated six-class public probabilities",
        "pre_injection_features": list(PRE_INJECTION_FEATURES),
        "probability_classes": list(CLASS_NAMES),
        "objectives": {
            "SIFA": {
                "base_probability": "p_clean_target_ineffective",
                "robust_utility": "mean(p_i under parameter jitter) - risk_penalty * std(p_i)",
            },
            "SEFA": {
                "base_probability": "p_clean_target_effective",
                "robust_utility": "mean(p_e under parameter jitter) - risk_penalty * std(p_e)",
            },
            "SHFA": {
                "base_probability": (
                    "harmonic mean of p_i/(81/256) and p_e/(175/256)"
                ),
                "robust_utility": "mean(normalized harmonic utility) - risk_penalty * std",
            },
        },
        "theoretical_ineffective_rate": THEORETICAL_INEFFECTIVE_RATE,
        "theoretical_effective_rate": THEORETICAL_EFFECTIVE_RATE,
        "risk_penalty": config.robustness_risk_penalty,
        "exploration_bonus": (
            "robust utility + exploration_disagreement_weight * cross-family disagreement"
        ),
        "exploration_disagreement_weight": config.exploration_disagreement_weight,
        "ground_truth_used": False,
    }
    write_json(model_directory / "optimizer_objective_contract.json", objective_contract)

    feature_contract = {
        "stage": 11,
        "role": "pre-injection glitch-parameter optimizer",
        "feature_names": list(PRE_INJECTION_FEATURES),
        "primitive_parameter_columns": list(PRIMITIVE_PARAMETER_COLUMNS),
        "selected_family": selected_family,
        "target_sboxes": list(TARGET_SBOXES),
        "forbidden_inputs": [
            "response traces",
            "ciphertext equality or Hamming distance",
            "response_received",
            "fault_model",
            "key_id or session_id",
            "private labels or internal states",
            "Test or Attack rows",
        ],
        "teacher_distillation_statement": (
            "Stage-10 post-injection public probabilities are training targets only; "
            "they are never optimizer input features."
        ),
    }
    write_json(model_directory / "pre_injection_feature_contract.json", feature_contract)

    stage12_policy = {
        "stage": 12,
        "recommended_total_experiments": config.stage12_recommended_experiments,
        "fractions": {
            "exploit_recommendations": config.stage12_exploitation_fraction,
            "explore_recommendations": config.stage12_exploration_fraction,
            "negative_and_safety_controls": config.stage12_control_fraction,
        },
        "target_allocation": {"S0": 0.5, "S5": 0.5},
        "objective_allocation": {"SIFA": 0.35, "SEFA": 0.35, "SHFA": 0.30},
        "closed_loop_rule": (
            "After each batch, score realized responses with the frozen Stage-10 quality "
            "classifier, update empirical yields, but do not retrain on Attack labels."
        ),
        "exploit_source": str(exploit_path),
        "explore_source": str(explore_path),
        "control_requirement": (
            "Retain missed, neighbor, multi-hit, and invalid controls so quality drift remains observable."
        ),
    }
    write_json(recommendation_directory / "stage_12_campaign_policy.json", stage12_policy)

    recommendation_plots = save_recommendation_plots(
        recommendation_directory,
        scored_by_target,
        config,
    )

    # Public-only access manifest.  No Stage-09 label file or Stage-10 Attack
    # probability file is opened by this stage.
    access_manifest = {
        "stage_10_files_read": [
            str(stage10_verification["summary_path"]),
            str(stage10_verification["manifest_path"]),
            str(
                stage10_directory
                / "predictions"
                / "development_quality_probabilities_public.csv"
            ),
        ],
        "stage_09_files_read": [
            str(stage9_verification["summary_path"]),
            str(stage9_verification["manifest_path"]),
            str(stage9_directory / "public_ml" / "ml_tabular_features_public.csv"),
        ],
        "files_explicitly_not_read": [
            str(stage9_directory / "private_labels"),
            str(stage9_directory / "locked_attack_labels"),
            str(
                stage10_directory
                / "predictions"
                / "attack_quality_probabilities_public.csv"
            ),
            str(stage10_directory / "validation_only"),
        ],
        "development_private_labels_accessed": False,
        "test_rows_accessed": False,
        "attack_rows_accessed": False,
        "attack_labels_accessed": False,
    }
    write_json(model_directory / "public_only_access_manifest.json", access_manifest)

    model_card = {
        "stage": 11,
        "model_role": "pre-injection multi-output glitch-quality surrogate",
        "selected_family": selected_family,
        "candidate_families": list(config.candidate_families),
        "feature_count": len(PRE_INJECTION_FEATURES),
        "feature_names": list(PRE_INJECTION_FEATURES),
        "soft_target_columns": list(PROBABILITY_COLUMNS),
        "selection_rule": (
            "Highest validation mean SIFA/SEFA Spearman rank correlation; ties broken "
            "by lower branch RMSE and lower all-class MAE."
        ),
        "validation_metrics": selected_validation_metrics,
        "stage_10_teacher_model_freeze_sha256": stage10_summary["model_freeze_sha256"],
        "teacher_selected_view": stage10_summary["selected_view"],
        "teacher_selected_family": stage10_summary["selected_family"],
        "private_labels_used": False,
        "test_or_attack_rows_used": False,
        "limitations": [
            "The optimizer distils a simulated post-injection teacher model.",
            "Recommended uplift is model-predicted and must be validated in Stage 12.",
            "Cross-family disagreement is only an uncertainty proxy.",
            "All candidate domains are bounded by observed public development parameters.",
        ],
    }
    write_json(model_directory / "optimizer_model_card.json", model_card)

    recommendation_diagnostics = {
        "candidate_pool_summary": candidate_pool_summary,
        "support_model": support_stats,
        "recommendation_counts": {
            "exploit": int(len(exploit_recommendations)),
            "explore": int(len(explore_recommendations)),
            "combined": int(len(combined_recommendations)),
        },
        "best_recommendations": [
            {
                "target_sbox": target,
                "objective": objective,
                "mode": mode,
                "robust_utility": float(
                    subset[f"robust_utility_{objective}"].iloc[0]
                ),
                "timing_offset_samples": float(subset["timing_offset_samples"].iloc[0]),
                "width_samples": float(subset["width_samples"].iloc[0]),
                "strength": float(subset["strength"].iloc[0]),
                "repeat": int(subset["repeat"].iloc[0]),
                "repeat_spacing_samples": float(subset["repeat_spacing_samples"].iloc[0]),
            }
            for target in TARGET_SBOXES
            for objective in OBJECTIVES
            for mode, source in (
                ("exploit", exploit_recommendations),
                ("explore", explore_recommendations),
            )
            for subset in [
                source[
                    (source["target_sbox"] == target)
                    & (source["objective"] == objective)
                ].sort_values("rank")
            ]
            if not subset.empty
        ],
    }
    write_json(
        validation_directory / "recommendation_diagnostics.json",
        recommendation_diagnostics,
    )

    validation_checks = {
        "stage_10_model_freeze_verified": {"passed": stage10_verification["passed"]},
        "stage_09_public_freeze_verified": {"passed": stage9_verification["passed"]},
        "public_only_data_access": {
            "passed": bool(
                not access_manifest["development_private_labels_accessed"]
                and not access_manifest["test_rows_accessed"]
                and not access_manifest["attack_rows_accessed"]
                and not access_manifest["attack_labels_accessed"]
            ),
        },
        "pre_injection_feature_leakage_audit": leakage_audit,
        "development_partitions_only": {
            "passed": set(development["campaign_partition"].unique())
            == {"train", "validation"},
            "partition_counts": data_audit["partition_counts"],
        },
        "selected_surrogate_branch_rank_quality": {
            "passed": bool(
                selected_validation_metrics["mean_branch_spearman"]
                >= config.minimum_branch_spearman
            ),
            "observed": selected_validation_metrics["mean_branch_spearman"],
            "minimum": config.minimum_branch_spearman,
        },
        "selected_surrogate_top_decile_uplift": {
            "passed": bool(
                selected_validation_metrics["mean_branch_top_decile_uplift"]
                >= config.minimum_top_decile_uplift
            ),
            "observed": selected_validation_metrics["mean_branch_top_decile_uplift"],
            "minimum": config.minimum_top_decile_uplift,
        },
        "model_reload_check": {"passed": reload_check_passed},
        "recommendation_counts": {
            "passed": bool(
                len(exploit_recommendations)
                == len(TARGET_SBOXES)
                * len(OBJECTIVES)
                * config.exploit_recommendations_per_target_objective
                and len(explore_recommendations)
                == len(TARGET_SBOXES)
                * len(OBJECTIVES)
                * config.explore_recommendations_per_target_objective
            ),
            "exploit": int(len(exploit_recommendations)),
            "explore": int(len(explore_recommendations)),
        },
        "recommendations_are_finite": {
            "passed": bool(
                np.all(
                    np.isfinite(
                        combined_recommendations.select_dtypes(include=[np.number])
                        .to_numpy(dtype=np.float64)
                    )
                )
            ),
        },
        "all_targets_and_objectives_present": {
            "passed": bool(
                set(combined_recommendations["target_sbox"].unique())
                == set(TARGET_SBOXES)
                and set(combined_recommendations["objective"].unique())
                == set(OBJECTIVES)
            ),
        },
    }
    all_checks_passed = all(
        bool(item["passed"]) for item in validation_checks.values()
    )
    write_json(
        validation_directory / "stage_11_validation_checks.json",
        {
            "all_checks_passed": all_checks_passed,
            "checks": validation_checks,
        },
    )

    # Freeze all model and recommendation artifacts.  This stage is public-only,
    # so there is no later private-label phase.
    freeze_files = sorted(
        [path for path in model_directory.rglob("*") if path.is_file()]
        + [path for path in recommendation_directory.rglob("*") if path.is_file()]
    )
    freeze_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": (
            "Stage-11 pre-injection optimizer, support model, objectives, and "
            "recommendations were frozen using public Train/Validation data only."
        ),
        "development_private_labels_accessed": False,
        "test_rows_accessed": False,
        "attack_rows_accessed": False,
        "attack_labels_accessed": False,
        "files": {
            str(path.relative_to(run_directory)).replace("\\", "/"): sha256_file(path)
            for path in freeze_files
        },
    }
    freeze_sha256 = stable_json_hash(freeze_payload)
    freeze_payload["freeze_sha256"] = freeze_sha256
    write_json(run_directory / "optimizer_freeze_manifest.json", freeze_payload)

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "stage": 11,
        "run_id": run_id,
        "run_directory": str(run_directory),
        "input_stage_10_run_directory": str(stage10_directory),
        "input_stage_09_run_directory": str(stage9_directory),
        "all_checks_passed": bool(all_checks_passed),
        "stage_10_model_freeze_verified": bool(stage10_verification["passed"]),
        "stage_09_public_freeze_verified": bool(stage9_verification["passed"]),
        "public_only_training": True,
        "development_private_labels_accessed": False,
        "test_rows_accessed": False,
        "attack_rows_accessed": False,
        "attack_labels_accessed": False,
        "development_row_count": int(len(development)),
        "partition_counts": data_audit["partition_counts"],
        "selected_family": selected_family,
        "candidate_families": list(config.candidate_families),
        "pre_injection_feature_count": len(PRE_INJECTION_FEATURES),
        "pre_injection_features": list(PRE_INJECTION_FEATURES),
        "validation_metrics": selected_validation_metrics,
        "candidate_count_per_target_requested": config.candidate_count_per_target,
        "candidate_pool_summary": candidate_pool_summary,
        "exploit_recommendation_count": int(len(exploit_recommendations)),
        "explore_recommendation_count": int(len(explore_recommendations)),
        "recommendation_targets": list(TARGET_SBOXES),
        "recommendation_objectives": list(OBJECTIVES),
        "support_thresholds": {
            "exploit": exploit_support_threshold,
            "explore": explore_support_threshold,
        },
        "model_reload_check_passed": reload_check_passed,
        "stage_10_teacher_model_freeze_sha256": stage10_summary["model_freeze_sha256"],
        "optimizer_freeze_sha256": freeze_sha256,
        "stage_12_recommended_experiments": config.stage12_recommended_experiments,
        "elapsed_seconds": float(elapsed_seconds),
        "model_files": sorted(path.name for path in model_directory.iterdir() if path.is_file()),
        "recommendation_files": sorted(
            path.name for path in recommendation_directory.iterdir() if path.is_file()
        ),
        "validation_files": sorted(
            path.name for path in validation_directory.iterdir() if path.is_file()
        ),
        "generated_plots": {
            "validation": validation_plots,
            "recommendations": recommendation_plots,
        },
    }
    write_json(run_directory / "stage_11_summary.json", summary)
    write_json(
        run_directory / "run_manifest.json",
        {
            "stage": 11,
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": serialize_config(config),
            "input_sha256": {
                "stage_10_summary.json": sha256_file(stage10_verification["summary_path"]),
                "stage_10_model_freeze_manifest.json": sha256_file(
                    stage10_verification["manifest_path"]
                ),
                "stage_09_summary.json": sha256_file(stage9_verification["summary_path"]),
                "stage_09_public_ml_freeze_manifest.json": sha256_file(
                    stage9_verification["manifest_path"]
                ),
            },
        },
    )

    print("\n" + "=" * 84)
    print("Stage 11 complete: leakage-safe pre-injection glitch optimizer")
    print("=" * 84)
    print("Run directory                    :", summary["run_directory"])
    print("All checks passed                :", summary["all_checks_passed"])
    print("Public-only training             :", summary["public_only_training"])
    print("Selected surrogate family        :", summary["selected_family"])
    print(
        "Validation branch Spearman      :",
        f"{summary['validation_metrics']['mean_branch_spearman']:.6f}",
    )
    print(
        "Validation top-decile uplift    :",
        f"{summary['validation_metrics']['mean_branch_top_decile_uplift']:.6f}",
    )
    print("Exploit recommendations          :", summary["exploit_recommendation_count"])
    print("Explore recommendations          :", summary["explore_recommendation_count"])
    print("Attack labels accessed           :", summary["attack_labels_accessed"])
    print("Optimizer freeze SHA-256         :", summary["optimizer_freeze_sha256"])
    print("Elapsed seconds                  :", f"{summary['elapsed_seconds']:.3f}")
    print("=" * 84)

    if not all_checks_passed:
        raise AssertionError(
            "Stage 11 failed validation. Inspect validation_only/"
            "stage_11_validation_checks.json"
        )
    return summary


def load_stage_11_config(path: str | Path) -> Stage11Config:
    raw = read_json(Path(path))
    if "candidate_families" in raw:
        raw["candidate_families"] = tuple(raw["candidate_families"])
    return Stage11Config(**raw)


if __name__ == "__main__":
    default_config = Stage11Config(
        input_stage10_run_directory=(
            r"C:\Users\SADRA\Desktop\LBlock\runs\stage_10"
            r"\stage10_20260718_185539_664419_seed20260718"
        ),
        output_root=r"C:\Users\SADRA\Desktop\LBlock\runs\stage_11",
    )
    run_stage_11(default_config)

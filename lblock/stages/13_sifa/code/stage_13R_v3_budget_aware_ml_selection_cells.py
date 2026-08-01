# %%
# ============================================================
# Stage 13R-v3 / Cell 1
# Budget-aware Stage-10 selection for paper-faithful SIFA
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display


@dataclass(frozen=True)
class Stage13RV3Config:
    """Configuration for public-only budget calibration and post-freeze validation."""

    input_run_directory: str = (
        './runs/stage_13R_paper_sifa_v2/stage13R_v2_20260718_210424_794132_seed20260722'
    )
    input_root_fallback: str = (
        './runs/stage_13R_paper_sifa_v2'
    )
    output_root: str = (
        './runs/stage_13R_paper_sifa_v3_budget_aware'
    )

    # These fractions are fixed before any truth/private file is opened.
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
    minimum_selected_per_task: int = 128
    required_loso_consensus: float = 0.75
    bootstrap_repetitions: int = 300
    matched_random_repetitions: int = 300
    random_seed: int = 20260723

    # Primary cryptanalytic score remains the exact, unweighted paper SEI.
    primary_statistic: str = "SEI"
    primary_selection_method: str = "session_balanced_probability_top_fraction"


stage13r_v3_config = Stage13RV3Config()


def v3_write_json(path: Path, payload: Any) -> None:
    """Write deterministic, readable JSON and handle NumPy scalar types."""

    def _default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_default,
        ),
        encoding="utf-8",
    )


def v3_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v3_resolve_input_run(config: Stage13RV3Config) -> Path:
    explicit = Path(config.input_run_directory).expanduser()
    if explicit.exists():
        return explicit.resolve()

    root = Path(config.input_root_fallback).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            "Stage 13R-v2 input directory was not found. Update "
            "stage13r_v3_config.input_run_directory."
        )

    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path / "public_campaign" / "paired_fault_campaign_public.csv").exists()
            and (path / "paper_sifa_public_attack_freeze_manifest.json").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No complete Stage 13R-v2 run was found")
    return candidates[0].resolve()


def v3_verify_stage13r_v2_freeze(run_directory: Path) -> Dict[str, Any]:
    """Verify every public file hash and the freeze-manifest self-hash."""

    manifest_path = run_directory / "paper_sifa_public_attack_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_manifest_hash = str(manifest.get("freeze_sha256", ""))
    unsigned_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "freeze_sha256"
    }
    encoded = json.dumps(
        unsigned_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_manifest_hash = hashlib.sha256(encoded).hexdigest()
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError("Stage 13R-v2 freeze-manifest SHA-256 mismatch")

    failures: List[str] = []
    checked = 0
    for relative_name, expected_hash in manifest.get("files", {}).items():
        path = run_directory / str(relative_name)
        if not path.exists():
            failures.append(f"missing:{relative_name}")
            continue
        actual_hash = v3_sha256_file(path)
        checked += 1
        if actual_hash != str(expected_hash):
            failures.append(f"hash:{relative_name}")

    if failures:
        raise RuntimeError(
            "Stage 13R-v2 public freeze verification failed: " + ", ".join(failures[:8])
        )

    return {
        "verified": True,
        "checked_file_count": int(checked),
        "freeze_sha256": expected_manifest_hash,
        "manifest_path": str(manifest_path),
    }


v3_input_run_directory = v3_resolve_input_run(stage13r_v3_config)
v3_input_freeze_check = v3_verify_stage13r_v2_freeze(v3_input_run_directory)

v3_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
v3_run_id = f"stage13R_v3_{v3_timestamp}_seed{stage13r_v3_config.random_seed}"
v3_run_directory = (
    Path(stage13r_v3_config.output_root).expanduser().resolve() / v3_run_id
)
v3_public_selection_directory = v3_run_directory / "public_selection"
v3_public_attack_directory = v3_run_directory / "public_attack"
v3_validation_directory = v3_run_directory / "validation_only"
for _directory in (
    v3_public_selection_directory,
    v3_public_attack_directory,
    v3_validation_directory,
):
    _directory.mkdir(parents=True, exist_ok=True)

v3_campaign_path = (
    v3_input_run_directory / "public_campaign" / "paired_fault_campaign_public.csv"
)
v3_campaign = pd.read_csv(v3_campaign_path)

v3_required_columns = {
    "experiment_id",
    "pair_id",
    "campaign_arm",
    "key_id",
    "session_id",
    "target_sbox",
    "target_sbox_index",
    "healthy_ciphertext_hex",
    "paper_ineffective",
    "ml_selected_ineffective",
    "p_clean_target_ineffective",
}
v3_missing_columns = sorted(v3_required_columns - set(v3_campaign.columns))
if v3_missing_columns:
    raise RuntimeError(f"Missing required public columns: {v3_missing_columns}")

# Normalize booleans robustly after CSV reload.
def v3_as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


v3_campaign["paper_ineffective"] = v3_as_bool(v3_campaign["paper_ineffective"])
v3_campaign["ml_selected_ineffective"] = v3_as_bool(
    v3_campaign["ml_selected_ineffective"]
)
v3_campaign["p_clean_target_ineffective"] = pd.to_numeric(
    v3_campaign["p_clean_target_ineffective"], errors="raise"
).clip(0.0, 1.0)

if len(v3_campaign) != 24000:
    raise RuntimeError(f"Expected 24000 public attempts, got {len(v3_campaign)}")
if set(v3_campaign["campaign_arm"].astype(str)) != {
    "guided_model",
    "random_uniform",
}:
    raise RuntimeError("Unexpected campaign arms")
if set(v3_campaign["target_sbox"].astype(str)) != {"S0", "S5"}:
    raise RuntimeError("Unexpected target set")
if sorted(v3_campaign["key_id"].astype(int).unique().tolist()) != [0, 1]:
    raise RuntimeError("Expected exactly two public key identifiers")

v3_write_json(
    v3_public_selection_directory / "stage_13R_v3_config.json",
    {
        **asdict(stage13r_v3_config),
        "resolved_input_run_directory": str(v3_input_run_directory),
        "resolved_output_run_directory": str(v3_run_directory),
        "input_freeze_verification": v3_input_freeze_check,
        "truth_or_private_files_opened": False,
    },
)

v3_write_json(
    v3_public_selection_directory / "budget_aware_selection_contract.json",
    {
        "reference_attack": "Dobraunig et al. SIFA, practical SEI ranking",
        "paper_ineffective_rule": "response returned and C_prime == C",
        "key_statistic": "SEI(k)=sum_x (p_hat_k(x)-1/16)^2",
        "probability_weighting_in_key_statistic": False,
        "stage10_role": (
            "Rank paper-ineffective ciphertexts before the unweighted paper SEI; "
            "the model probability never enters the SEI equation."
        ),
        "selection_method": stage13r_v3_config.primary_selection_method,
        "candidate_top_fractions": list(stage13r_v3_config.candidate_top_fractions),
        "minimum_selected_per_task": stage13r_v3_config.minimum_selected_per_task,
        "public_policy_rule": (
            "Choose one global fraction using only the four random-arm public tasks. "
            "Lexicographically maximize: robust task count, mean leave-one-session-out "
            "winner consensus, median relative SEI margin; break remaining ties by the "
            "smaller fraction. A robust task has a unique winner, full four-session "
            "coverage, and LOSO consensus >= required_loso_consensus."
        ),
        "matched_budget_control": (
            "For every random-arm task, compare the Stage-10 top-ranked subset with "
            "stratified random subsets of exactly the same size."
        ),
        "truth_or_private_files_available_to_selection": False,
    },
)

v3_started = time.perf_counter()
print("=" * 92)
print("Stage 13R-v3 configuration ready")
print("Verified Stage 13R-v2 input :", v3_input_run_directory)
print("Input public freeze SHA-256 :", v3_input_freeze_check["freeze_sha256"])
print("Output directory            :", v3_run_directory)
print("Public campaign rows         :", len(v3_campaign))
print("Candidate top fractions      :", stage13r_v3_config.candidate_top_fractions)
print("Minimum selected per task    :", stage13r_v3_config.minimum_selected_per_task)
print("Truth/private opened         : False")
print("=" * 92)

# %%
# ============================================================
# Stage 13R-v3 / Cell 2
# Exact paper SEI, session-balanced Top-k grid, public policy choice
# ============================================================

# Minimal validated LBlock constants needed by the final-round partial decryption.
V3_SBOX: Tuple[Tuple[int, ...], ...] = (
    (0xE, 0x9, 0xF, 0x0, 0xD, 0x4, 0xA, 0xB, 0x1, 0x2, 0x8, 0x3, 0x7, 0x6, 0xC, 0x5),
    (0x4, 0xB, 0xE, 0x9, 0xF, 0xD, 0x0, 0xA, 0x7, 0xC, 0x5, 0x6, 0x2, 0x8, 0x1, 0x3),
    (0x1, 0xE, 0x7, 0xC, 0xF, 0xD, 0x0, 0x6, 0xB, 0x5, 0x9, 0x3, 0x2, 0x4, 0x8, 0xA),
    (0x7, 0x6, 0x8, 0xB, 0x0, 0xF, 0x3, 0xE, 0x9, 0xA, 0xC, 0xD, 0x5, 0x2, 0x4, 0x1),
    (0xE, 0x5, 0xF, 0x0, 0x7, 0x2, 0xC, 0xD, 0x1, 0x8, 0x4, 0x9, 0xB, 0xA, 0x6, 0x3),
    (0x2, 0xD, 0xB, 0xC, 0xF, 0xE, 0x0, 0x9, 0x7, 0xA, 0x6, 0x3, 0x1, 0x8, 0x4, 0x5),
    (0xB, 0x9, 0x4, 0xE, 0x0, 0xF, 0xA, 0xD, 0x6, 0xC, 0x5, 0x7, 0x3, 0x8, 0x1, 0x2),
    (0xD, 0xA, 0xF, 0x0, 0xE, 0x4, 0x9, 0xB, 0x2, 0x1, 0x8, 0x3, 0x7, 0x5, 0xC, 0x6),
)
V3_P_SOURCE_FOR_OUTPUT: Tuple[int, ...] = (1, 3, 0, 2, 5, 7, 4, 6)
V3_SOURCE_TO_OUTPUT: Dict[int, int] = {
    int(source): int(output_index)
    for output_index, source in enumerate(V3_P_SOURCE_FOR_OUTPUT)
}


def v3_parse_ciphertext_words(ciphertext_hex: Any) -> Tuple[int, int]:
    text = str(ciphertext_hex).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 16:
        raise ValueError(f"Expected 64-bit ciphertext, got {text!r}")
    return int(text[:8], 16), int(text[8:], 16)


def v3_prepare_task_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    parsed = [v3_parse_ciphertext_words(value) for value in result["healthy_ciphertext_hex"]]
    result["x32_word"] = np.asarray([value[0] for value in parsed], dtype=np.uint64)
    result["x33_word"] = np.asarray([value[1] for value in parsed], dtype=np.uint64)
    return result


def v3_intermediate_matrix(task_frame: pd.DataFrame) -> np.ndarray:
    """Return X31_hat for every public sample and all 16 key hypotheses."""

    targets = task_frame["target_sbox_index"].astype(int).unique()
    if len(targets) != 1:
        raise ValueError("A task must contain one target S-box")
    target = int(targets[0])
    output_index = int(V3_SOURCE_TO_OUTPUT[target])

    x32 = task_frame["x32_word"].to_numpy(np.uint64)
    x33 = task_frame["x33_word"].to_numpy(np.uint64)
    x32_nibble = ((x32 >> np.uint64(4 * target)) & np.uint64(0xF)).astype(np.uint8)
    x33_nibble = ((x33 >> np.uint64(4 * output_index)) & np.uint64(0xF)).astype(np.uint8)
    sbox = np.asarray(V3_SBOX[target], dtype=np.uint8)

    matrix = np.empty((len(task_frame), 16), dtype=np.uint8)
    for key_guess in range(16):
        matrix[:, key_guess] = x33_nibble ^ sbox[x32_nibble ^ np.uint8(key_guess)]
    return matrix


def v3_scores_from_matrix(
    matrix: np.ndarray,
    row_indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Exact unweighted paper SEI for all 16 key hypotheses."""

    selected = matrix if row_indices is None else matrix[np.asarray(row_indices, dtype=int)]
    sample_count = int(selected.shape[0])
    rows: List[Dict[str, Any]] = []
    if sample_count == 0:
        return pd.DataFrame(
            {
                "key_guess": np.arange(16, dtype=int),
                "key_guess_hex": [f"{value:x}" for value in range(16)],
                "sample_count": 0,
                "sei": np.nan,
                "chi": np.nan,
            }
        )

    for key_guess in range(16):
        counts = np.bincount(selected[:, key_guess], minlength=16).astype(np.float64)
        empirical = counts / float(sample_count)
        sei = float(np.sum(np.square(empirical - (1.0 / 16.0))))
        rows.append(
            {
                "key_guess": int(key_guess),
                "key_guess_hex": f"{key_guess:x}",
                "sample_count": sample_count,
                "sei": sei,
                "chi": float(sample_count * 16.0 * sei),
            }
        )
    return pd.DataFrame(rows)


def v3_prediction_from_scores(scores: pd.DataFrame) -> Dict[str, Any]:
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


def v3_session_balanced_top_indices(
    task_frame: pd.DataFrame,
    fraction: float,
) -> np.ndarray:
    """Select highest Stage-10 probabilities while preserving session proportions."""

    if task_frame.empty:
        return np.asarray([], dtype=int)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    total = int(len(task_frame))
    requested = int(round(fraction * total))
    requested = min(total, max(1, requested))

    session_counts = task_frame.groupby("session_id").size().sort_index()
    exact_quotas = requested * session_counts.astype(float) / float(total)
    quotas = np.floor(exact_quotas).astype(int)
    remainder = requested - int(quotas.sum())
    remainder_order = sorted(
        session_counts.index.tolist(),
        key=lambda session: (
            -(float(exact_quotas.loc[session]) - float(quotas.loc[session])),
            int(session),
        ),
    )
    for session in remainder_order[:remainder]:
        quotas.loc[session] += 1

    selected: List[int] = []
    for session, quota in quotas.items():
        session_rows = task_frame[task_frame["session_id"].astype(int) == int(session)]
        session_rows = session_rows.sort_values(
            ["p_clean_target_ineffective", "experiment_id"],
            ascending=[False, True],
        )
        selected.extend(session_rows.index[: int(quota)].astype(int).tolist())

    selected_array = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected_array) != requested:
        # Deterministic fill in the unlikely event of a quota edge case.
        remaining = task_frame.loc[~task_frame.index.isin(selected_array)].sort_values(
            ["p_clean_target_ineffective", "experiment_id"],
            ascending=[False, True],
        )
        needed = requested - len(selected_array)
        if needed > 0:
            selected_array = np.concatenate(
                [selected_array, remaining.index[:needed].to_numpy(int)]
            )
    return np.asarray(sorted(selected_array.tolist()), dtype=int)


def v3_loso_consensus(
    task_frame: pd.DataFrame,
    matrix: np.ndarray,
    selected_indices: np.ndarray,
    full_winner: int,
) -> Tuple[float, int, str]:
    selected_sessions = task_frame.loc[selected_indices, "session_id"].astype(int)
    sessions = sorted(selected_sessions.unique().tolist())
    winners: List[int] = []
    for session in sessions:
        keep = selected_indices[
            task_frame.loc[selected_indices, "session_id"].to_numpy(int) != int(session)
        ]
        if len(keep) < 16:
            continue
        prediction = v3_prediction_from_scores(v3_scores_from_matrix(matrix, keep))
        winners.append(int(prediction["best_key_guess"]))
    if not winners:
        return 0.0, int(len(sessions)), ""
    consensus = float(np.mean(np.asarray(winners, dtype=int) == int(full_winner)))
    return consensus, int(len(sessions)), ",".join(f"{winner:x}" for winner in winners)


v3_prepared_campaign = v3_prepare_task_frame(v3_campaign)
v3_public_grid_score_parts: List[pd.DataFrame] = []
v3_public_grid_prediction_rows: List[Dict[str, Any]] = []
v3_task_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

for v3_arm in ("random_uniform", "guided_model"):
    for v3_key_id in sorted(v3_prepared_campaign["key_id"].astype(int).unique()):
        for v3_target_index in (0, 5):
            v3_task_all = (
                v3_prepared_campaign[
                    (v3_prepared_campaign["campaign_arm"].astype(str) == v3_arm)
                    & (v3_prepared_campaign["key_id"].astype(int) == int(v3_key_id))
                    & (
                        v3_prepared_campaign["target_sbox_index"].astype(int)
                        == int(v3_target_index)
                    )
                ]
                .sort_values(["pair_id", "experiment_id"])
                .reset_index(drop=True)
            )
            v3_ineffective = (
                v3_task_all[v3_task_all["paper_ineffective"]]
                .copy()
                .reset_index(drop=True)
            )
            if len(v3_ineffective) < stage13r_v3_config.minimum_selected_per_task:
                raise RuntimeError(
                    f"Too few paper-ineffective samples for {v3_arm}/key{v3_key_id}/S{v3_target_index}"
                )

            v3_matrix = v3_intermediate_matrix(v3_ineffective)
            v3_task_cache[(v3_arm, int(v3_key_id), int(v3_target_index))] = {
                "frame": v3_ineffective,
                "matrix": v3_matrix,
                "total_injections": int(len(v3_task_all)),
            }

            for v3_fraction in stage13r_v3_config.candidate_top_fractions:
                v3_selected_indices = v3_session_balanced_top_indices(
                    v3_ineffective,
                    float(v3_fraction),
                )
                v3_scores = v3_scores_from_matrix(v3_matrix, v3_selected_indices)
                v3_prediction = v3_prediction_from_scores(v3_scores)
                v3_loso, v3_session_coverage, v3_loso_winners = v3_loso_consensus(
                    v3_ineffective,
                    v3_matrix,
                    v3_selected_indices,
                    int(v3_prediction["best_key_guess"]),
                )

                v3_scores.insert(0, "campaign_arm", v3_arm)
                v3_scores.insert(1, "key_id", int(v3_key_id))
                v3_scores.insert(2, "target_sbox", f"S{int(v3_target_index)}")
                v3_scores.insert(3, "target_sbox_index", int(v3_target_index))
                v3_scores.insert(4, "top_fraction", float(v3_fraction))
                v3_public_grid_score_parts.append(v3_scores)

                selected_probabilities = v3_ineffective.loc[
                    v3_selected_indices,
                    "p_clean_target_ineffective",
                ].to_numpy(float)
                v3_public_grid_prediction_rows.append(
                    {
                        "campaign_arm": v3_arm,
                        "key_id": int(v3_key_id),
                        "target_sbox": f"S{int(v3_target_index)}",
                        "target_sbox_index": int(v3_target_index),
                        "top_fraction": float(v3_fraction),
                        "paper_ineffective_count": int(len(v3_ineffective)),
                        "selected_ciphertext_count": int(len(v3_selected_indices)),
                        "mean_selected_probability": float(np.mean(selected_probabilities)),
                        "minimum_selected_probability": float(np.min(selected_probabilities)),
                        "session_coverage": int(v3_session_coverage),
                        "loso_consensus": float(v3_loso),
                        "loso_winners_hex": v3_loso_winners,
                        **v3_prediction,
                    }
                )

v3_public_grid_scores = pd.concat(v3_public_grid_score_parts, ignore_index=True)
v3_public_grid_predictions = pd.DataFrame(v3_public_grid_prediction_rows)

# Public-only global policy selection uses the random arm exclusively.
v3_random_grid = v3_public_grid_predictions[
    v3_public_grid_predictions["campaign_arm"] == "random_uniform"
].copy()
v3_policy_rows: List[Dict[str, Any]] = []
for v3_fraction, v3_group in v3_random_grid.groupby("top_fraction"):
    v3_large_enough = bool(
        np.all(
            v3_group["selected_ciphertext_count"].to_numpy(int)
            >= stage13r_v3_config.minimum_selected_per_task
        )
    )
    v3_robust_mask = (
        v3_group["unique_best"].astype(bool)
        & (v3_group["session_coverage"].astype(int) == 4)
        & (
            v3_group["loso_consensus"].astype(float)
            >= stage13r_v3_config.required_loso_consensus
        )
    )
    v3_policy_rows.append(
        {
            "top_fraction": float(v3_fraction),
            "all_tasks_meet_minimum_count": v3_large_enough,
            "robust_task_count": int(v3_robust_mask.sum()),
            "unique_task_count": int(v3_group["unique_best"].astype(bool).sum()),
            "mean_loso_consensus": float(v3_group["loso_consensus"].mean()),
            "median_relative_margin": float(v3_group["relative_margin"].median()),
            "mean_selected_ciphertext_count": float(
                v3_group["selected_ciphertext_count"].mean()
            ),
        }
    )

v3_public_policy_grid = pd.DataFrame(v3_policy_rows)
v3_eligible_policy_grid = v3_public_policy_grid[
    v3_public_policy_grid["all_tasks_meet_minimum_count"].astype(bool)
    & (v3_public_policy_grid["top_fraction"].astype(float) < 1.0)
].copy()
if v3_eligible_policy_grid.empty:
    raise RuntimeError("No top-fraction candidate meets the public minimum-count rule")

v3_chosen_policy_row = (
    v3_eligible_policy_grid.sort_values(
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
v3_chosen_top_fraction = float(v3_chosen_policy_row["top_fraction"])

v3_public_grid_scores.to_csv(
    v3_public_selection_directory / "top_fraction_candidate_scores_public.csv",
    index=False,
)
v3_public_grid_predictions.to_csv(
    v3_public_selection_directory / "top_fraction_predictions_public.csv",
    index=False,
)
v3_public_policy_grid.to_csv(
    v3_public_selection_directory / "global_policy_selection_grid_public.csv",
    index=False,
)
v3_write_json(
    v3_public_selection_directory / "chosen_budget_aware_policy_public.json",
    {
        "chosen_top_fraction": v3_chosen_top_fraction,
        "selection_basis_arm": "random_uniform",
        "selection_uses_key_truth": False,
        "selection_uses_private_fault_labels": False,
        "selection_method": stage13r_v3_config.primary_selection_method,
        "public_policy_row": v3_chosen_policy_row.to_dict(),
    },
)

print("=" * 92)
print("Public-only budget policy selected")
print("Chosen top fraction          :", f"{v3_chosen_top_fraction:.3f}")
print("Robust random-arm tasks      :", int(v3_chosen_policy_row["robust_task_count"]), "/ 4")
print("Mean LOSO consensus          :", f"{float(v3_chosen_policy_row['mean_loso_consensus']):.3f}")
print("Median relative margin       :", f"{float(v3_chosen_policy_row['median_relative_margin']):.6f}")
print("Truth/private opened         : False")
print("=" * 92)
ipy_display(v3_public_policy_grid.sort_values("top_fraction").reset_index(drop=True))

# %%
# ============================================================
# Stage 13R-v3 / Cell 3
# Freeze primary pipelines, matched-budget controls, bootstrap, joint 8-bit attack
# ============================================================

V3_PIPELINES: Dict[str, Dict[str, Any]] = {
    "random_raw": {
        "campaign_arm": "random_uniform",
        "selection": "all_paper_ineffective",
        "uses_stage10": False,
        "uses_stage11": False,
    },
    "random_threshold": {
        "campaign_arm": "random_uniform",
        "selection": "previous_fixed_threshold",
        "uses_stage10": True,
        "uses_stage11": False,
    },
    "random_budget_aware": {
        "campaign_arm": "random_uniform",
        "selection": "chosen_session_balanced_top_fraction",
        "uses_stage10": True,
        "uses_stage11": False,
    },
    "guided_raw": {
        "campaign_arm": "guided_model",
        "selection": "all_paper_ineffective",
        "uses_stage10": False,
        "uses_stage11": True,
    },
    "guided_threshold": {
        "campaign_arm": "guided_model",
        "selection": "previous_fixed_threshold",
        "uses_stage10": True,
        "uses_stage11": True,
    },
    "guided_budget_aware": {
        "campaign_arm": "guided_model",
        "selection": "chosen_session_balanced_top_fraction",
        "uses_stage10": True,
        "uses_stage11": True,
    },
}


def v3_pipeline_indices(
    task_frame: pd.DataFrame,
    pipeline_name: str,
) -> np.ndarray:
    selection = str(V3_PIPELINES[pipeline_name]["selection"])
    if selection == "all_paper_ineffective":
        return np.arange(len(task_frame), dtype=int)
    if selection == "previous_fixed_threshold":
        return np.flatnonzero(task_frame["ml_selected_ineffective"].to_numpy(bool))
    if selection == "chosen_session_balanced_top_fraction":
        return v3_session_balanced_top_indices(task_frame, v3_chosen_top_fraction)
    raise ValueError(f"Unknown selection strategy: {selection}")


def v3_stratified_bootstrap_indices(
    task_frame: pd.DataFrame,
    selected_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: List[np.ndarray] = []
    selected_sessions = task_frame.loc[selected_indices, "session_id"].to_numpy(int)
    for session in sorted(np.unique(selected_sessions).tolist()):
        pool = selected_indices[selected_sessions == int(session)]
        if len(pool) == 0:
            continue
        sampled.append(rng.choice(pool, size=len(pool), replace=True).astype(int))
    if not sampled:
        return np.asarray([], dtype=int)
    return np.concatenate(sampled).astype(int)


def v3_stratified_matched_random_indices(
    task_frame: pd.DataFrame,
    reference_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: List[np.ndarray] = []
    reference_sessions = task_frame.loc[reference_indices, "session_id"].to_numpy(int)
    all_sessions = task_frame["session_id"].to_numpy(int)
    for session in sorted(np.unique(reference_sessions).tolist()):
        needed = int(np.sum(reference_sessions == int(session)))
        pool = np.flatnonzero(all_sessions == int(session))
        if needed > len(pool):
            raise RuntimeError("Matched-random quota exceeds the public session pool")
        sampled.append(rng.choice(pool, size=needed, replace=False).astype(int))
    if not sampled:
        return np.asarray([], dtype=int)
    return np.concatenate(sampled).astype(int)


v3_final_score_parts: List[pd.DataFrame] = []
v3_final_prediction_rows: List[Dict[str, Any]] = []
v3_selected_event_rows: List[Dict[str, Any]] = []
v3_bootstrap_score_parts: List[pd.DataFrame] = []
v3_bootstrap_summary_rows: List[Dict[str, Any]] = []
v3_matched_random_score_parts: List[pd.DataFrame] = []
v3_matched_random_summary_rows: List[Dict[str, Any]] = []

v3_rng = np.random.default_rng(stage13r_v3_config.random_seed)

for v3_pipeline_name, v3_pipeline_spec in V3_PIPELINES.items():
    v3_arm = str(v3_pipeline_spec["campaign_arm"])
    for v3_key_id in (0, 1):
        for v3_target_index in (0, 5):
            v3_cache = v3_task_cache[(v3_arm, v3_key_id, v3_target_index)]
            v3_task_frame = v3_cache["frame"]
            v3_matrix = v3_cache["matrix"]
            v3_selected_indices = v3_pipeline_indices(v3_task_frame, v3_pipeline_name)
            v3_scores = v3_scores_from_matrix(v3_matrix, v3_selected_indices)
            v3_prediction = v3_prediction_from_scores(v3_scores)
            v3_loso, v3_session_coverage, v3_loso_winners = v3_loso_consensus(
                v3_task_frame,
                v3_matrix,
                v3_selected_indices,
                int(v3_prediction["best_key_guess"]),
            )

            v3_scores.insert(0, "pipeline", v3_pipeline_name)
            v3_scores.insert(1, "key_id", int(v3_key_id))
            v3_scores.insert(2, "target_sbox", f"S{v3_target_index}")
            v3_scores.insert(3, "target_sbox_index", int(v3_target_index))
            v3_final_score_parts.append(v3_scores)

            v3_selected_probabilities = v3_task_frame.loc[
                v3_selected_indices,
                "p_clean_target_ineffective",
            ].to_numpy(float)
            v3_final_prediction_rows.append(
                {
                    "pipeline": v3_pipeline_name,
                    "key_id": int(v3_key_id),
                    "target_sbox": f"S{v3_target_index}",
                    "target_sbox_index": int(v3_target_index),
                    "total_injection_count": int(v3_cache["total_injections"]),
                    "paper_ineffective_count": int(len(v3_task_frame)),
                    "selected_ciphertext_count": int(len(v3_selected_indices)),
                    "selected_fraction_of_ineffective": float(
                        len(v3_selected_indices) / max(1, len(v3_task_frame))
                    ),
                    "mean_selected_probability": float(
                        np.mean(v3_selected_probabilities)
                    ) if len(v3_selected_probabilities) else np.nan,
                    "minimum_selected_probability": float(
                        np.min(v3_selected_probabilities)
                    ) if len(v3_selected_probabilities) else np.nan,
                    "session_coverage": int(v3_session_coverage),
                    "loso_consensus": float(v3_loso),
                    "loso_winners_hex": v3_loso_winners,
                    **v3_prediction,
                }
            )

            for v3_local_index in v3_selected_indices:
                v3_row = v3_task_frame.iloc[int(v3_local_index)]
                v3_selected_event_rows.append(
                    {
                        "pipeline": v3_pipeline_name,
                        "experiment_id": int(v3_row["experiment_id"]),
                        "pair_id": int(v3_row["pair_id"]),
                        "campaign_arm": v3_arm,
                        "key_id": int(v3_key_id),
                        "session_id": int(v3_row["session_id"]),
                        "target_sbox": f"S{v3_target_index}",
                        "target_sbox_index": int(v3_target_index),
                        "p_clean_target_ineffective": float(
                            v3_row["p_clean_target_ineffective"]
                        ),
                        "healthy_ciphertext_hex": str(v3_row["healthy_ciphertext_hex"]),
                    }
                )

            # Public bootstrap measures stability of the full-data winner only.
            v3_boot_winners: List[int] = []
            for v3_rep in range(stage13r_v3_config.bootstrap_repetitions):
                v3_sample_indices = v3_stratified_bootstrap_indices(
                    v3_task_frame,
                    v3_selected_indices,
                    v3_rng,
                )
                v3_rep_scores = v3_scores_from_matrix(v3_matrix, v3_sample_indices)
                v3_rep_prediction = v3_prediction_from_scores(v3_rep_scores)
                v3_boot_winners.append(int(v3_rep_prediction["best_key_guess"]))
                v3_rep_scores.insert(0, "pipeline", v3_pipeline_name)
                v3_rep_scores.insert(1, "key_id", int(v3_key_id))
                v3_rep_scores.insert(2, "target_sbox_index", int(v3_target_index))
                v3_rep_scores.insert(3, "repetition", int(v3_rep))
                v3_bootstrap_score_parts.append(v3_rep_scores)

            v3_bootstrap_summary_rows.append(
                {
                    "pipeline": v3_pipeline_name,
                    "key_id": int(v3_key_id),
                    "target_sbox": f"S{v3_target_index}",
                    "target_sbox_index": int(v3_target_index),
                    "selected_ciphertext_count": int(len(v3_selected_indices)),
                    "full_data_winner": int(v3_prediction["best_key_guess"]),
                    "full_data_winner_hex": str(v3_prediction["best_key_guess_hex"]),
                    "winner_stability_fraction": float(
                        np.mean(
                            np.asarray(v3_boot_winners, dtype=int)
                            == int(v3_prediction["best_key_guess"])
                        )
                    ),
                    "distinct_bootstrap_winner_count": int(
                        len(set(v3_boot_winners))
                    ),
                }
            )

            # Same-budget random controls are defined only for the random budget-aware arm.
            if v3_pipeline_name == "random_budget_aware":
                v3_control_winners: List[int] = []
                for v3_rep in range(stage13r_v3_config.matched_random_repetitions):
                    v3_control_indices = v3_stratified_matched_random_indices(
                        v3_task_frame,
                        v3_selected_indices,
                        v3_rng,
                    )
                    v3_control_scores = v3_scores_from_matrix(v3_matrix, v3_control_indices)
                    v3_control_prediction = v3_prediction_from_scores(v3_control_scores)
                    v3_control_winners.append(int(v3_control_prediction["best_key_guess"]))
                    v3_control_scores.insert(0, "key_id", int(v3_key_id))
                    v3_control_scores.insert(1, "target_sbox_index", int(v3_target_index))
                    v3_control_scores.insert(2, "repetition", int(v3_rep))
                    v3_matched_random_score_parts.append(v3_control_scores)

                v3_matched_random_summary_rows.append(
                    {
                        "key_id": int(v3_key_id),
                        "target_sbox": f"S{v3_target_index}",
                        "target_sbox_index": int(v3_target_index),
                        "matched_subset_size": int(len(v3_selected_indices)),
                        "model_topk_winner": int(v3_prediction["best_key_guess"]),
                        "model_topk_winner_hex": str(v3_prediction["best_key_guess_hex"]),
                        "matched_random_modal_winner": int(
                            pd.Series(v3_control_winners).value_counts().sort_values(
                                ascending=False
                            ).index[0]
                        ),
                        "matched_random_distinct_winner_count": int(
                            len(set(v3_control_winners))
                        ),
                    }
                )

v3_final_scores_public = pd.concat(v3_final_score_parts, ignore_index=True)
v3_final_predictions_public = pd.DataFrame(v3_final_prediction_rows)
v3_selected_events_public = pd.DataFrame(v3_selected_event_rows)
v3_bootstrap_scores_public = pd.concat(v3_bootstrap_score_parts, ignore_index=True)
v3_bootstrap_summary_public = pd.DataFrame(v3_bootstrap_summary_rows)
v3_matched_random_scores_public = pd.concat(
    v3_matched_random_score_parts,
    ignore_index=True,
)
v3_matched_random_summary_public = pd.DataFrame(v3_matched_random_summary_rows)

# Joint 8-bit ranking from the two independent four-bit SEI scores.
v3_joint_score_rows: List[Dict[str, Any]] = []
v3_joint_prediction_rows: List[Dict[str, Any]] = []
for v3_pipeline_name in V3_PIPELINES:
    for v3_key_id in (0, 1):
        v3_s0 = v3_final_scores_public[
            (v3_final_scores_public["pipeline"] == v3_pipeline_name)
            & (v3_final_scores_public["key_id"] == v3_key_id)
            & (v3_final_scores_public["target_sbox_index"] == 0)
        ].set_index("key_guess")
        v3_s5 = v3_final_scores_public[
            (v3_final_scores_public["pipeline"] == v3_pipeline_name)
            & (v3_final_scores_public["key_id"] == v3_key_id)
            & (v3_final_scores_public["target_sbox_index"] == 5)
        ].set_index("key_guess")

        for v3_k0 in range(16):
            for v3_k5 in range(16):
                v3_joint_score_rows.append(
                    {
                        "pipeline": v3_pipeline_name,
                        "key_id": int(v3_key_id),
                        "K32_0_guess": int(v3_k0),
                        "K32_5_guess": int(v3_k5),
                        "joint_guess_hex": f"{v3_k0:x}{v3_k5:x}",
                        "joint_sei": float(v3_s0.loc[v3_k0, "sei"] + v3_s5.loc[v3_k5, "sei"]),
                    }
                )

        v3_joint_for_key = pd.DataFrame(
            [
                row
                for row in v3_joint_score_rows
                if row["pipeline"] == v3_pipeline_name and row["key_id"] == v3_key_id
            ]
        ).sort_values(
            ["joint_sei", "K32_0_guess", "K32_5_guess"],
            ascending=[False, True, True],
        )
        v3_best_joint = v3_joint_for_key.iloc[0]
        v3_second_joint = v3_joint_for_key.iloc[1]
        v3_best_score = float(v3_best_joint["joint_sei"])
        v3_unique_joint = bool(
            np.sum(
                np.isclose(
                    v3_joint_for_key["joint_sei"].to_numpy(float),
                    v3_best_score,
                    atol=1.0e-15,
                    rtol=1.0e-12,
                )
            )
            == 1
        )
        v3_joint_prediction_rows.append(
            {
                "pipeline": v3_pipeline_name,
                "key_id": int(v3_key_id),
                "best_joint_guess_hex": str(v3_best_joint["joint_guess_hex"]),
                "best_K32_0_guess": int(v3_best_joint["K32_0_guess"]),
                "best_K32_5_guess": int(v3_best_joint["K32_5_guess"]),
                "best_joint_score": v3_best_score,
                "joint_score_margin": float(
                    v3_best_score - float(v3_second_joint["joint_sei"])
                ),
                "unique_best": v3_unique_joint,
            }
        )

v3_joint_scores_public = pd.DataFrame(v3_joint_score_rows)
v3_joint_predictions_public = pd.DataFrame(v3_joint_prediction_rows)

# Save all public artifacts before opening any truth or private category file.
v3_final_scores_public.to_csv(
    v3_public_attack_directory / "budget_aware_sifa_candidate_scores_public.csv",
    index=False,
)
v3_final_predictions_public.to_csv(
    v3_public_attack_directory / "budget_aware_sifa_predictions_public.csv",
    index=False,
)
v3_selected_events_public.to_csv(
    v3_public_selection_directory / "selected_events_public.csv",
    index=False,
)
v3_bootstrap_scores_public.to_csv(
    v3_public_attack_directory / "bootstrap_candidate_scores_public.csv",
    index=False,
)
v3_bootstrap_summary_public.to_csv(
    v3_public_attack_directory / "bootstrap_winner_stability_public.csv",
    index=False,
)
v3_matched_random_scores_public.to_csv(
    v3_public_attack_directory / "matched_random_candidate_scores_public.csv",
    index=False,
)
v3_matched_random_summary_public.to_csv(
    v3_public_attack_directory / "matched_random_summary_public.csv",
    index=False,
)
v3_joint_scores_public.to_csv(
    v3_public_attack_directory / "joint_8bit_candidate_scores_public.csv",
    index=False,
)
v3_joint_predictions_public.to_csv(
    v3_public_attack_directory / "joint_8bit_predictions_public.csv",
    index=False,
)

v3_write_json(
    v3_public_attack_directory / "paper_faithful_attack_contract.json",
    {
        "primary_statistic": "SEI",
        "formula": "sum_x (p_hat_k(x)-1/16)^2",
        "ranking_direction": "descending",
        "sample_weights_inside_SEI": False,
        "chosen_top_fraction": v3_chosen_top_fraction,
        "pipelines": V3_PIPELINES,
        "truth_or_private_access_before_freeze": False,
    },
)

# Freeze every public artifact generated by this stage.
v3_public_hashes: Dict[str, str] = {}
for v3_root in (v3_public_selection_directory, v3_public_attack_directory):
    for v3_path in sorted(v3_root.rglob("*")):
        if v3_path.is_file():
            v3_relative = (
                f"{v3_root.name}/"
                + str(v3_path.relative_to(v3_root)).replace("\\", "/")
            )
            v3_public_hashes[v3_relative] = v3_sha256_file(v3_path)

v3_freeze_unsigned = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "input_stage13r_v2_freeze_sha256": v3_input_freeze_check["freeze_sha256"],
    "chosen_top_fraction": v3_chosen_top_fraction,
    "statement": (
        "All budget candidates, the public-only selected fraction, all paper-SEI "
        "scores, public predictions, bootstrap scores, matched-budget controls, and "
        "joint 8-bit predictions were frozen before opening key truth or simulator labels."
    ),
    "files": v3_public_hashes,
}
v3_freeze_bytes = json.dumps(
    v3_freeze_unsigned,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
v3_public_freeze = {
    **v3_freeze_unsigned,
    "freeze_sha256": hashlib.sha256(v3_freeze_bytes).hexdigest(),
}
v3_write_json(
    v3_run_directory / "stage_13R_v3_public_freeze_manifest.json",
    v3_public_freeze,
)
v3_public_attack_frozen = True

print("=" * 92)
print("Stage 13R-v3 public attack frozen")
print("Chosen top fraction          :", f"{v3_chosen_top_fraction:.3f}")
print("Public freeze SHA-256        :", v3_public_freeze["freeze_sha256"])
print("Truth/private opened         : False")
print("Public predictions:")
print(
    v3_final_predictions_public[
        [
            "pipeline",
            "key_id",
            "target_sbox",
            "selected_ciphertext_count",
            "best_key_guess_hex",
            "score_margin",
            "loso_consensus",
            "unique_best",
        ]
    ].to_string(index=False)
)
print("=" * 92)

# %%
# ============================================================
# Stage 13R-v3 / Cell 4
# Post-freeze truth/private evaluation and final comparison
# ============================================================

if not globals().get("v3_public_attack_frozen", False):
    raise RuntimeError("The public Stage 13R-v3 attack must be frozen first")


def v3_rank_from_scores(
    scores: pd.DataFrame,
    true_guess: int,
    score_column: str = "sei",
) -> Dict[str, Any]:
    values = scores[score_column].to_numpy(float)
    guesses = scores["key_guess"].to_numpy(int)
    true_mask = guesses == int(true_guess)
    if np.sum(true_mask) != 1:
        raise RuntimeError("True candidate missing or duplicated")
    true_score = float(values[true_mask][0])
    tolerance = 1.0e-12
    true_rank = int(1 + np.sum(values > true_score + tolerance))
    best_score = float(np.nanmax(values))
    best_mask = np.isclose(values, best_score, atol=1.0e-15, rtol=1.0e-12)
    best_guesses = guesses[best_mask]
    best_guess = int(np.min(best_guesses))
    return {
        "true_rank": true_rank,
        "true_score": true_score,
        "best_guess": best_guess,
        "true_is_unique_rank1": bool(
            true_rank == 1 and len(best_guesses) == 1 and best_guess == int(true_guess)
        ),
    }


# Truth is opened only now, after the new public freeze.
v3_truth_path = (
    v3_input_run_directory / "locked_truth" / "paper_sifa_key_truth_LOCKED.json"
)
v3_private_path = (
    v3_input_run_directory
    / "private_after_freeze"
    / "paired_fault_ground_truth_after_freeze.csv"
)
if not v3_truth_path.exists():
    raise FileNotFoundError(v3_truth_path)
if not v3_private_path.exists():
    raise FileNotFoundError(v3_private_path)

v3_truth_payload = json.loads(v3_truth_path.read_text(encoding="utf-8"))
v3_truth = pd.DataFrame(v3_truth_payload["keys"])
v3_private = pd.read_csv(v3_private_path)

v3_truth_lookup: Dict[Tuple[int, int], int] = {}
for v3_row in v3_truth.itertuples(index=False):
    v3_truth_lookup[(int(v3_row.key_id), 0)] = int(v3_row.K32_0)
    v3_truth_lookup[(int(v3_row.key_id), 5)] = int(v3_row.K32_5)

# Evaluate the complete top-fraction grid after freeze.
v3_grid_truth_rows: List[Dict[str, Any]] = []
for (
    v3_arm,
    v3_key_id,
    v3_target_index,
    v3_fraction,
), v3_group in v3_public_grid_scores.groupby(
    ["campaign_arm", "key_id", "target_sbox_index", "top_fraction"]
):
    v3_true_guess = v3_truth_lookup[(int(v3_key_id), int(v3_target_index))]
    v3_rank = v3_rank_from_scores(v3_group, v3_true_guess)
    v3_prediction = v3_public_grid_predictions[
        (v3_public_grid_predictions["campaign_arm"] == v3_arm)
        & (v3_public_grid_predictions["key_id"] == int(v3_key_id))
        & (
            v3_public_grid_predictions["target_sbox_index"]
            == int(v3_target_index)
        )
        & np.isclose(
            v3_public_grid_predictions["top_fraction"].astype(float),
            float(v3_fraction),
        )
    ].iloc[0]
    v3_grid_truth_rows.append(
        {
            "campaign_arm": str(v3_arm),
            "key_id": int(v3_key_id),
            "target_sbox": f"S{int(v3_target_index)}",
            "target_sbox_index": int(v3_target_index),
            "top_fraction": float(v3_fraction),
            "selected_ciphertext_count": int(
                v3_prediction["selected_ciphertext_count"]
            ),
            "predicted_key_guess_hex": str(v3_prediction["best_key_guess_hex"]),
            "true_key_guess_hex": f"{v3_true_guess:x}",
            **v3_rank,
        }
    )

v3_grid_truth_evaluation = pd.DataFrame(v3_grid_truth_rows)

# Final pipeline ranks.
v3_final_rank_rows: List[Dict[str, Any]] = []
for (
    v3_pipeline_name,
    v3_key_id,
    v3_target_index,
), v3_group in v3_final_scores_public.groupby(
    ["pipeline", "key_id", "target_sbox_index"]
):
    v3_true_guess = v3_truth_lookup[(int(v3_key_id), int(v3_target_index))]
    v3_rank = v3_rank_from_scores(v3_group, v3_true_guess)
    v3_prediction = v3_final_predictions_public[
        (v3_final_predictions_public["pipeline"] == v3_pipeline_name)
        & (v3_final_predictions_public["key_id"] == int(v3_key_id))
        & (
            v3_final_predictions_public["target_sbox_index"]
            == int(v3_target_index)
        )
    ].iloc[0]
    v3_bootstrap = v3_bootstrap_summary_public[
        (v3_bootstrap_summary_public["pipeline"] == v3_pipeline_name)
        & (v3_bootstrap_summary_public["key_id"] == int(v3_key_id))
        & (
            v3_bootstrap_summary_public["target_sbox_index"]
            == int(v3_target_index)
        )
    ].iloc[0]
    v3_final_rank_rows.append(
        {
            "pipeline": str(v3_pipeline_name),
            "key_id": int(v3_key_id),
            "target_sbox": f"S{int(v3_target_index)}",
            "target_sbox_index": int(v3_target_index),
            "selected_ciphertext_count": int(
                v3_prediction["selected_ciphertext_count"]
            ),
            "predicted_key_guess_hex": str(v3_prediction["best_key_guess_hex"]),
            "true_key_guess_hex": f"{v3_true_guess:x}",
            "score_margin": float(v3_prediction["score_margin"]),
            "loso_consensus": float(v3_prediction["loso_consensus"]),
            "bootstrap_winner_stability": float(
                v3_bootstrap["winner_stability_fraction"]
            ),
            **v3_rank,
        }
    )

v3_final_rank_evaluation = pd.DataFrame(v3_final_rank_rows)

# Evaluate bootstrap true-winner fractions from the already frozen candidate scores.
v3_bootstrap_truth_rows: List[Dict[str, Any]] = []
for (
    v3_pipeline_name,
    v3_key_id,
    v3_target_index,
    v3_repetition,
), v3_group in v3_bootstrap_scores_public.groupby(
    ["pipeline", "key_id", "target_sbox_index", "repetition"]
):
    v3_true_guess = v3_truth_lookup[(int(v3_key_id), int(v3_target_index))]
    v3_rank = v3_rank_from_scores(v3_group, v3_true_guess)
    v3_bootstrap_truth_rows.append(
        {
            "pipeline": str(v3_pipeline_name),
            "key_id": int(v3_key_id),
            "target_sbox_index": int(v3_target_index),
            "repetition": int(v3_repetition),
            "true_rank": int(v3_rank["true_rank"]),
            "true_is_unique_rank1": bool(v3_rank["true_is_unique_rank1"]),
        }
    )
v3_bootstrap_truth = pd.DataFrame(v3_bootstrap_truth_rows)

# Same-budget random-control evaluation.
v3_matched_truth_rows: List[Dict[str, Any]] = []
for (
    v3_key_id,
    v3_target_index,
    v3_repetition,
), v3_group in v3_matched_random_scores_public.groupby(
    ["key_id", "target_sbox_index", "repetition"]
):
    v3_true_guess = v3_truth_lookup[(int(v3_key_id), int(v3_target_index))]
    v3_rank = v3_rank_from_scores(v3_group, v3_true_guess)
    v3_matched_truth_rows.append(
        {
            "key_id": int(v3_key_id),
            "target_sbox": f"S{int(v3_target_index)}",
            "target_sbox_index": int(v3_target_index),
            "repetition": int(v3_repetition),
            "true_rank": int(v3_rank["true_rank"]),
            "true_is_unique_rank1": bool(v3_rank["true_is_unique_rank1"]),
        }
    )
v3_matched_random_truth = pd.DataFrame(v3_matched_truth_rows)

# Selection purity and recall are evaluated only after the public attack freeze.
v3_public_private = v3_campaign.merge(
    v3_private,
    on=[
        "experiment_id",
        "pair_id",
        "campaign_arm",
        "key_id",
        "session_id",
        "target_sbox",
        "target_sbox_index",
    ],
    how="left",
    validate="one_to_one",
    suffixes=("", "_private"),
)
if v3_public_private["category"].isna().any():
    raise RuntimeError("Private category merge is incomplete")

v3_quality_rows: List[Dict[str, Any]] = []
for v3_pipeline_name, v3_spec in V3_PIPELINES.items():
    v3_arm = str(v3_spec["campaign_arm"])
    v3_selected_ids = set(
        v3_selected_events_public.loc[
            v3_selected_events_public["pipeline"] == v3_pipeline_name,
            "experiment_id",
        ].astype(int)
    )
    v3_arm_frame = v3_public_private[
        v3_public_private["campaign_arm"].astype(str) == v3_arm
    ]
    v3_selected_frame = v3_arm_frame[
        v3_arm_frame["experiment_id"].astype(int).isin(v3_selected_ids)
    ]
    v3_clean_total = int(
        np.sum(v3_arm_frame["category"].astype(str) == "clean_target_ineffective")
    )
    v3_clean_selected = int(
        np.sum(v3_selected_frame["category"].astype(str) == "clean_target_ineffective")
    )
    v3_quality_rows.append(
        {
            "pipeline": v3_pipeline_name,
            "selected_ciphertext_count": int(len(v3_selected_frame)),
            "actual_clean_target_ineffective_count": v3_clean_selected,
            "actual_clean_precision": float(
                v3_clean_selected / max(1, len(v3_selected_frame))
            ),
            "actual_clean_recall": float(v3_clean_selected / max(1, v3_clean_total)),
            "uses_stage10_classifier": bool(v3_spec["uses_stage10"]),
            "uses_stage11_optimizer": bool(v3_spec["uses_stage11"]),
        }
    )
v3_quality_evaluation = pd.DataFrame(v3_quality_rows)

# Joint 8-bit truth ranks.
v3_joint_truth_rows: List[Dict[str, Any]] = []
for (
    v3_pipeline_name,
    v3_key_id,
), v3_group in v3_joint_scores_public.groupby(["pipeline", "key_id"]):
    v3_true_k0 = v3_truth_lookup[(int(v3_key_id), 0)]
    v3_true_k5 = v3_truth_lookup[(int(v3_key_id), 5)]
    v3_true_mask = (
        (v3_group["K32_0_guess"].astype(int) == v3_true_k0)
        & (v3_group["K32_5_guess"].astype(int) == v3_true_k5)
    )
    v3_true_score = float(v3_group.loc[v3_true_mask, "joint_sei"].iloc[0])
    v3_true_rank = int(
        1
        + np.sum(
            v3_group["joint_sei"].to_numpy(float)
            > v3_true_score + 1.0e-12
        )
    )
    v3_prediction = v3_joint_predictions_public[
        (v3_joint_predictions_public["pipeline"] == v3_pipeline_name)
        & (v3_joint_predictions_public["key_id"] == int(v3_key_id))
    ].iloc[0]
    v3_joint_truth_rows.append(
        {
            "pipeline": str(v3_pipeline_name),
            "key_id": int(v3_key_id),
            "true_joint_guess_hex": f"{v3_true_k0:x}{v3_true_k5:x}",
            "predicted_joint_guess_hex": str(v3_prediction["best_joint_guess_hex"]),
            "true_joint_rank": v3_true_rank,
            "true_is_unique_rank1": bool(
                v3_true_rank == 1
                and bool(v3_prediction["unique_best"])
                and str(v3_prediction["best_joint_guess_hex"])
                == f"{v3_true_k0:x}{v3_true_k5:x}"
            ),
        }
    )
v3_joint_truth_evaluation = pd.DataFrame(v3_joint_truth_rows)

# Pipeline summaries.
v3_pipeline_summary_rows: List[Dict[str, Any]] = []
for v3_pipeline_name in V3_PIPELINES:
    v3_rank_subset = v3_final_rank_evaluation[
        v3_final_rank_evaluation["pipeline"] == v3_pipeline_name
    ]
    v3_joint_subset = v3_joint_truth_evaluation[
        v3_joint_truth_evaluation["pipeline"] == v3_pipeline_name
    ]
    v3_quality_row = v3_quality_evaluation[
        v3_quality_evaluation["pipeline"] == v3_pipeline_name
    ].iloc[0]
    v3_boot_subset = v3_bootstrap_truth[
        v3_bootstrap_truth["pipeline"] == v3_pipeline_name
    ]
    v3_pipeline_summary_rows.append(
        {
            "pipeline": v3_pipeline_name,
            "rank1_nibble_count": int(
                v3_rank_subset["true_is_unique_rank1"].astype(bool).sum()
            ),
            "rank1_nibble_success_rate": float(
                v3_rank_subset["true_is_unique_rank1"].astype(bool).mean()
            ),
            "mean_true_rank": float(v3_rank_subset["true_rank"].mean()),
            "median_true_rank": float(v3_rank_subset["true_rank"].median()),
            "rank1_joint_8bit_count": int(
                v3_joint_subset["true_is_unique_rank1"].astype(bool).sum()
            ),
            "mean_selected_ciphertexts_per_task": float(
                v3_rank_subset["selected_ciphertext_count"].mean()
            ),
            "mean_score_margin": float(v3_rank_subset["score_margin"].mean()),
            "mean_loso_consensus": float(v3_rank_subset["loso_consensus"].mean()),
            "mean_bootstrap_true_winner_fraction": float(
                v3_boot_subset["true_is_unique_rank1"].astype(bool).mean()
            ),
            "actual_clean_precision": float(v3_quality_row["actual_clean_precision"]),
            "actual_clean_recall": float(v3_quality_row["actual_clean_recall"]),
            "actual_clean_count": int(
                v3_quality_row["actual_clean_target_ineffective_count"]
            ),
        }
    )
v3_pipeline_summary = pd.DataFrame(v3_pipeline_summary_rows)

# Matched-budget evidence: model top-k versus random subsets of identical size.
v3_matched_comparison_rows: List[Dict[str, Any]] = []
for v3_key_id in (0, 1):
    for v3_target_index in (0, 5):
        v3_model_row = v3_final_rank_evaluation[
            (v3_final_rank_evaluation["pipeline"] == "random_budget_aware")
            & (v3_final_rank_evaluation["key_id"] == v3_key_id)
            & (v3_final_rank_evaluation["target_sbox_index"] == v3_target_index)
        ].iloc[0]
        v3_controls = v3_matched_random_truth[
            (v3_matched_random_truth["key_id"] == v3_key_id)
            & (v3_matched_random_truth["target_sbox_index"] == v3_target_index)
        ]
        v3_matched_comparison_rows.append(
            {
                "key_id": v3_key_id,
                "target_sbox": f"S{v3_target_index}",
                "target_sbox_index": v3_target_index,
                "model_topk_true_rank": int(v3_model_row["true_rank"]),
                "model_topk_rank1": bool(v3_model_row["true_is_unique_rank1"]),
                "matched_random_median_true_rank": float(
                    v3_controls["true_rank"].median()
                ),
                "matched_random_mean_true_rank": float(v3_controls["true_rank"].mean()),
                "matched_random_rank1_fraction": float(
                    v3_controls["true_is_unique_rank1"].astype(bool).mean()
                ),
                "model_better_than_random_median": bool(
                    int(v3_model_row["true_rank"])
                    < float(v3_controls["true_rank"].median())
                ),
            }
        )
v3_matched_comparison = pd.DataFrame(v3_matched_comparison_rows)

# Fixed, pre-declared assessment of whether the second model helped.
def v3_summary_row(pipeline: str) -> pd.Series:
    return v3_pipeline_summary[v3_pipeline_summary["pipeline"] == pipeline].iloc[0]


v3_random_raw_summary = v3_summary_row("random_raw")
v3_random_threshold_summary = v3_summary_row("random_threshold")
v3_random_budget_summary = v3_summary_row("random_budget_aware")

v3_classifier_helped_by_fixed_rule = bool(
    float(v3_random_budget_summary["mean_true_rank"])
    < float(v3_random_raw_summary["mean_true_rank"])
    and int(v3_random_budget_summary["rank1_nibble_count"])
    >= int(v3_random_threshold_summary["rank1_nibble_count"])
    and float(v3_random_budget_summary["actual_clean_precision"])
    > float(v3_random_raw_summary["actual_clean_precision"])
    and int(v3_matched_comparison["model_better_than_random_median"].sum()) >= 3
)

# Save post-freeze validation outputs.
v3_grid_truth_evaluation.to_csv(
    v3_validation_directory / "top_fraction_true_rank_curve.csv",
    index=False,
)
v3_final_rank_evaluation.to_csv(
    v3_validation_directory / "final_nibble_true_rank_evaluation.csv",
    index=False,
)
v3_bootstrap_truth.to_csv(
    v3_validation_directory / "bootstrap_true_rank_evaluation.csv",
    index=False,
)
v3_matched_random_truth.to_csv(
    v3_validation_directory / "matched_random_true_rank_distribution.csv",
    index=False,
)
v3_matched_comparison.to_csv(
    v3_validation_directory / "matched_budget_model_vs_random.csv",
    index=False,
)
v3_quality_evaluation.to_csv(
    v3_validation_directory / "selection_quality_after_freeze.csv",
    index=False,
)
v3_joint_truth_evaluation.to_csv(
    v3_validation_directory / "joint_8bit_true_rank_evaluation.csv",
    index=False,
)
v3_pipeline_summary.to_csv(
    v3_validation_directory / "pipeline_summary.csv",
    index=False,
)

# Diagnostic plots. These are validation-only and do not affect policy selection.
v3_random_curve = v3_grid_truth_evaluation[
    v3_grid_truth_evaluation["campaign_arm"] == "random_uniform"
]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for (v3_key_id, v3_target), v3_group in v3_random_curve.groupby(
    ["key_id", "target_sbox"]
):
    v3_group = v3_group.sort_values("top_fraction")
    ax.plot(
        v3_group["top_fraction"],
        v3_group["true_rank"],
        marker="o",
        label=f"key {v3_key_id} / {v3_target}",
    )
ax.axvline(v3_chosen_top_fraction, linestyle="--", label="chosen public fraction")
ax.set_xlabel("Top fraction of paper-ineffective ciphertexts")
ax.set_ylabel("True-key rank")
ax.set_title("Stage-10 budget-aware selection: true rank after public freeze")
ax.set_yticks(range(1, 17))
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    v3_validation_directory / "random_arm_true_rank_vs_top_fraction.png",
    dpi=170,
)
plt.close(fig)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
v3_summary_plot = v3_pipeline_summary.set_index("pipeline").loc[
    ["random_raw", "random_threshold", "random_budget_aware"],
    ["actual_clean_precision", "actual_clean_recall"],
]
v3_summary_plot.plot(kind="bar", ax=ax)
ax.set_ylabel("Rate")
ax.set_title("Random arm: Stage-10 selection precision and recall")
ax.set_ylim(0.0, 1.05)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(
    v3_validation_directory / "random_arm_precision_recall.png",
    dpi=170,
)
plt.close(fig)

v3_checks = {
    "input_stage13r_v2_freeze_verified": bool(v3_input_freeze_check["verified"]),
    "public_policy_selected_before_truth": True,
    "public_attack_frozen_before_truth": True,
    "probability_weighting_inside_sei": False,
    "exact_paper_sei_used": True,
    "chosen_policy_uses_public_random_arm_only": True,
    "matched_random_same_budget_controls_present": bool(
        len(v3_matched_random_scores_public) > 0
    ),
    "four_nibble_tasks_present_per_pipeline": bool(
        np.all(
            v3_final_rank_evaluation.groupby("pipeline").size().to_numpy(int)
            == 4
        )
    ),
    "two_joint_8bit_tasks_present_per_pipeline": bool(
        np.all(
            v3_joint_truth_evaluation.groupby("pipeline").size().to_numpy(int)
            == 2
        )
    ),
}
v3_all_integrity_checks_passed = bool(all(v3_checks.values()))

v3_summary = {
    "stage": "13R-v3",
    "title": "Budget-aware Stage-10 selection for paper-faithful LBlock SIFA",
    "run_id": v3_run_id,
    "input_stage13r_v2_run": str(v3_input_run_directory),
    "output_run_directory": str(v3_run_directory),
    "input_freeze_sha256": v3_input_freeze_check["freeze_sha256"],
    "public_freeze_sha256": v3_public_freeze["freeze_sha256"],
    "chosen_top_fraction": v3_chosen_top_fraction,
    "candidate_top_fractions": list(stage13r_v3_config.candidate_top_fractions),
    "primary_statistic": "SEI",
    "paper_faithful_unweighted_scoring": True,
    "classifier_helped_by_fixed_rule": v3_classifier_helped_by_fixed_rule,
    "all_integrity_checks_passed": v3_all_integrity_checks_passed,
    "integrity_checks": v3_checks,
    "pipeline_summary": v3_pipeline_summary.to_dict(orient="records"),
    "matched_budget_comparison": v3_matched_comparison.to_dict(orient="records"),
    "joint_8bit_evaluation": v3_joint_truth_evaluation.to_dict(orient="records"),
    "elapsed_seconds": float(time.perf_counter() - v3_started),
}
v3_summary_path = v3_run_directory / "stage_13R_v3_budget_aware_summary.json"
v3_write_json(v3_summary_path, v3_summary)
v3_write_json(
    v3_validation_directory / "stage_13R_v3_validation_checks.json",
    v3_checks,
)

print("\n" + "=" * 96)
print("Stage 13R-v3 completed — budget-aware Stage-10 SIFA selection")
print("=" * 96)
print("Run directory                  :", v3_run_directory)
print("All integrity checks passed    :", v3_all_integrity_checks_passed)
print("Chosen public top fraction     :", f"{v3_chosen_top_fraction:.3f}")
print("Classifier helped fixed rule   :", v3_classifier_helped_by_fixed_rule)
print("Random raw mean true rank      :", f"{float(v3_random_raw_summary['mean_true_rank']):.3f}")
print("Random threshold mean rank     :", f"{float(v3_random_threshold_summary['mean_true_rank']):.3f}")
print("Random budget-aware mean rank  :", f"{float(v3_random_budget_summary['mean_true_rank']):.3f}")
print("Random raw Rank-1 nibbles      :", int(v3_random_raw_summary["rank1_nibble_count"]), "/ 4")
print("Random threshold Rank-1        :", int(v3_random_threshold_summary["rank1_nibble_count"]), "/ 4")
print("Random budget-aware Rank-1     :", int(v3_random_budget_summary["rank1_nibble_count"]), "/ 4")
print("Public freeze SHA-256          :", v3_public_freeze["freeze_sha256"])
print("Summary file                   :", v3_summary_path)
print("Elapsed seconds                :", f"{v3_summary['elapsed_seconds']:.3f}")
print("=" * 96)

ipy_display(v3_pipeline_summary.sort_values("pipeline").reset_index(drop=True))
ipy_display(
    v3_final_rank_evaluation.sort_values(
        ["pipeline", "key_id", "target_sbox_index"]
    ).reset_index(drop=True)
)
ipy_display(v3_matched_comparison.reset_index(drop=True))

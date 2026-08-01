# %%
# ============================================================
# Stage 15R / Cell 1
# Load and verify the frozen Stage-14R campaign without opening truth
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display as ipy_display


if "engine" not in globals():
    raise RuntimeError(
        "Before Stage 15R, run the Stage-12 setup cell so the LBlock engine is available. "
        "Do not rerun the Stage-14R injection campaign."
    )


@dataclass(frozen=True)
class PaperSHFAConfig:
    input_stage14_run_directory: str
    output_root: str
    expected_stage14_public_attack_freeze_sha256: str = (
        "f0a0e2a3794059f4475a874a13d6ddfe974d42e54890f93b39eb2a409a9ef3df"
    )
    random_seed: int = 20260725
    target_sbox_indices: Tuple[int, ...] = (0, 5)
    number_of_keys: int = 2
    number_of_sessions: int = 4
    minimum_branch_samples: int = 16
    bootstrap_repetitions: int = 300
    matched_random_repetitions: int = 300
    stable_loso_threshold: float = 1.0
    injection_checkpoints: Tuple[int, ...] = (
        250,
        500,
        1000,
        1500,
        2000,
        3000,
        4000,
        6000,
        8000,
    )
    save_plots: bool = True


paper_shfa_config = PaperSHFAConfig(
    input_stage14_run_directory=os.environ.get(
        "LBLOCK_PAPER_SHFA_STAGE14",
        r"C:\Users\SADRA\Desktop\LBlock\runs\stage_14R_paper_sefa"
        r"\stage14R_20260718_220657_126236_seed20260724",
    ),
    expected_stage14_public_attack_freeze_sha256=os.environ.get(
        "LBLOCK_SHFA_EXPECTED_STAGE14_FREEZE",
        "f0a0e2a3794059f4475a874a13d6ddfe974d42e54890f93b39eb2a409a9ef3df",
    ),
    output_root=os.environ.get(
        "LBLOCK_PAPER_SHFA_OUTPUT",
        r"C:\Users\SADRA\Desktop\LBlock\runs\stage_15R_paper_shfa",
    ),
    bootstrap_repetitions=int(os.environ.get("LBLOCK_SHFA_BOOTSTRAP_REPS", "300")),
    matched_random_repetitions=int(os.environ.get("LBLOCK_SHFA_MATCHED_REPS", "300")),
    save_plots=os.environ.get("LBLOCK_SHFA_SAVE_PLOTS", "1").strip().lower()
    not in {"0", "false", "no"},
)


def shfa15_json_default(value: Any) -> Any:
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


def shfa15_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=shfa15_json_default,
        ),
        encoding="utf-8",
    )


def shfa15_read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shfa15_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shfa15_manifest_payload_hash(manifest: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "freeze_sha256"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=shfa15_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shfa15_verify_freeze_manifest(
    manifest_path: Path,
    frozen_directory: Path,
    expected_freeze_hash: Optional[str] = None,
) -> Dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = shfa15_read_json(manifest_path)
    stored_hash = str(manifest.get("freeze_sha256", ""))
    recomputed_manifest_hash = shfa15_manifest_payload_hash(manifest)
    if stored_hash != recomputed_manifest_hash:
        raise RuntimeError(
            f"Freeze-manifest hash mismatch: {manifest_path}\n"
            f"stored={stored_hash}\nrecomputed={recomputed_manifest_hash}"
        )
    if expected_freeze_hash and stored_hash != str(expected_freeze_hash):
        raise RuntimeError(
            f"Unexpected Stage-14R public freeze. Expected {expected_freeze_hash}, got {stored_hash}"
        )
    for relative_name, expected_file_hash in manifest.get("files", {}).items():
        path = frozen_directory / relative_name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_file_hash = shfa15_sha256_file(path)
        if actual_file_hash != str(expected_file_hash):
            raise RuntimeError(
                f"Frozen file hash mismatch: {path}\n"
                f"expected={expected_file_hash}\nactual={actual_file_hash}"
            )
    return manifest


def shfa15_stable_freeze(directory: Path, output_path: Path, statement: str) -> Dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statement": statement,
        "files": {
            str(path.relative_to(directory)).replace("\\", "/"): shfa15_sha256_file(path)
            for path in files
        },
    }
    manifest["freeze_sha256"] = shfa15_manifest_payload_hash(manifest)
    shfa15_write_json(output_path, manifest)
    return manifest


def shfa15_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


if paper_shfa_config.number_of_keys != 2:
    raise ValueError("Stage 15R is pre-registered for exactly two keys")
if set(paper_shfa_config.target_sbox_indices) != {0, 5}:
    raise ValueError("Stage 15R is pre-registered for S0 and S5")
if paper_shfa_config.number_of_sessions != 4:
    raise ValueError("Stage 15R requires four sessions")
if paper_shfa_config.minimum_branch_samples < 2:
    raise ValueError("minimum_branch_samples is too small")


shfa15_input_directory = Path(
    paper_shfa_config.input_stage14_run_directory
).expanduser().resolve()
if not shfa15_input_directory.is_dir():
    raise FileNotFoundError(
        f"Stage-14R run directory not found: {shfa15_input_directory}\n"
        "Set LBLOCK_PAPER_SHFA_STAGE14 or edit paper_shfa_config."
    )

shfa15_public_campaign_directory = shfa15_input_directory / "public_campaign"
shfa15_public_policy_directory = shfa15_input_directory / "public_policy"
shfa15_stage14_public_attack_directory = shfa15_input_directory / "public_attack"
shfa15_locked_directory = shfa15_input_directory / "locked_truth"
shfa15_stage14_validation_directory = shfa15_input_directory / "validation_only"

shfa15_stage14_attack_manifest = shfa15_verify_freeze_manifest(
    shfa15_input_directory / "stage14R_public_attack_freeze_manifest.json",
    shfa15_stage14_public_attack_directory,
    paper_shfa_config.expected_stage14_public_attack_freeze_sha256,
)
shfa15_stage14_policy_manifest = shfa15_verify_freeze_manifest(
    shfa15_input_directory / "stage14R_public_policy_freeze_manifest.json",
    shfa15_public_policy_directory,
)

shfa15_stage14_config = shfa15_read_json(
    shfa15_public_campaign_directory / "stage14R_config.json"
)
shfa15_stage14_attack_contract = shfa15_read_json(
    shfa15_stage14_public_attack_directory / "public_attack_contract.json"
)
shfa15_sefa_policy = shfa15_read_json(
    shfa15_public_policy_directory / "chosen_sefa_budget_policy_public.json"
)
shfa15_locked_access_manifest = shfa15_read_json(
    shfa15_public_campaign_directory / "locked_truth_access_manifest.json"
)

shfa15_effective_threshold = float(
    shfa15_stage14_config["stage10_effective_threshold"]
)
shfa15_ineffective_threshold = float(
    shfa15_stage14_config["stage10_ineffective_threshold"]
)
shfa15_effective_top_fraction = float(
    shfa15_sefa_policy["chosen_top_fraction"]
)
shfa15_ineffective_top_fraction = float(
    shfa15_stage14_attack_contract["frozen_sifa_top_fraction"]
)
shfa15_confirmation_attempts = int(
    shfa15_stage14_config["confirmation_injections_per_key_target_arm"]
)

if shfa15_confirmation_attempts != max(paper_shfa_config.injection_checkpoints):
    raise RuntimeError(
        "The pre-registered SHFA checkpoint grid does not end at the Stage-14R confirmation budget"
    )
if not (0.0 < shfa15_effective_top_fraction <= 1.0):
    raise RuntimeError("Invalid frozen effective top fraction")
if not (0.0 < shfa15_ineffective_top_fraction <= 1.0):
    raise RuntimeError("Invalid frozen ineffective top fraction")
if not (0.0 < shfa15_effective_threshold < 1.0):
    raise RuntimeError("Invalid effective threshold")
if not (0.0 < shfa15_ineffective_threshold < 1.0):
    raise RuntimeError("Invalid ineffective threshold")

shfa15_campaign_path = (
    shfa15_public_campaign_directory
    / "paired_sefa_campaign_with_public_probabilities.csv"
)
if not shfa15_campaign_path.is_file():
    raise FileNotFoundError(shfa15_campaign_path)
shfa15_public_scored = pd.read_csv(shfa15_campaign_path)

shfa15_required_columns = {
    "experiment_id",
    "pair_id",
    "campaign_partition",
    "campaign_arm",
    "confirmation_index",
    "key_id",
    "session_id",
    "target_sbox_index",
    "healthy_ciphertext_hex",
    "response_received",
    "ciphertext_equal",
    "p_clean_target_effective",
    "p_clean_target_ineffective",
}
shfa15_missing_columns = sorted(shfa15_required_columns - set(shfa15_public_scored.columns))
if shfa15_missing_columns:
    raise RuntimeError(
        "Stage-14R public campaign is missing required columns: "
        + ", ".join(shfa15_missing_columns)
    )

shfa15_public_scored["paper_effective"] = (
    shfa15_bool_series(shfa15_public_scored["response_received"])
    & (
        shfa15_public_scored["ciphertext_equal"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "false"
    )
)
shfa15_public_scored["paper_ineffective"] = (
    shfa15_bool_series(shfa15_public_scored["response_received"])
    & (
        shfa15_public_scored["ciphertext_equal"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "true"
    )
)
if bool((shfa15_public_scored["paper_effective"] & shfa15_public_scored["paper_ineffective"]).any()):
    raise AssertionError("Effective and ineffective event sets must be disjoint")


def shfa15_parse_ciphertext_words(ciphertext_hex: Any) -> Tuple[int, int]:
    text = str(ciphertext_hex).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if len(text) != 16:
        raise ValueError(f"Expected a 64-bit ciphertext, got {text!r}")
    return int(text[:8], 16), int(text[8:], 16)


shfa15_parsed_words = [
    shfa15_parse_ciphertext_words(value)
    for value in shfa15_public_scored["healthy_ciphertext_hex"]
]
shfa15_public_scored["x32_word"] = np.asarray(
    [value[0] for value in shfa15_parsed_words], dtype=np.uint64
)
shfa15_public_scored["x33_word"] = np.asarray(
    [value[1] for value in shfa15_parsed_words], dtype=np.uint64
)

_SHFA15_P_SOURCE_FOR_OUTPUT = tuple(int(value) for value in engine.P_SOURCE_FOR_OUTPUT)
_SHFA15_SOURCE_TO_OUTPUT = {
    int(source): int(output_index)
    for output_index, source in enumerate(_SHFA15_P_SOURCE_FOR_OUTPUT)
}
_SHFA15_STATE_NIBBLE_BY_CHANNEL = {
    int(source): int((output_index - 2) % 8)
    for source, output_index in _SHFA15_SOURCE_TO_OUTPUT.items()
}
if _SHFA15_STATE_NIBBLE_BY_CHANNEL[0] != 0:
    raise AssertionError("S0 must map to X31[0]")
if _SHFA15_STATE_NIBBLE_BY_CHANNEL[5] != 2:
    raise AssertionError("S5 must map to X31[2]")


def shfa15_reconstruct_x31_target(
    x32: int,
    x33: int,
    target_sbox_index: int,
    key_guess: int,
) -> int:
    sbox_index = int(target_sbox_index)
    output_index = int(_SHFA15_SOURCE_TO_OUTPUT[sbox_index])
    x32_nibble = int(engine.get_nibble(int(x32), sbox_index))
    x33_nibble = int(engine.get_nibble(int(x33), output_index))
    return int(
        x33_nibble
        ^ int(engine.SBOX[sbox_index][x32_nibble ^ int(key_guess)])
    )


def shfa15_intermediate_matrix(task_frame: pd.DataFrame) -> np.ndarray:
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
                shfa15_reconstruct_x31_target(
                    int(x32), int(x33), target_index, key_guess
                )
                for x32, x33 in zip(x32_values, x33_values)
            ),
            dtype=np.uint8,
            count=len(task_frame),
        )
    return matrix


def shfa15_session_balanced_top_indices(
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
        rows = task_frame[
            task_frame["session_id"].astype(int) == int(session)
        ].sort_values(
            [probability_column, "experiment_id"],
            ascending=[False, True],
        )
        selected.extend(rows.index[: int(quota)].astype(int).tolist())
    selected_array = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected_array) != requested:
        remaining = task_frame.loc[
            ~task_frame.index.isin(selected_array)
        ].sort_values(
            [probability_column, "experiment_id"],
            ascending=[False, True],
        )
        needed = requested - len(selected_array)
        if needed > 0:
            selected_array = np.concatenate(
                [selected_array, remaining.index[:needed].to_numpy(int)]
            )
    return np.asarray(sorted(selected_array.tolist()), dtype=int)


_SHFA15_UNIFORM_16 = np.full(16, 1.0 / 16.0, dtype=float)
_SHFA15_UNIFORM_256 = np.full((16, 16), 1.0 / 256.0, dtype=float)


def shfa15_individual_sei_scores(
    matrix: np.ndarray,
    selected_indices: np.ndarray,
) -> pd.DataFrame:
    selected_indices = np.asarray(selected_indices, dtype=int)
    sample_count = int(len(selected_indices))
    rows: List[Dict[str, Any]] = []
    for key_guess in range(16):
        values = matrix[selected_indices, key_guess]
        counts = np.bincount(values, minlength=16).astype(float)
        empirical = counts / float(sample_count)
        sei = float(np.sum(np.square(empirical - _SHFA15_UNIFORM_16)))
        rows.append({"key_guess": key_guess, "sei": sei})
    return pd.DataFrame(rows)


# SHFA uses the product distribution q_h(x,y)=q_e(x)q_i(y).  The primary
# score is the 16x16 joint SEI from the paper.  Stage-10 probabilities never
# enter this equation.
def shfa15_scores_from_matrices(
    effective_matrix: np.ndarray,
    ineffective_matrix: np.ndarray,
    effective_indices: np.ndarray,
    ineffective_indices: np.ndarray,
) -> pd.DataFrame:
    effective_indices = np.asarray(effective_indices, dtype=int)
    ineffective_indices = np.asarray(ineffective_indices, dtype=int)
    number_effective = int(len(effective_indices))
    number_ineffective = int(len(ineffective_indices))
    if number_effective == 0 or number_ineffective == 0:
        raise ValueError("SHFA requires non-empty effective and ineffective branches")

    rows: List[Dict[str, Any]] = []
    for key_guess in range(16):
        values_effective = effective_matrix[effective_indices, key_guess]
        values_ineffective = ineffective_matrix[ineffective_indices, key_guess]
        counts_effective = np.bincount(values_effective, minlength=16).astype(float)
        counts_ineffective = np.bincount(values_ineffective, minlength=16).astype(float)
        distribution_effective = counts_effective / float(number_effective)
        distribution_ineffective = counts_ineffective / float(number_ineffective)

        sei_effective = float(
            np.sum(np.square(distribution_effective - _SHFA15_UNIFORM_16))
        )
        sei_ineffective = float(
            np.sum(np.square(distribution_ineffective - _SHFA15_UNIFORM_16))
        )
        joint_distribution = np.outer(
            distribution_effective,
            distribution_ineffective,
        )
        joint_sei = float(
            np.sum(np.square(joint_distribution - _SHFA15_UNIFORM_256))
        )

        # Algebraic audit of the exact outer-product statistic:
        # ||qe qi^T-U256||² = SEIe*SEIi + (SEIe+SEIi)/16.
        equivalent_joint_sei = float(
            sei_effective * sei_ineffective
            + (sei_effective + sei_ineffective) / 16.0
        )
        if not np.isclose(
            joint_sei,
            equivalent_joint_sei,
            atol=1.0e-15,
            rtol=1.0e-12,
        ):
            raise AssertionError("SHFA joint-SEI algebraic identity failed")

        rows.append(
            {
                "key_guess": int(key_guess),
                "key_guess_hex": f"{key_guess:x}",
                "effective_sample_count": number_effective,
                "ineffective_sample_count": number_ineffective,
                "sei_effective_component": sei_effective,
                "sei_ineffective_component": sei_ineffective,
                "shfa_sei_joint": joint_sei,
                "shfa_sei_joint_equivalent": equivalent_joint_sei,
                "alternative_sei_sum": float(sei_effective + sei_ineffective),
                "alternative_sei_product": float(sei_effective * sei_ineffective),
            }
        )
    return pd.DataFrame(rows)


def shfa15_prediction(scores: pd.DataFrame) -> Dict[str, Any]:
    ordered = scores.sort_values(
        ["shfa_sei_joint", "key_guess"],
        ascending=[False, True],
    ).reset_index(drop=True)
    if ordered.empty or not np.isfinite(float(ordered.iloc[0]["shfa_sei_joint"])):
        return {
            "best_key_guess": -1,
            "best_key_guess_hex": "",
            "best_score": np.nan,
            "second_score": np.nan,
            "score_margin": np.nan,
            "relative_margin": np.nan,
            "unique_best": False,
        }
    best_score = float(ordered.iloc[0]["shfa_sei_joint"])
    second_score = float(ordered.iloc[1]["shfa_sei_joint"])
    ties = np.isclose(
        scores["shfa_sei_joint"].to_numpy(float),
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


# Six pre-registered SHFA pipelines.  Both branch policies were already frozen
# before the Stage-14R confirmation partition was attacked.
_SHFA15_PIPELINES: Dict[str, Dict[str, Any]] = {}
for arm_name, campaign_arm in (
    ("random", "random_uniform"),
    ("guided", "guided_model"),
):
    for selector in ("raw", "threshold", "budget"):
        pipeline_name = f"shfa_{arm_name}_{selector}"
        _SHFA15_PIPELINES[pipeline_name] = {
            "attack_type": "SHFA",
            "campaign_arm": campaign_arm,
            "selector": selector,
            "uses_stage11": campaign_arm == "guided_model",
            "uses_stage10": selector != "raw",
            "effective_probability_column": "p_clean_target_effective",
            "ineffective_probability_column": "p_clean_target_ineffective",
            "effective_threshold": shfa15_effective_threshold,
            "ineffective_threshold": shfa15_ineffective_threshold,
            "effective_budget_fraction": shfa15_effective_top_fraction,
            "ineffective_budget_fraction": shfa15_ineffective_top_fraction,
        }


def shfa15_event_frames(
    pipeline: str,
    key_id: int,
    target_index: int,
    maximum_confirmation_index: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spec = _SHFA15_PIPELINES[pipeline]
    base_mask = (
        (shfa15_public_scored["campaign_partition"].astype(str) == "confirmation")
        & (
            shfa15_public_scored["campaign_arm"].astype(str)
            == str(spec["campaign_arm"])
        )
        & (shfa15_public_scored["key_id"].astype(int) == int(key_id))
        & (
            shfa15_public_scored["target_sbox_index"].astype(int)
            == int(target_index)
        )
    )
    if maximum_confirmation_index is not None:
        base_mask &= (
            shfa15_public_scored["confirmation_index"].astype(int)
            < int(maximum_confirmation_index)
        )
    effective_frame = (
        shfa15_public_scored[base_mask & shfa15_public_scored["paper_effective"]]
        .sort_values(["confirmation_index", "experiment_id"])
        .reset_index(drop=True)
    )
    ineffective_frame = (
        shfa15_public_scored[base_mask & shfa15_public_scored["paper_ineffective"]]
        .sort_values(["confirmation_index", "experiment_id"])
        .reset_index(drop=True)
    )
    return effective_frame, ineffective_frame


def shfa15_select_branch_indices(
    event_frame: pd.DataFrame,
    selector: str,
    probability_column: str,
    threshold: float,
    budget_fraction: float,
) -> np.ndarray:
    if selector == "raw":
        return np.arange(len(event_frame), dtype=int)
    if selector == "threshold":
        mask = (
            event_frame[probability_column].astype(float).to_numpy()
            >= float(threshold)
        )
        return np.where(mask)[0].astype(int)
    if selector == "budget":
        return shfa15_session_balanced_top_indices(
            event_frame,
            float(budget_fraction),
            probability_column,
        )
    raise ValueError(f"Unknown selector: {selector}")


def shfa15_select_both_branches(
    effective_frame: pd.DataFrame,
    ineffective_frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    selector = str(spec["selector"])
    effective_indices = shfa15_select_branch_indices(
        effective_frame,
        selector,
        str(spec["effective_probability_column"]),
        float(spec["effective_threshold"]),
        float(spec["effective_budget_fraction"]),
    )
    ineffective_indices = shfa15_select_branch_indices(
        ineffective_frame,
        selector,
        str(spec["ineffective_probability_column"]),
        float(spec["ineffective_threshold"]),
        float(spec["ineffective_budget_fraction"]),
    )
    return effective_indices, ineffective_indices


def shfa15_loso_consensus(
    effective_frame: pd.DataFrame,
    ineffective_frame: pd.DataFrame,
    effective_matrix: np.ndarray,
    ineffective_matrix: np.ndarray,
    effective_indices: np.ndarray,
    ineffective_indices: np.ndarray,
    full_winner: int,
) -> Tuple[float, int, str]:
    sessions = sorted(
        set(effective_frame.loc[effective_indices, "session_id"].astype(int).tolist())
        | set(ineffective_frame.loc[ineffective_indices, "session_id"].astype(int).tolist())
    )
    winners: List[int] = []
    for session in sessions:
        keep_effective = effective_indices[
            effective_frame.loc[effective_indices, "session_id"].to_numpy(int)
            != int(session)
        ]
        keep_ineffective = ineffective_indices[
            ineffective_frame.loc[ineffective_indices, "session_id"].to_numpy(int)
            != int(session)
        ]
        if (
            len(keep_effective) < paper_shfa_config.minimum_branch_samples
            or len(keep_ineffective) < paper_shfa_config.minimum_branch_samples
        ):
            continue
        prediction = shfa15_prediction(
            shfa15_scores_from_matrices(
                effective_matrix,
                ineffective_matrix,
                keep_effective,
                keep_ineffective,
            )
        )
        winners.append(int(prediction["best_key_guess"]))
    if not winners:
        return 0.0, int(len(sessions)), ""
    consensus = float(np.mean(np.asarray(winners, dtype=int) == int(full_winner)))
    return consensus, int(len(sessions)), ",".join(f"{winner:x}" for winner in winners)


# A full reconstruction of the frozen Stage-14R SIFA/SEFA candidate scores is
# used as a campaign-integrity guard.  It proves that the public campaign being
# loaded is the one that generated the frozen Stage-14R attack tables.
def shfa15_validate_stage14_candidate_scores() -> None:
    frozen_scores = pd.read_csv(
        shfa15_stage14_public_attack_directory
        / "sifa_sefa_candidate_scores_public.csv"
    )
    recreated_rows: List[Dict[str, Any]] = []
    for attack_type, event_column, probability_column, threshold, fraction in (
        (
            "SEFA",
            "paper_effective",
            "p_clean_target_effective",
            shfa15_effective_threshold,
            shfa15_effective_top_fraction,
        ),
        (
            "SIFA",
            "paper_ineffective",
            "p_clean_target_ineffective",
            shfa15_ineffective_threshold,
            shfa15_ineffective_top_fraction,
        ),
    ):
        for arm_name, campaign_arm in (
            ("random", "random_uniform"),
            ("guided", "guided_model"),
        ):
            for selector in ("raw", "threshold", "budget"):
                pipeline = f"{attack_type.lower()}_{arm_name}_{selector}"
                for key_id in range(paper_shfa_config.number_of_keys):
                    for target_index in paper_shfa_config.target_sbox_indices:
                        mask = (
                            (shfa15_public_scored["campaign_partition"].astype(str) == "confirmation")
                            & (shfa15_public_scored["campaign_arm"].astype(str) == campaign_arm)
                            & (shfa15_public_scored["key_id"].astype(int) == key_id)
                            & (shfa15_public_scored["target_sbox_index"].astype(int) == target_index)
                            & shfa15_public_scored[event_column].astype(bool)
                        )
                        frame = (
                            shfa15_public_scored[mask]
                            .sort_values(["confirmation_index", "experiment_id"])
                            .reset_index(drop=True)
                        )
                        matrix = shfa15_intermediate_matrix(frame)
                        indices = shfa15_select_branch_indices(
                            frame,
                            selector,
                            probability_column,
                            threshold,
                            fraction,
                        )
                        scores = shfa15_individual_sei_scores(matrix, indices)
                        for row in scores.itertuples(index=False):
                            recreated_rows.append(
                                {
                                    "pipeline": pipeline,
                                    "key_id": key_id,
                                    "target_sbox_index": target_index,
                                    "key_guess": int(row.key_guess),
                                    "sei_recreated": float(row.sei),
                                }
                            )
    recreated = pd.DataFrame(recreated_rows)
    compare = frozen_scores[
        ["pipeline", "key_id", "target_sbox_index", "key_guess", "sei"]
    ].merge(
        recreated,
        on=["pipeline", "key_id", "target_sbox_index", "key_guess"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not bool((compare["_merge"] == "both").all()):
        raise RuntimeError("Stage-14R frozen candidate-score row identity check failed")
    if not np.allclose(
        compare["sei"].to_numpy(float),
        compare["sei_recreated"].to_numpy(float),
        atol=1.0e-14,
        rtol=1.0e-10,
        equal_nan=True,
    ):
        maximum_difference = float(
            np.nanmax(
                np.abs(
                    compare["sei"].to_numpy(float)
                    - compare["sei_recreated"].to_numpy(float)
                )
            )
        )
        raise RuntimeError(
            "Stage-14R public campaign does not reproduce its frozen candidate scores; "
            f"max difference={maximum_difference}"
        )


shfa15_validate_stage14_candidate_scores()

shfa15_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
shfa15_run_id = f"stage15R_{shfa15_timestamp}_seed{paper_shfa_config.random_seed}"
shfa15_run_directory = (
    Path(paper_shfa_config.output_root).expanduser().resolve() / shfa15_run_id
)
shfa15_public_attack_directory = shfa15_run_directory / "public_attack"
shfa15_validation_directory = shfa15_run_directory / "validation_only"
for directory in (shfa15_public_attack_directory, shfa15_validation_directory):
    directory.mkdir(parents=True, exist_ok=True)

shfa15_write_json(
    shfa15_public_attack_directory / "stage15R_pre_registered_contract.json",
    {
        "input_stage14_run_directory": str(shfa15_input_directory),
        "verified_stage14_public_attack_freeze_sha256": str(
            shfa15_stage14_attack_manifest["freeze_sha256"]
        ),
        "attack": "paper-faithful SHFA",
        "event_sets": {
            "effective": "response_received and C_prime != C",
            "ineffective": "response_received and C_prime == C",
            "disjoint": True,
        },
        "key_recovery_ciphertext": "correct/non-faulty C in both branches",
        "primary_score": (
            "SEI_h(k)=sum_x sum_y "
            "(p_hat_e,k(x)*p_hat_i,k(y)-1/256)^2; highest score wins"
        ),
        "stage10_probability_inside_shfa_score": False,
        "branch_selection_policies": {
            "effective_threshold": shfa15_effective_threshold,
            "ineffective_threshold": shfa15_ineffective_threshold,
            "effective_top_fraction": shfa15_effective_top_fraction,
            "ineffective_top_fraction": shfa15_ineffective_top_fraction,
            "policies_were_frozen_before_stage14_confirmation": True,
        },
        "pipelines": _SHFA15_PIPELINES,
        "equal_injection_budget": shfa15_confirmation_attempts,
        "comparison_principle": (
            "Model impact is evaluated within each attack at the same total fault-attempt budget. "
            "Effective and ineffective event counts are not used as a direct cross-attack superiority metric."
        ),
        "truth_opened": False,
        "private_labels_opened": False,
        "configuration": asdict(paper_shfa_config),
    },
)

print("=" * 100)
print("Stage 15R configuration ready — paper-faithful SHFA")
print("Verified Stage-14R input         :", shfa15_input_directory)
print("Stage-14R public freeze SHA-256  :", shfa15_stage14_attack_manifest["freeze_sha256"])
print("Output directory                 :", shfa15_run_directory)
print("Confirmation attempts per task   :", shfa15_confirmation_attempts)
print("Effective frozen top fraction    :", f"{shfa15_effective_top_fraction:.3f}")
print("Ineffective frozen top fraction  :", f"{shfa15_ineffective_top_fraction:.3f}")
print("Effective frozen threshold       :", f"{shfa15_effective_threshold:.6f}")
print("Ineffective frozen threshold     :", f"{shfa15_ineffective_threshold:.6f}")
print("Stage-14 score reproduction      : passed")
print("Truth/private opened             : False")
print("=" * 100)

# %%
# ============================================================
# Stage 15R / Cell 2
# Public SHFA scoring, injection-prefix curves, and attack freeze
# ============================================================

shfa15_started = time.perf_counter()
shfa15_score_parts: List[pd.DataFrame] = []
shfa15_prediction_rows: List[Dict[str, Any]] = []
shfa15_task_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

for pipeline, spec in _SHFA15_PIPELINES.items():
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            effective_frame, ineffective_frame = shfa15_event_frames(
                pipeline,
                key_id,
                target_index,
            )
            effective_matrix = shfa15_intermediate_matrix(effective_frame)
            ineffective_matrix = shfa15_intermediate_matrix(ineffective_frame)
            effective_indices, ineffective_indices = shfa15_select_both_branches(
                effective_frame,
                ineffective_frame,
                spec,
            )
            if len(effective_indices) < paper_shfa_config.minimum_branch_samples:
                raise RuntimeError(
                    f"Too few effective samples for {pipeline}/key{key_id}/S{target_index}: "
                    f"{len(effective_indices)}"
                )
            if len(ineffective_indices) < paper_shfa_config.minimum_branch_samples:
                raise RuntimeError(
                    f"Too few ineffective samples for {pipeline}/key{key_id}/S{target_index}: "
                    f"{len(ineffective_indices)}"
                )

            scores = shfa15_scores_from_matrices(
                effective_matrix,
                ineffective_matrix,
                effective_indices,
                ineffective_indices,
            )
            prediction = shfa15_prediction(scores)
            loso, coverage, loso_winners = shfa15_loso_consensus(
                effective_frame,
                ineffective_frame,
                effective_matrix,
                ineffective_matrix,
                effective_indices,
                ineffective_indices,
                int(prediction["best_key_guess"]),
            )

            scores.insert(0, "pipeline", pipeline)
            scores.insert(1, "attack_type", "SHFA")
            scores.insert(2, "key_id", int(key_id))
            scores.insert(3, "target_sbox", f"S{int(target_index)}")
            scores.insert(4, "target_sbox_index", int(target_index))
            shfa15_score_parts.append(scores)

            selected_effective_probabilities = effective_frame.loc[
                effective_indices,
                str(spec["effective_probability_column"]),
            ].to_numpy(float)
            selected_ineffective_probabilities = ineffective_frame.loc[
                ineffective_indices,
                str(spec["ineffective_probability_column"]),
            ].to_numpy(float)
            shfa15_prediction_rows.append(
                {
                    "pipeline": pipeline,
                    "attack_type": "SHFA",
                    "campaign_arm": str(spec["campaign_arm"]),
                    "selector": str(spec["selector"]),
                    "uses_stage11": bool(spec["uses_stage11"]),
                    "uses_stage10": bool(spec["uses_stage10"]),
                    "key_id": int(key_id),
                    "target_sbox": f"S{int(target_index)}",
                    "target_sbox_index": int(target_index),
                    "confirmation_injection_count": shfa15_confirmation_attempts,
                    "observable_effective_count": int(len(effective_frame)),
                    "observable_ineffective_count": int(len(ineffective_frame)),
                    "selected_effective_count": int(len(effective_indices)),
                    "selected_ineffective_count": int(len(ineffective_indices)),
                    "mean_selected_effective_probability": float(
                        np.mean(selected_effective_probabilities)
                    ),
                    "mean_selected_ineffective_probability": float(
                        np.mean(selected_ineffective_probabilities)
                    ),
                    "loso_consensus": float(loso),
                    "session_coverage": int(coverage),
                    "loso_winners_hex": loso_winners,
                    **prediction,
                }
            )
            shfa15_task_cache[(pipeline, int(key_id), int(target_index))] = {
                "effective_frame": effective_frame,
                "ineffective_frame": ineffective_frame,
                "effective_matrix": effective_matrix,
                "ineffective_matrix": ineffective_matrix,
                "effective_indices": effective_indices,
                "ineffective_indices": ineffective_indices,
            }

shfa15_candidate_scores_public = pd.concat(shfa15_score_parts, ignore_index=True)
shfa15_predictions_public = pd.DataFrame(shfa15_prediction_rows)

# Prefix curves use TOTAL confirmation fault attempts on the x-axis.  The two
# event classes are never treated as interchangeable sample counts.
shfa15_prefix_score_parts: List[pd.DataFrame] = []
shfa15_prefix_prediction_rows: List[Dict[str, Any]] = []
for pipeline, spec in _SHFA15_PIPELINES.items():
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            for checkpoint in paper_shfa_config.injection_checkpoints:
                effective_frame, ineffective_frame = shfa15_event_frames(
                    pipeline,
                    key_id,
                    target_index,
                    maximum_confirmation_index=int(checkpoint),
                )
                effective_matrix = shfa15_intermediate_matrix(effective_frame)
                ineffective_matrix = shfa15_intermediate_matrix(ineffective_frame)
                effective_indices, ineffective_indices = shfa15_select_both_branches(
                    effective_frame,
                    ineffective_frame,
                    spec,
                )
                if (
                    len(effective_indices) < paper_shfa_config.minimum_branch_samples
                    or len(ineffective_indices) < paper_shfa_config.minimum_branch_samples
                ):
                    continue
                scores = shfa15_scores_from_matrices(
                    effective_matrix,
                    ineffective_matrix,
                    effective_indices,
                    ineffective_indices,
                )
                prediction = shfa15_prediction(scores)
                loso, coverage, loso_winners = shfa15_loso_consensus(
                    effective_frame,
                    ineffective_frame,
                    effective_matrix,
                    ineffective_matrix,
                    effective_indices,
                    ineffective_indices,
                    int(prediction["best_key_guess"]),
                )
                scores.insert(0, "pipeline", pipeline)
                scores.insert(1, "attack_type", "SHFA")
                scores.insert(2, "key_id", int(key_id))
                scores.insert(3, "target_sbox", f"S{int(target_index)}")
                scores.insert(4, "target_sbox_index", int(target_index))
                scores.insert(5, "injection_checkpoint", int(checkpoint))
                shfa15_prefix_score_parts.append(scores)
                shfa15_prefix_prediction_rows.append(
                    {
                        "pipeline": pipeline,
                        "attack_type": "SHFA",
                        "key_id": int(key_id),
                        "target_sbox": f"S{int(target_index)}",
                        "target_sbox_index": int(target_index),
                        "injection_checkpoint": int(checkpoint),
                        "observable_effective_count": int(len(effective_frame)),
                        "observable_ineffective_count": int(len(ineffective_frame)),
                        "selected_effective_count": int(len(effective_indices)),
                        "selected_ineffective_count": int(len(ineffective_indices)),
                        "loso_consensus": float(loso),
                        "session_coverage": int(coverage),
                        "loso_winners_hex": loso_winners,
                        **prediction,
                    }
                )

shfa15_prefix_scores_public = pd.concat(shfa15_prefix_score_parts, ignore_index=True)
shfa15_prefix_predictions_public = pd.DataFrame(shfa15_prefix_prediction_rows)

# Independent nibble attacks are combined into the same 8-bit key-pair ranking
# convention used in Stage 13R and Stage 14R.  This 8-bit combination is not the
# effective/ineffective joint distribution; it is a final key-candidate product
# across S0 and S5, scored by the sum of their SHFA nibble scores.
shfa15_joint_8bit_score_rows: List[Dict[str, Any]] = []
shfa15_joint_8bit_prediction_rows: List[Dict[str, Any]] = []
for pipeline in _SHFA15_PIPELINES:
    for key_id in range(paper_shfa_config.number_of_keys):
        left = shfa15_candidate_scores_public[
            (shfa15_candidate_scores_public["pipeline"] == pipeline)
            & (shfa15_candidate_scores_public["key_id"].astype(int) == int(key_id))
            & (shfa15_candidate_scores_public["target_sbox_index"].astype(int) == 0)
        ][["key_guess", "shfa_sei_joint"]].rename(
            columns={"key_guess": "guess_s0", "shfa_sei_joint": "score_s0"}
        )
        right = shfa15_candidate_scores_public[
            (shfa15_candidate_scores_public["pipeline"] == pipeline)
            & (shfa15_candidate_scores_public["key_id"].astype(int) == int(key_id))
            & (shfa15_candidate_scores_public["target_sbox_index"].astype(int) == 5)
        ][["key_guess", "shfa_sei_joint"]].rename(
            columns={"key_guess": "guess_s5", "shfa_sei_joint": "score_s5"}
        )
        joint = (
            left.assign(_join=1)
            .merge(right.assign(_join=1), on="_join")
            .drop(columns="_join")
        )
        joint["joint_guess"] = (
            joint["guess_s0"].astype(int) * 16
            + joint["guess_s5"].astype(int)
        )
        joint["joint_guess_hex"] = [
            f"{int(a):x}{int(b):x}"
            for a, b in zip(joint["guess_s0"], joint["guess_s5"])
        ]
        joint["joint_8bit_score"] = (
            joint["score_s0"].astype(float) + joint["score_s5"].astype(float)
        )
        ordered = joint.sort_values(
            ["joint_8bit_score", "joint_guess"],
            ascending=[False, True],
        ).reset_index(drop=True)
        margin = float(
            ordered.iloc[0]["joint_8bit_score"]
            - ordered.iloc[1]["joint_8bit_score"]
        )
        shfa15_joint_8bit_prediction_rows.append(
            {
                "pipeline": pipeline,
                "attack_type": "SHFA",
                "key_id": int(key_id),
                "best_joint_guess": int(ordered.iloc[0]["joint_guess"]),
                "best_joint_guess_hex": str(ordered.iloc[0]["joint_guess_hex"]),
                "best_joint_score": float(ordered.iloc[0]["joint_8bit_score"]),
                "joint_score_margin": margin,
            }
        )
        for row in joint.itertuples(index=False):
            shfa15_joint_8bit_score_rows.append(
                {
                    "pipeline": pipeline,
                    "attack_type": "SHFA",
                    "key_id": int(key_id),
                    "guess_s0": int(row.guess_s0),
                    "guess_s5": int(row.guess_s5),
                    "joint_guess": int(row.joint_guess),
                    "joint_guess_hex": str(row.joint_guess_hex),
                    "score_s0": float(row.score_s0),
                    "score_s5": float(row.score_s5),
                    "joint_8bit_score": float(row.joint_8bit_score),
                }
            )

shfa15_joint_8bit_scores_public = pd.DataFrame(shfa15_joint_8bit_score_rows)
shfa15_joint_8bit_predictions_public = pd.DataFrame(
    shfa15_joint_8bit_prediction_rows
)

shfa15_score_ranges = (
    shfa15_candidate_scores_public.groupby(
        ["pipeline", "key_id", "target_sbox_index"]
    )["shfa_sei_joint"]
    .agg(lambda values: float(np.nanmax(values) - np.nanmin(values)))
)
if np.any(shfa15_score_ranges.to_numpy(float) <= 1.0e-15):
    failing = shfa15_score_ranges[shfa15_score_ranges <= 1.0e-15]
    raise RuntimeError(
        "SHFA identifiability guard failed before truth was opened:\n"
        + failing.to_string()
    )

shfa15_candidate_scores_public.to_csv(
    shfa15_public_attack_directory / "shfa_candidate_scores_public.csv",
    index=False,
)
shfa15_predictions_public.to_csv(
    shfa15_public_attack_directory / "shfa_predictions_public.csv",
    index=False,
)
shfa15_prefix_scores_public.to_csv(
    shfa15_public_attack_directory / "injection_prefix_shfa_candidate_scores_public.csv",
    index=False,
)
shfa15_prefix_predictions_public.to_csv(
    shfa15_public_attack_directory / "injection_prefix_shfa_predictions_public.csv",
    index=False,
)
shfa15_joint_8bit_scores_public.to_csv(
    shfa15_public_attack_directory / "joint_8bit_shfa_candidate_scores_public.csv",
    index=False,
)
shfa15_joint_8bit_predictions_public.to_csv(
    shfa15_public_attack_directory / "joint_8bit_shfa_predictions_public.csv",
    index=False,
)
shfa15_write_json(
    shfa15_public_attack_directory / "public_attack_contract.json",
    {
        "truth_opened": False,
        "private_labels_opened": False,
        "input_stage14_public_attack_freeze_sha256": str(
            shfa15_stage14_attack_manifest["freeze_sha256"]
        ),
        "primary_statistic": "joint-distribution unweighted SEI_h",
        "joint_distribution": "p_hat_e,k(x) * p_hat_i,k(y)",
        "uniform_reference": "1/256",
        "probability_weighting_inside_score": False,
        "correct_ciphertext_used_in_both_branches": True,
        "faulty_ciphertext_values_used_for_key_scoring": False,
        "confirmation_partition_only": True,
        "equal_total_fault_attempt_budget": shfa15_confirmation_attempts,
        "pipelines": _SHFA15_PIPELINES,
    },
)

shfa15_public_attack_freeze = shfa15_stable_freeze(
    shfa15_public_attack_directory,
    shfa15_run_directory / "stage15R_public_attack_freeze_manifest.json",
    "All SHFA scores, predictions, injection-prefix curves, and 8-bit rankings were frozen before Stage-14R key truth or private labels were opened by Stage 15R.",
)

print("=" * 100)
print("Stage 15R public SHFA attack frozen")
print("Public freeze SHA-256            :", shfa15_public_attack_freeze["freeze_sha256"])
print("Truth/private opened             : False")
print("Public predictions:")
print(
    shfa15_predictions_public[
        [
            "pipeline",
            "key_id",
            "target_sbox",
            "selected_effective_count",
            "selected_ineffective_count",
            "best_key_guess_hex",
            "score_margin",
            "loso_consensus",
            "unique_best",
        ]
    ].to_string(index=False)
)
print("=" * 100)

# %%
# ============================================================
# Stage 15R / Cell 3
# Post-freeze truth opening, bootstrap, controls, and rank metrics
# ============================================================

if not (shfa15_run_directory / "stage15R_public_attack_freeze_manifest.json").is_file():
    raise RuntimeError("Public SHFA attack must be frozen before truth is opened")

shfa15_locked_label_path = shfa15_locked_directory / "fault_labels_LOCKED.csv"
shfa15_locked_key_path = shfa15_locked_directory / "key_truth_LOCKED.json"
if shfa15_sha256_file(shfa15_locked_label_path) != str(
    shfa15_locked_access_manifest["fault_labels_sha256"]
):
    raise RuntimeError("Stage-14R locked label hash mismatch")
if shfa15_sha256_file(shfa15_locked_key_path) != str(
    shfa15_locked_access_manifest["key_truth_sha256"]
):
    raise RuntimeError("Stage-14R locked key hash mismatch")

shfa15_private_labels = pd.read_csv(shfa15_locked_label_path)
shfa15_key_truth = shfa15_read_json(shfa15_locked_key_path)

shfa15_true_nibbles: Dict[Tuple[int, int], int] = {}
shfa15_true_joint_8bit: Dict[int, int] = {}
for item in shfa15_key_truth["keys"]:
    key_id = int(item["key_id"])
    round_key_32 = int(str(item["round_key_32_hex"]), 16)
    nibble_s0 = int((round_key_32 >> 0) & 0xF)
    nibble_s5 = int((round_key_32 >> (4 * 5)) & 0xF)
    shfa15_true_nibbles[(key_id, 0)] = nibble_s0
    shfa15_true_nibbles[(key_id, 5)] = nibble_s5
    shfa15_true_joint_8bit[key_id] = int(nibble_s0 * 16 + nibble_s5)


def shfa15_rank_from_scores(
    scores: pd.DataFrame,
    true_guess: int,
    score_column: str = "shfa_sei_joint",
    guess_column: str = "key_guess",
) -> Tuple[int, float, bool]:
    true_row = scores[
        scores[guess_column].astype(int) == int(true_guess)
    ]
    if len(true_row) != 1:
        raise RuntimeError("True candidate missing or duplicated")
    true_score = float(true_row.iloc[0][score_column])
    all_scores = scores[score_column].to_numpy(float)
    rank = int(1 + np.sum(all_scores > true_score + 1.0e-15))
    tie_count = int(
        np.sum(
            np.isclose(
                all_scores,
                true_score,
                atol=1.0e-15,
                rtol=1.0e-12,
            )
        )
    )
    return rank, true_score, bool(rank == 1 and tie_count == 1)


shfa15_rank_rows: List[Dict[str, Any]] = []
for prediction in shfa15_predictions_public.itertuples(index=False):
    scores = shfa15_candidate_scores_public[
        (shfa15_candidate_scores_public["pipeline"] == prediction.pipeline)
        & (shfa15_candidate_scores_public["key_id"].astype(int) == int(prediction.key_id))
        & (
            shfa15_candidate_scores_public["target_sbox_index"].astype(int)
            == int(prediction.target_sbox_index)
        )
    ]
    true_guess = shfa15_true_nibbles[
        (int(prediction.key_id), int(prediction.target_sbox_index))
    ]
    true_rank, true_score, true_unique_rank1 = shfa15_rank_from_scores(
        scores,
        true_guess,
    )
    shfa15_rank_rows.append(
        {
            **prediction._asdict(),
            "true_key_guess": int(true_guess),
            "true_key_guess_hex": f"{true_guess:x}",
            "true_rank": int(true_rank),
            "true_score": float(true_score),
            "true_is_unique_rank1": bool(true_unique_rank1),
        }
    )
shfa15_rank_evaluation = pd.DataFrame(shfa15_rank_rows)

shfa15_prefix_rank_rows: List[Dict[str, Any]] = []
for prediction in shfa15_prefix_predictions_public.itertuples(index=False):
    scores = shfa15_prefix_scores_public[
        (shfa15_prefix_scores_public["pipeline"] == prediction.pipeline)
        & (shfa15_prefix_scores_public["key_id"].astype(int) == int(prediction.key_id))
        & (
            shfa15_prefix_scores_public["target_sbox_index"].astype(int)
            == int(prediction.target_sbox_index)
        )
        & (
            shfa15_prefix_scores_public["injection_checkpoint"].astype(int)
            == int(prediction.injection_checkpoint)
        )
    ]
    true_guess = shfa15_true_nibbles[
        (int(prediction.key_id), int(prediction.target_sbox_index))
    ]
    true_rank, true_score, true_unique_rank1 = shfa15_rank_from_scores(
        scores,
        true_guess,
    )
    shfa15_prefix_rank_rows.append(
        {
            **prediction._asdict(),
            "true_key_guess": int(true_guess),
            "true_rank": int(true_rank),
            "true_score": float(true_score),
            "true_is_unique_rank1": bool(true_unique_rank1),
        }
    )
shfa15_prefix_rank_evaluation = pd.DataFrame(shfa15_prefix_rank_rows)


def shfa15_stratified_bootstrap_indices(
    frame: pd.DataFrame,
    selected_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_indices = np.asarray(selected_indices, dtype=int)
    output: List[int] = []
    selected_sessions = frame.loc[selected_indices, "session_id"].astype(int)
    for session in sorted(selected_sessions.unique().tolist()):
        pool = selected_indices[
            selected_sessions.to_numpy(int) == int(session)
        ]
        if len(pool) == 0:
            continue
        sampled = rng.choice(pool, size=len(pool), replace=True)
        output.extend(int(value) for value in sampled)
    return np.asarray(output, dtype=int)


shfa15_bootstrap_rows: List[Dict[str, Any]] = []
for pipeline in _SHFA15_PIPELINES:
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            cache = shfa15_task_cache[(pipeline, key_id, target_index)]
            true_guess = shfa15_true_nibbles[(key_id, target_index)]
            for repetition in range(paper_shfa_config.bootstrap_repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_shfa_config.random_seed,
                            sum(ord(ch) for ch in pipeline),
                            key_id,
                            target_index,
                            repetition,
                            15031,
                        ]
                    )
                )
                effective_bootstrap = shfa15_stratified_bootstrap_indices(
                    cache["effective_frame"],
                    cache["effective_indices"],
                    rng,
                )
                ineffective_bootstrap = shfa15_stratified_bootstrap_indices(
                    cache["ineffective_frame"],
                    cache["ineffective_indices"],
                    rng,
                )
                scores = shfa15_scores_from_matrices(
                    cache["effective_matrix"],
                    cache["ineffective_matrix"],
                    effective_bootstrap,
                    ineffective_bootstrap,
                )
                prediction = shfa15_prediction(scores)
                true_rank, _, true_unique = shfa15_rank_from_scores(
                    scores,
                    true_guess,
                )
                shfa15_bootstrap_rows.append(
                    {
                        "pipeline": pipeline,
                        "key_id": key_id,
                        "target_sbox": f"S{target_index}",
                        "target_sbox_index": target_index,
                        "repetition": repetition,
                        "best_key_guess": int(prediction["best_key_guess"]),
                        "true_rank": int(true_rank),
                        "true_is_unique_rank1": bool(true_unique),
                    }
                )
shfa15_bootstrap = pd.DataFrame(shfa15_bootstrap_rows)
shfa15_bootstrap_summary = (
    shfa15_bootstrap.groupby(
        ["pipeline", "key_id", "target_sbox", "target_sbox_index"],
        as_index=False,
    )
    .agg(
        bootstrap_true_winner_fraction=("true_is_unique_rank1", "mean"),
        bootstrap_mean_true_rank=("true_rank", "mean"),
        bootstrap_median_true_rank=("true_rank", "median"),
    )
)
shfa15_rank_evaluation = shfa15_rank_evaluation.merge(
    shfa15_bootstrap_summary,
    on=["pipeline", "key_id", "target_sbox", "target_sbox_index"],
    how="left",
    validate="one_to_one",
)


def shfa15_random_same_session_budget(
    frame: pd.DataFrame,
    selected_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_sessions = frame.loc[selected_indices, "session_id"].astype(int)
    output: List[int] = []
    for session in sorted(selected_sessions.unique().tolist()):
        quota = int(np.sum(selected_sessions.to_numpy(int) == int(session)))
        pool = frame.index[
            frame["session_id"].astype(int) == int(session)
        ].to_numpy(int)
        if quota > len(pool):
            raise RuntimeError("Matched-random session quota exceeds branch pool")
        chosen = rng.choice(pool, size=quota, replace=False)
        output.extend(int(value) for value in chosen)
    return np.asarray(sorted(output), dtype=int)


# Matched-random controls preserve the selected count of EACH event branch and
# each session.  This is the direct test of Stage-10 ranking quality for SHFA.
shfa15_matched_random_rows: List[Dict[str, Any]] = []
for pipeline, spec in _SHFA15_PIPELINES.items():
    if not bool(spec["uses_stage10"]):
        continue
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            cache = shfa15_task_cache[(pipeline, key_id, target_index)]
            true_guess = shfa15_true_nibbles[(key_id, target_index)]
            model_scores = shfa15_candidate_scores_public[
                (shfa15_candidate_scores_public["pipeline"] == pipeline)
                & (shfa15_candidate_scores_public["key_id"].astype(int) == key_id)
                & (
                    shfa15_candidate_scores_public["target_sbox_index"].astype(int)
                    == target_index
                )
            ]
            model_rank, _, model_rank1 = shfa15_rank_from_scores(
                model_scores,
                true_guess,
            )
            random_ranks: List[int] = []
            random_rank1: List[bool] = []
            for repetition in range(paper_shfa_config.matched_random_repetitions):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            paper_shfa_config.random_seed,
                            sum(ord(ch) for ch in pipeline),
                            key_id,
                            target_index,
                            repetition,
                            15041,
                        ]
                    )
                )
                effective_random = shfa15_random_same_session_budget(
                    cache["effective_frame"],
                    cache["effective_indices"],
                    rng,
                )
                ineffective_random = shfa15_random_same_session_budget(
                    cache["ineffective_frame"],
                    cache["ineffective_indices"],
                    rng,
                )
                scores = shfa15_scores_from_matrices(
                    cache["effective_matrix"],
                    cache["ineffective_matrix"],
                    effective_random,
                    ineffective_random,
                )
                random_rank, _, random_unique = shfa15_rank_from_scores(
                    scores,
                    true_guess,
                )
                random_ranks.append(int(random_rank))
                random_rank1.append(bool(random_unique))
            median_random_rank = float(np.median(random_ranks))
            shfa15_matched_random_rows.append(
                {
                    "pipeline": pipeline,
                    "selector": str(spec["selector"]),
                    "campaign_arm": str(spec["campaign_arm"]),
                    "key_id": key_id,
                    "target_sbox": f"S{target_index}",
                    "target_sbox_index": target_index,
                    "model_true_rank": int(model_rank),
                    "model_rank1": bool(model_rank1),
                    "matched_random_median_true_rank": median_random_rank,
                    "matched_random_mean_true_rank": float(np.mean(random_ranks)),
                    "matched_random_rank1_fraction": float(np.mean(random_rank1)),
                    "model_better_than_random_median": bool(
                        model_rank < median_random_rank
                    ),
                    "model_not_worse_than_random_median": bool(
                        model_rank <= median_random_rank
                    ),
                }
            )
shfa15_matched_random = pd.DataFrame(shfa15_matched_random_rows)

# Branch-specific selection quality.  Effective and ineffective counts are kept
# separate; they are not treated as directly comparable sample currencies.
shfa15_labels_by_experiment = shfa15_private_labels.set_index("experiment_id")
shfa15_selection_quality_rows: List[Dict[str, Any]] = []
for pipeline in _SHFA15_PIPELINES:
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            cache = shfa15_task_cache[(pipeline, key_id, target_index)]
            row: Dict[str, Any] = {
                "pipeline": pipeline,
                "key_id": key_id,
                "target_sbox": f"S{target_index}",
                "target_sbox_index": target_index,
            }
            for branch_name, clean_category in (
                ("effective", "clean_target_effective"),
                ("ineffective", "clean_target_ineffective"),
            ):
                frame = cache[f"{branch_name}_frame"]
                selected_indices = cache[f"{branch_name}_indices"]
                experiment_ids = frame["experiment_id"].astype(int).to_numpy()
                selected_experiment_ids = frame.loc[
                    selected_indices,
                    "experiment_id",
                ].astype(int).to_numpy()
                all_categories = shfa15_labels_by_experiment.loc[
                    experiment_ids,
                    "category",
                ].astype(str).to_numpy()
                selected_categories = shfa15_labels_by_experiment.loc[
                    selected_experiment_ids,
                    "category",
                ].astype(str).to_numpy()
                actual_clean_total = int(np.sum(all_categories == clean_category))
                selected_clean = int(np.sum(selected_categories == clean_category))
                selected_count = int(len(selected_indices))
                precision = (
                    float(selected_clean / selected_count)
                    if selected_count > 0
                    else np.nan
                )
                recall = (
                    float(selected_clean / actual_clean_total)
                    if actual_clean_total > 0
                    else np.nan
                )
                row[f"actual_clean_{branch_name}_count"] = actual_clean_total
                row[f"selected_clean_{branch_name}_count"] = selected_clean
                row[f"clean_{branch_name}_precision"] = precision
                row[f"clean_{branch_name}_recall"] = recall
            shfa15_selection_quality_rows.append(row)
shfa15_selection_quality = pd.DataFrame(shfa15_selection_quality_rows)
shfa15_rank_evaluation = shfa15_rank_evaluation.merge(
    shfa15_selection_quality,
    on=["pipeline", "key_id", "target_sbox", "target_sbox_index"],
    how="left",
    validate="one_to_one",
)

# 8-bit key-pair evaluation.
shfa15_joint_8bit_rank_rows: List[Dict[str, Any]] = []
for prediction in shfa15_joint_8bit_predictions_public.itertuples(index=False):
    scores = shfa15_joint_8bit_scores_public[
        (shfa15_joint_8bit_scores_public["pipeline"] == prediction.pipeline)
        & (shfa15_joint_8bit_scores_public["key_id"].astype(int) == int(prediction.key_id))
    ]
    true_joint = shfa15_true_joint_8bit[int(prediction.key_id)]
    true_rank, true_score, true_unique = shfa15_rank_from_scores(
        scores,
        true_joint,
        score_column="joint_8bit_score",
        guess_column="joint_guess",
    )
    shfa15_joint_8bit_rank_rows.append(
        {
            **prediction._asdict(),
            "true_joint_guess": int(true_joint),
            "true_joint_guess_hex": f"{true_joint:02x}",
            "true_joint_score": float(true_score),
            "true_joint_rank": int(true_rank),
            "true_joint_is_unique_rank1": bool(true_unique),
        }
    )
shfa15_joint_8bit_rank_evaluation = pd.DataFrame(
    shfa15_joint_8bit_rank_rows
)


def shfa15_curve_metrics(
    prefix_frame: pd.DataFrame,
    pipeline: str,
    key_id: int,
    target_index: int,
) -> Dict[str, Any]:
    subset = prefix_frame[
        (prefix_frame["pipeline"] == pipeline)
        & (prefix_frame["key_id"].astype(int) == int(key_id))
        & (prefix_frame["target_sbox_index"].astype(int) == int(target_index))
    ].copy()
    lookup = {
        int(row.injection_checkpoint): row
        for row in subset.itertuples(index=False)
    }
    checkpoints = list(paper_shfa_config.injection_checkpoints)
    ranks = np.asarray(
        [
            int(lookup[checkpoint].true_rank)
            if checkpoint in lookup
            else 16
            for checkpoint in checkpoints
        ],
        dtype=float,
    )
    unique_rank1 = np.asarray(
        [
            bool(lookup[checkpoint].true_is_unique_rank1)
            if checkpoint in lookup
            else False
            for checkpoint in checkpoints
        ],
        dtype=bool,
    )
    loso = np.asarray(
        [
            float(lookup[checkpoint].loso_consensus)
            if checkpoint in lookup
            else 0.0
            for checkpoint in checkpoints
        ],
        dtype=float,
    )

    first_rank1 = np.nan
    for checkpoint, is_rank1 in zip(checkpoints, unique_rank1):
        if is_rank1:
            first_rank1 = float(checkpoint)
            break

    first_persistent_rank1 = np.nan
    first_stable_rank1 = np.nan
    for index, checkpoint in enumerate(checkpoints):
        if bool(np.all(unique_rank1[index:])):
            first_persistent_rank1 = float(checkpoint)
            if bool(np.all(loso[index:] >= paper_shfa_config.stable_loso_threshold)):
                first_stable_rank1 = float(checkpoint)
            break

    x_values = np.asarray([0] + checkpoints, dtype=float)
    y_values = np.concatenate([np.asarray([16.0]), ranks])
    integration = (
        np.trapezoid(y_values, x_values)
        if hasattr(np, "trapezoid")
        else np.trapz(y_values, x_values)
    )
    normalized_aurc = float(integration / float(checkpoints[-1]))
    return {
        "first_injection_checkpoint_rank1": first_rank1,
        "first_persistent_rank1_checkpoint": first_persistent_rank1,
        "first_stable_rank1_checkpoint": first_stable_rank1,
        "normalized_aurc": normalized_aurc,
    }


shfa15_curve_metric_rows: List[Dict[str, Any]] = []
for pipeline in _SHFA15_PIPELINES:
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            shfa15_curve_metric_rows.append(
                {
                    "pipeline": pipeline,
                    "key_id": key_id,
                    "target_sbox": f"S{target_index}",
                    "target_sbox_index": target_index,
                    **shfa15_curve_metrics(
                        shfa15_prefix_rank_evaluation,
                        pipeline,
                        key_id,
                        target_index,
                    ),
                }
            )
shfa15_curve_metrics_frame = pd.DataFrame(shfa15_curve_metric_rows)
shfa15_rank_evaluation = shfa15_rank_evaluation.merge(
    shfa15_curve_metrics_frame,
    on=["pipeline", "key_id", "target_sbox", "target_sbox_index"],
    how="left",
    validate="one_to_one",
)

shfa15_joint_summary = (
    shfa15_joint_8bit_rank_evaluation.groupby("pipeline", as_index=False)
    .agg(
        rank1_joint_8bit_count=("true_joint_is_unique_rank1", "sum"),
        mean_joint_true_rank=("true_joint_rank", "mean"),
    )
)
shfa15_pipeline_summary = (
    shfa15_rank_evaluation.groupby(["pipeline", "attack_type"], as_index=False)
    .agg(
        task_count=("true_rank", "size"),
        rank1_nibble_count=("true_is_unique_rank1", "sum"),
        rank1_nibble_success_rate=("true_is_unique_rank1", "mean"),
        mean_true_rank=("true_rank", "mean"),
        median_true_rank=("true_rank", "median"),
        mean_score_margin=("score_margin", "mean"),
        mean_loso_consensus=("loso_consensus", "mean"),
        mean_bootstrap_true_winner_fraction=(
            "bootstrap_true_winner_fraction",
            "mean",
        ),
        mean_selected_effective_count=("selected_effective_count", "mean"),
        mean_selected_ineffective_count=("selected_ineffective_count", "mean"),
        mean_clean_effective_precision=("clean_effective_precision", "mean"),
        mean_clean_effective_recall=("clean_effective_recall", "mean"),
        mean_clean_ineffective_precision=("clean_ineffective_precision", "mean"),
        mean_clean_ineffective_recall=("clean_ineffective_recall", "mean"),
        median_first_rank1_checkpoint=("first_injection_checkpoint_rank1", "median"),
        median_first_persistent_rank1_checkpoint=(
            "first_persistent_rank1_checkpoint",
            "median",
        ),
        median_first_stable_rank1_checkpoint=(
            "first_stable_rank1_checkpoint",
            "median",
        ),
        mean_normalized_aurc=("normalized_aurc", "mean"),
    )
    .merge(shfa15_joint_summary, on="pipeline", how="left", validate="one_to_one")
)

# Save all Stage-15R validation outputs before loading Stage-14R validation
# tables for cross-attack reporting.
shfa15_rank_evaluation.to_csv(
    shfa15_validation_directory / "final_nibble_true_rank_evaluation.csv",
    index=False,
)
shfa15_prefix_rank_evaluation.to_csv(
    shfa15_validation_directory / "injection_prefix_true_rank_curves.csv",
    index=False,
)
shfa15_bootstrap.to_csv(
    shfa15_validation_directory / "bootstrap_repetitions.csv",
    index=False,
)
shfa15_bootstrap_summary.to_csv(
    shfa15_validation_directory / "bootstrap_summary.csv",
    index=False,
)
shfa15_matched_random.to_csv(
    shfa15_validation_directory / "matched_branch_budget_model_vs_random.csv",
    index=False,
)
shfa15_selection_quality.to_csv(
    shfa15_validation_directory / "branch_selection_quality_after_freeze.csv",
    index=False,
)
shfa15_joint_8bit_rank_evaluation.to_csv(
    shfa15_validation_directory / "joint_8bit_true_rank_evaluation.csv",
    index=False,
)
shfa15_curve_metrics_frame.to_csv(
    shfa15_validation_directory / "persistent_rank1_and_aurc_metrics.csv",
    index=False,
)
shfa15_pipeline_summary.to_csv(
    shfa15_validation_directory / "shfa_pipeline_summary.csv",
    index=False,
)

print("=" * 100)
print("Stage 15R truth validation completed")
print("Public freeze remains            :", shfa15_public_attack_freeze["freeze_sha256"])
print("SHFA pipeline summary:")
ipy_display(shfa15_pipeline_summary)
print("=" * 100)

# %%
# ============================================================
# Stage 15R / Cell 4
# Equal-injection model-impact comparison and final summary
# ============================================================

# Stage-14R validation products are opened only now, after the SHFA public
# attack has been frozen.  They are used for reporting under the same total
# 8,000-attempt confirmation budget, not for selecting any SHFA policy.
shfa15_stage14_pipeline_summary = pd.read_csv(
    shfa15_stage14_validation_directory / "pipeline_summary.csv"
)
shfa15_stage14_prefix_ranks = pd.read_csv(
    shfa15_stage14_validation_directory / "injection_prefix_true_rank_curves.csv"
)
shfa15_stage14_rank_evaluation = pd.read_csv(
    shfa15_stage14_validation_directory / "final_nibble_true_rank_evaluation.csv"
)


def shfa15_external_curve_metrics(
    prefix_frame: pd.DataFrame,
    pipeline: str,
    key_id: int,
    target_index: int,
) -> Dict[str, Any]:
    subset = prefix_frame[
        (prefix_frame["pipeline"] == pipeline)
        & (prefix_frame["key_id"].astype(int) == int(key_id))
        & (prefix_frame["target_sbox_index"].astype(int) == int(target_index))
    ].copy()
    lookup = {
        int(row.injection_checkpoint): row
        for row in subset.itertuples(index=False)
    }
    checkpoints = list(paper_shfa_config.injection_checkpoints)
    ranks = np.asarray(
        [
            int(lookup[checkpoint].true_rank)
            if checkpoint in lookup
            else 16
            for checkpoint in checkpoints
        ],
        dtype=float,
    )
    unique_rank1 = np.asarray(
        [
            bool(lookup[checkpoint].true_is_unique_rank1)
            if checkpoint in lookup
            else False
            for checkpoint in checkpoints
        ],
        dtype=bool,
    )
    first_persistent = np.nan
    for index, checkpoint in enumerate(checkpoints):
        if bool(np.all(unique_rank1[index:])):
            first_persistent = float(checkpoint)
            break
    x_values = np.asarray([0] + checkpoints, dtype=float)
    y_values = np.concatenate([np.asarray([16.0]), ranks])
    return {
        "first_persistent_rank1_checkpoint": first_persistent,
        "normalized_aurc": float(
            (
                np.trapezoid(y_values, x_values)
                if hasattr(np, "trapezoid")
                else np.trapz(y_values, x_values)
            )
            / float(checkpoints[-1])
        ),
    }


shfa15_stage14_curve_rows: List[Dict[str, Any]] = []
for pipeline in sorted(shfa15_stage14_pipeline_summary["pipeline"].astype(str).unique()):
    for key_id in range(paper_shfa_config.number_of_keys):
        for target_index in paper_shfa_config.target_sbox_indices:
            shfa15_stage14_curve_rows.append(
                {
                    "pipeline": pipeline,
                    "key_id": key_id,
                    "target_sbox_index": target_index,
                    **shfa15_external_curve_metrics(
                        shfa15_stage14_prefix_ranks,
                        pipeline,
                        key_id,
                        target_index,
                    ),
                }
            )
shfa15_stage14_curve_metrics = pd.DataFrame(shfa15_stage14_curve_rows)
shfa15_stage14_curve_summary = (
    shfa15_stage14_curve_metrics.groupby("pipeline", as_index=False)
    .agg(
        median_first_persistent_rank1_checkpoint=(
            "first_persistent_rank1_checkpoint",
            "median",
        ),
        mean_normalized_aurc=("normalized_aurc", "mean"),
    )
)
shfa15_stage14_enhanced_summary = shfa15_stage14_pipeline_summary.merge(
    shfa15_stage14_curve_summary,
    on="pipeline",
    how="left",
    validate="one_to_one",
)

# Cross-attack table deliberately excludes effective/ineffective selected-count
# comparisons.  Every row uses the same total confirmation fault-attempt budget.
comparison_columns = [
    "pipeline",
    "attack_type",
    "rank1_nibble_count",
    "mean_true_rank",
    "median_true_rank",
    "rank1_joint_8bit_count",
    "mean_joint_true_rank",
    "mean_loso_consensus",
    "mean_bootstrap_true_winner_fraction",
    "median_first_persistent_rank1_checkpoint",
    "mean_normalized_aurc",
]
shfa15_equal_injection_comparison = pd.concat(
    [
        shfa15_stage14_enhanced_summary[comparison_columns],
        shfa15_pipeline_summary[comparison_columns],
    ],
    ignore_index=True,
).sort_values(["attack_type", "pipeline"]).reset_index(drop=True)
shfa15_equal_injection_comparison["total_fault_attempt_budget_per_task"] = (
    shfa15_confirmation_attempts
)


def shfa15_pipeline_metric(pipeline: str, column: str) -> float:
    row = shfa15_equal_injection_comparison[
        shfa15_equal_injection_comparison["pipeline"] == pipeline
    ]
    if len(row) != 1:
        raise RuntimeError(f"Missing or duplicated pipeline summary: {pipeline}")
    return float(row.iloc[0][column])


# Model-impact deltas are calculated WITHIN each attack.  They never compare
# effective and ineffective ciphertext counts as if they were the same sample.
shfa15_model_impact_rows: List[Dict[str, Any]] = []
for attack_type in ("SIFA", "SEFA", "SHFA"):
    prefix = attack_type.lower()
    random_raw = f"{prefix}_random_raw"
    random_threshold = f"{prefix}_random_threshold"
    random_budget = f"{prefix}_random_budget"
    guided_raw = f"{prefix}_guided_raw"
    guided_budget = f"{prefix}_guided_budget"
    for comparison_name, baseline, model_pipeline, model_stage in (
        ("stage11_guidance", random_raw, guided_raw, "Stage 11"),
        ("stage10_threshold", random_raw, random_threshold, "Stage 10"),
        ("stage10_budget", random_raw, random_budget, "Stage 10"),
        ("combined_models", random_raw, guided_budget, "Stage 11 + Stage 10"),
    ):
        shfa15_model_impact_rows.append(
            {
                "attack_type": attack_type,
                "comparison": comparison_name,
                "model_stage": model_stage,
                "baseline_pipeline": baseline,
                "model_pipeline": model_pipeline,
                "equal_total_fault_attempt_budget": shfa15_confirmation_attempts,
                "baseline_rank1_nibbles": int(
                    shfa15_pipeline_metric(baseline, "rank1_nibble_count")
                ),
                "model_rank1_nibbles": int(
                    shfa15_pipeline_metric(model_pipeline, "rank1_nibble_count")
                ),
                "rank1_nibble_gain": int(
                    shfa15_pipeline_metric(model_pipeline, "rank1_nibble_count")
                    - shfa15_pipeline_metric(baseline, "rank1_nibble_count")
                ),
                "baseline_mean_true_rank": shfa15_pipeline_metric(
                    baseline, "mean_true_rank"
                ),
                "model_mean_true_rank": shfa15_pipeline_metric(
                    model_pipeline, "mean_true_rank"
                ),
                "mean_true_rank_reduction": float(
                    shfa15_pipeline_metric(baseline, "mean_true_rank")
                    - shfa15_pipeline_metric(model_pipeline, "mean_true_rank")
                ),
                "baseline_bootstrap_stability": shfa15_pipeline_metric(
                    baseline, "mean_bootstrap_true_winner_fraction"
                ),
                "model_bootstrap_stability": shfa15_pipeline_metric(
                    model_pipeline, "mean_bootstrap_true_winner_fraction"
                ),
                "baseline_normalized_aurc": shfa15_pipeline_metric(
                    baseline, "mean_normalized_aurc"
                ),
                "model_normalized_aurc": shfa15_pipeline_metric(
                    model_pipeline, "mean_normalized_aurc"
                ),
                "normalized_aurc_reduction": float(
                    shfa15_pipeline_metric(baseline, "mean_normalized_aurc")
                    - shfa15_pipeline_metric(model_pipeline, "mean_normalized_aurc")
                ),
            }
        )
shfa15_model_impact_comparison = pd.DataFrame(shfa15_model_impact_rows)

shfa15_budget_controls = shfa15_matched_random[
    shfa15_matched_random["pipeline"] == "shfa_random_budget"
]
shfa15_stage11_helped = bool(
    (
        shfa15_pipeline_metric("shfa_guided_raw", "rank1_nibble_count")
        > shfa15_pipeline_metric("shfa_random_raw", "rank1_nibble_count")
    )
    or (
        shfa15_pipeline_metric("shfa_guided_raw", "mean_true_rank")
        < shfa15_pipeline_metric("shfa_random_raw", "mean_true_rank")
    )
)
shfa15_stage10_helped = bool(
    (
        shfa15_pipeline_metric("shfa_random_budget", "rank1_nibble_count")
        >= shfa15_pipeline_metric("shfa_random_raw", "rank1_nibble_count")
    )
    and (
        shfa15_pipeline_metric("shfa_random_budget", "mean_true_rank")
        <= shfa15_pipeline_metric("shfa_random_raw", "mean_true_rank")
    )
    and (
        int(
            shfa15_budget_controls[
                "model_not_worse_than_random_median"
            ].astype(bool).sum()
        )
        >= 3
    )
)
shfa15_combined_all_nibbles_rank1 = bool(
    shfa15_pipeline_metric("shfa_guided_budget", "rank1_nibble_count") == 4
)
shfa15_combined_all_joint_rank1 = bool(
    shfa15_pipeline_metric("shfa_guided_budget", "rank1_joint_8bit_count") == 2
)

shfa15_integrity_checks = {
    "stage14_public_attack_freeze_verified": True,
    "stage14_public_policy_freeze_verified": True,
    "stage14_frozen_scores_reproduced_from_public_campaign": True,
    "effective_and_ineffective_event_sets_disjoint": bool(
        not (
            shfa15_public_scored["paper_effective"]
            & shfa15_public_scored["paper_ineffective"]
        ).any()
    ),
    "confirmation_partition_only": True,
    "both_branch_policies_frozen_before_confirmation": True,
    "shfa_attack_frozen_before_truth_opened": True,
    "correct_ciphertexts_used_for_both_branches": True,
    "faulty_ciphertext_values_not_used_by_key_scorer": True,
    "no_stage10_probability_weighting_inside_shfa_score": True,
    "exact_joint_outer_product_sei_used": True,
    "joint_sei_algebraic_identity_checked": True,
    "all_public_tasks_identifiable": bool(
        np.all(shfa15_score_ranges.to_numpy(float) > 1.0e-15)
    ),
    "six_pipelines_four_nibble_tasks_each": bool(
        len(shfa15_rank_evaluation) == 6 * 4
    ),
    "six_pipelines_two_joint_tasks_each": bool(
        len(shfa15_joint_8bit_rank_evaluation) == 6 * 2
    ),
    "matched_random_controls_for_all_stage10_tasks": bool(
        len(shfa15_matched_random) == 4 * 4
    ),
    "locked_truth_hashes_verified": True,
    "cross_attack_comparison_uses_equal_total_injection_budget": bool(
        shfa15_equal_injection_comparison[
            "total_fault_attempt_budget_per_task"
        ].nunique()
        == 1
    ),
    "cross_attack_primary_table_excludes_event_count_comparison": bool(
        not any(
            "selected_effective" in column or "selected_ineffective" in column
            for column in shfa15_equal_injection_comparison.columns
        )
    ),
}
shfa15_all_integrity_checks_passed = bool(all(shfa15_integrity_checks.values()))

shfa15_equal_injection_comparison.to_csv(
    shfa15_validation_directory / "equal_injection_sifa_sefa_shfa_comparison.csv",
    index=False,
)
shfa15_model_impact_comparison.to_csv(
    shfa15_validation_directory / "within_attack_model_impact_comparison.csv",
    index=False,
)
shfa15_write_json(
    shfa15_validation_directory / "integrity_checks.json",
    {
        "all_integrity_checks_passed": shfa15_all_integrity_checks_passed,
        "checks": shfa15_integrity_checks,
    },
)

if paper_shfa_config.save_plots:
    fig, ax = plt.subplots(figsize=(11, 6))
    for pipeline, group in shfa15_prefix_rank_evaluation.groupby("pipeline"):
        curve = group.groupby("injection_checkpoint")["true_rank"].mean().sort_index()
        ax.plot(curve.index, curve.values, marker="o", label=pipeline)
    ax.set_xlabel("Total confirmation fault attempts per key/target/arm")
    ax.set_ylabel("Mean true-key nibble rank")
    ax.set_title("Stage 15R — SHFA rank convergence under equal injection budgets")
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(
        shfa15_validation_directory / "shfa_rank_convergence.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    impact_plot = shfa15_model_impact_comparison[
        shfa15_model_impact_comparison["comparison"].isin(
            ["stage11_guidance", "stage10_budget", "combined_models"]
        )
    ].copy()
    labels = (
        impact_plot["attack_type"].astype(str)
        + " / "
        + impact_plot["comparison"].astype(str)
    )
    positions = np.arange(len(impact_plot))
    ax.bar(positions, impact_plot["mean_true_rank_reduction"].to_numpy(float))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Reduction in mean true-key rank")
    ax.set_title("Model impact within each attack at 8,000 fault attempts")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        shfa15_validation_directory / "within_attack_model_rank_gain.png",
        dpi=180,
    )
    plt.close(fig)

shfa15_elapsed = time.perf_counter() - shfa15_started
shfa15_summary = {
    "stage": "15R",
    "attack": "paper-faithful Statistical Hybrid Fault Attack on LBlock",
    "run_id": shfa15_run_id,
    "run_directory": str(shfa15_run_directory),
    "input_stage14_run_directory": str(shfa15_input_directory),
    "random_seed": paper_shfa_config.random_seed,
    "total_confirmation_fault_attempts_per_key_target_arm": shfa15_confirmation_attempts,
    "new_fault_injections_performed": 0,
    "same_campaign_reason": (
        "SHFA is defined to use the disjoint effective and ineffective subsets collected concurrently from the same fault-attempt campaign."
    ),
    "paper_method": {
        "effective_set": "C_prime != C",
        "ineffective_set": "C_prime == C",
        "key_recovery_ciphertext": "correct C for both sets",
        "joint_distribution": "p_hat_e,k(x) * p_hat_i,k(y)",
        "primary_score": "sum_x sum_y (joint_hat_k(x,y)-1/256)^2",
        "stage10_probability_inside_score": False,
        "partial_decryption_target": "X31 nibble",
    },
    "frozen_branch_policies": {
        "effective_threshold": shfa15_effective_threshold,
        "ineffective_threshold": shfa15_ineffective_threshold,
        "effective_top_fraction": shfa15_effective_top_fraction,
        "ineffective_top_fraction": shfa15_ineffective_top_fraction,
    },
    "input_stage14_public_attack_freeze_sha256": str(
        shfa15_stage14_attack_manifest["freeze_sha256"]
    ),
    "public_attack_freeze_sha256": str(
        shfa15_public_attack_freeze["freeze_sha256"]
    ),
    "truth_opened_only_after_shfa_public_freeze": True,
    "all_integrity_checks_passed": shfa15_all_integrity_checks_passed,
    "integrity_checks": shfa15_integrity_checks,
    "stage11_helped_by_fixed_rule": shfa15_stage11_helped,
    "stage10_helped_by_fixed_rule": shfa15_stage10_helped,
    "guided_budget_all_nibbles_rank1": shfa15_combined_all_nibbles_rank1,
    "guided_budget_all_joint_8bit_rank1": shfa15_combined_all_joint_rank1,
    "shfa_pipeline_summary": shfa15_pipeline_summary.to_dict(orient="records"),
    "shfa_final_nibble_results": shfa15_rank_evaluation.to_dict(orient="records"),
    "shfa_joint_8bit_results": shfa15_joint_8bit_rank_evaluation.to_dict(orient="records"),
    "matched_random_controls": shfa15_matched_random.to_dict(orient="records"),
    "equal_injection_sifa_sefa_shfa_comparison": (
        shfa15_equal_injection_comparison.to_dict(orient="records")
    ),
    "within_attack_model_impact": shfa15_model_impact_comparison.to_dict(
        orient="records"
    ),
    "elapsed_seconds": float(shfa15_elapsed),
    "important_interpretation": (
        "The primary claim is the within-attack gain from Stage 11 and Stage 10 at an equal total fault-attempt budget. "
        "The effective and ineffective sample counts are reported only as branch diagnostics and are not used as a direct SIFA-versus-SEFA-versus-SHFA superiority metric."
    ),
}
shfa15_summary_path = shfa15_run_directory / "stage_15R_paper_shfa_summary.json"
shfa15_write_json(shfa15_summary_path, shfa15_summary)

print("\n" + "=" * 104)
print("Stage 15R completed — paper-faithful SHFA and equal-injection model comparison")
print("=" * 104)
print("Run directory                    :", shfa15_run_directory)
print("All integrity checks passed      :", shfa15_all_integrity_checks_passed)
print("Stage-11 helped fixed rule       :", shfa15_stage11_helped)
print("Stage-10 helped fixed rule       :", shfa15_stage10_helped)
print("Guided+budget all nibble Rank-1  :", shfa15_combined_all_nibbles_rank1)
print("Guided+budget all 8-bit Rank-1   :", shfa15_combined_all_joint_rank1)
print("Public attack freeze SHA-256     :", shfa15_public_attack_freeze["freeze_sha256"])
print("Summary file                     :", shfa15_summary_path)
print("Elapsed seconds                  :", f"{shfa15_elapsed:.3f}")
print("=" * 104)
ipy_display(shfa15_pipeline_summary)
ipy_display(
    shfa15_rank_evaluation[
        [
            "pipeline",
            "key_id",
            "target_sbox",
            "selected_effective_count",
            "selected_ineffective_count",
            "best_key_guess_hex",
            "true_key_guess_hex",
            "true_rank",
            "true_is_unique_rank1",
            "bootstrap_true_winner_fraction",
            "loso_consensus",
            "first_persistent_rank1_checkpoint",
            "first_stable_rank1_checkpoint",
            "normalized_aurc",
        ]
    ].rename(columns={"best_key_guess_hex": "predicted_key_guess_hex"})
)
ipy_display(shfa15_matched_random)
ipy_display(shfa15_model_impact_comparison)
ipy_display(shfa15_equal_injection_comparison)

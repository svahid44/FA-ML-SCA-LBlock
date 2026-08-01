# Stage 04 — Last-round S-box timing map

**Phase:** B  
**Canonical source run:** `stage04_20260718_171931_573552_seed20260718`

## Objective

Recover timing centers for logical S-box lanes S0–S7.

## Directory contents

- `code/`: canonical implementation.
- `artifacts/`: compact public contracts, configurations, figures, tables, and freeze records.
- `PROVENANCE.json`: source run ID and SHA-256 provenance.

## Data boundary

Large traces, full campaigns, serialized models, and private truth are not
stored in Git. See [`../../../data/DATA_ARCHIVE_MANIFEST.csv`](../../../data/DATA_ARCHIVE_MANIFEST.csv).

## Execution note

Run from the repository root. Stages 13–15 are Jupyter/VS Code cell exports and
reuse the validated Stage-12 kernel context used by the official experiment.

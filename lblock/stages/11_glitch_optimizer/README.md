# Stage 11 — Pre-injection glitch optimizer

**Phase:** C  
**Canonical source run:** `stage11_20260718_191029_290452_seed20260718`

## Objective

Recommend parameter vectors before injection for SIFA, SEFA, and SHFA.

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

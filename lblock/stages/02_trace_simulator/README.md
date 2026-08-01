# Stage 02 — Full internal trace simulator

**Phase:** A  
**Canonical source run:** `stage02_20260718_165358_459976_seed20260718`

## Objective

Generate deterministic healthy traces with one global trigger and separated public/private metadata.

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

# Stage 13 — Paper-faithful SIFA

**Phase:** D  
**Canonical source run:** `stage13R_v2_20260718_210424_794132_seed20260722`

## Objective

Recover selected last-round nibbles using ineffective events and identifiable X31 reconstruction.

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

## Canonical correction

Stage 13R-v2 rejects the permutation-invariant `X32 XOR key` adaptation and
reconstructs an identifiable X31 nibble. The v3 script adds the pre-frozen
budget-aware Stage-10 selection analysis while retaining exact unweighted SEI.

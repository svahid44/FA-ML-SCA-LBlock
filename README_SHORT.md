# Learning-Guided Statistical Fault Attacks on LBlock-64/80

This repository contains a simulation-based 15-stage workflow for learning-guided SIFA, SEFA, and SHFA on LBlock-64/80.

## Repository map

```text
lblock/stages/
├── 01_reference/          # LBlock implementation and test vectors
├── 02_trace_simulator/    # Healthy trace generation
├── 03_alignment/          # Trace alignment and final-round localization
├── 04_sbox_timing/        # Timing map for S0–S7
├── 05_target_selection/   # Selection of S0 and S5
├── 06_fault_engine/       # Parametric random-AND-4 fault model
├── 07_bias_capacity/      # SIFA/SEFA/SHFA bias analysis
├── 08_large_campaign/     # Large fault-injection campaign
├── 09_ml_dataset/         # ML dataset and leakage audit
├── 10_fault_classifier/   # Post-injection sample selection
├── 11_glitch_optimizer/   # Pre-injection parameter optimization
├── 12_closed_loop/        # Guided closed-loop campaign
├── 13_sifa/               # Paper-faithful SIFA
├── 14_sefa/               # Paper-faithful SEFA
└── 15_shfa/               # Paper-faithful SHFA

results/
├── official_summaries/    # Main JSON results
├── tables/                # Final CSV tables
└── figures/               # Final plots
```

Each stage contains:

```text
code/          # Main implementation
artifacts/     # Configurations, public outputs, tables, and plots
README.md      # Stage-specific notes
```

## How to run

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib threadpoolctl ipython jupyterlab
```

Run the stages in order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
                         ↓
                         09 → 10 → 11 → 12
                                      ↓
                                      13 → 14 → 15
```

For Stages 1–12, run the Python file inside each stage's `code/` directory.

Stages 13–15 are cell-based attack scripts. Run Stage 12 first, keep the same Jupyter or VS Code kernel active, and then execute the SIFA, SEFA, or SHFA cells.

## Where to see the results

Start with:

```text
results/official_summaries/
results/tables/
results/figures/
```

For stage-level details, open:

```text
lblock/stages/<stage_name>/artifacts/
```

## Recommended reviewer path

```text
Stage 01  → verify the cipher
Stage 06  → inspect the fault model
Stage 10  → inspect post-injection selection
Stage 11  → inspect pre-injection optimization
Stage 13  → inspect SIFA
Stage 14  → inspect SEFA
Stage 15  → inspect SHFA
results/  → inspect the final comparisons
```


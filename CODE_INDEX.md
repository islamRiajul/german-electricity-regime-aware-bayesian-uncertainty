# Code Index

Map of the standalone scripts in `src/` to the parts of the main notebook
(`German_Electricity_Uncertainity.ipynb`). Each script mirrors one notebook
section and can be run on its own thanks to the shipped intermediate data in
`data/`.

| Script (`src/`) | Notebook part | Purpose |
|-----------------|---------------|---------|
| `Setup & Imports.py` | Setup | Loads libraries, resolves the data folder |
| `Part 1 — Load Dataset.py` | 1 | Load raw SMARD day-ahead prices |
| `Part 2 — Preprocessing & Feature Engineering.py` | 2 | Clean data, engineer features |
| `Part 3 — Visualize Data.py` | 3 | Exploratory data dashboard |
| `Part 4 — Feature Selection.py` | 4 | Select predictive features |
| `Part 5 — Visualize Feature Selection.py` | 5 | Feature-selection plots |
| `Part 6 — Signed-Log Transform.py` | 6 | Variance-stabilizing transform |
| `Part 7 — Regime Detection.py` | 7 | Normal / Elevated / Crisis regimes |
| `Part 8 — Model Training & Optuna Tuning.py` | 8 | Train models, tune with Optuna |
| `Part 9 — Show Results Dashboard.py` | 9 | Prediction dashboard |
| `Part 10 — Result Analysis Dashboard.py` | 10 | Detailed result analysis |
| `Part 11 — Per-Regime Feature Analysis.py` | 11 | Feature importance by regime |
| `Part 12 — Battery Profit & Uncertainty-Aware Trading.py` | 12 | Battery valuation |
| `Part 13 — Three Model Improvements.py` | 13 | Model improvement experiments |
| `Part 14 — Advanced Upgrades & Full-History Battery.py` | 14 | Upgrades + full-history battery |
| `Part 15 — Scale-Aware Conformal Calibration.py` | 15 | Conformal calibration |
| `Part 15b — Multi-Level Conformal Calibration.py` | 15b | Multi-level calibration |
| `Part 16 — Calibrated Battery Simulation.py` | 16 | Battery with calibrated bands |
| `Part 17 — Variance-Stabilizing Transform Study.py` | 17 | signed-log vs asinh study |
| `Part 18 — Unified Model Comparison.py` | 18 | Unified comparison |
| `Part 19 — Final Model Evaluation & Battery Valuation.py` | 19 | Final evaluation |

Supporting modules:

| File | Purpose |
|------|---------|
| `german_epf_research.py` | Shared models + helper functions (imported by the notebook) |
| `vst_concepts.py` | Variance-stabilizing-transform helpers |

> The `part*.py` files (lowercase) are earlier script versions of the same steps,
> kept for reference.

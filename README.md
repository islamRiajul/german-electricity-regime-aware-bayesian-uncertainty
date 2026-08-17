# German Electricity Prices — Regime-Aware Bayesian Uncertainty Quantification[cite: 1]

Bayesian and deep-learning uncertainty quantification for the German (SMARD)
day-ahead electricity market, with a regime-aware battery-storage valuation that
turns forecast uncertainty into euros[cite: 1].

## 📊 Interactive Dashboards & Plots
View all 30 interactive forecast models and regime analysis charts live:
👉 [Open Interactive Plotly Dashboard](https://islamriajul.github.io/german-electricity-regime-aware-bayesian-uncertainty/)

## Highlights[cite: 1]

- **Heavy-tailed Student-t forecasting** — the final model reaches[cite: 1]
  **MAE ≈ €7.9/MWh** on the full 35-feature set (≈ €15.4 on the compact[cite: 1]
  15-feature set) by modelling price spikes instead of being surprised by them[cite: 1].
- **Multi-level conformal calibration** — cuts calibration error (MAACE) from[cite: 1]
  **~25% down to ~1%**, so the uncertainty bands are trustworthy[cite: 1].
- **Variance-stabilizing transforms** — a fair head-to-head of signed-log vs[cite: 1]
  asinh scaling across every model (best: robust-std → asinh)[cite: 1].
- **Regime-aware battery valuation** — well-calibrated uncertainty unlocks up to[cite: 1]
  **~€150M/year** of extra battery profit in the Crisis regime across Germany's[cite: 1]
  fleet[cite: 1].

## Results at a glance[cite: 1]

Final model — **Student-t + asinh transform + multi-level conformal[cite: 1]
calibration** (notebook Part 19)[cite: 1]:

| Feature set   | MAE (€/MWh) | RMSE  | PICP@90 | MPIW (med) | MAACE  |
|---------------|-------------|-------|---------|------------|--------|
| Compact (15)  | 15.44       | 22.39 | 90.3%   | 60.92      | 1.01%  |
| Full (35)     | **7.87**    | 14.66 | 92.7%   | 32.63      | 3.70%  |

*Lower MAE/RMSE = more accurate · PICP closest to 90% = best coverage ·[cite: 1]
lower MPIW = sharper bands · lower MAACE = better calibrated[cite: 1].*

The calibration step leaves accuracy untouched while collapsing the[cite: 1]
miscalibration: on the same Student-t model, MAACE drops from ~25% (raw) to[cite: 1]
~1% (multi-level conformal), and median interval width tightens from ~€195 to[cite: 1]
~€72[cite: 1].

## Market regimes[cite: 1]

Prices are labelled into three regimes, which drive both the analysis and the[cite: 1]
battery valuation[cite: 1]:

| Regime   | Hours   | Avg price | Mostly |
|----------|---------|-----------|--------|
| Normal   | 30,641  | €64/MWh   | 2020   |
| Elevated | 19,321  | €122/MWh  | 2025   |
| Crisis   | 5,357   | €266/MWh  | 2022   |

## Battery valuation[cite: 1]

Extra daily profit from uncertainty-aware ("smart") trading vs a naive baseline,[cite: 1]
using the calibrated final model, scaled to Germany's projected battery fleet[cite: 1]
(notebook Part 16)[cite: 1]:

| Regime   | Extra profit        | Annualized (fleet) |
|----------|---------------------|--------------------|
| Normal   | +€6.47 /MWh/day     | ≈ €23.6M/yr        |
| Elevated | +€22.29 /MWh/day    | ≈ €81.4M/yr        |
| Crisis   | +€40.98 /MWh/day    | ≈ €149.6M/yr       |

The better-calibrated the uncertainty, the more of this value is captured —[cite: 1]
especially in the high-price Crisis regime[cite: 1].

## Models compared[cite: 1]

| Model | Type |
|-------|------|
| DDNN | Gaussian deep distributional network |
| EvDNN | Deep evidential regression |
| VI-DDNN | Variational (Bayesian) network |
| VI+CP | VI-DDNN + conformal prediction |
| BSSM | Bayesian state-space (Kalman/EM) baseline |
| Student-t | Heavy-tailed final model (best) |

## Repository layout[cite: 1]

```[cite: 1]
.
├── German_Electricity_Uncertainity.ipynb   # main notebook (run top to bottom)
├── run_all.py                               # one-command full pipeline run
├── src/
│   ├── german_epf_research.py               # models + helpers (imported by the notebook)
│   ├── Part 1 — Load Dataset.py             # each notebook part as a standalone script
│   ├── ...                                  #   ... through Part 19
│   └── part*.py                             # earlier script versions of the same steps
├── data/
│   ├── Day-ahead_prices_*.csv               # raw SMARD day-ahead prices
│   ├── data_part2.pkl                       # pre-built engineered features
│   ├── data_part7_regimes.pkl               # pre-built regime labels
│   ├── features_part2.json
│   └── selected_part4.json
├── figures/                                 # exported plots (PNG + interactive HTML)
├── files/                                   # intermediate results (.pkl / .npy / .json)
├── HOW_TO_PUBLISH_WITH_PLOTS.md             # making plots visible on GitHub
├── CODE_INDEX.md                            # map of scripts to notebook parts
├── requirements.txt
├── LICENSE
└── .gitignore
```[cite: 1]

## Quick start[cite: 1]

```bash[cite: 1]
pip install -r requirements.txt
jupyter notebook German_Electricity_Uncertainity.ipynb
```[cite: 1]

Run the **Setup** cell first — it loads all libraries and auto-detects the data[cite: 1]
folder (checking `data/`, the Kaggle mount, then the current directory)[cite: 1].

**Ready-to-run data included.** The `data/` folder ships with pre-built[cite: 1]
`data_part2.pkl`, `data_part7_regimes.pkl`, `features_part2.json` and[cite: 1]
`selected_part4.json`, so you can run **any part standalone** without[cite: 1]
regenerating from scratch[cite: 1].

> **Note on the shipped data files:** these are a convenience reconstruction of[cite: 1]
> the preprocessing/feature-engineering steps, built from the raw SMARD CSV so[cite: 1]
> the notebook runs out of the box. For the exact pipeline, run the notebook[cite: 1]
> top-to-bottom (or `python run_all.py`), which regenerates them from the CSV[cite: 1].

### One-command full run[cite: 1]

```bash[cite: 1]
python run_all.py
```[cite: 1]

This executes the entire notebook top-to-bottom and writes[cite: 1]
`German_Electricity_Uncertainity.executed.ipynb` with all outputs[cite: 1].

> **Note:** the Setup cell adds `src/` to `sys.path`, so[cite: 1]
> `from german_epf_research import ...` works out of the box as long as you run[cite: 1]
> the notebook from the repository root[cite: 1].

## Data[cite: 1]

Day-ahead prices from the German electricity market (SMARD / Bundesnetzagentur),[cite: 1]
hourly, Jan 2020 – Apr 2026. See `data/`[cite: 1].

## Method & credits[cite: 1]

The approach builds on Lebedev, Das, Pappert & Schlüter (2026),[cite: 1]
*"Probabilistic Electricity Price Forecasting"* (IEEE Access, arXiv:2509.19417)[cite: 1].
This repository extends it with regime-aware, calibration-focused, and[cite: 1]
battery-valuation contributions[cite: 1].

## License[cite: 1]

MIT — see `LICENSE`[cite: 1].

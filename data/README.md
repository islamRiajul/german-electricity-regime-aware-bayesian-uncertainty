# Data

**File:** `Day-ahead_prices_202001010000_202605010000_Hour.csv`

- **Source:** [SMARD](https://www.smard.de/en) — Bundesnetzagentur (German Federal Network Agency)
- **Market:** Germany/Luxembourg day-ahead bidding zone
- **Resolution:** hourly
- **Span:** 2020-01-01 → 2026-04-30
- **Format:** semicolon-separated; European number format (comma = thousands);
  a title row precedes the header (the loader uses `skiprows=1`).

## Raw columns in the CSV
`Start date`, `End date`, `Germany/Luxembourg [€/MWh]` (the price target),
`grid load [MWh]`, `Residual load [MWh]`, `Total [MWh]`,
`Photovoltaics and wind [MWh]`, `Wind offshore [MWh]`, `Wind onshore [MWh]`,
`Photovoltaics [MWh]`, `Other [MWh]`, `Volume (+) [MWh]`, `Volume (-) [MWh]`,
`Price [€/MWh]`, `Net income [€]`.

## What the model actually uses
The target is the day-ahead price (`Germany/Luxembourg [€/MWh]`). The current
feature set is **derived from the price series itself** — autoregressive lags,
rolling statistics, and calendar encodings (see `features_part2.json`):

`lag_{1,2,3,6,12,24,48,168}`, `roll_mean_24`, `roll_std_24`, `roll_mean_168`,
`price_change`, and cyclical `hour/dow/month` sin–cos terms.

The load and generation columns above are present in the raw CSV and available
for extension, but are **not** part of the shipped feature set.

## Columns deliberately EXCLUDED (data leakage)
`Volume (+)`, `Volume (-)`, the balancing `Price`, and `Net income` — because
`Net income = Volume × Price` is circular with the target.

## License
Data © Bundesnetzagentur | SMARD.de. Redistributed here for research
reproducibility under SMARD's terms of use. Please cite SMARD as the source.

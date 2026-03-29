# TV-IFE Estimator

Structured pipeline for the router flagship-family sales forecasting project.

## What This Project Does

- Validates that the enhanced flagship file is a strict subset of the raw market file.
- Aggregates 77 flagship listings into brand-model families.
- Builds monthly family panels for the full months from July 2023 through June 2024.
- Runs three forecasting tracks:
  - `naive_last_month`
  - `direct_ridge`
  - `paper_inspired_factor`
- Produces reusable CSV outputs under `outputs/`.

## Project Layout

```text
data/
  router_market_raw.xlsx
  router_market_enhanced.xlsx
outputs/
src/tv_ife_estimator/
  cli.py
  config.py
  constants.py
  io.py
  preprocessing.py
  features.py
  backtest.py
  pipeline.py
  models/
```

## Data Assumptions

- `data/router_market_raw.xlsx` points to the raw market workbook.
- `data/router_market_enhanced.xlsx` points to the enhanced flagship workbook.
- The pipeline excludes `202407` from training because it is not a full month.

## Run

```bash
python -m tv_ife_estimator
```

or

```bash
tv-ife-estimator
```

## Main Outputs

- `outputs/family_lookup.csv`
- `outputs/family_static_profile.csv`
- `outputs/family_month_panel.csv`
- `outputs/direct_training_frame.csv`
- `outputs/paper_training_frame.csv`
- `outputs/forecast_eval.csv`
- `outputs/model_summary.csv`
- `outputs/next_month_forecast.csv`
- `outputs/paper_track_coefficients.csv`
- `outputs/paper_track_factors.csv`
- `outputs/paper_track_loadings.csv`
- `outputs/paper_track_ic.csv`
- `outputs/paper_track_fitted.csv`
- `outputs/paper_track_fit_summary.csv`
- `outputs/diagnostics.json`

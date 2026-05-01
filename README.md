# Revenue Forecasting — Competition Pipeline

Daily revenue forecasting using LightGBM + XGBoost ensemble with Vietnam-specific
calendar features, post-event decay signals, and Optuna hyperparameter tuning.

---

## Folder Structure

```
├── config.py               # All paths, model params, CV settings
├── data_loader.py          # Raw CSV loading + basic preprocessing
├── feature_engineering.py  # All feature creation (lag, rolling, calendar, external)
├── model.py                # LightGBM / XGBoost definitions + Optuna tuning
├── train.py                # CV training pipeline + ensemble blending
├── predict.py              # Inference + submission.csv generation
├── utils.py                # Metrics, CV fold builder, logging
├── main.py                 # Single entry point
│
├── /kaggle/input/          # Raw data (read-only)
│   ├── sales.csv
│   ├── sample_submission.csv
│   ├── web_traffic.csv
│   ├── promotions.csv
│   ├── inventory.csv
│   ├── returns.csv
│   ├── orders.csv
│   └── customers.csv
│
└── /kaggle/working/        # All outputs
    ├── features_train_v3.csv
    ├── features_predict_v3.csv
    ├── feature_meta_v3.json
    └── submission.csv
```

---

## How to Run

### Step 1 — Build features (one-time, ~5 min)

```bash
python feature_engineering.py
```

Outputs `features_train_v3.csv`, `features_predict_v3.csv`, `feature_meta_v3.json`
to `/kaggle/working/`.

### Step 2 — Train + predict

```bash
python main.py
```

This runs:
1. Load pre-built feature CSVs
2. Add extra features (post-event decay, COVID flag, smooth seasonal)
3. Time-series CV (5 folds, expanding window)
4. Train LightGBM + XGBoost
5. Find optimal blend weight via RMSE minimisation
6. Retrain on full data
7. Generate `submission.csv`

### Optional — Optuna hyperparameter tuning (~40 min on Kaggle P100)

Set `USE_OPTUNA = True` in `config.py`, then run `python main.py`.

---

## Model Description

| Component | Detail |
|---|---|
| Base learners | LightGBM (gbdt) + XGBoost (gbtree) |
| Target | log1p(Revenue), expm1 at inference |
| Ensemble | Scalar blend weight optimised on OOF RMSE |
| CV scheme | Expanding window, 5 folds, 90-day gap, 182-day val |
| COGS estimate | Monthly seasonal Revenue/COGS ratio (not global) |

### Key Feature Groups

- **Calendar** — day, week, month, quarter, Fourier (annual + weekly)
- **YoY lags** — lag_365, lag_728, rolling means/stds around same day last year
- **Year-ratio adjustment** — scales lag_365 by year-over-year growth trend
- **Vietnam holidays** — Tết proximity, fixed holidays, shopping events (11/11, 12/12)
- **Post-event decay** — exponential decay signal after major events (fixes post-Tết over-prediction)
- **COVID structural break** — YoY ratio lag, trend acceleration flag
- **External signals** — web traffic, promotions, inventory fill rate, returns, customer orders

### Known Error Patterns Fixed

| Issue | Fix |
|---|---|
| Post-holiday over-prediction | `days_after_event`, `post_event_decay` |
| COVID 2020 MAPE=38% | `yoy_ratio_lag`, `trend_accel`, `is_covid_period` |
| Tết / Dec volatility | `rev_lag365_roll60`, `rev_same_weekday_ly` |
| COGS seasonal bias | Monthly COGS ratio instead of 10-year global |

---

## Baseline Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LGBM tuned | 1,190,163 | 860,272 | 0.8141 |
| XGB tuned | 1,186,363 | 850,231 | 0.8153 |
| Ensemble | 1,181,298 | 849,264 | 0.8168 |

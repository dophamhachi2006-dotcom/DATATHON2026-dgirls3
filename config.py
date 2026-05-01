"""
config.py — Central configuration for all paths, model params, and CV settings.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/kaggle/input/datathon-2026-round-1/"
WORK_DIR = "/kaggle/working/"

RAW_FILES = {
    "sales":       DATA_DIR + "sales.csv",
    "submission":  DATA_DIR + "sample_submission.csv",
    "web_traffic": DATA_DIR + "web_traffic.csv",
    "promotions":  DATA_DIR + "promotions.csv",
    "inventory":   DATA_DIR + "inventory.csv",
    "returns":     DATA_DIR + "returns.csv",
    "orders":      DATA_DIR + "orders.csv",
    "customers":   DATA_DIR + "customers.csv",
}

FEATURE_TRAIN_FILE = WORK_DIR + "features_train.csv"
FEATURE_PRED_FILE  = WORK_DIR + "features_pred.csv"
FEATURE_META_FILE  = WORK_DIR + "feature_meta.json"
SUBMISSION_FILE    = WORK_DIR + "submission.csv"

# ── Target & date column ──────────────────────────────────────────────────────
TARGET     = "Revenue"
DATE_COL   = "Date"
RANDOM_SEED = 42

# ── Time-series CV settings ───────────────────────────────────────────────────
CV = {
    "n_folds":        5,
    "gap_days":       90,   # gap between train end and val start (avoid leakage)
    "val_days":       182,  # ~6 months per validation window
    "min_train_days": 730,  # ~2 years minimum training history
}

# ── LightGBM parameters ───────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "boosting_type":     "gbdt",
    "verbose":           -1,
    "num_leaves":        63,
    "min_child_samples": 50,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_estimators":      2000,
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,
}

# ── XGBoost parameters ────────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "booster":          "gbtree",
    "max_depth":        5,
    "min_child_weight": 50,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "alpha":            0.1,
    "lambda":           1.0,
    "n_estimators":     2000,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbosity":        0,
}

EARLY_STOPPING = 100

# ── Optuna settings ───────────────────────────────────────────────────────────
OPTUNA = {
    "n_trials_lgbm": 50,
    "n_trials_xgb":  30,
}

# ── Feature column list ───────────────────────────────────────────────────────
# Defined here as the authoritative source; feature_engineering.py will
# filter to only columns that actually exist in the DataFrame.
FEATURE_COLS = [
    # Calendar
    "year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "is_month_end", "is_month_start", "is_quarter_end", "dom_bin",
    "trend", "days_to_payday", "is_payday_window",
    # Fourier seasonality
    "sin_365d_k1", "cos_365d_k1", "sin_365d_k2", "cos_365d_k2",
    "sin_365d_k3", "cos_365d_k3",
    "sin_7d_k1",   "cos_7d_k1",   "sin_7d_k2",   "cos_7d_k2",
    "sin_7d_k3",   "cos_7d_k3",
    # YoY lags & rolling
    "rev_lag_365", "rev_lag_364", "rev_lag_728", "cogs_lag_365",
    "log_rev_lag_365", "log_monthly_avg", "monthly_avg_rev",
    # Year-scale adjustment
    "year_scale_ratio", "rev_lag365_scaled", "rev_lag365_roll7_scaled",
    "rev_lag365_roll30_scaled", "log_rev_lag365_scaled", "lag365_scale_deviation",
    # Rolling stats on lag
    "rev_lag365_roll7", "rev_lag365_roll14", "rev_lag365_roll30",
    "rev_lag365_std7",  "rev_lag365_std30",  "rev_mom_yoy",
    # Smooth seasonal
    "rev_lag365_roll60", "rev_same_weekday_ly", "seasonal_cv",
    # Vietnam holidays
    "days_to_tet", "pre_tet_7d", "pre_tet_14d", "post_tet_7d",
    "is_fixed_holiday", "is_tet_holiday",
    "is_black_friday_month", "is_1111", "is_1212", "is_0808",
    # Post-event decay
    "days_after_event", "is_post_event_3d", "is_post_event_7d", "is_post_event_14d",
    "post_event_decay", "days_to_next_event", "pre_event_surge",
    # Structural breaks
    "is_covid_period", "yoy_ratio_lag", "log_yoy_ratio_lag",
    "trend_accel", "trend_accel_norm",
    # Web traffic
    "sessions_roll7", "sessions_roll14", "sessions_lag_365",
    "visitors_lag_365", "traffic_momentum",
    # Promotions
    "n_active_promos", "max_promo_discount", "has_stackable_promo",
    "promo_is_percentage", "promo_online_channel", "days_since_promo_end",
    # Inventory
    "avg_fill_rate", "stockout_rate", "avg_days_of_supply", "avg_sell_through",
    "fill_rate_lag_1y", "stockout_rate_lag_1y",
    # Returns
    "return_amount_roll7", "return_amount_roll30", "return_qty_roll7",
    "return_rate_roll7", "return_amount_lag_365",
    # Customer behaviour
    "orders_roll7", "orders_roll30",
    "new_cust_ratio_roll7", "new_cust_ratio_roll30",
    "new_cust_ratio_lag14", "new_cust_ratio_lag21", "orders_lag_365",
]

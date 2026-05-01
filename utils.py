"""
utils.py
--------
Shared helper utilities: metrics, CV fold generation, logging.
"""

import logging
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt= "%H:%M:%S",
    )
    return logging.getLogger(name)


# ── Metrics ───────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — robust to low-revenue days."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mape_above_percentile(y_true: np.ndarray, y_pred: np.ndarray, pct: int = 25) -> float:
    """MAPE computed only on days where revenue exceeds the given percentile.
    Filters out post-holiday low-revenue days that inflate standard MAPE."""
    threshold = np.percentile(y_true, pct)
    mask = y_true > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse"    : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae"     : float(mean_absolute_error(y_true, y_pred)),
        "r2"      : float(r2_score(y_true, y_pred)),
        "smape"   : smape(y_true, y_pred),
        "mape_p25": mape_above_percentile(y_true, y_pred, pct=25),
    }


def print_metrics(m: dict, label: str = "") -> None:
    prefix = f"{label}: " if label else ""
    print(
        f"{prefix}"
        f"RMSE={m['rmse']:>12,.0f}  "
        f"MAE={m['mae']:>12,.0f}  "
        f"R²={m['r2']:.4f}  "
        f"SMAPE={m['smape']:.2f}%  "
        f"MAPE>P25={m['mape_p25']:.2f}%"
    )


# ── Time-series CV ────────────────────────────────────────────────────────────

def make_time_series_folds(df, date_col, n_folds, gap_days, val_days, min_train_days):
    """
    Expanding-window time-series cross-validation.

    Schema for each fold:
        [0 .. train_end)  →  gap  →  [val_start .. val_end)

    The gap prevents data leakage from lag features (e.g. lag_90 needs a
    90-day buffer between train end and validation start).
    """
    df        = df.sort_values(date_col).reset_index(drop=True)
    total     = len(df)
    usable_end= total - val_days - gap_days
    step      = max(1, (usable_end - min_train_days) // n_folds)

    folds = []
    for i in range(n_folds):
        train_end = min_train_days + i * step
        val_start = train_end + gap_days
        val_end   = val_start + val_days
        if val_end > total:
            break
        folds.append((list(range(0, train_end)), list(range(val_start, val_end))))

    return folds

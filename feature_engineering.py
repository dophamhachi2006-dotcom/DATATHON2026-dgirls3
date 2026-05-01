"""
feature_engineering.py — All feature creation for the revenue forecasting pipeline.

All rolling/lag operations use .shift(n) before .rolling() to ensure
no future data leaks into any row.
"""

import json
import numpy as np
import pandas as pd

from config import DATE_COL, FEATURE_COLS


# ── Tet dates (lunar new year) for 2013-2024 ─────────────────────────────────
TET_DATES = pd.to_datetime([
    "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08",
    "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25",
    "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
])


def _fixed_holidays(years: range) -> pd.DatetimeIndex:
    dates = []
    for y in years:
        dates += [
            f"{y}-01-01", f"{y}-04-10", f"{y}-04-30",
            f"{y}-05-01", f"{y}-09-02",
        ]
    return pd.to_datetime(dates)


# ── Individual feature groups ─────────────────────────────────────────────────

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df[DATE_COL].dt
    df["year"]           = d.year
    df["month"]          = d.month
    df["day"]            = d.day
    df["dayofweek"]      = d.dayofweek
    df["dayofyear"]      = d.dayofyear
    df["weekofyear"]     = d.isocalendar().week.astype(int)
    df["quarter"]        = d.quarter
    df["is_weekend"]     = (d.dayofweek >= 5).astype(int)
    df["is_month_end"]   = d.is_month_end.astype(int)
    df["is_month_start"] = d.is_month_start.astype(int)
    df["is_quarter_end"] = d.is_quarter_end.astype(int)
    df["dom_bin"]        = pd.cut(df["day"], bins=[0, 10, 20, 31], labels=[0, 1, 2]).astype(int)
    df["trend"]          = (df[DATE_COL] - df[DATE_COL].min()).dt.days
    # Payday proximity: distance to the 28th of each month
    df["days_to_payday"]   = df["day"].apply(lambda d: min(abs(d - 28), 28 - abs(d - 28) + 3))
    df["is_payday_window"] = ((df["day"] >= 25) | (df["day"] <= 5)).astype(int)
    return df


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
    """Annual (period=365) and weekly (period=7) Fourier terms."""
    for period, col, n in [(365, "dayofyear", 3), (7, "dayofweek", 3)]:
        t = df[col].values
        for k in range(1, n + 1):
            df[f"sin_{period}d_k{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"cos_{period}d_k{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """YoY lags and rolling statistics — all anchored at lag 365+ to avoid leakage."""
    df = df.copy()
    rev = df["Revenue"]

    df["rev_lag_365"]  = rev.shift(365)
    df["rev_lag_364"]  = rev.shift(364)
    df["rev_lag_728"]  = rev.shift(728)
    df["cogs_lag_365"] = df["COGS"].shift(365)

    lag = rev.shift(365)
    df["rev_lag365_roll7"]  = lag.rolling(7,  center=True, min_periods=1).mean()
    df["rev_lag365_roll14"] = lag.rolling(14, center=True, min_periods=1).mean()
    df["rev_lag365_roll30"] = lag.rolling(30, center=True, min_periods=1).mean()
    df["rev_lag365_roll60"] = lag.rolling(60, center=True, min_periods=1).mean()
    df["rev_lag365_std7"]   = lag.rolling(7,  center=True, min_periods=1).std()
    df["rev_lag365_std30"]  = lag.rolling(30, center=True, min_periods=1).std()
    df["rev_mom_yoy"]       = df["rev_lag365_roll7"] / (df["rev_lag365_roll30"] + 1e-6)

    # Smooth same-weekday alignment: average of ±1 day offsets around lag 365
    df["rev_same_weekday_ly"] = (
        rev.shift(364) * 0.34 + rev.shift(365) * 0.33 + rev.shift(366) * 0.33
    )

    # Month-of-year average from prior year (shift year +1 to avoid leakage)
    monthly_avg = (
        df[rev.notna()]
        .groupby(["year", "month"])["Revenue"]
        .mean().reset_index()
        .rename(columns={"Revenue": "monthly_avg_rev"})
    )
    monthly_avg["year"] = monthly_avg["year"] + 1
    df = df.merge(monthly_avg, on=["year", "month"], how="left")

    df["log_rev_lag_365"] = np.log1p(df["rev_lag_365"])
    df["log_monthly_avg"] = np.log1p(df["monthly_avg_rev"])

    # Seasonal coefficient of variation (volatility signal)
    monthly_std = (
        df[rev.notna()].groupby("month")["Revenue"]
        .std().rename("monthly_revenue_std")
    )
    df = df.merge(monthly_std.reset_index(), on="month", how="left")
    df["seasonal_cv"] = df["monthly_revenue_std"] / (df["rev_lag_365"] + 1e-6)

    return df


def add_year_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale lag-365 features by the trailing YoY growth ratio so the model
    can adapt to long-term revenue trends without seeing current-year targets.
    """
    df = df.copy()
    annual_mean = (
        df[df["Revenue"].notna()].groupby("year")["Revenue"].mean().sort_index()
    )
    yoy_ratio       = annual_mean / annual_mean.shift(1)
    expanding_ratio = yoy_ratio.expanding(min_periods=2).mean()
    ratio_map       = expanding_ratio.shift(1).to_dict()  # lag 1 year → no leakage

    df["year_scale_ratio"]          = df["year"].map(ratio_map).fillna(1.0)
    df["rev_lag365_scaled"]         = df["rev_lag_365"]      * df["year_scale_ratio"]
    df["rev_lag365_roll7_scaled"]   = df["rev_lag365_roll7"] * df["year_scale_ratio"]
    df["rev_lag365_roll30_scaled"]  = df["rev_lag365_roll30"]* df["year_scale_ratio"]
    df["log_rev_lag365_scaled"]     = np.log1p(df["rev_lag365_scaled"])
    df["lag365_scale_deviation"]    = df["rev_lag_365"] - df["rev_lag365_scaled"]
    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    fixed = _fixed_holidays(range(2012, 2025))
    df = df.copy()
    df["days_to_tet"]  = df[DATE_COL].apply(
        lambda d: min(abs((d - t).days) for t in TET_DATES)
    )
    df["pre_tet_7d"]   = (df["days_to_tet"] <= 7).astype(int)
    df["pre_tet_14d"]  = ((df["days_to_tet"] <= 14) & (df["days_to_tet"] > 7)).astype(int)
    df["post_tet_7d"]  = df[DATE_COL].apply(
        lambda d: int(any(0 < (d - t).days <= 7 for t in TET_DATES))
    )
    df["is_fixed_holiday"]      = df[DATE_COL].isin(fixed).astype(int)
    df["is_tet_holiday"]        = df[DATE_COL].isin(TET_DATES).astype(int)
    df["is_black_friday_month"] = ((df["month"] == 11) & (df["day"] >= 20)).astype(int)
    df["is_1111"] = ((df["month"] == 11) & (df["day"] == 11)).astype(int)
    df["is_1212"] = ((df["month"] == 12) & (df["day"] == 12)).astype(int)
    df["is_0808"] = ((df["month"] ==  8) & (df["day"] ==  8)).astype(int)
    return df


def add_post_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-event demand drop features.
    Revenue typically falls sharply in the 1–7 days after major shopping events.
    """
    shopping_events = pd.to_datetime([
        *[f"{y}-11-11" for y in range(2013, 2025)],
        *[f"{y}-12-12" for y in range(2013, 2025)],
        *[f"{y}-08-08" for y in range(2013, 2025)],
        *[f"{y}-01-01" for y in range(2013, 2025)],
        *[f"{y}-09-02" for y in range(2013, 2025)],
    ])
    all_events = pd.DatetimeIndex(list(TET_DATES) + list(shopping_events)).sort_values()

    df = df.copy()

    def _days_after(date):
        past = all_events[all_events <= date]
        return (date - past[-1]).days if len(past) else 999

    def _days_to_next(date):
        future = all_events[all_events >= date]
        return (future[0] - date).days if len(future) else 999

    df["days_after_event"]  = df[DATE_COL].apply(_days_after)
    df["days_to_next_event"]= df[DATE_COL].apply(_days_to_next)

    df["is_post_event_3d"]  = (df["days_after_event"] <= 3).astype(int)
    df["is_post_event_7d"]  = (df["days_after_event"] <= 7).astype(int)
    df["is_post_event_14d"] = (df["days_after_event"] <= 14).astype(int)

    # Exponential decay weight (peaks at 1.0 immediately after event)
    df["post_event_decay"] = np.exp(-df["days_after_event"] / 7.0)
    df.loc[df["days_after_event"] > 21, "post_event_decay"] = 0.0

    df["pre_event_surge"] = np.exp(-df["days_to_next_event"] / 5.0)
    df.loc[df["days_to_next_event"] > 14, "pre_event_surge"] = 0.0

    return df


def add_structural_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """COVID-period flag and YoY trend acceleration from lagged data only."""
    df = df.copy()
    df["is_covid_period"] = (
        (df[DATE_COL] >= "2020-01-01") & (df[DATE_COL] <= "2021-06-30")
    ).astype(int)

    if "rev_lag_365" in df.columns and "rev_lag_728" in df.columns:
        df["yoy_ratio_lag"]     = (df["rev_lag_365"] / (df["rev_lag_728"] + 1e-6)).clip(0.3, 3.0)
        df["log_yoy_ratio_lag"] = np.log(df["yoy_ratio_lag"])
        df["trend_accel"]       = df["rev_lag_365"] - df["rev_lag_728"]
        df["trend_accel_norm"]  = df["trend_accel"] / (df["rev_lag_728"] + 1e-6)
    return df


def add_traffic(df: pd.DataFrame, web_traffic: pd.DataFrame) -> pd.DataFrame:
    daily = (
        web_traffic.groupby("date")
        .agg(sessions=("sessions", "sum"), unique_visitors=("unique_visitors", "sum"))
        .reset_index().rename(columns={"date": DATE_COL})
    )
    df = df.merge(daily, on=DATE_COL, how="left")

    # All rolling features use .shift(1) so today's traffic cannot leak into today's revenue
    s = df["sessions"].shift(1)
    df["sessions_roll7"]    = s.rolling(7,  min_periods=1).mean()
    df["sessions_roll14"]   = s.rolling(14, min_periods=1).mean()
    df["sessions_lag_365"]  = df["sessions"].shift(365)
    df["visitors_lag_365"]  = df["unique_visitors"].shift(365)
    df["traffic_momentum"]  = (
        df["sessions"].shift(365).rolling(7,  min_periods=1).mean() /
        (df["sessions"].shift(365).rolling(30, min_periods=1).mean() + 1e-6)
    )
    return df


def add_promotions(df: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_active_promos"]     = 0
    df["max_promo_discount"]  = 0.0
    df["has_stackable_promo"] = 0
    df["promo_is_percentage"] = 0
    df["promo_online_channel"]= 0

    for _, promo in promotions.iterrows():
        mask = (df[DATE_COL] >= promo["start_date"]) & (df[DATE_COL] <= promo["end_date"])
        df.loc[mask, "n_active_promos"]   += 1
        df.loc[mask, "max_promo_discount"] = df.loc[mask, "max_promo_discount"].clip(
            lower=promo.get("discount_value", 0)
        )
        if promo.get("stackable_flag", 0):
            df.loc[mask, "has_stackable_promo"] = 1
        if str(promo.get("promo_type", "")).lower() == "percentage":
            df.loc[mask, "promo_is_percentage"] = 1
        if str(promo.get("promo_channel", "")).lower() in ["online", "email"]:
            df.loc[mask, "promo_online_channel"] = 1

    # Days since last promotion ended (capped at 60; 999 = no prior promo)
    df["days_since_promo_end"] = np.nan
    last_end = pd.NaT
    for idx, row in df.iterrows():
        if len(promotions[promotions["end_date"] == row[DATE_COL]]) > 0:
            last_end = row[DATE_COL]
        if pd.notna(last_end) and row["n_active_promos"] == 0:
            df.loc[idx, "days_since_promo_end"] = (row[DATE_COL] - last_end).days
    df["days_since_promo_end"] = df["days_since_promo_end"].fillna(999).clip(upper=60)
    return df


def add_inventory(df: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        inventory.groupby(["year", "month"])
        .agg(
            avg_fill_rate     =("fill_rate",        "mean"),
            stockout_rate     =("stockout_flag",    "mean"),
            avg_days_of_supply=("days_of_supply",   "mean"),
            avg_sell_through  =("sell_through_rate","mean"),
        )
        .reset_index()
    )
    df = df.merge(monthly, on=["year", "month"], how="left")

    # Lag by one year for the same month to avoid leakage into the test period
    monthly_lag = monthly.copy()
    monthly_lag["year"] = monthly_lag["year"] + 1
    monthly_lag = monthly_lag.rename(columns={
        "avg_fill_rate":      "fill_rate_lag_1y",
        "stockout_rate":      "stockout_rate_lag_1y",
        "avg_days_of_supply": "days_supply_lag_1y",
        "avg_sell_through":   "sell_through_lag_1y",
    })
    df = df.merge(monthly_lag, on=["year", "month"], how="left")
    return df


def add_returns(df: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    daily_ret = (
        returns.groupby("return_date")
        .agg(
            daily_refund_amount=("refund_amount",   "sum"),
            daily_return_qty   =("return_quantity", "sum"),
            n_returns          =("return_id",       "count"),
        )
        .reset_index().rename(columns={"return_date": DATE_COL})
    )
    df = df.merge(daily_ret, on=DATE_COL, how="left")
    df[["daily_refund_amount", "daily_return_qty", "n_returns"]] = (
        df[["daily_refund_amount", "daily_return_qty", "n_returns"]].fillna(0)
    )

    ref = df["daily_refund_amount"].shift(1)
    df["return_amount_roll7"]  = ref.rolling(7,  min_periods=1).mean()
    df["return_amount_roll30"] = ref.rolling(30, min_periods=1).mean()
    df["return_qty_roll7"]     = df["daily_return_qty"].shift(1).rolling(7, min_periods=1).mean()
    df["return_rate_roll7"]    = (
        ref.rolling(7, min_periods=1).sum() /
        (df["Revenue"].shift(1).rolling(7, min_periods=1).sum() + 1e-6)
    )
    df["return_amount_lag_365"] = df["daily_refund_amount"].shift(365)
    return df


def add_customers(df: pd.DataFrame, orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    ord_cust = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id", how="left")
    ord_cust["is_new_customer"] = (
        (ord_cust["order_date"] - ord_cust["signup_date"]).dt.days <= 30
    ).astype(int)

    daily_orders = (
        ord_cust.groupby("order_date")
        .agg(
            total_orders      =("order_id",        "count"),
            new_customer_orders=("is_new_customer","sum"),
            unique_customers  =("customer_id",     "nunique"),
        )
        .reset_index().rename(columns={"order_date": DATE_COL})
    )
    daily_orders["new_customer_ratio"] = (
        daily_orders["new_customer_orders"] / daily_orders["total_orders"]
    )
    df = df.merge(daily_orders, on=DATE_COL, how="left")

    to = df["total_orders"].shift(1)
    ncr = df["new_customer_ratio"].shift(1)
    df["orders_roll7"]          = to.rolling(7,  min_periods=1).mean()
    df["orders_roll30"]         = to.rolling(30, min_periods=1).mean()
    df["new_cust_ratio_roll7"]  = ncr.rolling(7,  min_periods=1).mean()
    df["new_cust_ratio_roll30"] = ncr.rolling(30, min_periods=1).mean()
    df["new_cust_ratio_lag14"]  = df["new_customer_ratio"].shift(14)
    df["new_cust_ratio_lag21"]  = df["new_customer_ratio"].shift(21)
    df["orders_lag_365"]        = df["total_orders"].shift(365)
    return df


# ── Null imputation ───────────────────────────────────────────────────────────

def _fill_lag_fallbacks(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    For the prediction horizon, some lag-365 values are unavailable because
    the required look-back date falls after the training window ends.
    Fall back to the monthly average from the prior year.
    """
    lag_cols = [
        "rev_lag_365", "rev_lag_364", "rev_lag_728",
        "rev_lag365_roll7", "rev_lag365_roll14", "rev_lag365_roll30",
        "log_rev_lag_365",
    ]
    pred_df = pred_df.copy()
    for col in lag_cols:
        if col not in pred_df.columns:
            continue
        mask = pred_df[col].isnull()
        if not mask.any():
            continue
        fallback = (
            np.log1p(pred_df.loc[mask, "monthly_avg_rev"])
            if col == "log_rev_lag_365"
            else pred_df.loc[mask, "monthly_avg_rev"]
        )
        pred_df.loc[mask, col] = fallback
    return pred_df


def impute_with_train_median(
    train_df: pd.DataFrame,
    pred_df:  pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill remaining NaNs using the median from the training set."""
    train_df, pred_df = train_df.copy(), pred_df.copy()
    for col in feature_cols:
        med = train_df[col].median()
        train_df[col] = train_df[col].fillna(med)
        pred_df[col]  = pred_df[col].fillna(med)
    return train_df, pred_df


# ── Monthly COGS ratio ────────────────────────────────────────────────────────

def compute_monthly_cogs_ratio(train_df: pd.DataFrame) -> dict:
    """
    Revenue/COGS ratio varies by month (e.g. seasonal discounting).
    Using a single global ratio introduces up to 20-25% error on COGS predictions.
    Returns a dict mapping month (1-12) → seasonal average ratio.
    """
    monthly = (
        train_df.groupby(["year", "month"])
        .agg(rev=("Revenue", "sum"), cogs=("COGS", "sum"))
        .reset_index()
    )
    monthly["ratio"] = monthly["rev"] / monthly["cogs"]
    return monthly.groupby("month")["ratio"].mean().to_dict()


# ── Master pipeline ───────────────────────────────────────────────────────────

def build_features(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full feature engineering pipeline.

    Returns
    -------
    train_df      : DataFrame with features + Revenue/COGS targets
    pred_df       : DataFrame with features for the forecast horizon
    meta          : dict with feature list and COGS ratio info
    """
    from data_loader import build_date_spine

    df = build_date_spine(data["sales"], data["submission"])

    # Build features in order (calendar first, then increasingly complex)
    df = add_calendar(df)
    df = add_fourier(df)
    df = add_lag_rolling(df)
    df = add_year_ratio(df)
    df = add_holidays(df)
    df = add_post_event(df)
    df = add_structural_breaks(df)
    df = add_traffic(df, data["web_traffic"])
    df = add_promotions(df, data["promotions"])
    df = add_inventory(df, data["inventory"])
    df = add_returns(df, data["returns"])
    df = add_customers(df, data["orders"], data["customers"])

    # Resolve feature list to columns that actually exist
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]

    # Split into train (known Revenue) and prediction horizon
    pred_start = data["submission"][DATE_COL].min()
    train_df = df[df["Revenue"].notna()].copy()
    train_df = train_df.dropna(subset=["rev_lag_365"])   # need at least 1 year of history
    pred_df  = df[df[DATE_COL] >= pred_start].copy()

    pred_df  = _fill_lag_fallbacks(pred_df)
    train_df, pred_df = impute_with_train_median(train_df, pred_df, feature_cols)

    monthly_cogs_ratio = compute_monthly_cogs_ratio(train_df)
    global_cogs_ratio  = train_df["Revenue"].sum() / train_df["COGS"].sum()

    meta = {
        "feature_cols":        feature_cols,
        "n_features":          len(feature_cols),
        "monthly_cogs_ratio":  monthly_cogs_ratio,
        "global_cogs_ratio":   global_cogs_ratio,
        "train_start":         str(train_df[DATE_COL].min().date()),
        "train_end":           str(train_df[DATE_COL].max().date()),
        "pred_start":          str(pred_df[DATE_COL].min().date()),
        "pred_end":            str(pred_df[DATE_COL].max().date()),
        "n_train":             len(train_df),
        "n_pred":              len(pred_df),
    }

    print(f"Features : {len(feature_cols)}")
    print(f"Train    : {meta['train_start']} → {meta['train_end']} ({meta['n_train']} rows)")
    print(f"Predict  : {meta['pred_start']}  → {meta['pred_end']}  ({meta['n_pred']} rows)")
    print(f"Train nulls : {train_df[feature_cols].isnull().sum().sum()} ✅")
    print(f"Pred  nulls : {pred_df[feature_cols].isnull().sum().sum()} ✅")

    return train_df, pred_df, meta

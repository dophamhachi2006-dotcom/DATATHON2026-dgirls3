"""
data_loader.py — Load and minimally preprocess raw CSV files.
"""

import pandas as pd
from config import RAW_FILES, DATE_COL


def load_all() -> dict[str, pd.DataFrame]:
    """Load every raw file and return as a dict of DataFrames."""
    parse_map = {
        "sales":       [DATE_COL],
        "submission":  [DATE_COL],
        "web_traffic": ["date"],
        "promotions":  ["start_date", "end_date"],
        "inventory":   ["snapshot_date"],
        "returns":     ["return_date"],
        "orders":      ["order_date"],
        "customers":   ["signup_date"],
    }
    data = {}
    for name, path in RAW_FILES.items():
        data[name] = pd.read_csv(path, parse_dates=parse_map[name])

    data["sales"] = data["sales"].sort_values(DATE_COL).reset_index(drop=True)
    return data


def build_date_spine(sales: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    """
    Build a continuous daily date range spanning train + prediction period,
    left-joining known Revenue/COGS values from the training set.
    """
    pred_end = submission[DATE_COL].max()
    all_dates = pd.date_range(sales[DATE_COL].min(), pred_end, freq="D")
    df = pd.DataFrame({DATE_COL: all_dates})
    df = df.merge(sales[[DATE_COL, "Revenue", "COGS"]], on=DATE_COL, how="left")
    return df

"""
predict.py
----------
Inference on the test set and submission file generation.
Reads artefacts produced by train.py and writes submission.csv.
Also generates SHAP summary plot for model explainability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config import DATE_COL, SUBMISSION_FILE, WORK_DIR
from utils import get_logger

logger = get_logger(__name__)


def predict(artefacts: dict) -> pd.DataFrame:
    """
    Generate Revenue and COGS predictions for the prediction split.

    Args:
        artefacts: dict returned by train.run()

    Returns:
        submission DataFrame with columns [Date, Revenue, COGS]
    """
    final_lgbm   = artefacts["final_lgbm"]
    final_xgb    = artefacts["final_xgb"]
    blend_w      = artefacts["blend_w"]
    feature_cols = artefacts["feature_cols"]
    monthly_cogs = artefacts["monthly_cogs"]
    pred_df      = artefacts["pred_df"]

    X_pred = pred_df[feature_cols].values

    lgbm_pred = np.expm1(final_lgbm.predict(X_pred))
    xgb_pred  = np.expm1(final_xgb.predict(X_pred))

    # Weighted blend — same weight found during CV
    pred_revenue = blend_w * lgbm_pred + (1 - blend_w) * xgb_pred
    pred_revenue = np.clip(pred_revenue, 0, None)

    # COGS: seasonal ratio by month-of-year (avoids 10-year global ratio error)
    pred_month   = pred_df[DATE_COL].dt.month
    global_ratio = np.mean(list(monthly_cogs.values()))
    cogs_ratio   = pred_month.map(
        {int(k): v for k, v in monthly_cogs.items()}
    ).fillna(global_ratio).values
    pred_cogs = pred_revenue / cogs_ratio

    submission = pd.DataFrame({
        "Date"   : pred_df[DATE_COL].dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(pred_revenue, 2),
        "COGS"   : np.round(pred_cogs, 2),
    })

    logger.info(f"Predictions  Revenue: mean={pred_revenue.mean():,.0f}  "
                f"min={pred_revenue.min():,.0f}  max={pred_revenue.max():,.0f}")
    return submission


def explain(artefacts: dict) -> None:
    """
    Generate SHAP feature importance plots for the final LightGBM model.
    Saves:
      - shap_summary_bar.png  : mean |SHAP| bar chart (top 20 features)
      - shap_summary_dot.png  : beeswarm plot showing direction + magnitude
    Also logs top-10 features with business interpretation.
    """
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed — skipping explainability. Run: pip install shap")
        return

    final_lgbm   = artefacts["final_lgbm"]
    train_df     = artefacts["train_df"]
    feature_cols = artefacts["feature_cols"]

    # Sample 1000 rows for speed (representative of full train set)
    sample = train_df[feature_cols].sample(n=min(1000, len(train_df)), random_state=42)
    X_sample = sample.values

    logger.info("Computing SHAP values (sample=1000)...")
    explainer   = shap.TreeExplainer(final_lgbm)
    shap_values = explainer.shap_values(X_sample)

    # ── Bar chart: mean |SHAP| top 20
    shap.summary_plot(
        shap_values, sample,
        plot_type="bar", max_display=20, show=False
    )
    plt.title("Top 20 Features — Mean |SHAP Value| (LightGBM)", fontsize=12)
    plt.tight_layout()
    plt.savefig(WORK_DIR + "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {WORK_DIR}shap_summary_bar.png")

    # ── Beeswarm: direction + magnitude top 20
    shap.summary_plot(
        shap_values, sample,
        plot_type="dot", max_display=20, show=False
    )
    plt.title("SHAP Beeswarm — Feature Impact Direction (LightGBM)", fontsize=12)
    plt.tight_layout()
    plt.savefig(WORK_DIR + "shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved → {WORK_DIR}shap_summary_dot.png")

    # ── Log top-10 với business interpretation
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols
    ).sort_values(ascending=False)

    # Business label mapping
    biz_labels = {
        "rev_lag365_scaled"     : "Doanh thu cùng kỳ năm ngoái (đã điều chỉnh trend)",
        "rev_lag365_roll30"     : "Trung bình 30 ngày cùng kỳ năm ngoái",
        "rev_lag365_roll7"      : "Trung bình 7 ngày cùng kỳ năm ngoái",
        "monthly_avg_rev"       : "Doanh thu trung bình tháng năm ngoái",
        "rev_lag_365"           : "Doanh thu ngày này năm ngoái",
        "rev_lag_728"           : "Doanh thu ngày này 2 năm trước",
        "post_event_decay"      : "Mức độ suy giảm sau sự kiện lớn (Tết, 11/11...)",
        "days_after_event"      : "Số ngày kể từ sự kiện gần nhất",
        "yoy_ratio_lag"         : "Tốc độ tăng trưởng YoY năm ngoái (COVID signal)",
        "trend"                 : "Xu hướng tuyến tính dài hạn",
        "sessions_roll7"        : "Lượng truy cập web 7 ngày gần nhất",
        "n_active_promos"       : "Số chương trình khuyến mãi đang chạy",
        "is_covid_period"       : "Giai đoạn COVID (2020–2021)",
        "days_to_tet"           : "Số ngày tới Tết Nguyên Đán",
        "sin_365d_k1"           : "Thành phần Fourier mùa vụ năm (k=1)",
        "avg_fill_rate"         : "Tỷ lệ đáp ứng đơn hàng từ tồn kho",
        "orders_roll7"          : "Số đơn hàng trung bình 7 ngày gần nhất",
        "return_rate_roll7"     : "Tỷ lệ hoàn trả 7 ngày gần nhất",
        "year_scale_ratio"      : "Hệ số điều chỉnh tăng trưởng theo năm",
        "is_payday_window"      : "Cửa sổ ngày lương (25–5 hàng tháng)",
    }

    logger.info("\n" + "=" * 60)
    logger.info("TOP 10 FEATURES — Business Interpretation")
    logger.info("=" * 60)
    for i, (feat, val) in enumerate(mean_shap.head(10).items(), 1):
        label = biz_labels.get(feat, feat)
        logger.info(f"  {i:>2}. {feat:<35} |SHAP|={val:.4f}")
        logger.info(f"      → {label}")
    logger.info("=" * 60)


def save_submission(submission: pd.DataFrame, path: str = SUBMISSION_FILE):
    submission.to_csv(path, index=False)
    logger.info(f"Submission saved → {path}  ({len(submission)} rows)")


if __name__ == "__main__":
    from train import run as train_run
    artefacts  = train_run()
    submission = predict(artefacts)
    save_submission(submission)
    explain(artefacts)

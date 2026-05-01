"""
main.py
-------
Entry point. Runs the full pipeline:
  load → feature engineering → train → evaluate → predict → save
"""

from train import run as train_run
from predict import predict, save_submission
from utils import get_logger

logger = get_logger("main")


def main():
    logger.info("=" * 60)
    logger.info("Revenue Forecasting Pipeline")
    logger.info("=" * 60)

    artefacts  = train_run()
    submission = predict(artefacts)
    save_submission(submission)

    ens = artefacts["oof_metrics"]["ensemble"]
    logger.info(f"Final OOF  RMSE={ens['rmse']:,.0f}  MAE={ens['mae']:,.0f}  R²={ens['r2']:.4f}")
    logger.info("Done.")


if __name__ == "__main__":
    main()

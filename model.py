"""
model.py
--------
LightGBM and XGBoost model definitions.
Optuna hyperparameter tuning with time-series CV.
"""

import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error

from config import LGBM_PARAMS, XGB_PARAMS, EARLY_STOPPING, RANDOM_SEED, OPTUNA
from utils import compute_metrics, get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger(__name__)


# ── Training ──────────────────────────────────────────────────────────────────

def train_lgbm(params, X_tr, y_tr, X_vl, y_vl, feature_names):
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, free_raw_data=False)
    dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        params, dtrain,
        num_boost_round = params["n_estimators"],
        valid_sets      = [dval],
        callbacks       = [lgb.early_stopping(EARLY_STOPPING, verbose=False),
                           lgb.log_evaluation(-1)],
    )
    return model, model.best_iteration


def train_xgb(params, X_tr, y_tr, X_vl, y_vl):
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set              = [(X_vl, y_vl)],
        early_stopping_rounds = EARLY_STOPPING,
        verbose               = False,
    )
    return model, model.best_iteration


def train_base_model(model_type, params, folds, X, y, feature_names):
    """
    Expanding-window CV for a single base learner.
    Returns OOF predictions (Revenue scale), per-fold models, metrics, best iters.
    """
    oof_preds  = np.full(len(X), np.nan)
    models     = []
    metrics    = []
    best_iters = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx],   y[val_idx]

        if model_type == "lgbm":
            model, best_iter = train_lgbm(params, X_tr, y_tr, X_vl, y_vl, feature_names)
            log_pred = model.predict(X_vl, num_iteration=best_iter)
        elif model_type == "xgb":
            model, best_iter = train_xgb(params, X_tr, y_tr, X_vl, y_vl)
            log_pred = model.predict(X_vl)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        pred_val   = np.expm1(log_pred)
        actual_val = np.expm1(y_vl)
        oof_preds[val_idx] = pred_val

        m = compute_metrics(actual_val, pred_val)
        metrics.append(m)
        models.append(model)
        best_iters.append(best_iter)

        logger.info(f"  [{model_type.upper()}] Fold {fold_idx+1}: RMSE={m['rmse']:,.0f}  R²={m['r2']:.4f}  best_iter={best_iter}")

    return oof_preds, models, metrics, best_iters


def retrain_final(model_type, params, n_estimators, X, y, feature_names):
    """Retrain on full training set with fixed n_estimators (no early stopping)."""
    if model_type == "lgbm":
        dtrain = lgb.Dataset(X, label=y, feature_name=feature_names)
        final_params = {**params, "n_estimators": n_estimators}
        model = lgb.train(final_params, dtrain, num_boost_round=n_estimators,
                          callbacks=[lgb.log_evaluation(-1)])
    elif model_type == "xgb":
        model = xgb.XGBRegressor(**{**params, "n_estimators": n_estimators,
                                    "early_stopping_rounds": None})
        model.fit(X, y, verbose=False)
    return model


# ── Blend weight ──────────────────────────────────────────────────────────────

def find_optimal_blend(lgbm_oof, xgb_oof, y_true):
    """
    Grid-search free scalar blend: pred = w * lgbm + (1-w) * xgb.
    Optimised over OOF RMSE on Revenue scale.
    """
    def blend_rmse(w):
        blended = w * lgbm_oof + (1 - w) * xgb_oof
        return np.sqrt(mean_squared_error(y_true, blended))

    result = minimize_scalar(blend_rmse, bounds=(0.0, 1.0), method="bounded")
    return result.x   # optimal LGBM weight


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _oof_rmse(params, model_type, folds, X, y):
    """Objective function shared by LGBM and XGB Optuna objectives."""
    oof = np.full(len(X), np.nan)
    for train_idx, val_idx in folds:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_vl, y_vl = X[val_idx],   y[val_idx]

        if model_type == "lgbm":
            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain, free_raw_data=False)
            model  = lgb.train(params, dtrain,
                               num_boost_round=params.get("n_estimators", 2000),
                               valid_sets=[dval],
                               callbacks=[lgb.early_stopping(100, verbose=False),
                                          lgb.log_evaluation(-1)])
            oof[val_idx] = np.expm1(model.predict(X_vl, num_iteration=model.best_iteration))
        elif model_type == "xgb":
            model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            oof[val_idx] = np.expm1(model.predict(X_vl))

    mask = ~np.isnan(oof)
    return np.sqrt(mean_squared_error(np.expm1(y[mask]), oof[mask]))


def tune_lgbm(folds, X, y):
    def objective(trial):
        params = {
            **LGBM_PARAMS,
            "num_leaves"       : trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.0, 5.0),
            "min_split_gain"   : trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        return _oof_rmse(params, "lgbm", folds, X, y)

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner    = optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    # Seed with known-good baseline
    study.enqueue_trial({
        "num_leaves": 63, "min_child_samples": 50, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "min_split_gain": 0.0,
    })
    study.optimize(objective, n_trials=OPTUNA["n_trials_lgbm"])
    logger.info(f"Best LGBM RMSE: {study.best_value:,.0f}")
    return study.best_params


def tune_xgb(folds, X, y):
    def objective(trial):
        params = {
            **XGB_PARAMS,
            "max_depth"        : trial.suggest_int("max_depth", 3, 8),
            "min_child_weight" : trial.suggest_int("min_child_weight", 10, 100),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "alpha"            : trial.suggest_float("alpha", 0.0, 2.0),
            "lambda"           : trial.suggest_float("lambda", 0.0, 5.0),
            "gamma"            : trial.suggest_float("gamma", 0.0, 1.0),
        }
        return _oof_rmse(params, "xgb", folds, X, y)

    study = optuna.create_study(
        direction = "minimize",
        sampler   = optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial({
        "max_depth": 5, "min_child_weight": 50, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "colsample_bylevel": 1.0,
        "alpha": 0.1, "lambda": 1.0, "gamma": 0.0,
    })
    study.optimize(objective, n_trials=OPTUNA["n_trials_xgb"])
    logger.info(f"Best XGB RMSE: {study.best_value:,.0f}")
    return study.best_params

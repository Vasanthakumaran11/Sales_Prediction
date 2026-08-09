"""
Consolidated retraining pipeline — callable from the admin API instead of
running notebooks 01-08 by hand. Transcribes the exact same steps and
hyperparameters used in ml_workspace/notebooks/03-08 (not a redesign), and
reuses engineer_features_batch from feature_engineering.py so the retrained
models stay feature-compatible with the live prediction path.

Models are written to a staging directory first and only promoted to the
live backend/ml/models + backend/ml/encoders paths if the new ensemble R^2
doesn't regress past PROMOTE_TOLERANCE against the currently-deployed model.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from ml.feature_engineering import engineer_features_batch, FEATURE_COLUMNS, CAT_FEATURE_INDICES

BACKEND_DIR = Path(__file__).parent.parent
REPO_ROOT = BACKEND_DIR.parent
ML_WORKSPACE_DIR = REPO_ROOT / "ml_workspace"
DEFAULT_DATASET_PATH = ML_WORKSPACE_DIR / "datasets" / "retailai_finalized_dataset.csv"
METRICS_DIR = ML_WORKSPACE_DIR / "metrics"

LIVE_MODELS_DIR = Path(__file__).parent / "models"
LIVE_ENCODERS_DIR = Path(__file__).parent / "encoders"
STAGING_DIR = Path(__file__).parent / "_staging"

TEST_DAYS = 20
PROMOTE_TOLERANCE = 0.01  # allow up to a 1-point R^2 regression before refusing to promote

MODEL_FILENAMES = {
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl",
    "lightgbm": "lightbgm.pkl",  # matches the (typo'd) filename model_loader.py expects
    "catboost": "catboost.pkl",
}


def _train_models(X_train, y_train):
    models = {}

    models["random_forest"] = RandomForestRegressor(
        n_estimators=150, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    models["random_forest"].fit(X_train, y_train)

    models["xgboost"] = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42, verbosity=0
    )
    models["xgboost"].fit(X_train, y_train)

    models["lightgbm"] = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42
    )
    models["lightgbm"].fit(X_train, y_train)

    models["catboost"] = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42
    )
    models["catboost"].fit(X_train, y_train, cat_features=CAT_FEATURE_INDICES)

    return models


def _evaluate(models, X_test, y_test):
    preds = {name: m.predict(X_test) for name, m in models.items()}
    individual = {
        name: {
            "r2": float(r2_score(y_test, p)),
            "mae": float(mean_absolute_error(y_test, p)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, p))),
        }
        for name, p in preds.items()
    }

    # R^2-weighted ensemble, same formula as notebook 08
    r2_floor = {k: max(v["r2"], 0.01) for k, v in individual.items()}
    total_r2 = sum(r2_floor.values())
    weights = {k: v / total_r2 for k, v in r2_floor.items()}

    ensemble_pred = sum(weights[name] * preds[name] for name in preds)
    ensemble_metrics = {
        "r2": float(r2_score(y_test, ensemble_pred)),
        "mae": float(mean_absolute_error(y_test, ensemble_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
    }

    return individual, weights, ensemble_metrics


def _current_ensemble_r2() -> float:
    path = METRICS_DIR / "ensemble_metrics.json"
    if path.exists():
        return json.loads(path.read_text()).get("r2", 0.0)
    return 0.0


def run_training(dataset_path: str | None = None) -> dict:
    """
    Runs the full pipeline end-to-end. Returns a dict with per-model and
    ensemble metrics, and whether the new models were promoted to production.
    Raises on any hard failure (bad dataset, training error).
    """
    csv_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Store_ID", "Item_ID", "Date", "Units_Sold", "Units_Stocked", "Units_Remaining"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    staging_models_dir = STAGING_DIR / "models"
    staging_encoders_dir = STAGING_DIR / "encoders"
    staging_models_dir.mkdir(parents=True)
    staging_encoders_dir.mkdir(parents=True)

    # Feature engineering + fresh encoders (fit on the full new dataset)
    df_feat, _encoders = engineer_features_batch(df, fit_encoders=True, save_dir=str(staging_encoders_dir))

    # Chronological split — never shuffle time-series data
    df_feat["Date"] = pd.to_datetime(df_feat["Date"])
    split_date = df_feat["Date"].max() - pd.Timedelta(days=TEST_DAYS)
    train_df = df_feat[df_feat["Date"] <= split_date].reset_index(drop=True)
    test_df = df_feat[df_feat["Date"] > split_date].reset_index(drop=True)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Units_Sold"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Units_Sold"]

    models = _train_models(X_train, y_train)
    individual_metrics, weights, ensemble_metrics = _evaluate(models, X_test, y_test)

    for name, model in models.items():
        joblib.dump(model, staging_models_dir / MODEL_FILENAMES[name], compress=3)
    (staging_models_dir / "ensemble_weights.json").write_text(json.dumps(weights, indent=2))

    current_r2 = _current_ensemble_r2()
    promoted = ensemble_metrics["r2"] >= current_r2 - PROMOTE_TOLERANCE

    if promoted:
        LIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        LIVE_ENCODERS_DIR.mkdir(parents=True, exist_ok=True)
        for f in staging_models_dir.iterdir():
            shutil.copy2(f, LIVE_MODELS_DIR / f.name)
        for f in staging_encoders_dir.iterdir():
            shutil.copy2(f, LIVE_ENCODERS_DIR / f.name)

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        (METRICS_DIR / "ensemble_metrics.json").write_text(json.dumps(ensemble_metrics, indent=2))

        comparison_rows = [
            {"model": "Hybrid Ensemble", **ensemble_metrics},
            *[{"model": name, **m} for name, m in individual_metrics.items()],
        ]
        pd.DataFrame(comparison_rows).to_csv(METRICS_DIR / "all_model_comparison.csv", index=False)

        from ml.model_loader import load_all
        load_all(force=True)

    return {
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "datasetRows": len(df),
        "individualMetrics": individual_metrics,
        "ensembleMetrics": ensemble_metrics,
        "ensembleWeights": weights,
        "previousEnsembleR2": current_r2,
        "promoted": promoted,
        "reason": None if promoted else (
            f"New ensemble R^2 ({ensemble_metrics['r2']:.4f}) regressed more than "
            f"{PROMOTE_TOLERANCE} below the currently deployed model ({current_r2:.4f}) — not promoted."
        ),
    }

"""
Hybrid ML/DL Ensemble for Retail Sales Forecasting
Integrates XGBoost, LightGBM, Random Forest, and LSTM (if available)
"""

import numpy as np
import joblib
from pathlib import Path

class HybridEnsemblePredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent.parent / "models" / "algorithms"
        self.model_dir = Path(model_dir)
        self.models = self._load_models()
        self.lstm_model = self._load_lstm()

    def _load_models(self):
        models = {}
        for name in ["xgboost_model.pkl", "lightgbm_model.pkl", "random_forest_model.pkl"]:
            path = self.model_dir / name
            if path.exists():
                try:
                    models[name.split("_")[0]] = joblib.load(path)
                except Exception:
                    models[name.split("_")[0]] = None
            else:
                models[name.split("_")[0]] = None
        return models

    def _load_lstm(self):
        # Placeholder for LSTM model loading (e.g., Keras/TensorFlow)
        # Return None if not available
        return None

    def predict(self, features: np.ndarray):
        preds = []
        for model in self.models.values():
            if model is not None:
                try:
                    pred = model.predict([features])[0]
                    preds.append(pred)
                except Exception:
                    continue
        # Optionally, add LSTM prediction here
        if self.lstm_model is not None:
            try:
                lstm_pred = self.lstm_model.predict(features.reshape(1, -1))[0]
                preds.append(lstm_pred)
            except Exception:
                pass
        if preds:
            return float(np.mean(preds))
        return 0.0

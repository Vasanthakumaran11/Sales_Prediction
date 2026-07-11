"""
Loads all 4 models + encoders + ensemble weights ONCE at FastAPI startup.
Never reload per-request -- that's slow and wasteful.
"""
import joblib
import json
from pathlib import Path

# Resolve paths relative to THIS file (ml/model_loader.py)
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ENCODERS_DIR = BASE_DIR / "encoders"

_models = None
_weights = None
_encoders = None

# Categorical columns that have saved encoders
CATEGORICAL_COLS = [
    "Store_Type", "Location_Type", "Category",
    "Day_Type", "Demand_Level", "Supplier_ID", "Item_Name"
]

def _load_encoders():
    encoders = {}
    for col in CATEGORICAL_COLS:
        path = ENCODERS_DIR / f"{col}_encoder.pkl"
        if path.exists():
            encoders[col] = joblib.load(str(path))
    return encoders


def load_all():
    global _models, _weights, _encoders
    if _models is not None:
        return _models, _weights, _encoders

    # Note: lightbgm.pkl has a typo in filename — map it correctly
    model_files = {
        "random_forest": MODELS_DIR / "random_forest.pkl",
        "xgboost":       MODELS_DIR / "xgboost.pkl",
        "lightgbm":      MODELS_DIR / "lightbgm.pkl",   # note: typo in saved file name
        "catboost":      MODELS_DIR / "catboost.pkl",
    }

    missing = [k for k, p in model_files.items() if not p.exists()]
    if missing:
        print(f"[model_loader] WARNING: Missing model files: {missing}. Predictions will use rule-based fallback.")
        _models = {}
    else:
        _models = {name: joblib.load(str(path)) for name, path in model_files.items()}
        print(f"[model_loader] Loaded {len(_models)} models successfully.")

    weights_path = MODELS_DIR / "ensemble_weights.json"
    _weights = json.loads(weights_path.read_text()) if weights_path.exists() else {
        "random_forest": 0.25, "xgboost": 0.25, "lightgbm": 0.25, "catboost": 0.25
    }

    _encoders = _load_encoders()
    print(f"[model_loader] Loaded {len(_encoders)} encoders. Ready.")
    return _models, _weights, _encoders


def get_models():
    if _models is None:
        load_all()
    return _models, _weights, _encoders
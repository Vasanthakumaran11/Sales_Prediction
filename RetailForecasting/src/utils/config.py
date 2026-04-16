"""
Configuration file for the Retail Forecasting System
Contains all constants, paths, and configuration parameters
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data directories
RAW_DATA_DIR = DATA_DIR / "raw"
USER_DATA_DIR = DATA_DIR / "user"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
for directory in [RAW_DATA_DIR, USER_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset paths
BASE_DATASET_PATH = RAW_DATA_DIR / "base_dataset.csv"
BASE_PROCESSED_PATH = PROCESSED_DATA_DIR / "base_processed.csv"
USER_PROCESSED_PATH = PROCESSED_DATA_DIR / "user_processed.csv"
FINAL_DATASET_PATH = PROCESSED_DATA_DIR / "final_dataset.csv"

# User data paths
PRODUCTS_FILE = USER_DATA_DIR / "products.csv"
PURCHASES_FILE = USER_DATA_DIR / "purchases.csv"
SALES_FILE = USER_DATA_DIR / "sales.csv"

# Model paths
BASE_MODEL_PATH = MODELS_DIR / "base_model.pkl"
PERSONALIZED_MODEL_PATH = MODELS_DIR / "personalized_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ========== DATASET GENERATION PARAMETERS ==========
# Time range for synthetic data
DAYS_RANGE = 90
START_DATE = "01-01-2025"  # DD-MM-YYYY

# Store details
STORE_TYPES = ["Small", "Medium", "Supermarket"]
LOCATION_TYPES = ["Urban", "Semi-Urban", "Rural"]
NUM_STORES = 3

# Product catalog for Tamil Nadu region
PRODUCTS = {
    "Milk": {"category": "Perishable", "price_range": (22, 35)},
    "Curd": {"category": "Perishable", "price_range": (40, 60)},
    "Paneer": {"category": "Perishable", "price_range": (180, 250)},
    "Rice": {"category": "Non-Perishable", "price_range": (50, 90)},
    "Toor Dal": {"category": "Non-Perishable", "price_range": (100, 150)},
    "Wheat Flour": {"category": "Non-Perishable", "price_range": (25, 45)},
    "Cooking Oil": {"category": "Non-Perishable", "price_range": (120, 200)},
    "Biscuits": {"category": "Non-Perishable", "price_range": (5, 30)},
    "Snacks": {"category": "Non-Perishable", "price_range": (20, 80)},
    "Masala": {"category": "Non-Perishable", "price_range": (40, 120)},
    "Sugar": {"category": "Non-Perishable", "price_range": (40, 60)},
    "Salt": {"category": "Non-Perishable", "price_range": (15, 25)},
    "Bread": {"category": "Perishable", "price_range": (30, 60)},
    "Eggs": {"category": "Perishable", "price_range": (30, 50)},
}

# ========== DEMAND PATTERNS ==========
WEEKEND_MULTIPLIER = 1.25  # +25% on weekends
FESTIVAL_MULTIPLIER = 1.35  # +35% on festival days
FESTIVAL_DATES = ["14-01-2025"]  # Pongal festival
DEMAND_VARIABILITY = 0.15  # ±15% variability

# Stockout probability by store type
STOCKOUT_PROB = {
    "Small": 0.15,
    "Medium": 0.08,
    "Supermarket": 0.03,
}

# Turnover rates (days to sell average stock)
TURNOVER_DAYS = {
    "Perishable": (1, 3),  # 1-3 days
    "Non-Perishable": (7, 14),  # 7-14 days
}

# ========== INVENTORY PARAMETERS ==========
# Safety stock calculation parameters
Z_SCORE = 1.96  # 95% service level
LEAD_TIME_DAYS = 2

# ========== PREPROCESSING PARAMETERS ==========
# Look-back periods for lag and rolling features
LAG_PERIODS = [1, 7]  # 1 day, 1 week
ROLLING_WINDOW = 7  # 7-day rolling window
ROLLING_METRICS = ["mean", "std"]

# ========== MODEL TRAINING PARAMETERS ==========
# Test-train split
TEST_SPLIT = 0.2
RANDOM_STATE = 42

# Model hyperparameters
MODEL_PARAMS = {
    "linear_regression": {},
    "decision_tree": {
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    },
}

# ========== RETRAINING PARAMETERS ==========
# Minimum data points required for retraining
MIN_DATA_POINTS_FOR_RETRAIN = 14  # 2 weeks
RETRAIN_CHECK_INTERVAL = 7  # Check every 7 days

# ========== TARGET VARIABLE ==========
TARGET_VARIABLE = "Units_Sold"

# ========== FEATURE COLUMNS ==========
CATEGORICAL_FEATURES = ["Store_Type", "Location_Type", "Category", "Day_Type"]
NUMERICAL_FEATURES = [
    "Units_Stocked",
    "Unit_Price",
    "Discount",
    "Day_of_Week",
    "Is_Weekend",
    "Is_Festival",
]

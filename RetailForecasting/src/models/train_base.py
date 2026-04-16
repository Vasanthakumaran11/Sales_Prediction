"""
Base Model Training Module
Trains initial ML models on synthetic dataset
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# XGBoost and LightGBM
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    BASE_PROCESSED_PATH,
    BASE_MODEL_PATH,
    MODEL_METADATA_PATH,
    TEST_SPLIT,
    RANDOM_STATE,
    TARGET_VARIABLE,
    MODEL_PARAMS,
)
from preprocessing.preprocess import DataPreprocessor


class BaseModelTrainer:
    """Train base ML models on processed dataset"""

    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = []
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf

    def load_processed_data(self):
        """Load preprocessed data"""
        print("📂 Loading processed data...")

        if not os.path.exists(BASE_PROCESSED_PATH):
            print("⚠️  Processed data not found. Running preprocessing...")
            preprocessor = DataPreprocessor()
            self.df, self.feature_columns = preprocessor.process()
        else:
            self.df = pd.read_csv(BASE_PROCESSED_PATH)
            print(f"✅ Loaded data: {self.df.shape}")

        return self.df

    def prepare_features(self):
        """Prepare features for training"""
        print("📊 Preparing features...")

        # Identify numeric columns (exclude identifiers and target)
        exclude_cols = [
            "Date",
            "Store_ID",
            "Item_Name",
            TARGET_VARIABLE,
            "Day",
            "Month",
            "Store_Type",  # Exclude original categorical columns
            "Location_Type",
            "Category",
            "Day_Type",
        ]
        numeric_cols = []

        for col in self.df.columns:
            # Only include numeric columns and encoded versions
            if col not in exclude_cols:
                if self.df[col].dtype != "object":  # Numeric columns
                    numeric_cols.append(col)
                elif "_Encoded" in col:  # Encoded categorical columns
                    numeric_cols.append(col)

        # Filter to columns that actually exist
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        self.feature_columns = numeric_cols

        print(f"  ✓ Total features: {len(numeric_cols)}")
        print(f"  ✓ Features: {numeric_cols[:10]}...")  # Show first 10

        return numeric_cols

    def split_data(self):
        """Split data into train and test sets"""
        print("✂️  Splitting data into train/test...")

        X = self.df[self.feature_columns]
        y = self.df[TARGET_VARIABLE]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
        )

        print(f"  ✓ Train: {self.X_train.shape}")
        print(f"  ✓ Test: {self.X_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self, model_name, model_class, params):
        """Train a single model"""
        print(f"\n🤖 Training {model_name}...")

        try:
            model = model_class(**params)
            model.fit(self.X_train, self.y_train)

            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Metrics
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            test_r2 = r2_score(self.y_test, y_pred_test)

            # Store results
            self.models[model_name] = model
            self.results[model_name] = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
            }

            # Track best model
            if test_r2 > self.best_score:
                self.best_score = test_r2
                self.best_model = model
                self.best_model_name = model_name

            print(f"  ✓ Train MAE: {train_mae:.2f}")
            print(f"  ✓ Test MAE: {test_mae:.2f}")
            print(f"  ✓ Test RMSE: {test_rmse:.2f}")
            print(f"  ✓ Test R²: {test_r2:.4f}")

            return model

        except Exception as e:
            print(f"  ❌ Error training {model_name}: {str(e)}")
            return None

    def train_all_models(self):
        """Train all models"""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        # Linear Regression
        self.train_model(
            "Linear Regression",
            LinearRegression,
            MODEL_PARAMS["linear_regression"],
        )

        # Decision Tree
        self.train_model(
            "Decision Tree", DecisionTreeRegressor, MODEL_PARAMS["decision_tree"]
        )

        # Random Forest
        self.train_model(
            "Random Forest", RandomForestRegressor, MODEL_PARAMS["random_forest"]
        )

        # XGBoost
        if XGBOOST_AVAILABLE:
            self.train_model("XGBoost", xgb.XGBRegressor, MODEL_PARAMS["xgboost"])
        else:
            print("⚠️  XGBoost not available. Install with: pip install xgboost")

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            self.train_model("LightGBM", lgb.LGBMRegressor, MODEL_PARAMS["lightgbm"])
        else:
            print("⚠️  LightGBM not available. Install with: pip install lightgbm")

        return self.models

    def save_best_model(self):
        """Save the best performing model"""
        print("\n" + "=" * 80)
        print("SAVING BEST MODEL")
        print("=" * 80)

        if self.best_model is None:
            print("❌ No model trained successfully. Cannot save.")
            return None

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Test R²: {self.best_score:.4f}")

        # Create models directory
        os.makedirs(os.path.dirname(BASE_MODEL_PATH), exist_ok=True)

        # Save model
        with open(BASE_MODEL_PATH, "wb") as f:
            pickle.dump(self.best_model, f)

        print(f"  ✓ Model saved: {BASE_MODEL_PATH}")

        # Save metadata
        metadata = {
            "model_name": self.best_model_name,
            "features": self.feature_columns,
            "training_date": datetime.now().isoformat(),
            "test_split": TEST_SPLIT,
            "random_state": RANDOM_STATE,
            "metrics": self.results[self.best_model_name],
            "all_models_metrics": {k: v for k, v in self.results.items()},
            "training_data_shape": self.df.shape,
            "feature_count": len(self.feature_columns),
        }

        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Metadata saved: {MODEL_METADATA_PATH}")

        return self.best_model

    def print_results_summary(self):
        """Print summary of all model results"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80 + "\n")

        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)

        print(results_df)
        print()

    def train(self):
        """Execute full training pipeline"""
        print("\n" + "=" * 80)
        print("BASE MODEL TRAINING PIPELINE")
        print("=" * 80 + "\n")

        self.load_processed_data()
        self.prepare_features()
        self.split_data()
        self.train_all_models()
        self.print_results_summary()
        self.save_best_model()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)


def main():
    """Main training function"""
    trainer = BaseModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()

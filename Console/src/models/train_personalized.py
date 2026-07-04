"""
Personalized Model Retraining Module
Retrains model using combined base + user data after sufficient data collected
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.train_base import BaseModelTrainer
from utils.config import (
    BASE_PROCESSED_PATH,
    FINAL_DATASET_PATH,
    USER_DATA_DIR,
    PERSONALIZED_MODEL_PATH,
    MODEL_METADATA_PATH,
    MIN_DATA_POINTS_FOR_RETRAIN,
)


class PersonalizedModelTrainer(BaseModelTrainer):
    """Train personalized model using combined dataset"""

    def __init__(self):
        super().__init__()
        self.user_df = None
        self.combined_df = None

    def check_retraining_required(self):
        """Check if we have sufficient user data for retraining"""
        sales_file = os.path.join(USER_DATA_DIR, "sales.csv")

        if not os.path.exists(sales_file):
            return False, 0

        sales_df = pd.read_csv(sales_file)
        user_data_points = len(sales_df)

        print(f"📊 User data points: {user_data_points}")
        print(f"📊 Required for retraining: {MIN_DATA_POINTS_FOR_RETRAIN}")

        if user_data_points >= MIN_DATA_POINTS_FOR_RETRAIN:
            return True, user_data_points
        return False, user_data_points

    def load_user_data(self):
        """Load and preprocess user data"""
        print("📂 Loading user data...")

        sales_file = os.path.join(USER_DATA_DIR, "sales.csv")

        if not os.path.exists(sales_file):
            print("⚠️  No user sales data found")
            return None

        user_sales = pd.read_csv(sales_file)
        print(f"  ✓ Loaded {len(user_sales)} user transactions")

        # Create features similar to base dataset
        user_sales["Date"] = pd.to_datetime(user_sales["Date"], format="%d-%m-%Y")
        user_sales = user_sales.sort_values("Date").reset_index(drop=True)

        # Add time features
        user_sales["Day"] = user_sales["Date"].dt.day
        user_sales["Month"] = user_sales["Date"].dt.month
        user_sales["Day_of_Week"] = user_sales["Date"].dt.weekday
        user_sales["Is_Weekend"] = (user_sales["Day_of_Week"] >= 5).astype(int)
        user_sales["Day_Type"] = user_sales["Is_Weekend"].map(
            {1: "Weekend", 0: "Weekday"}
        )
        user_sales["Is_Festival"] = 0  # Placeholder
        user_sales["Date"] = user_sales["Date"].dt.strftime("%d-%m-%Y")

        # Add required columns if missing
        if "Store_Type" not in user_sales.columns:
            user_sales["Store_Type"] = "User_Store"
        if "Location_Type" not in user_sales.columns:
            user_sales["Location_Type"] = "Urban"
        if "Category" not in user_sales.columns:
            user_sales["Category"] = "Non-Perishable"
        if "Units_Stocked" not in user_sales.columns:
            user_sales["Units_Stocked"] = user_sales["Units_Sold"] * 1.2
        if "Units_Remaining" not in user_sales.columns:
            user_sales["Units_Remaining"] = (
                user_sales["Units_Stocked"] - user_sales["Units_Sold"]
            )

        self.user_df = user_sales
        return user_sales

    def combine_datasets(self):
        """Combine base and user datasets"""
        print("🔗 Combining datasets...")

        if self.df is None:
            self.load_data()

        if self.user_df is None:
            print("⚠️  No user data to combine")
            return self.df

        # Convert base dataset dates to datetime for comparison
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%d-%m-%Y")
        self.user_df["Date"] = pd.to_datetime(self.user_df["Date"], format="%d-%m-%Y")

        # Combine datasets
        combined = pd.concat([self.df, self.user_df], ignore_index=True, sort=True)
        combined = combined.sort_values(["Store_ID", "Date"]).reset_index(drop=True)

        # Convert dates back to string
        combined["Date"] = combined["Date"].dt.strftime("%d-%m-%Y")

        self.combined_df = combined
        print(f"  ✓ Combined shape: {combined.shape}")

        return combined

    def retrain(self):
        """Execute retraining pipeline"""
        print("\n" + "=" * 80)
        print("PERSONALIZED MODEL RETRAINING PIPELINE")
        print("=" * 80 + "\n")

        # Check if retraining is required
        should_retrain, user_data_count = self.check_retraining_required()

        if not should_retrain:
            print(
                f"⚠️  Insufficient data for retraining. Got {user_data_count}, need {MIN_DATA_POINTS_FOR_RETRAIN}"
            )
            return False

        print(f"✅ Sufficient data for retraining ({user_data_count} points)")

        # Load and prepare data
        self.load_user_data()
        self.load_data()
        combined = self.combine_datasets()

        # Preprocess combined data
        from preprocessing.preprocess import DataPreprocessor

        preprocessor = DataPreprocessor(dataset_path=None)
        preprocessor.df = combined
        preprocessor.clean_data()
        preprocessor.encode_categorical()
        preprocessor.create_lag_features()
        preprocessor.create_rolling_features()
        preprocessor.create_derived_features()
        features = preprocessor.prepare_features()

        preprocessor.df = preprocessor.df.dropna()

        print(f"✅ Data prepared: {preprocessor.df.shape}")

        # Train model
        self.df = preprocessor.df
        self.feature_columns = features
        self.split_data()
        self.train_all_models()
        self.print_results_summary()

        # Save personalized model
        self.save_personalized_model()

        print("\n" + "=" * 80)
        print("RETRAINING COMPLETE")
        print("=" * 80)

        return True

    def save_personalized_model(self):
        """Save the personalized model"""
        print("\n" + "=" * 80)
        print("SAVING PERSONALIZED MODEL")
        print("=" * 80)

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Test R²: {self.best_score:.4f}")

        # Save model
        with open(PERSONALIZED_MODEL_PATH, "wb") as f:
            pickle.dump(self.best_model, f)

        print(f"  ✓ Model saved: {PERSONALIZED_MODEL_PATH}")

        # Update metadata
        with open(MODEL_METADATA_PATH, "r") as f:
            metadata = json.load(f)

        metadata.update(
            {
                "personalized_model_name": self.best_model_name,
                "personalized_training_date": datetime.now().isoformat(),
                "personalized_metrics": self.results[self.best_model_name],
                "personalized_data_shape": self.df.shape,
                "personalized": True,
            }
        )

        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Metadata updated: {MODEL_METADATA_PATH}")


def main():
    """Main retraining function"""
    trainer = PersonalizedModelTrainer()
    trainer.retrain()


if __name__ == "__main__":
    main()

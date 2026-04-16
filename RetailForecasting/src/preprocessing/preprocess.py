"""
Data Preprocessing and Feature Engineering Module
Handles data cleaning, encoding, and feature creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    BASE_DATASET_PATH,
    BASE_PROCESSED_PATH,
    LAG_PERIODS,
    ROLLING_WINDOW,
    ROLLING_METRICS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_VARIABLE,
)


class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""

    def __init__(self, dataset_path=BASE_DATASET_PATH):
        self.dataset_path = dataset_path
        self.df = None
        self.encoders = {}
        self.feature_columns = []

    def load_data(self):
        """Load raw dataset"""
        print(f"📂 Loading data from {self.dataset_path}...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ Data loaded: {self.df.shape}")
        return self.df

    def clean_data(self):
        """Clean data - handle missing values"""
        print("🧹 Cleaning data...")

        # Check for missing values
        missing_before = self.df.isnull().sum().sum()

        if missing_before > 0:
            print(f"⚠️  Found {missing_before} missing values")

            # Forward fill for time series data
            self.df = self.df.sort_values(["Store_ID", "Date"])
            self.df = self.df.fillna(method="ffill", limit=2)

            # Fill remaining with backward fill
            self.df = self.df.fillna(method="bfill")

            missing_after = self.df.isnull().sum().sum()
            print(f"✅ Missing values after fillna: {missing_after}")

        # Remove duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicates...")
            self.df = self.df.drop_duplicates()

        return self.df

    def encode_categorical(self):
        """Encode categorical variables"""
        print("🔤 Encoding categorical features...")

        for feature in CATEGORICAL_FEATURES:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f"{feature}_Encoded"] = le.fit_transform(
                    self.df[feature].astype(str)
                )
                self.encoders[feature] = le
                print(f"  ✓ {feature}: {len(le.classes_)} classes")

        return self.df

    def create_lag_features(self):
        """Create lag features for time-series prediction"""
        print("⏱️  Creating lag features...")

        # Sort by store and date for proper lag calculation
        self.df = self.df.sort_values(["Store_ID", "Date"]).reset_index(drop=True)

        for lag in LAG_PERIODS:
            # Lag for Units_Sold
            self.df[f"Lag_{lag}_Units_Sold"] = (
                self.df.groupby("Store_ID")[TARGET_VARIABLE]
                .shift(lag)
                .fillna(method="bfill")
                .fillna(0)
            )

            # Lag for Revenue
            self.df[f"Lag_{lag}_Revenue"] = (
                self.df.groupby("Store_ID")["Revenue"]
                .shift(lag)
                .fillna(method="bfill")
                .fillna(0)
            )

            print(f"  ✓ Lag {lag} features created")

        return self.df

    def create_rolling_features(self):
        """Create rolling window features"""
        print(f"🔄 Creating rolling features (window={ROLLING_WINDOW})...")

        # Sort by store and date
        self.df = self.df.sort_values(["Store_ID", "Date"]).reset_index(drop=True)

        for metric in ROLLING_METRICS:
            # Rolling mean/std for Units_Sold
            if metric == "mean":
                self.df[f"Rolling_Mean_{ROLLING_WINDOW}d_Units_Sold"] = (
                    self.df.groupby("Store_ID")[TARGET_VARIABLE]
                    .rolling(ROLLING_WINDOW, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                self.df[f"Rolling_Mean_{ROLLING_WINDOW}d_Revenue"] = (
                    self.df.groupby("Store_ID")["Revenue"]
                    .rolling(ROLLING_WINDOW, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )

            elif metric == "std":
                self.df[f"Rolling_Std_{ROLLING_WINDOW}d_Units_Sold"] = (
                    self.df.groupby("Store_ID")[TARGET_VARIABLE]
                    .rolling(ROLLING_WINDOW, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                    .fillna(0)
                )

                self.df[f"Rolling_Std_{ROLLING_WINDOW}d_Revenue"] = (
                    self.df.groupby("Store_ID")["Revenue"]
                    .rolling(ROLLING_WINDOW, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                    .fillna(0)
                )

            print(f"  ✓ Rolling {metric} features created")

        return self.df

    def create_derived_features(self):
        """Create derived features"""
        print("🔧 Creating derived features...")

        # Sell-through ratio
        self.df["Sell_Through_Ratio"] = np.where(
            self.df["Units_Stocked"] > 0,
            self.df["Units_Sold"] / self.df["Units_Stocked"],
            0,
        )

        # Stock remaining ratio
        self.df["Stock_Remaining_Ratio"] = np.where(
            self.df["Units_Stocked"] > 0,
            self.df["Units_Remaining"] / self.df["Units_Stocked"],
            0,
        )

        # Revenue per unit stocked
        self.df["Revenue_Per_Unit_Stocked"] = np.where(
            self.df["Units_Stocked"] > 0,
            self.df["Revenue"] / self.df["Units_Stocked"],
            0,
        )

        # Discount impact
        self.df["Discount_Applied"] = (self.df["Discount"] > 0).astype(int)

        # High demand flag
        self.df["High_Demand_Flag"] = (self.df["Demand_Level"] == "High").astype(int)

        # Low stock flag
        self.df["Low_Stock_Flag"] = (self.df["Units_Remaining"] < 5).astype(int)

        print("  ✓ Derived features created")
        return self.df

    def prepare_features(self):
        """Prepare feature set for modeling"""
        print("📊 Preparing feature set...")

        # Feature list
        features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

        # Add encoded categorical features
        features += [f"{feat}_Encoded" for feat in CATEGORICAL_FEATURES]

        # Add lag features
        for lag in LAG_PERIODS:
            features += [f"Lag_{lag}_Units_Sold", f"Lag_{lag}_Revenue"]

        # Add rolling features
        features += [
            f"Rolling_Mean_{ROLLING_WINDOW}d_Units_Sold",
            f"Rolling_Mean_{ROLLING_WINDOW}d_Revenue",
            f"Rolling_Std_{ROLLING_WINDOW}d_Units_Sold",
            f"Rolling_Std_{ROLLING_WINDOW}d_Revenue",
        ]

        # Add derived features
        features += [
            "Sell_Through_Ratio",
            "Stock_Remaining_Ratio",
            "Revenue_Per_Unit_Stocked",
            "Discount_Applied",
            "High_Demand_Flag",
            "Low_Stock_Flag",
        ]

        # Filter to features that exist
        features = [f for f in features if f in self.df.columns]
        self.feature_columns = features

        print(f"  ✓ Total features: {len(features)}")
        print(f"  ✓ Feature list saved")

        return features

    def process(self):
        """Execute full preprocessing pipeline"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING PIPELINE")
        print("=" * 80 + "\n")

        self.load_data()
        self.clean_data()
        self.encode_categorical()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_derived_features()
        features = self.prepare_features()

        # Remove rows with NaN that might still exist
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        final_rows = len(self.df)

        if initial_rows > final_rows:
            print(
                f"⚠️  Removed {initial_rows - final_rows} rows with missing values"
            )

        # Save processed data
        os.makedirs(os.path.dirname(BASE_PROCESSED_PATH), exist_ok=True)
        self.df.to_csv(BASE_PROCESSED_PATH, index=False)

        print(f"\n✅ Processed data saved: {BASE_PROCESSED_PATH}")
        print(f"📊 Final shape: {self.df.shape}")
        print(f"📋 Sample processed data:\n{self.df.head()}")

        return self.df, features


def main():
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    df, features = preprocessor.process()

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

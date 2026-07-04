"""
Prediction Module
Makes predictions using trained models
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    BASE_MODEL_PATH,
    PERSONALIZED_MODEL_PATH,
    MODEL_METADATA_PATH,
    TARGET_VARIABLE,
)


class PredictionEngine:
    """Load trained models and make predictions"""

    def __init__(self, use_personalized=False):
        self.model = None
        self.metadata = None
        self.feature_columns = []
        self.model_name = ""
        self.use_personalized = use_personalized

        self.load_model()

    def load_model(self):
        """Load trained model and metadata"""
        print("Loading model...")

        # Determine which model to use
        if self.use_personalized and os.path.exists(PERSONALIZED_MODEL_PATH):
            model_path = PERSONALIZED_MODEL_PATH
            print(f"  v Using personalized model")
        else:
            model_path = BASE_MODEL_PATH
            print(f"  v Using base model")

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load metadata
        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH, "r") as f:
                self.metadata = json.load(f)
                self.feature_columns = self.metadata.get("features", [])
                self.model_name = self.metadata.get("model_name", "Unknown")

        print(f"  v Model loaded: {self.model_name}")
        print(f"  v Features: {len(self.feature_columns)}")

    def prepare_input(self, input_df):
        """Prepare input data for prediction"""
        # Ensure all required features are present
        missing_features = [f for f in self.feature_columns if f not in input_df.columns]

        if missing_features:
            # Fill missing features with 0
            for feat in missing_features:
                input_df[feat] = 0

        # Select only required features and maintain order
        input_df = input_df[self.feature_columns]

        return input_df

    def predict(self, input_df):
        """Make predictions"""
        input_prepared = self.prepare_input(input_df.copy())
        predictions = self.model.predict(input_prepared)
        return np.maximum(predictions, 0)  # Ensure non-negative predictions

    def predict_single(self, features_dict):
        """Predict for a single sample"""
        df = pd.DataFrame([features_dict])
        return self.predict(df)[0]


def load_prediction_engine(use_personalized=False):
    """Helper function to load prediction engine"""
    return PredictionEngine(use_personalized=use_personalized)

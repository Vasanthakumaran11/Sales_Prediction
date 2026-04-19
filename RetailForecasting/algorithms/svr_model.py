"""
Support Vector Regression Model
Handles non-linear relationships with high dimensionality
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import TARGET_VARIABLE


class SVRModel:
    """Support Vector Regression Model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        self.model_path = model_path or "models/algorithms/svr_model.pkl"
        
    def load_data(self, df):
        """Load and prepare data"""
        print("📚 [SVR] Loading data...")
        
        exclude_cols = [
            "Date", "Store_ID", "Item_Name", TARGET_VARIABLE,
            "Day", "Month", "Store_Type", "Location_Type",
            "Category", "Day_Type",
            "Revenue", "Units_Remaining", "Suggested_Next_Stock",
            "Demand_Level", "High_Demand_Flag", "Low_Stock_Flag",
            "Sell_Through_Ratio", "Stock_Remaining_Ratio",
            "Revenue_Per_Unit_Stocked"
        ]
        
        numeric_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype != "object"]
        
        self.feature_columns = numeric_cols
        return df[numeric_cols], df[TARGET_VARIABLE]
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        print(f"  ✓ Train size: {self.X_train.shape[0]}")
        print(f"  ✓ Test size: {self.X_test.shape[0]}")
    
    def train(self, X=None, y=None):
        """Train the SVR model"""
        print("\n🤖 [SVR] Training model...")
        
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # Using a C value and gamma appropriate generally for this scaling
        self.model = SVR(kernel='rbf', C=10.0, gamma='scale')
        self.model.fit(X_train_scaled, self.y_train)
        
        print("  ✅ Model trained successfully")
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n📊 [SVR] Evaluating model...")
        
        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        y_pred_train = self.model.predict(X_train_scaled)
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        y_pred_test = self.model.predict(X_test_scaled)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        self.metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'model_name': 'SVR'
        }
        
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f} ⭐")
        
        return self.metrics
    
    def save_model(self):
        """Save trained model and scaler to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Save both model and scaler as a tuple
        with open(self.model_path, 'wb') as f:
            pickle.dump((self.model, self.scaler), f)
        print(f"  ✅ Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, tuple) and len(data) == 2:
                    self.model, self.scaler = data
                else: # Fallback mechanism if loaded old unscaled layout
                    self.model = data
            print(f"  ✅ Model loaded from {self.model_path}")
        else:
            print(f"  ❌ Model not found at {self.model_path}")
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_metrics(self):
        """Return evaluation metrics"""
        return self.metrics

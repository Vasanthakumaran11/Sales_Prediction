"""
Random Forest Model
Ensemble of decision trees with high accuracy and robustness
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import BASE_PROCESSED_PATH, BASE_MODEL_PATH, TARGET_VARIABLE


class RandomForestModel:
    """Random Forest Model for demand forecasting"""
    
    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=5, 
                 min_samples_leaf=2, model_path=None):
        self.model = None
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model_path = model_path or "models/algorithms/random_forest.pkl"
        
    def load_data(self, df):
        """Load and prepare data"""
        print("📚 [Random Forest] Loading data...")
        
        # Prepare features (exclude non-numeric and target)
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
    
    def train(self):
        """Train the Random Forest model"""
        print("\n🤖 [Random Forest] Training model...")
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        
        print("  ✅ Model trained successfully")
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n📊 [Random Forest] Evaluating model...")
        
        # Training predictions
        y_pred_train = self.model.predict(self.X_train)
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        # Testing predictions
        y_pred_test = self.model.predict(self.X_test)
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
            'model_name': 'Random Forest'
        }
        
        print(f"  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f} ⭐")
        
        return self.metrics
    
    def save_model(self):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"  ✅ Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"  ✅ Model loaded from {self.model_path}")
        else:
            print(f"  ❌ Model not found at {self.model_path}")
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        return self.model.predict(X)
    
    def get_metrics(self):
        """Return evaluation metrics"""
        return self.metrics
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

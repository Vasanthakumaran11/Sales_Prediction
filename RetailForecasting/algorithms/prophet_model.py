"""
Facebook Prophet Model
Specialized time-series implementation for retail forecasting.
Models trend, seasonality, and holiday effects organically.
"""

import pandas as pd
import numpy as np
import pickle
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import TARGET_VARIABLE


class ProphetModel:
    """Facebook Prophet Model"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df_ref = None # Used to pull true Dates
        self.metrics = {}
        self.model_path = model_path or "models/algorithms/prophet_model.pkl"
        
    def load_data(self, df):
        """Load and prepare data"""
        print("📚 [Prophet] Loading data...")
        self.df_ref = df.copy()
        
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
        
    def _build_prophet_df(self, X, y):
        """Reconstruct dataframe with 'ds' and 'y' required by Prophet"""
        if self.df_ref is None:
            raise ValueError("Prophet requires df_ref. Ensure load_data() or hybrid_ensemble passed df_ref.")
        
        # Pull original dates based on index
        dates = pd.to_datetime(self.df_ref.loc[X.index, 'Date'], format="%d-%m-%Y")
        
        # Add hour offset based on cumulative count to ensure unique timestamps for panel data
        hour_offset = pd.to_timedelta(dates.groupby(dates).cumcount(), unit='h')
        unique_dates = dates + hour_offset
        
        prophet_df = pd.DataFrame({
            'ds': unique_dates,
            'y': y.values
        })
        
        # Add regressors
        for col in self.feature_columns:
            prophet_df[col] = X[col].values
            
        return prophet_df
    
    def train(self, X=None, y=None):
        """Train the Prophet model"""
        print("\n🤖 [Prophet] Training model...")
        
        train_df = self._build_prophet_df(self.X_train, self.y_train)
        
        self.model = Prophet(yearly_seasonality=False, daily_seasonality=True)
        
        for col in self.feature_columns:
            self.model.add_regressor(col)
            
        self.model.fit(train_df)
        print("  ✅ Model trained successfully")
    
    def evaluate(self):
        """Evaluate model performance"""
        print("\n📊 [Prophet] Evaluating model...")
        
        train_df = self._build_prophet_df(self.X_train, self.y_train)
        test_df = self._build_prophet_df(self.X_test, self.y_test)
        
        train_pred_df = self.model.predict(train_df.drop(columns=['y']))
        y_pred_train = train_pred_df['yhat'].values
        
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        test_pred_df = self.model.predict(test_df.drop(columns=['y']))
        y_pred_test = test_pred_df['yhat'].values
        
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
            'model_name': 'Prophet'
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
        # Prophet models shouldn't be pickled optimally, but we'll try standard pickle. Wait, Prophet explicitly supports pickling!
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
            raise ValueError("Model not trained.")
        
        # For prediction, we need to rebuild prophet df. 
        # But if X doesn't have an index matching df_ref (like in cold start predict()), 
        # we generate dummy timestamps since Prophet requires ds
        
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = [pd.Timestamp.now()] * len(X)
        for col in self.feature_columns:
            prophet_df[col] = X[col].values if hasattr(X, 'values') else np.array(X)[col] # handle dicts vs df
            
        pred_df = self.model.predict(prophet_df)
        return pred_df['yhat'].values
    
    def get_metrics(self):
        return self.metrics

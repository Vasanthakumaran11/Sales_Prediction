"""
Hybrid Ensemble Model
Combines predictions from all 8 models using weighted averaging
Based on each model's R² score for optimal performance
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.linear_regression import LinearRegressionModel
from algorithms.decision_tree import DecisionTreeModel
from algorithms.random_forest import RandomForestModel
from algorithms.xgboost_model import XGBoostModel
from algorithms.lightgbm_model import LightGBMModel
from algorithms.svr_model import SVRModel
from algorithms.knn_model import KNNModel
from algorithms.prophet_model import ProphetModel
from src.utils.config import TARGET_VARIABLE


class HybridEnsembleModel:
    """Hybrid Ensemble combining all 8 ML models"""
    
    def __init__(self, model_path=None):
        self.models = {}
        self.weights = {}
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.metrics = {}
        self.individual_metrics = {}
        self.model_path = model_path or "models/algorithms/hybrid_ensemble.pkl"
        
        # Initialize individual models
        self.models['Linear Regression'] = LinearRegressionModel()
        self.models['Decision Tree'] = DecisionTreeModel()
        self.models['Random Forest'] = RandomForestModel()
        self.models['XGBoost'] = XGBoostModel()
        self.models['LightGBM'] = LightGBMModel()
        self.models['SVR'] = SVRModel()
        self.models['KNN'] = KNNModel()
        self.models['Prophet'] = ProphetModel()
    
    def load_data(self, df):
        """Load and prepare data"""
        print("📚 [Hybrid Ensemble] Loading data...")
        
        # Get features from first model
        X, y = self.models['Linear Regression'].load_data(df)
        self.feature_columns = self.models['Linear Regression'].feature_columns
        
        self.df_ref = df
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Split data for all models
        for model in self.models.values():
            model.X_train = self.X_train
            model.X_test = self.X_test
            model.y_train = self.y_train
            model.y_test = self.y_test
            model.feature_columns = self.feature_columns
            if hasattr(model, 'df_ref') or type(model).__name__ == "ProphetModel":
                model.df_ref = getattr(self, 'df_ref', None)
        
        print(f"  ✓ Train size: {self.X_train.shape[0]}")
        print(f"  ✓ Test size: {self.X_test.shape[0]}")
    
    def train(self):
        """Train all 8 individual models"""
        print("\n" + "="*80)
        print("🤖 [HYBRID ENSEMBLE] Training all 8 models...")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\n📌 Training {model_name}...")
            model.train()
    
    def evaluate(self):
        """Evaluate all models and calculate weights based on R² scores"""
        print("\n" + "="*80)
        print("📊 [HYBRID ENSEMBLE] Evaluating all models...")
        print("="*80)
        
        r2_scores = {}
        
        for model_name, model in self.models.items():
            print(f"\n📌 Evaluating {model_name}...")
            metrics = model.evaluate()
            self.individual_metrics[model_name] = metrics
            r2_scores[model_name] = metrics['test_r2']
        
        # Calculate weights based on R² scores (normalized)
        # Ensure all R² scores are positive by adding offset if needed
        min_r2 = min(r2_scores.values())
        if min_r2 < 0:
            adjusted_r2 = {k: v - min_r2 + 0.01 for k, v in r2_scores.items()}
        else:
            adjusted_r2 = r2_scores
        
        total_r2 = sum(adjusted_r2.values())
        self.weights = {k: v / total_r2 for k, v in adjusted_r2.items()}
        
        print("\n" + "-"*80)
        print("📊 MODEL WEIGHTS (Based on R² Score):")
        print("-"*80)
        for model_name, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model_name:20s}: {weight:6.2%}")
        
        return self.individual_metrics
    
    def train_all(self, X, y):
        """Train all models with given data"""
        print("\n" + "="*80)
        print("🚀 [HYBRID ENSEMBLE] FULL TRAINING PIPELINE")
        print("="*80)
        
        # Prepare data
        self.split_data(X, y)
        
        # Train all models
        self.train()
        
        # Evaluate all models
        self.evaluate()
        
        # Make predictions and evaluate ensemble
        self._evaluate_ensemble()
    
    def _evaluate_ensemble(self):
        """Evaluate the ensemble performance"""
        print("\n" + "="*80)
        print("🎯 [HYBRID ENSEMBLE] ENSEMBLE PERFORMANCE")
        print("="*80)
        
        # Get predictions from all models
        y_pred_train = self._ensemble_predict(self.X_train)
        y_pred_test = self._ensemble_predict(self.X_test)
        
        # Calculate ensemble metrics
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        train_r2 = r2_score(self.y_train, y_pred_train)
        
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
            'model_name': 'Hybrid Ensemble'
        }
        
        print(f"\n  Train MAE: {train_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²: {test_r2:.4f} ⭐⭐⭐ ENSEMBLE")
        
        return self.metrics
    
    def _ensemble_predict(self, X):
        """Make ensemble predictions using weighted averaging"""
        predictions = []
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 1/len(self.models))
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
    
    def predict(self, X):
        """Make predictions using the ensemble"""
        return self._ensemble_predict(X)
    
    def save_models(self):
        """Save all models to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save individual models
        for model_name, model in self.models.items():
            model.save_model()
        
        # Save ensemble metadata
        ensemble_data = {
            'weights': self.weights,
            'individual_metrics': self.individual_metrics,
            'ensemble_metrics': self.metrics,
            'feature_columns': self.feature_columns
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"\n  ✅ Hybrid Ensemble saved to {self.model_path}")
    
    def load_models(self):
        """Load all models from disk"""
        for model in self.models.values():
            try:
                model.load_model()
            except:
                pass
        
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                ensemble_data = pickle.load(f)
                self.weights = ensemble_data.get('weights', {})
                self.individual_metrics = ensemble_data.get('individual_metrics', {})
                self.metrics = ensemble_data.get('ensemble_metrics', {})
                self.feature_columns = ensemble_data.get('feature_columns', [])
            print(f"  ✅ Hybrid Ensemble loaded from {self.model_path}")
    
    def get_metrics(self):
        """Return ensemble metrics"""
        return self.metrics
    
    def get_all_metrics(self):
        """Return all metrics (individual + ensemble)"""
        return {
            'individual_models': self.individual_metrics,
            'ensemble': self.metrics,
            'weights': self.weights
        }
    
    def compare_models(self):
        """Get detailed comparison of all models"""
        comparison = pd.DataFrame(self.individual_metrics).T
        
        # Add ensemble metrics
        ensemble_df = pd.DataFrame([self.metrics])
        comparison = pd.concat([comparison, ensemble_df], ignore_index=False)
        
        return comparison

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.linear_regression import LinearRegressionModel
from algorithms.decision_tree import DecisionTreeModel
from algorithms.random_forest import RandomForestModel
from algorithms.xgboost_model import XGBoostModel
from algorithms.lightgbm_model import LightGBMModel
from algorithms.svr_model import SVRModel
from algorithms.knn_model import KNNModel
from algorithms.hybrid_ensemble import HybridEnsembleModel
from algorithms.model_comparison import ModelComparison
from src.preprocessing.preprocess import DataPreprocessor
from src.utils.config import BASE_PROCESSED_PATH


class ModelTrainer:
    """Orchestrator for model training and comparison"""
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.hybrid_model = None
        self.comparison = ModelComparison()
    
    def load_data(self):
        """Load processed data"""
        print("\n" + "="*80)
        print("📂 LOADING DATA")
        print("="*80)
        
        if not os.path.exists(BASE_PROCESSED_PATH):
            print("⚠️  Processed data not found. Running preprocessing...")
            preprocessor = DataPreprocessor()
            self.df, _ = preprocessor.process()
        else:
            self.df = pd.read_csv(BASE_PROCESSED_PATH)
            print(f"✅ Loaded data: {self.df.shape}")
        
        return self.df
    
    def train_single_model(self, model_name):
        """Train a single model"""
        print("\n" + "="*80)
        print(f"🚀 TRAINING SINGLE MODEL: {model_name}")
        print("="*80)
        
        if model_name == 'Linear Regression':
            model = LinearRegressionModel()
        elif model_name == 'Decision Tree':
            model = DecisionTreeModel()
        elif model_name == 'Random Forest':
            model = RandomForestModel()
        elif model_name == 'XGBoost':
            model = XGBoostModel()
        elif model_name == 'LightGBM':
            model = LightGBMModel()
        elif model_name == 'SVR':
            model = SVRModel()
        elif model_name == 'KNN':
            model = KNNModel()
        else:
            print(f"❌ Unknown model: {model_name}")
            return None
        
        # Load data
        X, y = model.load_data(self.df)
        
        # Split data
        model.split_data(X, y)
        
        # Train
        model.train()
        
        # Evaluate
        metrics = model.evaluate()
        
        # Save
        model.save_model()
        
        self.models[model_name] = model
        self.comparison.add_model_metrics(model_name, metrics)
        
        return model
    
    def train_all_individual_models(self):
        """Train all 7 individual models"""
        print("\n" + "="*80)
        print("🚀 TRAINING ALL 7 INDIVIDUAL MODELS")
        print("="*80)
        
        model_names = [
            'Linear Regression',
            'Decision Tree',
            'Random Forest',
            'XGBoost',
            'LightGBM',
            'SVR',
            'KNN'
        ]
        
        for model_name in model_names:
            self.train_single_model(model_name)
        
        print("\n" + "="*80)
        print("✅ ALL INDIVIDUAL MODELS TRAINED SUCCESSFULLY")
        print("="*80)
    
    def train_hybrid_ensemble(self):
        """Train hybrid ensemble combining all models"""
        print("\n" + "="*80)
        print("🚀 TRAINING HYBRID ENSEMBLE MODEL")
        print("="*80)
        
        self.hybrid_model = HybridEnsembleModel()
        
        # Load data
        X, y = self.hybrid_model.load_data(self.df)
        
        # Train all and get comparison
        self.hybrid_model.train_all(X, y)
        
        # Save models
        self.hybrid_model.save_models()
        
        # Add to comparison
        self.comparison.add_model_metrics('Hybrid Ensemble', 
                                         self.hybrid_model.metrics)
        
        print("\n" + "="*80)
        print("✅ HYBRID ENSEMBLE TRAINED SUCCESSFULLY")
        print("="*80)
    
    def run_full_pipeline(self):
        """Run complete training pipeline"""
        print("\n" + "="*100)
        print("🎯 COMPLETE MODEL TRAINING & COMPARISON PIPELINE")
        print("="*100)
        
        # Load data
        self.load_data()
        
        # Train individual models
        self.train_all_individual_models()
        
        # Train hybrid ensemble
        self.train_hybrid_ensemble()
        
        # Print comparison report
        self.print_summary_report()
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        self.comparison.print_summary_report()
    
    def print_detailed_comparison(self):
        """Print detailed metrics for all models"""
        self.comparison.print_detailed_comparison()
    
    def get_comparison_dataframe(self):
        """Get comparison as DataFrame"""
        return self.comparison.get_comparison_dataframe()
    
    def export_report(self, filename='model_comparison.csv'):
        """Export comparison report to CSV"""
        self.comparison.export_to_csv(filename)


def main():
    """Main execution function"""
    trainer = ModelTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train all models and ensemble
    trainer.run_full_pipeline()
    
    # Print detailed comparison
    trainer.print_detailed_comparison()
    
    # Export report
    trainer.export_report()


if __name__ == "__main__":
    main()

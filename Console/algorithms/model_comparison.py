"""
Model Comparison Utility
Compare performance metrics of all models side by side
Generate detailed comparison reports and visualizations
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
import os


class ModelComparison:
    """Compare multiple models and display detailed metrics"""
    
    def __init__(self):
        self.metrics_data = {}
        self.comparison_df = None
    
    def add_model_metrics(self, model_name, metrics):
        """Add metrics for a model"""
        self.metrics_data[model_name] = metrics
    
    def create_comparison_table(self):
        """Create a comparison DataFrame"""
        df_list = []
        
        for model_name, metrics in self.metrics_data.items():
            df_list.append({
                'Model': model_name,
                'Train MAE': metrics.get('train_mae', 0),
                'Test MAE': metrics.get('test_mae', 0),
                'Train RMSE': metrics.get('train_rmse', 0),
                'Test RMSE': metrics.get('test_rmse', 0),
                'Train R²': metrics.get('train_r2', 0),
                'Test R²': metrics.get('test_r2', 0)
            })
        
        self.comparison_df = pd.DataFrame(df_list)
        return self.comparison_df
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        print("\n" + "="*120)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*120)
        
        # Sort by Test R² (descending)
        sorted_df = self.comparison_df.sort_values('Test R²', ascending=False)
        
        # Format for display
        display_df = sorted_df.copy()
        display_df['Train MAE'] = display_df['Train MAE'].apply(lambda x: f"{x:.4f}")
        display_df['Test MAE'] = display_df['Test MAE'].apply(lambda x: f"{x:.4f}")
        display_df['Train RMSE'] = display_df['Train RMSE'].apply(lambda x: f"{x:.4f}")
        display_df['Test RMSE'] = display_df['Test RMSE'].apply(lambda x: f"{x:.4f}")
        display_df['Train R²'] = display_df['Train R²'].apply(lambda x: f"{x:.4f}")
        display_df['Test R²'] = display_df['Test R²'].apply(lambda x: f"{x:.4f} ⭐")
        
        print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        print("="*120 + "\n")
    
    def print_detailed_comparison(self):
        """Print detailed metrics for each model"""
        if not self.metrics_data:
            print("No metrics data available")
            return
        
        print("\n" + "="*100)
        print("📈 DETAILED MODEL METRICS")
        print("="*100)
        
        for model_name, metrics in self.metrics_data.items():
            print(f"\n┌─ {model_name}")
            print(f"├─ Training Performance:")
            print(f"│  ├─ MAE:  {metrics.get('train_mae', 0):.4f}")
            print(f"│  ├─ RMSE: {metrics.get('train_rmse', 0):.4f}")
            print(f"│  └─ R²:   {metrics.get('train_r2', 0):.4f}")
            print(f"├─ Testing Performance:")
            print(f"│  ├─ MAE:  {metrics.get('test_mae', 0):.4f}")
            print(f"│  ├─ RMSE: {metrics.get('test_rmse', 0):.4f}")
            print(f"│  └─ R²:   {metrics.get('test_r2', 0):.4f} ⭐")
            print(f"└─")
    
    def get_best_model(self):
        """Get the best model based on Test R²"""
        if not self.metrics_data:
            return None
        
        best_model = max(
            self.metrics_data.items(),
            key=lambda x: x[1].get('test_r2', 0)
        )
        
        return best_model
    
    def print_best_model(self):
        """Print best model summary"""
        best_model, metrics = self.get_best_model()
        
        print("\n" + "="*80)
        print(f"🏆 BEST PERFORMING MODEL: {best_model}")
        print("="*80)
        print(f"  Test R² Score: {metrics.get('test_r2', 0):.4f}")
        print(f"  Test MAE:      {metrics.get('test_mae', 0):.4f}")
        print(f"  Test RMSE:     {metrics.get('test_rmse', 0):.4f}")
        print(f"  Train R²:      {metrics.get('train_r2', 0):.4f}")
        print("="*80 + "\n")
    
    def print_metric_rankings(self):
        """Print models ranked by each metric"""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        print("\n" + "="*100)
        print("🎯 MODEL RANKINGS BY METRIC")
        print("="*100)
        
        metrics_to_rank = ['Test R²', 'Test RMSE', 'Test MAE']
        
        for metric in metrics_to_rank:
            print(f"\n📌 Ranked by {metric} (Lower is Better for Error, Higher for R²):")
            
            if metric == 'Test R²':
                ranked = self.comparison_df.nlargest(6, metric)[['Model', metric]]
            else:
                ranked = self.comparison_df.nsmallest(6, metric)[['Model', metric]]
            
            for idx, (_, row) in enumerate(ranked.iterrows(), 1):
                print(f"   {idx}. {row['Model']:20s} - {row[metric]:8.4f}")
    
    def calculate_improvement(self):
        """Calculate improvement of ensemble over individual models"""
        if 'Hybrid Ensemble' not in self.metrics_data:
            print("Hybrid Ensemble metrics not available")
            return None
        
        ensemble_r2 = self.metrics_data['Hybrid Ensemble']['test_r2']
        ensemble_mae = self.metrics_data['Hybrid Ensemble']['test_mae']
        
        print("\n" + "="*100)
        print("📈 HYBRID ENSEMBLE IMPROVEMENT OVER INDIVIDUAL MODELS")
        print("="*100)
        
        individual_models = {k: v for k, v in self.metrics_data.items() 
                            if k != 'Hybrid Ensemble'}
        
        print(f"\nEnsemble Test R²: {ensemble_r2:.4f}")
        print(f"Ensemble Test MAE: {ensemble_mae:.4f}\n")
        
        print("Model                    | Δ R²      | Δ MAE     | Δ RMSE    | Status")
        print("-"*100)
        
        improvements = {}
        for model_name, metrics in individual_models.items():
            model_r2 = metrics['test_r2']
            model_mae = metrics['test_mae']
            model_rmse = metrics['test_rmse']
            
            delta_r2 = ensemble_r2 - model_r2
            delta_mae = ensemble_mae - model_mae
            delta_rmse = metrics['test_rmse'] - ensemble_r2  # Approximation
            
            status = "✅ BETTER" if delta_r2 > 0 else "⚠️ EQUAL/WORSE"
            
            print(f"{model_name:24s} | {delta_r2:+7.4f}   | {delta_mae:+7.4f}   | {model_rmse:7.4f}   | {status}")
            
            improvements[model_name] = {
                'delta_r2': delta_r2,
                'delta_mae': delta_mae,
                'status': status
            }
        
        print("-"*100)
        
        return improvements
    
    def print_summary_report(self):
        """Print comprehensive summary report"""
        print("\n" + "="*120)
        print("📊 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*120)
        
        self.print_comparison_table()
        self.print_best_model()
        self.print_metric_rankings()
        self.calculate_improvement()
        
        print("\n" + "="*120)
        print("KEY INSIGHTS:")
        print("="*120)
        print("  • Each model has different strengths and weaknesses")
        print("  • R² Score: Measure of variance explained (1.0 = perfect)")
        print("  • MAE: Average absolute error (lower is better)")
        print("  • RMSE: Root mean squared error (lower is better)")
        print("  • Hybrid Ensemble combines all models for robust predictions")
        print("="*120 + "\n")
    
    def export_to_csv(self, filename='model_comparison.csv'):
        """Export comparison to CSV file"""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        self.comparison_df.to_csv(filename, index=False)
        print(f"✅ Comparison exported to {filename}")
    
    def get_comparison_dataframe(self):
        """Return the comparison DataFrame"""
        if self.comparison_df is None:
            self.create_comparison_table()
        
        return self.comparison_df

"""
Main Interactive Application
Terminal-based interface for the Retail Forecasting System
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

from data.data_engine import UserDataEngine
from models.predict import PredictionEngine
from models.train_personalized import PersonalizedModelTrainer
from pipeline.run_pipeline import CompletePipeline
from utils.inventory import InventoryOptimizer, print_inventory_report
from algorithms.model_trainer import ModelTrainer
from algorithms.model_comparison import ModelComparison
import pandas as pd
from datetime import datetime


class RetailForecastingApp:
    """Interactive application for retail forecasting"""

    def __init__(self):
        self.user_engine = None
        self.prediction_engine = None
        self.use_personalized = False

    def initialize_system(self):
        """Initialize the system"""
        print("\n" + "=" * 80)
        print("🏪 RETAIL FORECASTING SYSTEM")
        print("=" * 80)

        # Load user engine
        try:
            self.user_engine = UserDataEngine()
            print("✅ User data engine loaded")
        except Exception as e:
            print(f"❌ Error loading user engine: {e}")
            return False

        # Load prediction engine
        try:
            self.prediction_engine = PredictionEngine(
                use_personalized=self.use_personalized
            )
            print("✅ Prediction engine loaded")
        except Exception as e:
            print(f"❌ Error loading prediction engine: {e}")
            return False

        return True

    def menu(self):
        """Main menu"""
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("""
1. Model Training & Execution
   1.1 - Run model training pipeline and display evaluation metrics
   
2. Smart Grocery System
   2.1 - Open interactive Smart Grocery App
   
3. Model Analysis
   3.1 - Show evaluation metrics and accuracy of all models
   
4. Exit

Enter your choice (e.g., 1.1, 2.1, 3.1):
        """)

    def run_complete_pipeline(self):
        """Run complete pipeline"""
        pipeline = CompletePipeline()
        pipeline.run_complete_pipeline()

    def record_sale(self):
        """Record a sale"""
        print("\n📝 RECORD SALE")
        print("-" * 80)

        product_name = input("Product name: ").strip()
        try:
            units_sold = int(input("Units sold: "))
            unit_price = float(input("Unit price (₹): "))
            discount = float(input("Discount (0-1, e.g., 0.1 for 10% discount): "))
            promo = input("Promotional sale? (y/n): ").lower() == "y"
            holiday = input("Holiday? (y/n): ").lower() == "y"

            if self.user_engine.record_sale(
                product_name=product_name,
                units_sold=units_sold,
                unit_price=unit_price,
                discount=discount,
                promo=promo,
                holiday=holiday,
            ):
                print("✅ Sale recorded successfully")
            else:
                print("❌ Error recording sale")
        except ValueError:
            print("❌ Invalid input format")

    def view_sales_summary(self):
        """View sales summary"""
        print("\n📊 SALES SUMMARY")
        print("-" * 80)

        self.user_engine.reload()
        summary = self.user_engine.get_sales_summary()

        if summary is None or summary.empty:
            print("No sales data available")
        else:
            print(summary)

    def get_inventory_recommendations(self):
        """Get inventory recommendations"""
        print("\n📦 INVENTORY RECOMMENDATIONS")
        print("-" * 80)

        self.user_engine.reload()

        if self.user_engine.sales.empty:
            print("No sales data available for analysis")
            return

        # Calculate recommendations
        recommendations = []

        for product in self.user_engine.sales["Product_Name"].unique():
            product_data = self.user_engine.sales[
                self.user_engine.sales["Product_Name"] == product
            ]
            mean_demand = product_data["Units_Sold"].mean()
            demand_std = product_data["Units_Sold"].std()

            if pd.isna(demand_std):
                demand_std = 0

            current_inventory = int(mean_demand * 3)  # Estimate

            rec = InventoryOptimizer.get_inventory_recommendation(
                product, current_inventory, mean_demand, demand_std
            )
            recommendations.append(rec)

        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            print_inventory_report(rec_df)
        else:
            print("No recommendations available")

    def get_forecast(self):
        """Get demand forecast"""
        print("\n🔮 DEMAND FORECAST")
        print("-" * 80)

        product_name = input("Enter product name: ").strip()

        # Create sample features for prediction
        features = {
            "Store_Type_Encoded": 0,
            "Location_Type_Encoded": 0,
            "Category_Encoded": 0,
            "Units_Stocked": 50,
            "Unit_Price": 100,
            "Discount": 0,
            "Day_of_Week": 2,
            "Is_Weekend": 0,
            "Lag_1_Units_Sold": 30,
            "Lag_7_Units_Sold": 28,
            "Rolling_Mean_7d_Units_Sold": 29,
            "Rolling_Std_7d_Units_Sold": 5,
            "Sell_Through_Ratio": 0.6,
            "Stock_Remaining_Ratio": 0.4,
            "Revenue_Per_Unit_Stocked": 50,
            "Discount_Applied": 0,
            "High_Demand_Flag": 0,
            "Low_Stock_Flag": 0,
            "Is_Festival": 0,
            "Day_Type_Encoded": 0,
        }

        try:
            forecast = self.prediction_engine.predict_single(features)
            print(f"\n🎯 Forecast for {product_name}:")
            print(f"   Predicted Units to Sell: {forecast:.0f} units")
        except Exception as e:
            print(f"❌ Error generating forecast: {e}")

    def check_model_status(self):
        """Check model status"""
        print("\n📊 MODEL STATUS")
        print("-" * 80)

        import json
        from utils.config import MODEL_METADATA_PATH

        if os.path.exists(MODEL_METADATA_PATH):
            with open(MODEL_METADATA_PATH, "r") as f:
                metadata = json.load(f)

            print(f"Base Model: {metadata.get('model_name')}")
            print(f"Training Date: {metadata.get('training_date')}")
            print(f"Test R²: {metadata.get('metrics', {}).get('test_r2', 'N/A')}")

            if metadata.get("personalized"):
                print(f"\n✅ Personalized Model (ACTIVE)")
                print(
                    f"   Training Date: {metadata.get('personalized_training_date')}"
                )
                print(
                    f"   Test R²: {metadata.get('personalized_metrics', {}).get('test_r2', 'N/A')}"
                )
            else:
                print("\n⚠️  Personalized Model: Not yet trained")
                print("   Collecting user data for personalization...")
        else:
            print("No model metadata found")

    def trigger_retraining(self):
        """Trigger model retraining"""
        print("\n🔄 MODEL RETRAINING")
        print("-" * 80)

        trainer = PersonalizedModelTrainer()
        should_retrain, data_count = trainer.check_retraining_required()

        if should_retrain:
            print(f"✅ Retraining triggered ({data_count} user data points)")
            trainer.retrain()
            self.use_personalized = True
            self.prediction_engine = PredictionEngine(use_personalized=True)
            print("✅ Now using personalized model for predictions")
        else:
            print(f"⚠️  Insufficient data ({data_count} points collected)")
            print(f"   Required: {trainer.check_retraining_required()} points")

    def train_single_algorithm(self):
        """Train a single model from algorithms folder"""
        print("\n" + "="*80)
        print("🤖 TRAIN SINGLE ALGORITHM MODEL")
        print("="*80)
        
        models = [
            'Linear Regression',
            'Decision Tree',
            'Random Forest',
            'XGBoost',
            'LightGBM'
        ]
        
        print("\nAvailable Models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
        
        try:
            choice = int(input("\nSelect model (1-5): "))
            if 1 <= choice <= 5:
                model_name = models[choice - 1]
                
                trainer = ModelTrainer()
                trainer.load_data()
                trainer.train_single_model(model_name)
                
                print(f"\n✅ {model_name} trained and saved successfully!")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    def train_all_algorithms(self):
        """Train all 8 individual algorithms"""
        print("\n" + "="*80)
        print("🤖 TRAIN ALL 8 INDIVIDUAL MODELS")
        print("="*80)
        
        confirm = input("\nThis will train all 8 models. Continue? (y/n): ").lower()
        if confirm == 'y':
            trainer = ModelTrainer()
            trainer.load_data()
            trainer.train_all_individual_models()
            
            print("\n" + "="*80)
            print("✅ ALL 8 MODELS TRAINED SUCCESSFULLY")
            print("="*80)
    
    def train_hybrid_ensemble_model(self):
        """Train the hybrid ensemble model"""
        print("\n" + "="*80)
        print("🎯 TRAIN HYBRID ENSEMBLE MODEL")
        print("="*80)
        
        print("\nThe Hybrid Ensemble combines predictions from all 8 models")
        print("using weighted averaging based on their R² scores.")
        
        confirm = input("\nContinue with hybrid ensemble training? (y/n): ").lower()
        if confirm == 'y':
            trainer = ModelTrainer()
            trainer.load_data()
            trainer.train_hybrid_ensemble()
            
            # The model_trainer already prints the success message.
    
    def view_model_comparison(self):
        """View detailed model comparison report"""
        print("\n" + "="*100)
        print("📊 MODEL COMPARISON REPORT")
        print("="*100)
        
        trainer = ModelTrainer()
        trainer.load_data()
        
        # Check if individual metrics exist
        print("\nLoading model metrics...")
        print("(Make sure you have trained the models first using options 6.2 or 6.3)")
        
        try:
            # Try to load metrics from saved models
            from algorithms.linear_regression import LinearRegressionModel
            from algorithms.decision_tree import DecisionTreeModel
            from algorithms.random_forest import RandomForestModel
            from algorithms.xgboost_model import XGBoostModel
            from algorithms.lightgbm_model import LightGBMModel
            
            models = [
                LinearRegressionModel(),
                DecisionTreeModel(),
                RandomForestModel(),
                XGBoostModel(),
                LightGBMModel()
            ]
            
            comparison = ModelComparison()
            
            for model in models:
                try:
                    model.load_model()
                    if model.metrics:
                        comparison.add_model_metrics(
                            model.metrics.get('model_name', 'Unknown'),
                            model.metrics
                        )
                except:
                    pass
            
            if comparison.metrics_data:
                comparison.print_summary_report()
            else:
                print("⚠️  No model metrics found. Please train models first.")
        
        except Exception as e:
            print(f"❌ Error loading model metrics: {e}")
    
    def view_detailed_metrics(self):
        """View detailed metrics for all trained models"""
        print("\n" + "="*100)
        print("📈 DETAILED MODEL METRICS")
        print("="*100)
        
        try:
            from algorithms.linear_regression import LinearRegressionModel
            from algorithms.decision_tree import DecisionTreeModel
            from algorithms.random_forest import RandomForestModel
            from algorithms.xgboost_model import XGBoostModel
            from algorithms.lightgbm_model import LightGBMModel
            
            models = [
                ('Linear Regression', LinearRegressionModel()),
                ('Decision Tree', DecisionTreeModel()),
                ('Random Forest', RandomForestModel()),
                ('XGBoost', XGBoostModel()),
                ('LightGBM', LightGBMModel())
            ]
            
            for model_name, model in models:
                try:
                    model.load_model()
                    if model.metrics:
                        print(f"\n┌─ {model_name}")
                        print(f"├─ Training Performance:")
                        print(f"│  ├─ MAE:  {model.metrics.get('train_mae', 0):.4f}")
                        print(f"│  ├─ RMSE: {model.metrics.get('train_rmse', 0):.4f}")
                        print(f"│  └─ R²:   {model.metrics.get('train_r2', 0):.4f}")
                        print(f"├─ Testing Performance:")
                        print(f"│  ├─ MAE:  {model.metrics.get('test_mae', 0):.4f}")
                        print(f"│  ├─ RMSE: {model.metrics.get('test_rmse', 0):.4f}")
                        print(f"│  └─ R²:   {model.metrics.get('test_r2', 0):.4f} ⭐")
                        print(f"└─")
                except:
                    print(f"\n⚠️  {model_name}: Metrics not available (train model first)")
        
        except Exception as e:
            print(f"❌ Error loading metrics: {e}")

    def run(self):
        """Main application loop"""
        if not self.initialize_system():
            print("❌ Failed to initialize system")
            return

        while True:
            self.menu()
            choice = input().strip()

            if choice == "1.1":
                self.run_complete_pipeline()
                self.train_all_algorithms()
                self.train_hybrid_ensemble_model()
                # Training and evaluating are handled entirely within the functions.
            elif choice == "2.1":
                print("\n" + "=" * 80)
                print("🏪 LAUNCHING SMART GROCERY SYSTEM")
                print("=" * 80)
                try:
                    from app_enhanced import SmartGroceryApp
                    app = SmartGroceryApp()
                    app.run()
                except KeyboardInterrupt:
                    print("\nReturned to Main Menu")
                except Exception as e:
                    print(f"❌ Error running Smart Grocery App: {e}")
            elif choice == "3.1":
                self.view_model_comparison()
                self.view_detailed_metrics()
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")


def main():
    """Main entry point"""
    app = RetailForecastingApp()
    app.run()


if __name__ == "__main__":
    main()

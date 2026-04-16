"""
Main Interactive Application
Terminal-based interface for the Retail Forecasting System
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.data_engine import UserDataEngine
from models.predict import PredictionEngine
from models.train_personalized import PersonalizedModelTrainer
from pipeline.run_pipeline import CompletePipeline
from utils.inventory import InventoryOptimizer, print_inventory_report
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
1. Setup & Pipeline
   1.1 - Run complete pipeline (Dataset + Training)
   
2. Sales Management
   2.1 - Record daily sales
   2.2 - View sales summary
   
3. inventory Management
   3.1 - Get inventory recommendations
   3.2 - View inventory status
   
4. Demand Forecasting
   4.1 - Get demand forecast
   4.2 - View forecast insights
   
5. Model Management
   5.1 - Check personalized model status
   5.2 - Trigger retraining
   
6. Exit

Enter your choice (e.g., 1.1, 2.1, etc.):
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
            elif choice == "2.1":
                self.record_sale()
            elif choice == "2.2":
                self.view_sales_summary()
            elif choice == "3.1":
                self.get_inventory_recommendations()
            elif choice == "3.2":
                print("Inventory status: Use 3.1 for detailed recommendations")
            elif choice == "4.1":
                self.get_forecast()
            elif choice == "4.2":
                print("Forecast insights available through 4.1")
            elif choice == "5.1":
                self.check_model_status()
            elif choice == "5.2":
                self.trigger_retraining()
            elif choice == "6":
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

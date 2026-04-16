"""
Main Pipeline Runner
Orchestrates the complete ML pipeline from data generation to prediction
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import SyntheticDataGenerator
from preprocessing.preprocess import DataPreprocessor
from models.train_base import BaseModelTrainer
from models.train_personalized import PersonalizedModelTrainer
from data.data_engine import UserDataEngine
from utils.inventory import InventoryOptimizer, print_inventory_report
from models.predict import PredictionEngine


class CompletePipeline:
    """Execute complete ML pipeline"""

    def __init__(self):
        pass

    def stage_1_generate_base_dataset(self):
        """Stage 1: Generate synthetic base dataset"""
        print("\n🎯 STAGE 1: BASE DATASET GENERATION")
        generator = SyntheticDataGenerator()
        df = generator.generate_data()
        return df

    def stage_2_preprocess_data(self):
        """Stage 2: Preprocess and engineer features"""
        print("\n🎯 STAGE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
        preprocessor = DataPreprocessor()
        df, features = preprocessor.process()
        return df, features

    def stage_3_train_base_model(self):
        """Stage 3: Train base ML model"""
        print("\n🎯 STAGE 3: BASE MODEL TRAINING")
        trainer = BaseModelTrainer()
        trainer.train()
        print("\n✅ Base model training complete")

    def stage_4_initialize_user_system(self):
        """Stage 4: Initialize user data collection system"""
        print("\n🎯 STAGE 4: USER DATA SYSTEM INITIALIZATION")
        user_engine = UserDataEngine()
        print("✅ User data system ready")
        print(f"   - Products initialized: {len(user_engine.products)}")
        print(f"   - Ready to accept sales data")
        return user_engine

    def stage_5_test_predictions(self):
        """Stage 5: Test predictions with base model"""
        print("\n🎯 STAGE 5: TEST PREDICTIONS (COLD START)")
        try:
            engine = PredictionEngine(use_personalized=False)
            print("✅ Prediction engine loaded successfully")
            print(f"   - Model: {engine.model_name}")
            print(f"   - Features: {len(engine.feature_columns)}")
        except Exception as e:
            print(f"❌ Error loading prediction engine: {str(e)}")

    def run_complete_pipeline(self):
        """Execute complete pipeline"""
        print("\n" + "=" * 100)
        print("AI-BASED SMART GROCERY ACCOUNT & DEMAND FORECASTING SYSTEM")
        print("=" * 100)

        # Stage 1: Generate base dataset
        self.stage_1_generate_base_dataset()

        # Stage 2: Preprocess data
        self.stage_2_preprocess_data()

        # Stage 3: Train base model
        self.stage_3_train_base_model()

        # Stage 4: Initialize user system
        self.stage_4_initialize_user_system()

        # Stage 5: Test predictions
        self.stage_5_test_predictions()

        print("\n" + "=" * 100)
        print("COMPLETE PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 100)
        print(
            """
System is now ready for:
✅ Making predictions with base model (Cold Start)
✅ Collecting user data (sales, purchases)
✅ Retraining with personalized data (after 2-4 weeks)

📖 Next Steps:
1. Use app/app.py for Streamlit UI
2. Or use main.py for terminal-based interaction
        """
        )


def main():
    """Main entry point"""
    pipeline = CompletePipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()

"""
Synthetic Dataset Generator for Retail Forecasting System
Generates realistic grocery retail data for Tamil Nadu region
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    BASE_DATASET_PATH,
    DAYS_RANGE,
    START_DATE,
    STORE_TYPES,
    LOCATION_TYPES,
    NUM_STORES,
    PRODUCTS,
    WEEKEND_MULTIPLIER,
    FESTIVAL_MULTIPLIER,
    FESTIVAL_DATES,
    DEMAND_VARIABILITY,
    STOCKOUT_PROB,
    TURNOVER_DAYS,
)


class SyntheticDataGenerator:
    """Generate synthetic grocery retail dataset"""

    def __init__(self):
        self.data = []
        self.start_date = datetime.strptime(START_DATE, "%d-%m-%Y")
        self.np_random = np.random.RandomState(42)

    def generate_store_info(self):
        """Generate store information"""
        stores = []
        for i in range(NUM_STORES):
            store_type = STORE_TYPES[i % len(STORE_TYPES)]
            location = LOCATION_TYPES[i % len(LOCATION_TYPES)]

            # Sales base by store type
            if store_type == "Supermarket":
                base_demand = self.np_random.normal(100, 15)
            elif store_type == "Medium":
                base_demand = self.np_random.normal(60, 10)
            else:
                base_demand = self.np_random.normal(35, 8)

            stores.append(
                {
                    "Store_ID": f"STORE_{i+1:03d}",
                    "Store_Type": store_type,
                    "Location_Type": location,
                    "Base_Demand": max(10, base_demand),
                }
            )
        return stores

    def generate_date_features(self, date):
        """Generate date-based features"""
        day_of_week = date.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        day_type = "Weekend" if is_weekend else "Weekday"

        # Festival check
        date_str = date.strftime("%d-%m-%Y")
        is_festival = 1 if date_str in FESTIVAL_DATES else 0

        return {
            "Date": date_str,
            "Day": date.day,
            "Month": date.month,
            "Day_of_Week": day_of_week,
            "Is_Weekend": is_weekend,
            "Day_Type": day_type,
            "Is_Festival": is_festival,
        }

    def calculate_units_stocked(self, item_name, store_type):
        """Calculate units to stock based on item and store type"""
        category = PRODUCTS[item_name]["category"]
        min_days, max_days = TURNOVER_DAYS[category]

        # Base units
        if store_type == "Supermarket":
            base_units = self.np_random.uniform(50, 150)
        elif store_type == "Medium":
            base_units = self.np_random.uniform(30, 80)
        else:
            base_units = self.np_random.uniform(15, 40)

        # Adjust for turnover
        turnover_factor = self.np_random.uniform(min_days, max_days)
        units_stocked = max(1, int(base_units * (turnover_factor / 7)))

        return units_stocked

    def calculate_units_sold(self, units_stocked, store_type, item_name, date_features):
        """Calculate units sold with realistic patterns"""
        base_demand = self.np_random.uniform(0.6, 0.95)

        # Demand multipliers
        demand = base_demand

        if date_features["Is_Weekend"]:
            demand *= WEEKEND_MULTIPLIER
        if date_features["Is_Festival"]:
            demand *= FESTIVAL_MULTIPLIER

        # Random variability
        demand *= self.np_random.normal(1, DEMAND_VARIABILITY)

        # Calculate sold units
        units_sold = int(demand * units_stocked)

        # Check for stockout
        stockout_prob = STOCKOUT_PROB[store_type]
        if self.np_random.random() < stockout_prob and units_stocked < 20:
            units_sold = units_stocked

        return min(units_stocked, max(0, units_sold))

    def determine_demand_level(self, units_sold, units_stocked):
        """Determine demand level based on sales"""
        if units_stocked == 0:
            return "Stockout"

        sell_through = units_sold / units_stocked

        if sell_through >= 0.9:
            return "High"
        elif sell_through >= 0.5:
            return "Medium"
        else:
            return "Low"

    def generate_data(self):
        """Generate complete synthetic dataset"""
        print("🏭 Generating synthetic retail dataset...")

        stores = self.generate_store_info()
        data = []

        for day in range(DAYS_RANGE):
            current_date = self.start_date + timedelta(days=day)
            date_features = self.generate_date_features(current_date)

            for store in stores:
                for item_name, item_info in PRODUCTS.items():
                    # Pricing
                    price_min, price_max = item_info["price_range"]
                    unit_price = self.np_random.uniform(price_min, price_max)

                    # Discount (occasional, 5-15%)
                    discount = 0
                    if self.np_random.random() < 0.15:
                        discount = self.np_random.uniform(0.05, 0.15)

                    # Inventory
                    units_stocked = self.calculate_units_stocked(
                        item_name, store["Store_Type"]
                    )
                    units_sold = self.calculate_units_sold(
                        units_stocked,
                        store["Store_Type"],
                        item_name,
                        date_features,
                    )
                    units_remaining = units_stocked - units_sold

                    # Revenue
                    revenue = units_sold * unit_price * (1 - discount)

                    # Demand level
                    demand_level = self.determine_demand_level(units_sold, units_stocked)

                    # Suggested next stock (with safety buffer)
                    suggested_next_stock = int(units_sold * 1.1 + 5)

                    # Create record
                    record = {
                        **date_features,
                        "Store_ID": store["Store_ID"],
                        "Store_Type": store["Store_Type"],
                        "Location_Type": store["Location_Type"],
                        "Item_Name": item_name,
                        "Category": item_info["category"],
                        "Units_Stocked": units_stocked,
                        "Units_Sold": units_sold,
                        "Units_Remaining": units_remaining,
                        "Unit_Price": round(unit_price, 2),
                        "Discount": round(discount, 3),
                        "Revenue": round(revenue, 2),
                        "Demand_Level": demand_level,
                        "Suggested_Next_Stock": suggested_next_stock,
                    }

                    data.append(record)

        df = pd.DataFrame(data)

        # Save dataset
        os.makedirs(os.path.dirname(BASE_DATASET_PATH), exist_ok=True)
        df.to_csv(BASE_DATASET_PATH, index=False)

        print(f"✅ Dataset generated: {BASE_DATASET_PATH}")
        print(f"📊 Shape: {df.shape}")
        print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"🏬 Stores: {df['Store_ID'].nunique()}")
        print(f"🛒 Products: {df['Item_Name'].nunique()}")
        print(f"\n📋 Sample data:\n{df.head(10)}")

        return df


def main():
    """Main function to generate synthetic data"""
    generator = SyntheticDataGenerator()
    df = generator.generate_data()

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

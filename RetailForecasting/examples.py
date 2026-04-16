"""
Example Usage Script for Retail Forecasting System
Shows how to use the system programmatically
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.data_engine import UserDataEngine
from models.predict import PredictionEngine
from utils.inventory import InventoryOptimizer, print_inventory_report
import pandas as pd

print("\n" + "=" * 80)
print("EXAMPLE: AI-Based Retail Forecasting System Usage")
print("=" * 80)

# ============================================================================
# EXAMPLE 1: Initialize the System
# ============================================================================
print("\n🎯 EXAMPLE 1: Initialize System Components")
print("-" * 80)

# Load user data engine
user_engine = UserDataEngine()
print(f"✅ User data engine loaded")
print(f"   - Products: {len(user_engine.products)}")
print(f"   - Sales: {len(user_engine.sales)}")
print(f"   - Purchases: {len(user_engine.purchases)}")

# Load prediction engine
prediction_engine = PredictionEngine(use_personalized=False)
print(f"\n✅ Prediction engine loaded")
print(f"   - Model: {prediction_engine.model_name}")
print(f"   - Features: {len(prediction_engine.feature_columns)}")

# ============================================================================
# EXAMPLE 2: Record Sales
# ============================================================================
print("\n\n🎯 EXAMPLE 2: Record Sales Transactions")
print("-" * 80)

sales_data = [
    {"product": "Milk", "units": 20, "price": 25, "discount": 0.1},
    {"product": "Rice", "units": 15, "price": 60, "discount": 0.0},
    {"product": "Cooking Oil", "units": 8, "price": 150, "discount": 0.05},
    {"product": "Bread", "units": 12, "price": 40, "discount": 0.0},
]

print(f"Recording {len(sales_data)} sales transactions...\n")

for sale in sales_data:
    user_engine.record_sale(
        product_name=sale["product"],
        units_sold=sale["units"],
        unit_price=sale["price"],
        discount=sale["discount"],
        promo=False,
        holiday=False,
    )
    print(f"  ✅ {sale['product']}: {sale['units']} units @ ₹{sale['price']}/unit")

# Reload to get updated data
user_engine.reload()
print(f"\n📊 Total sales recorded: {len(user_engine.sales)}")

# ============================================================================
# EXAMPLE 3: View Sales Summary
# ============================================================================
print("\n\n🎯 EXAMPLE 3: Sales Summary Analysis")
print("-" * 80)

sales_summary = user_engine.get_sales_summary()

if sales_summary is not None and not sales_summary.empty:
    print("\nSales by Product:")
    print(sales_summary)
    print(f"\n💰 Total Revenue: ₹{sales_summary['Revenue'].sum():.2f}")
    print(f"📦 Total Units Sold: {int(sales_summary['Units_Sold'].sum())}")
else:
    print("No sales data available")

# ============================================================================
# EXAMPLE 4: Make Demand Predictions
# ============================================================================
print("\n\n🎯 EXAMPLE 4: Demand Forecasting")
print("-" * 80)

print("\nGenerating demand forecasts for the next day...\n")

# Create feature vectors for predictions
forecast_scenarios = [
    {
        "name": "Weekday (Normal Stock)",
        "features": {
            "Day_of_Week": 2,
            "Is_Weekend": 0,
            "Day_Type_Encoded": 0,
            "Is_Festival": 0,
            "Store_Type_Encoded": 1,
            "Location_Type_Encoded": 1,
            "Category_Encoded": 0,
            "Units_Stocked": 50,
            "Units_Remaining": 40,
            "Unit_Price": 100,
            "Discount": 0,
            "Discount_Applied": 0,
            "Revenue": 4000,
            "Suggested_Next_Stock": 55,
            "Lag_1_Units_Sold": 30,
            "Lag_7_Units_Sold": 28,
            "Rolling_Mean_7d_Units_Sold": 29,
            "Rolling_Std_7d_Units_Sold": 5,
            "Rolling_Mean_7d_Revenue": 2900,
            "Rolling_Std_7d_Revenue": 500,
            "Sell_Through_Ratio": 0.6,
            "Stock_Remaining_Ratio": 0.4,
            "Revenue_Per_Unit_Stocked": 50,
            "High_Demand_Flag": 0,
            "Low_Stock_Flag": 0,
        }
    },
    {
        "name": "Weekend (High Demand)",
        "features": {
            "Day_of_Week": 5,
            "Is_Weekend": 1,
            "Day_Type_Encoded": 1,
            "Is_Festival": 0,
            "Store_Type_Encoded": 1,
            "Location_Type_Encoded": 1,
            "Category_Encoded": 0,
            "Units_Stocked": 50,
            "Units_Remaining": 40,
            "Unit_Price": 100,
            "Discount": 0,
            "Discount_Applied": 0,
            "Revenue": 4000,
            "Suggested_Next_Stock": 55,
            "Lag_1_Units_Sold": 35,
            "Lag_7_Units_Sold": 32,
            "Rolling_Mean_7d_Units_Sold": 34,
            "Rolling_Std_7d_Units_Sold": 6,
            "Rolling_Mean_7d_Revenue": 3400,
            "Rolling_Std_7d_Revenue": 600,
            "Sell_Through_Ratio": 0.7,
            "Stock_Remaining_Ratio": 0.3,
            "Revenue_Per_Unit_Stocked": 55,
            "High_Demand_Flag": 1,
            "Low_Stock_Flag": 0,
        }
    },
]

for scenario in forecast_scenarios:
    try:
        forecast = prediction_engine.predict_single(scenario["features"])
        print(f"📈 {scenario['name']}")
        print(f"   Predicted Units to Sell: {forecast:.0f} units")
        print(f"   Revenue Estimate @ ₹100/unit: ₹{forecast * 100:.0f}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")

# ============================================================================
# EXAMPLE 5: Inventory Optimization
# ============================================================================
print("\n🎯 EXAMPLE 5: Inventory Optimization Recommendations")
print("-" * 80)

products_to_analyze = [
    {"name": "Milk", "inventory": 45, "mean_demand": 12, "std_demand": 2},
    {"name": "Rice", "inventory": 25, "mean_demand": 22, "std_demand": 4},
    {"name": "Cooking Oil", "inventory": 5, "mean_demand": 8, "std_demand": 2},
    {"name": "Bread", "inventory": 15, "mean_demand": 18, "std_demand": 3},
]

recommendations = []

print("\nCalculating inventory recommendations...\n")

for product in products_to_analyze:
    rec = InventoryOptimizer.get_inventory_recommendation(
        product["name"],
        product["inventory"],
        product["mean_demand"],
        product["std_demand"],
    )
    recommendations.append(rec)

rec_df = pd.DataFrame(recommendations)

print("Detailed Recommendations:")
print("-" * 80)

for idx, row in rec_df.iterrows():
    print(f"\n{row['product_name']}")
    print(f"  Current Stock: {int(row['current_inventory'])} units")
    print(f"  Daily Demand: {row['mean_daily_demand']:.1f} ± {row['demand_std']:.1f} units")
    print(f"  Safety Stock: {row['safety_stock']:.0f} units")
    print(f"  Reorder Point: {row['reorder_point']:.0f} units")
    print(f"  Quantity to Order: {int(row['quantity_to_order'])} units")
    print(f"  Risk Level: {row['risk_level']}")
    print(f"  Action: {row['action']}")

# ============================================================================
# EXAMPLE 6: Advanced Analytics
# ============================================================================
print("\n\n🎯 EXAMPLE 6: Advanced Analytics")
print("-" * 80)

print("\nCalculating key metrics...\n")

# Get sales dataframe
sales = user_engine.sales

if not sales.empty:
    print(f"📊 Total Transactions: {len(sales)}")
    print(f"💰 Total Revenue: ₹{sales['Revenue'].sum():.2f}")
    print(f"📦 Total Units Sold: {int(sales['Units_Sold'].sum())}")
    print(f"🏪 Avg Price per Unit: ₹{(sales['Revenue'].sum() / sales['Units_Sold'].sum()):.2f}")
    print(f"📈 Avg Sale Size: {sales['Units_Sold'].mean():.1f} units")

    # By product
    print("\n📋 Top Products by Revenue:")
    by_revenue = sales.groupby("Product_Name")["Revenue"].sum().sort_values(ascending=False)
    for i, (product, revenue) in enumerate(by_revenue.head(5).items(), 1):
        print(f"  {i}. {product}: ₹{revenue:.0f}")

# ============================================================================
# EXAMPLE 7: Batch Operations
# ============================================================================
print("\n\n🎯 EXAMPLE 7: Batch Operations (Week of Sales)")
print("-" * 80)

# Simulate a week of sales
week_data = {
    "Monday": [
        ("Milk", 18, 25.0),
        ("Rice", 12, 60.0),
    ],
    "Tuesday": [
        ("Bread", 20, 40.0),
        ("Milk", 15, 25.0),
    ],
    "Wednesday": [
        ("Cooking Oil", 5, 150.0),
        ("Masala", 8, 80.0),
    ],
    "Thursday": [
        ("Milk", 22, 25.0),
        ("Biscuits", 30, 20.0),
    ],
    "Friday": [
        ("Rice", 25, 60.0),
        ("Eggs", 12, 40.0),
    ],
    "Saturday": [
        ("Milk", 28, 25.0),
        ("Snacks", 15, 50.0),
    ],
    "Sunday": [
        ("Bread", 25, 40.0),
        ("Curd", 10, 50.0),
    ],
}

print(f"\nRecording a week of sales ({len(week_data)} days)...\n")

total_items = 0

for day, transactions in week_data.items():
    for product, units, price in transactions:
        user_engine.record_sale(product, units, price)
        total_items += 1
    print(f"  ✅ {day}: {len(transactions)} transactions")

print(f"\n✅ Total items added: {total_items}")
user_engine.reload()
print(f"📊 Total inventory: {len(user_engine.sales)} transactions")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY OF EXAMPLES")
print("=" * 80)

print("""
✅ Example 1: Initialized user and prediction engines
✅ Example 2: Recorded sales transactions  
✅ Example 3: Analyzed sales summary
✅ Example 4: Generated demand forecasts
✅ Example 5: Calculated inventory recommendations
✅ Example 6: Performed advanced analytics
✅ Example 7: Batch processed a week of sales

📚 Next Steps:
- Use main.py for interactive terminal interface
- Use app/app.py for web-based dashboard
- Review README.md for detailed documentation
- Customize config.py for your store parameters
- Collect 2-4 weeks of data, then trigger retraining

💡 Key Files:
- src/data/data_engine.py:      Record and manage sales
- src/models/predict.py:        Make demand predictions
- src/utils/inventory.py:       Calculate inventory levels
- src/utils/config.py:          Configure system parameters
""")

print("\n" + "=" * 80)
print("END OF EXAMPLES")
print("=" * 80)

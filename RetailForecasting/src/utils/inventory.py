"""
Inventory Optimization Module
Calculates safety stock, reorder quantities, and risk levels
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Z_SCORE, LEAD_TIME_DAYS


class InventoryOptimizer:
    """Optimize inventory levels"""

    @staticmethod
    def calculate_safety_stock(mean_demand, demand_std, lead_time=LEAD_TIME_DAYS):
        """
        Calculate safety stock

        Safety Stock = Z × σ × √LeadTime

        Parameters:
        - mean_demand: Average daily demand
        - demand_std: Standard deviation of demand
        - lead_time: Lead time in days
        """
        if demand_std == 0:
            demand_std = mean_demand * 0.1  # Use 10% of mean if std is 0

        safety_stock = Z_SCORE * demand_std * np.sqrt(lead_time)
        return max(0, safety_stock)

    @staticmethod
    def calculate_reorder_quantity(
        mean_demand,
        demand_std,
        current_inventory,
        lead_time=LEAD_TIME_DAYS,
    ):
        """
        Calculate reorder quantity

        Reorder Quantity = (Demand × LeadTime) + Safety Stock - Current Inventory
        """
        # Lead time demand
        lead_time_demand = mean_demand * lead_time

        # Safety stock
        safety_stock = InventoryOptimizer.calculate_safety_stock(
            mean_demand, demand_std, lead_time
        )

        # Reorder point
        reorder_point = lead_time_demand + safety_stock

        # Quantity to order
        quantity_to_order = max(0, reorder_point - current_inventory)

        return {
            "reorder_point": max(0, reorder_point),
            "safety_stock": safety_stock,
            "lead_time_demand": lead_time_demand,
            "quantity_to_order": quantity_to_order,
        }

    @staticmethod
    def determine_risk_level(
        current_inventory, reorder_point, 
        mean_demand, safety_stock
    ):
        """Determine stock risk level"""

        threshold_high = reorder_point * 1.5
        threshold_medium = reorder_point
        threshold_low = safety_stock

        if current_inventory >= threshold_high:
            return "LOW"  # Well-stocked
        elif current_inventory >= threshold_medium:
            return "MEDIUM"  # Normal levels
        elif current_inventory >= threshold_low:
            return "HIGH"  # Below reorder point
        else:
            return "CRITICAL"  # Critical stock level

    @staticmethod
    def get_inventory_recommendation(
        product_name,
        current_inventory,
        mean_demand,
        demand_std,
        lead_time=LEAD_TIME_DAYS,
    ):
        """Get complete inventory recommendation"""

        reorder_calc = InventoryOptimizer.calculate_reorder_quantity(
            mean_demand, demand_std, current_inventory, lead_time
        )

        risk_level = InventoryOptimizer.determine_risk_level(
            current_inventory,
            reorder_calc["reorder_point"],
            mean_demand,
            reorder_calc["safety_stock"],
        )

        recommendation = {
            "product_name": product_name,
            "current_inventory": current_inventory,
            "mean_daily_demand": round(mean_demand, 2),
            "demand_std": round(demand_std, 2),
            "lead_time_days": lead_time,
            "safety_stock": round(reorder_calc["safety_stock"], 2),
            "reorder_point": round(reorder_calc["reorder_point"], 2),
            "lead_time_demand": round(reorder_calc["lead_time_demand"], 2),
            "quantity_to_order": int(reorder_calc["quantity_to_order"]),
            "risk_level": risk_level,
            "action": InventoryOptimizer._get_action(risk_level),
        }

        return recommendation

    @staticmethod
    def _get_action(risk_level):
        """Get recommended action based on risk level"""
        actions = {
            "LOW": "Monitor - No immediate action needed",
            "MEDIUM": "Normal operations - Order as planned",
            "HIGH": "Priority reorder - Increase purchase immediately",
            "CRITICAL": "URGENT: Restock immediately to avoid stockout",
        }
        return actions.get(risk_level, "Unknown")

    @staticmethod
    def analyze_dataframe(df, product_col="Product_Name", demand_col="Units_Sold"):
        """Analyze inventory for DataFrame of products"""
        recommendations = []

        for product in df[product_col].unique():
            product_data = df[df[product_col] == product][demand_col]

            mean_demand = product_data.mean()
            demand_std = product_data.std()
            current_inventory = df[df[product_col] == product].iloc[-1].get(
                "Units_Remaining", 0
            )

            if mean_demand > 0:
                rec = InventoryOptimizer.get_inventory_recommendation(
                    product, current_inventory, mean_demand, demand_std
                )
                recommendations.append(rec)

        return pd.DataFrame(recommendations) if recommendations else pd.DataFrame()


def print_inventory_report(recommendations_df):
    """Print formatted inventory report"""
    if recommendations_df.empty:
        print("No inventory data available")
        return

    print("\n" + "=" * 100)
    print("INVENTORY OPTIMIZATION REPORT")
    print("=" * 100 + "\n")

    # Color codes for risk levels
    risk_colors = {
        "LOW": "🟢",
        "MEDIUM": "🟡",
        "HIGH": "🔴",
        "CRITICAL": "🔴🔴",
    }

    for idx, row in recommendations_df.iterrows():
        risk_color = risk_colors.get(row["risk_level"], "⚪")

        print(f"\n{risk_color} {row['product_name']}")
        print(f"   Current Inventory: {int(row['current_inventory'])} units")
        print(f"   Mean Daily Demand: {row['mean_daily_demand']:.2f} units")
        print(f"   Safety Stock: {row['safety_stock']:.0f} units")
        print(f"   Reorder Point: {row['reorder_point']:.0f} units")
        print(f"   Quantity to Order: {int(row['quantity_to_order'])} units")
        print(f"   Risk Level: {row['risk_level']}")
        print(f"   Action: {row['action']}")

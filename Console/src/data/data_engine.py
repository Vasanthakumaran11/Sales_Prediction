"""
User Data Engine
Handles collection and management of user data (products, purchases, sales)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PRODUCTS_FILE, PURCHASES_FILE, SALES_FILE, PRODUCTS


class UserDataEngine:
    """Manage user-specific data collection"""

    def __init__(self):
        self.products = self._initialize_products()
        self.purchases = self._initialize_purchases()
        self.sales = self._initialize_sales()

    def _initialize_products(self):
        """Initialize products CSV"""
        if os.path.exists(PRODUCTS_FILE):
            return pd.read_csv(PRODUCTS_FILE)

        # Create products dataframe
        products_data = []
        for item_name, item_info in PRODUCTS.items():
            products_data.append(
                {
                    "Product_ID": f"PROD_{len(products_data)+1:03d}",
                    "Product_Name": item_name,
                    "Category": item_info["category"],
                    "Default_Price_Min": item_info["price_range"][0],
                    "Default_Price_Max": item_info["price_range"][1],
                    "Date_Added": datetime.now().strftime("%d-%m-%Y"),
                }
            )

        df = pd.DataFrame(products_data)
        os.makedirs(os.path.dirname(PRODUCTS_FILE), exist_ok=True)
        df.to_csv(PRODUCTS_FILE, index=False)

        return df

    def _initialize_purchases(self):
        """Initialize purchases CSV"""
        if os.path.exists(PURCHASES_FILE):
            return pd.read_csv(PURCHASES_FILE)

        # Create empty purchases dataframe with schema
        df = pd.DataFrame(
            columns=[
                "Purchase_ID",
                "Date",
                "Product_ID",
                "Product_Name",
                "Quantity",
                "Cost_Price",
                "Total_Cost",
            ]
        )
        os.makedirs(os.path.dirname(PURCHASES_FILE), exist_ok=True)
        df.to_csv(PURCHASES_FILE, index=False)

        return df

    def _initialize_sales(self):
        """Initialize sales CSV"""
        if os.path.exists(SALES_FILE):
            return pd.read_csv(SALES_FILE)

        # Create empty sales dataframe with schema
        df = pd.DataFrame(
            columns=[
                "Sale_ID",
                "Date",
                "Product_Name",
                "Units_Sold",
                "Unit_Price",
                "Discount",
                "Revenue",
                "Promo",
                "Holiday",
                "Shop_Closed",
            ]
        )
        os.makedirs(os.path.dirname(SALES_FILE), exist_ok=True)
        df.to_csv(SALES_FILE, index=False)

        return df

    def add_product(self, product_name, category, price_min, price_max):
        """Add new product"""
        if product_name in self.products["Product_Name"].values:
            return False  # Product already exists

        new_product = pd.DataFrame(
            [
                {
                    "Product_ID": f"PROD_{len(self.products)+1:03d}",
                    "Product_Name": product_name,
                    "Category": category,
                    "Default_Price_Min": price_min,
                    "Default_Price_Max": price_max,
                    "Date_Added": datetime.now().strftime("%d-%m-%Y"),
                }
            ]
        )

        self.products = pd.concat(
            [self.products, new_product], ignore_index=True
        )
        self.products.to_csv(PRODUCTS_FILE, index=False)

        return True

    def record_purchase(
        self,
        product_name,
        quantity,
        cost_price,
        date=None,
    ):
        """Record a purchase"""
        if date is None:
            date = datetime.now().strftime("%d-%m-%Y")

        # Find product ID
        product_row = self.products[self.products["Product_Name"] == product_name]
        if product_row.empty:
            return False

        product_id = product_row.iloc[0]["Product_ID"]
        total_cost = quantity * cost_price

        new_purchase = pd.DataFrame(
            [
                {
                    "Purchase_ID": f"PURCH_{len(self.purchases)+1:06d}",
                    "Date": date,
                    "Product_ID": product_id,
                    "Product_Name": product_name,
                    "Quantity": quantity,
                    "Cost_Price": cost_price,
                    "Total_Cost": total_cost,
                }
            ]
        )

        self.purchases = pd.concat([self.purchases, new_purchase], ignore_index=True)
        self.purchases.to_csv(PURCHASES_FILE, index=False)

        return True

    def record_sale(
        self,
        product_name,
        units_sold,
        unit_price,
        discount=0,
        promo=False,
        holiday=False,
        shop_closed=False,
        date=None,
    ):
        """Record a sale"""
        if date is None:
            date = datetime.now().strftime("%d-%m-%Y")

        revenue = units_sold * unit_price * (1 - discount)

        new_sale = pd.DataFrame(
            [
                {
                    "Sale_ID": f"SALE_{len(self.sales)+1:06d}",
                    "Date": date,
                    "Product_Name": product_name,
                    "Units_Sold": units_sold,
                    "Unit_Price": unit_price,
                    "Discount": discount,
                    "Revenue": revenue,
                    "Promo": int(promo),
                    "Holiday": int(holiday),
                    "Shop_Closed": int(shop_closed),
                }
            ]
        )

        self.sales = pd.concat([self.sales, new_sale], ignore_index=True)
        self.sales.to_csv(SALES_FILE, index=False)

        return True

    def get_sales_summary(self):
        """Get sales summary"""
        if self.sales.empty:
            return None

        summary = (
            self.sales.groupby("Product_Name")
            .agg(
                {
                    "Units_Sold": "sum",
                    "Revenue": "sum",
                    "Sale_ID": "count",
                }
            )
            .rename(columns={"Sale_ID": "Total_Sales"})
        )

        return summary

    def get_data_count(self):
        """Get count of user data"""
        return {
            "products": len(self.products),
            "purchases": len(self.purchases),
            "sales": len(self.sales),
        }

    def reload(self):
        """Reload data from files"""
        self.products = (
            pd.read_csv(PRODUCTS_FILE) if os.path.exists(PRODUCTS_FILE) else None
        )
        self.purchases = (
            pd.read_csv(PURCHASES_FILE) if os.path.exists(PURCHASES_FILE) else None
        )
        self.sales = (
            pd.read_csv(SALES_FILE) if os.path.exists(SALES_FILE) else None
        )


def create_user_data_engine():
    """Helper function to create user data engine"""
    return UserDataEngine()

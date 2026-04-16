"""
Product Suggestion and Management System
Recommends products based on investment and manages product catalog
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class ProductManager:
    """Manages product suggestions and inventory"""
    
    # Product catalog with categories
    PRODUCT_CATALOG = {
        "Perishables": {
            "Milk": {"default_price": 28, "lead_time": 1, "turnover": "High"},
            "Curd": {"default_price": 45, "lead_time": 2, "turnover": "High"},
            "Paneer": {"default_price": 280, "lead_time": 2, "turnover": "Medium"},
            "Bread": {"default_price": 35, "lead_time": 1, "turnover": "High"},
            "Eggs": {"default_price": 45, "lead_time": 1, "turnover": "High"},
        },
        "Non-Perishables": {
            "Rice": {"default_price": 65, "lead_time": 3, "turnover": "High"},
            "Wheat Flour": {"default_price": 45, "lead_time": 3, "turnover": "High"},
            "Oil": {"default_price": 160, "lead_time": 2, "turnover": "High"},
            "Sugar": {"default_price": 50, "lead_time": 3, "turnover": "Medium"},
            "Salt": {"default_price": 20, "lead_time": 5, "turnover": "Low"},
            "Dal": {"default_price": 85, "lead_time": 3, "turnover": "High"},
            "Spices Mix": {"default_price": 150, "lead_time": 4, "turnover": "Medium"},
        },
        "Snacks & Biscuits": {
            "Biscuits": {"default_price": 40, "lead_time": 2, "turnover": "High"},
            "Chips": {"default_price": 50, "lead_time": 2, "turnover": "High"},
            "Namkeen": {"default_price": 80, "lead_time": 3, "turnover": "Medium"},
            "Chocolate": {"default_price": 60, "lead_time": 2, "turnover": "High"},
            "Cookies": {"default_price": 50, "lead_time": 2, "turnover": "High"},
        },
        "Beverages": {
            "Tea": {"default_price": 250, "lead_time": 3, "turnover": "High"},
            "Coffee": {"default_price": 180, "lead_time": 3, "turnover": "Medium"},
            "Juice": {"default_price": 80, "lead_time": 2, "turnover": "Medium"},
            "Soft Drinks": {"default_price": 45, "lead_time": 2, "turnover": "High"},
        },
        "Frozen Foods": {
            "Ice Cream": {"default_price": 120, "lead_time": 1, "turnover": "Medium"},
            "Frozen Vegetables": {"default_price": 100, "lead_time": 1, "turnover": "Low"},
        },
        "Personal Care": {
            "Soap": {"default_price": 35, "lead_time": 3, "turnover": "High"},
            "Shampoo": {"default_price": 120, "lead_time": 3, "turnover": "Medium"},
            "Toothpaste": {"default_price": 80, "lead_time": 3, "turnover": "High"},
            "Deodorant": {"default_price": 150, "lead_time": 3, "turnover": "Low"},
        }
    }
    
    # Investment-based recommendations
    INVESTMENT_RECOMMENDATIONS = {
        "Budget": {
            "description": "Essential items for basic store",
            "categories": ["Perishables", "Non-Perishables"],
            "min_products": 12,
            "initial_stock_value": 20000
        },
        "Moderate": {
            "description": "Complete inventory with new categories",
            "categories": ["Perishables", "Non-Perishables", "Snacks & Biscuits", "Beverages"],
            "min_products": 20,
            "initial_stock_value": 80000
        },
        "Premium": {
            "description": "Full range with premium items",
            "categories": ["Perishables", "Non-Perishables", "Snacks & Biscuits", "Beverages", "Frozen Foods", "Personal Care"],
            "min_products": 35,
            "initial_stock_value": 300000
        },
        "Enterprise": {
            "description": "Complete supermarket inventory",
            "categories": list(PRODUCT_CATALOG.keys()),
            "min_products": 50,
            "initial_stock_value": 1000000
        }
    }
    
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.products_file = self.store_path / "products.csv"
    
    def get_suggested_products(self, investment: int) -> Dict:
        """
        Get product suggestions based on investment amount
        
        Args:
            investment: Investment amount in INR
            
        Returns:
            Dictionary with suggested categories and products
        """
        category = self._get_investment_category(investment)
        recommendations = self.INVESTMENT_RECOMMENDATIONS[category]
        
        suggested = {
            "investment_category": category,
            "description": recommendations["description"],
            "categories": recommendations["categories"],
            "suggested_products": {},
            "estimated_initial_value": recommendations["initial_stock_value"],
            "recommended_product_count": recommendations["min_products"]
        }
        
        # Get products from suggested categories
        for category in recommendations["categories"]:
            if category in self.PRODUCT_CATALOG:
                suggested["suggested_products"][category] = list(self.PRODUCT_CATALOG[category].keys())
        
        return suggested
    
    def add_product_to_store(self, product_name: str, category: str, 
                             initial_stock: int, unit_price: float = None) -> bool:
        """Add product to store inventory"""
        try:
            # Get default price if not provided
            if unit_price is None:
                if category in self.PRODUCT_CATALOG and product_name in self.PRODUCT_CATALOG[category]:
                    unit_price = self.PRODUCT_CATALOG[category][product_name]["default_price"]
                else:
                    unit_price = 100  # Default price
            
            # Check if product already exists
            if self._product_exists(product_name):
                print(f"⚠️  Product '{product_name}' already exists!")
                return False
            
            # Add to CSV
            product_id = f"PROD_{int(datetime.now().timestamp())}"
            with open(self.products_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    product_id,
                    product_name,
                    category,
                    initial_stock,
                    unit_price,
                    datetime.now().strftime("%d-%m-%Y")
                ])
            
            return True
        except Exception as e:
            print(f"❌ Error adding product: {str(e)}")
            return False
    
    def initialize_with_suggestions(self, investment: int) -> bool:
        """Initialize store with suggested products"""
        try:
            suggestions = self.get_suggested_products(investment)
            
            # Add recommended products
            for category, products in suggestions["suggested_products"].items():
                for product in products[:5]:  # Add first 5 products from each category
                    unit_price = self.PRODUCT_CATALOG[category][product]["default_price"]
                    initial_stock = 50  # Default initial stock
                    self.add_product_to_store(product, category, initial_stock, unit_price)
            
            return True
        except Exception as e:
            print(f"❌ Error initializing products: {str(e)}")
            return False
    
    def get_all_products(self) -> List[Dict]:
        """Get all products for this store"""
        try:
            products = []
            if not self.products_file.exists():
                return products
            
            with open(self.products_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    products.append(row)
            
            return products
        except Exception:
            return []
    
    def get_product_by_name(self, product_name: str) -> Dict:
        """Get product details by name"""
        products = self.get_all_products()
        for product in products:
            if product.get('Product_Name') == product_name:
                return product
        return None
    
    def update_stock(self, product_name: str, new_stock: int) -> bool:
        """Update product stock quantity"""
        try:
            products = self.get_all_products()
            updated = False
            
            with open(self.products_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Product_ID', 'Product_Name', 'Category', 'Stock_Quantity', 'Unit_Price', 'Last_Updated'])
                writer.writeheader()
                
                for product in products:
                    if product['Product_Name'] == product_name:
                        product['Stock_Quantity'] = new_stock
                        product['Last_Updated'] = datetime.now().strftime("%d-%m-%Y")
                        updated = True
                    writer.writerow(product)
            
            return updated
        except Exception:
            return False
    
    def _product_exists(self, product_name: str) -> bool:
        """Check if product already exists"""
        return self.get_product_by_name(product_name) is not None
    
    @staticmethod
    def _get_investment_category(investment: int) -> str:
        """Categorize investment amount"""
        if investment < 50000:
            return "Budget"
        elif investment < 150000:
            return "Moderate"
        elif investment < 500000:
            return "Premium"
        else:
            return "Enterprise"
    
    def display_product_suggestions(self, investment: int) -> None:
        """Display product suggestions in formatted way"""
        suggestions = self.get_suggested_products(investment)
        
        print("\n" + "="*60)
        print("🛍️  PRODUCT SUGGESTIONS FOR YOUR STORE")
        print("="*60)
        print(f"\n💰 Investment Category: {suggestions['investment_category']}")
        print(f"📝 Description: {suggestions['description']}")
        print(f"📦 Recommended Product Count: {suggestions['recommended_product_count']}")
        print(f"💵 Estimated Initial Inventory Value: ₹{suggestions['estimated_initial_value']:,}")
        
        print("\n📂 SUGGESTED CATEGORIES:")
        for idx, category in enumerate(suggestions["categories"], 1):
            products = suggestions["suggested_products"][category]
            print(f"\n{idx}. {category}")
            print(f"   Products: {', '.join(products[:5])}...")
            if len(products) > 5:
                print(f"   ... and {len(products) - 5} more")
        
        print("\n" + "="*60)

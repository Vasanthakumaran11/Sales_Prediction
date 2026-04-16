#!/usr/bin/env python3
"""
AI-Based Smart Grocery Account & Demand Forecasting System
Enhanced Terminal-Based Application

Main entry point for the complete system with interactive menus,
user management, sales tracking, and ML-based predictions.
"""

import sys
import os
from pathlib import Path
import csv
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from users.user_manager import UserManager, StoreInfo
from users.product_manager import ProductManager
from interface.dashboard import Dashboard
from interface.input_handler import InputHandler
from analytics.sales_analytics import SalesAnalytics
from inventory.inventory_manager import InventoryManager


class SmartGroceryApp:
    """Main application class"""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.current_store = None
        self.current_store_path = None
        self.product_manager = None
        self.sales_analytics = None
    
    def run(self):
        """Run the main application"""
        Dashboard.print_welcome()
        
        while True:
            options = [
                ('1', 'New Grocery (First-time user)'),
                ('2', 'Existing Grocery (Returning user)'),
                ('3', 'Exit')
            ]
            
            choice = Dashboard.print_menu(
                "🏪 SMART GROCERY AI SYSTEM",
                options
            )
            
            if choice == '1':
                self.register_new_store()
            elif choice == '2':
                self.load_existing_store()
            elif choice == '3':
                print(f"\n{Dashboard.COLORS['GREEN']}Thank you for using Smart Grocery AI System!{Dashboard.COLORS['END']}\n")
                break
    
    def register_new_store(self):
        """Register a new store"""
        store_info = InputHandler.get_store_registration()
        
        if store_info is None:
            Dashboard.print_error("Registration cancelled")
            return
        
        # Check if store already exists
        if self.user_manager.store_exists(store_info['store_name']):
            Dashboard.print_error(f"Store '{store_info['store_name']}' already exists!")
            return
        
        # Create store profile
        Dashboard.loading_animation("Creating store profile", 2)
        if not self.user_manager.create_new_store(store_info):
            Dashboard.print_error("Failed to create store")
            return
        
        Dashboard.print_success(f"Store '{store_info['store_name']}' created successfully!")
        
        # Show product suggestions
        self.current_store_path = self.user_manager.get_store_path(store_info['store_name'])
        self.product_manager = ProductManager(str(self.current_store_path))
        
        self.product_manager.display_product_suggestions(store_info['investment'])
        
        # Initialize with suggested products
        if Dashboard.get_yes_no("Would you like to initialize with suggested products?"):
            Dashboard.loading_animation("Initializing products", 2)
            self.product_manager.initialize_with_suggestions(store_info['investment'])
            Dashboard.print_success("Products initialized successfully!")
        
        # Load the store for operations
        self.current_store = store_info['store_name']
        self.sales_analytics = SalesAnalytics(str(self.current_store_path))
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
        self.store_menu()
    
    def load_existing_store(self):
        """Load an existing store"""
        stores = self.user_manager.get_store_list()
        
        if not stores:
            Dashboard.print_error("No existing stores found. Please create a new store first.")
            return
        
        # Display store list
        Dashboard.clear_screen()
        Dashboard.print_header("📦 SELECT YOUR STORE", 70)
        
        for idx, store in enumerate(stores, 1):
            print(f"  {idx}) {store}")
        
        while True:
            try:
                choice = int(input(f"\n{Dashboard.COLORS['BOLD']}Enter store number: {Dashboard.COLORS['END']}"))
                if 1 <= choice <= len(stores):
                    store_name = stores[choice - 1]
                    break
                Dashboard.print_warning(f"Please select between 1 and {len(stores)}")
            except ValueError:
                Dashboard.print_warning("Please enter a valid number")
        
        # Load store profile
        store_profile = self.user_manager.load_store_profile(store_name)
        if not store_profile:
            Dashboard.print_error(f"Failed to load store '{store_name}'")
            return
        
        self.current_store = store_name
        self.current_store_path = self.user_manager.get_store_path(store_name)
        self.product_manager = ProductManager(str(self.current_store_path))
        self.sales_analytics = SalesAnalytics(str(self.current_store_path))
        
        Dashboard.print_success(f"Loaded store: {store_name}")
        Dashboard.print_store_profile(store_profile)
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
        self.store_menu()
    
    def store_menu(self):
        """Main store operations menu"""
        while True:
            options = [
                ('1', 'Daily Sales Entry'),
                ('2', 'View Sales History & Analytics'),
                ('3', 'Inventory Management'),
                ('4', 'Product Management'),
                ('5', 'Monthly Prediction'),
                ('6', 'View Store Profile'),
                ('7', 'Back to Main Menu')
            ]
            
            choice = Dashboard.print_menu(
                f"🏪 {self.current_store} - Main Menu",
                options
            )
            
            if choice == '1':
                self.daily_sales_entry()
            elif choice == '2':
                self.view_sales_history()
            elif choice == '3':
                self.inventory_management()
            elif choice == '4':
                self.product_management()
            elif choice == '5':
                self.monthly_prediction()
            elif choice == '6':
                profile = self.user_manager.load_store_profile(self.current_store)
                if profile:
                    Dashboard.print_store_profile(profile)
                    input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
            elif choice == '7':
                break
    
    def daily_sales_entry(self):
        """Process daily sales entry"""
        products = self.product_manager.get_all_products()
        if not products:
            Dashboard.print_warning("No products found. Please add products first.")
            return
        
        product_names = [p['Product_Name'] for p in products]
        
        sales_entry = InputHandler.get_daily_sales_entry(product_names)
        if not sales_entry:
            Dashboard.print_error("Sales entry cancelled")
            return
        
        # Save sales entry
        try:
            sale_id = f"SALE_{int(datetime.now().timestamp())}"
            
            sales_file = self.current_store_path / "sales.csv"
            with open(sales_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    sale_id,
                    sales_entry['date'],
                    sales_entry['product_name'],
                    sales_entry['units_sold'],
                    sales_entry['unit_price'],
                    sales_entry['discount'],
                    sales_entry['revenue'],
                    sales_entry['promo'],
                    sales_entry['holiday'],
                    sales_entry['shop_closed']
                ])
            
            Dashboard.print_success(f"Sale recorded successfully! Sale ID: {sale_id}")
            
            # Update store stats
            self.user_manager.update_store_stats(
                self.current_store,
                sales_entry['units_sold'],
                sales_entry['revenue']
            )
            
        except Exception as e:
            Dashboard.print_error(f"Failed to save sale: {str(e)}")
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
    
    def view_sales_history(self):
        """View sales history and analytics"""
        Dashboard.clear_screen()
        
        # Get analytics
        analytics = self.sales_analytics.get_insights()
        SalesAnalytics.display_analytics(analytics)
        
        # Show products performance
        performance = analytics.get('product_performance', {})
        if performance:
            Dashboard.print_section("📊 PRODUCT PERFORMANCE")
            
            headers = ['Product', 'Units Sold', 'Revenue', 'Avg Price', 'Transactions']
            rows = []
            
            for product, data in sorted(performance.items(), 
                                       key=lambda x: x[1]['revenue'], reverse=True)[:10]:
                rows.append([
                    product,
                    str(int(data['units_sold'])),
                    f"₹{data['revenue']:.2f}",
                    f"₹{data['avg_price']:.2f}",
                    str(data['transactions'])
                ])
            
            Dashboard.print_table(headers, rows, [20, 12, 15, 12, 12])
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
    
    def inventory_management(self):
        """Manage inventory"""
        Dashboard.clear_screen()
        Dashboard.print_header("📦 INVENTORY MANAGEMENT", 70)
        
        products = self.product_manager.get_all_products()
        if not products:
            Dashboard.print_warning("No products found")
            return
        
        # Display products
        print("\n🏷️  Available Products:\n")
        for idx, product in enumerate(products, 1):
            print(f"  {idx}) {product['Product_Name']} (Stock: {product['Stock_Quantity']})")
        
        # Select product
        while True:
            try:
                choice = int(input(f"\n{Dashboard.COLORS['BOLD']}Select product number: {Dashboard.COLORS['END']}"))
                if 1 <= choice <= len(products):
                    selected_product = products[choice - 1]
                    break
                Dashboard.print_warning(f"Please select between 1 and {len(products)}")
            except ValueError:
                Dashboard.print_warning("Please enter a valid number")
        
        # Get sales history for this product
        sales = self.sales_analytics.load_sales_data()
        product_sales = [int(s['Units_Sold']) for s in sales 
                        if s['Product_Name'] == selected_product['Product_Name']]
        
        if not product_sales:
            product_sales = [10]  # Default for new products
        
        # Get recommendation
        recommendation = InventoryManager.get_inventory_recommendation(
            product_name=selected_product['Product_Name'],
            current_stock=int(selected_product['Stock_Quantity']),
            demand_history=product_sales,
            unit_cost=float(selected_product['Unit_Price']) * 0.5,  # Assume 50% cost
            unit_price=float(selected_product['Unit_Price'])
        )
        
        # Display recommendation
        InventoryManager.display_inventory_recommendation(recommendation)
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
    
    def product_management(self):
        """Manage products"""
        while True:
            options = [
                ('1', 'View All Products'),
                ('2', 'Add New Product'),
                ('3', 'Update Stock'),
                ('4', 'Back to Store Menu')
            ]
            
            choice = Dashboard.print_menu("📦 PRODUCT MANAGEMENT", options)
            
            if choice == '1':
                self.view_all_products()
            elif choice == '2':
                self.add_new_product()
            elif choice == '3':
                self.update_product_stock()
            elif choice == '4':
                break
    
    def view_all_products(self):
        """View all products"""
        products = self.product_manager.get_all_products()
        
        Dashboard.clear_screen()
        Dashboard.print_header("📦 ALL PRODUCTS", 70)
        
        if not products:
            Dashboard.print_info("No products found")
            return
        
        headers = ['Product Name', 'Category', 'Stock', 'Price', 'Last Updated']
        rows = []
        
        for product in products:
            rows.append([
                product['Product_Name'],
                product.get('Category', 'N/A'),
                product['Stock_Quantity'],
                f"₹{float(product['Unit_Price']):.2f}",
                product.get('Last_Updated', 'N/A')
            ])
        
        Dashboard.print_table(headers, rows, [20, 15, 10, 12, 15])
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
    
    def add_new_product(self):
        """Add a new product"""
        product_details = InputHandler.get_new_product()
        
        if product_details:
            if self.product_manager.add_product_to_store(
                product_details['product_name'],
                product_details['category'],
                product_details['initial_stock'],
                product_details['unit_price']
            ):
                Dashboard.print_success(f"Product '{product_details['product_name']}' added successfully!")
            else:
                Dashboard.print_error("Failed to add product")
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
    
    def update_product_stock(self):
        """Update product stock"""
        products = self.product_manager.get_all_products()
        
        if not products:
            Dashboard.print_warning("No products found")
            return
        
        Dashboard.clear_screen()
        Dashboard.print_header("📦 UPDATE STOCK", 70)
        
        print("\n🏷️  Products:\n")
        for idx, product in enumerate(products, 1):
            print(f"  {idx}) {product['Product_Name']} (Current: {product['Stock_Quantity']})")
        
        while True:
            try:
                choice = int(input(f"\n{Dashboard.COLORS['BOLD']}Select product: {Dashboard.COLORS['END']}"))
                if 1 <= choice <= len(products):
                    selected = products[choice - 1]
                    break
                Dashboard.print_warning(f"Please select between 1 and {len(products)}")
            except ValueError:
                Dashboard.print_warning("Please enter a valid number")
        
        new_stock = Dashboard.get_input("Enter new stock quantity: ", input_type='int')
        
        if new_stock is not None and new_stock >= 0:
            if self.product_manager.update_stock(selected['Product_Name'], new_stock):
                Dashboard.print_success(f"Stock updated for '{selected['Product_Name']}'!")
            else:
                Dashboard.print_error("Failed to update stock")
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")
    
    def monthly_prediction(self):
        """Get monthly demand prediction"""
        Dashboard.clear_screen()
        Dashboard.print_header("🔮 MONTHLY DEMAND PREDICTION", 70)
        
        # Note: In production, this would use the ML models
        # For now, showing the workflow
        
        Dashboard.print_info("Monthly prediction feature requires historical data accumulation.")
        Dashboard.print_info("After 14+ days of sales data, personalized predictions will be available.")
        
        # Show current data status
        sales = self.sales_analytics.load_sales_data()
        if len(sales) > 0:
            unique_dates = len(set(s['Date'] for s in sales))
            Dashboard.print_success(f"Current data: {len(sales)} transactions, {unique_dates} days")
        
        input(f"\n{Dashboard.COLORS['BOLD']}Press Enter to continue...{Dashboard.COLORS['END']}")


def main():
    """Main entry point"""
    try:
        app = SmartGroceryApp()
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n{Dashboard.COLORS['YELLOW']}Application interrupted by user{Dashboard.COLORS['END']}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Dashboard.COLORS['RED']}Fatal error: {str(e)}{Dashboard.COLORS['END']}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()

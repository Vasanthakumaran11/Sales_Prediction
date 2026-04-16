"""
Input Handling and Form Validation
Manages user input forms with validation
"""

from typing import Dict, Optional, List
from datetime import datetime
from .dashboard import Dashboard


class InputHandler:
    """Handles user input and form validation"""
    
    # Validation patterns
    VALID_LOCATIONS = ['Urban', 'Semi-Urban', 'Rural']
    VALID_STORE_TYPES = ['Small', 'Medium', 'Supermarket']
    
    @staticmethod
    def get_store_registration() -> Optional[Dict]:
        """
        Get store registration details from user
        
        Returns:
            Dictionary with store information or None if cancelled
        """
        Dashboard.print_header("📝 NEW STORE REGISTRATION", 70)
        
        try:
            # Store name
            while True:
                store_name = Dashboard.get_input("Enter Store Name: ")
                if store_name and len(store_name) >= 2:
                    break
                Dashboard.print_warning("Store name must be at least 2 characters")
            
            # Location
            while True:
                print("\n📍 Select Location:")
                for idx, loc in enumerate(InputHandler.VALID_LOCATIONS, 1):
                    print(f"  {idx}) {loc}")
                
                choice = Dashboard.get_input("Enter choice (1-3): ")
                if choice in ['1', '2', '3']:
                    location = InputHandler.VALID_LOCATIONS[int(choice) - 1]
                    break
                Dashboard.print_warning("Invalid choice")
            
            # Store type
            while True:
                print("\n🏬 Select Store Type:")
                for idx, stype in enumerate(InputHandler.VALID_STORE_TYPES, 1):
                    print(f"  {idx}) {stype}")
                
                choice = Dashboard.get_input("Enter choice (1-3): ")
                if choice in ['1', '2', '3']:
                    store_type = InputHandler.VALID_STORE_TYPES[int(choice) - 1]
                    break
                Dashboard.print_warning("Invalid choice")
            
            # Investment amount
            while True:
                investment = Dashboard.get_input("Enter Initial Investment (₹): ", input_type='int')
                if investment and investment >= 10000:
                    break
                Dashboard.print_warning("Minimum investment is ₹10,000")
            
            return {
                'store_name': store_name,
                'location': location,
                'store_type': store_type,
                'investment': investment
            }
        
        except Exception as e:
            Dashboard.print_error(f"Error in registration: {str(e)}")
            return None
    
    @staticmethod
    def get_daily_sales_entry(products: List[str]) -> Optional[Dict]:
        """
        Get daily sales entry from user
        
        Args:
            products: List of available products
        
        Returns:
            Dictionary with sales details or None if cancelled
        """
        Dashboard.print_header("💰 DAILY SALES ENTRY", 70)
        
        try:
            # Date
            print("\n📅 Sale Date (DD-MM-YYYY) or press Enter for today:")
            date_input = Dashboard.get_input("Date: ", allow_empty=True)
            
            if date_input:
                datetime.strptime(date_input, "%d-%m-%Y")  # Validate format
                sale_date = date_input
            else:
                sale_date = datetime.now().strftime("%d-%m-%Y")
            
            # Product selection
            print("\n📦 Available Products:")
            for idx, product in enumerate(products, 1):
                print(f"  {idx}) {product}")
            
            while True:
                product_choice = Dashboard.get_input("Select product (enter number): ", input_type='int')
                if 1 <= product_choice <= len(products):
                    product_name = products[product_choice - 1]
                    break
                Dashboard.print_warning(f"Please select between 1 and {len(products)}")
            
            # Units sold
            while True:
                units_sold = Dashboard.get_input("Units Sold: ", input_type='int')
                if units_sold and units_sold > 0:
                    break
                Dashboard.print_warning("Units must be greater than 0")
            
            # Unit price
            while True:
                unit_price = Dashboard.get_input("Unit Price (₹): ", input_type='float')
                if unit_price and unit_price > 0:
                    break
                Dashboard.print_warning("Price must be greater than 0")
            
            # Discount
            print("\n🎉 Discount (%) or press Enter for no discount:")
            discount_input = Dashboard.get_input("Discount: ", input_type='float', allow_empty=True)
            discount = discount_input if discount_input else 0.0
            
            if discount < 0 or discount > 100:
                Dashboard.print_warning("Discount must be between 0-100%")
                discount = 0
            
            # Promotional sale
            promo = 1 if Dashboard.get_yes_no("Was this a promotional sale?") else 0
            
            # Holiday
            holiday = 1 if Dashboard.get_yes_no("Was this on a holiday?") else 0
            
            # Shop closed
            shop_closed = 1 if Dashboard.get_yes_no("Was shop closed this day?") else 0
            
            # Calculate revenue
            revenue = units_sold * unit_price * (1 - discount / 100)
            
            return {
                'date': sale_date,
                'product_name': product_name,
                'units_sold': units_sold,
                'unit_price': unit_price,
                'discount': discount,
                'revenue': revenue,
                'promo': promo,
                'holiday': holiday,
                'shop_closed': shop_closed
            }
        
        except ValueError as e:
            Dashboard.print_error(f"Invalid input format: {str(e)}")
            return None
        except Exception as e:
            Dashboard.print_error(f"Error in sales entry: {str(e)}")
            return None
    
    @staticmethod
    def get_prediction_parameters() -> Optional[Dict]:
        """
        Get parameters for monthly prediction
        
        Returns:
            Dictionary with prediction parameters
        """
        Dashboard.print_header("🔮 MONTHLY PREDICTION SETUP", 70)
        
        try:
            # Month
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            print("\n📅 Select Prediction Month:")
            for idx, month in enumerate(months, 1):
                print(f"  {idx:2d}) {month}", end="  ")
                if idx % 3 == 0:
                    print()
            
            while True:
                month_choice = Dashboard.get_input("\nEnter month number (1-12): ", input_type='int')
                if 1 <= month_choice <= 12:
                    prediction_month = months[month_choice - 1]
                    break
                Dashboard.print_warning("Please select between 1 and 12")
            
            # Festivals
            festivals = Dashboard.get_yes_no("Are there any festivals planned this month?")
            
            # Promotion plans
            promotion = Dashboard.get_yes_no("Are you planning promotional sales?")
            
            # Expected demand change
            print("\n📊 Expected Demand Change (Optional):")
            print("  Leave empty for no change, or enter percentage (e.g., +10 or -5)")
            demand_change_input = Dashboard.get_input("Demand Change (%): ", allow_empty=True)
            
            if demand_change_input:
                try:
                    demand_change = float(demand_change_input)
                except ValueError:
                    Dashboard.print_warning("Invalid percentage. Using 0%")
                    demand_change = 0.0
            else:
                demand_change = 0.0
            
            return {
                'month': prediction_month,
                'has_festivals': festivals,
                'has_promotion': promotion,
                'demand_change_percent': demand_change
            }
        
        except Exception as e:
            Dashboard.print_error(f"Error in prediction setup: {str(e)}")
            return None
    
    @staticmethod
    def confirm_action(action_description: str) -> bool:
        """
        Get user confirmation for an action
        
        Args:
            action_description: Description of the action
        
        Returns:
            True if user confirms, False otherwise
        """
        print(f"\n\n⏸️  {action_description}")
        return Dashboard.get_yes_no("Do you want to continue?")
    
    @staticmethod
    def get_new_product() -> Optional[Dict]:
        """
        Get details for adding a new product
        
        Returns:
            Dictionary with product details or None
        """
        Dashboard.print_header("➕ ADD NEW PRODUCT", 70)
        
        try:
            product_name = Dashboard.get_input("Product Name: ")
            category = Dashboard.get_input("Category: ")
            
            initial_stock = None
            while initial_stock is None:
                initial_stock = Dashboard.get_input("Initial Stock Quantity: ", input_type='int')
                if initial_stock and initial_stock > 0:
                    break
                Dashboard.print_warning("Stock must be greater than 0")
                initial_stock = None
            
            unit_price = None
            while unit_price is None:
                unit_price = Dashboard.get_input("Unit Price (₹): ", input_type='float')
                if unit_price and unit_price > 0:
                    break
                Dashboard.print_warning("Price must be greater than 0")
                unit_price = None
            
            return {
                'product_name': product_name,
                'category': category,
                'initial_stock': initial_stock,
                'unit_price': unit_price
            }
        
        except Exception as e:
            Dashboard.print_error(f"Error adding product: {str(e)}")
            return None

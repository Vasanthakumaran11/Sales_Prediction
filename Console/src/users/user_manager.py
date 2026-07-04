"""
User and Store Profile Management System
Handles user registration, profile storage, and retrieval
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


class UserManager:
    """Manages user/store profiles and data directories"""
    
    def __init__(self, base_path: str = "data/user"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_new_store(self, store_info: Dict) -> bool:
        """
        Create new store profile with investment-based suggestions
        
        Args:
            store_info: Dictionary with store_name, location, store_type, investment
        
        Returns:
            bool: Success status
        """
        try:
            store_name = store_info['store_name']
            store_path = self.base_path / store_name
            store_path.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            store_info['created_date'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            store_info['last_accessed'] = store_info['created_date']
            store_info['total_sales'] = 0.0
            store_info['total_revenue'] = 0.0
            
            # Save profile
            profile_path = store_path / "profile.json"
            with open(profile_path, 'w') as f:
                json.dump(store_info, f, indent=4)
            
            # Initialize CSV files
            self._initialize_csv_files(store_path)
            
            return True
        except Exception as e:
            print(f"❌ Error creating store: {str(e)}")
            return False
    
    def get_store_list(self) -> List[str]:
        """Get list of all existing stores"""
        try:
            stores = [d.name for d in self.base_path.iterdir() if d.is_dir()]
            return sorted(stores)
        except Exception:
            return []
    
    def load_store_profile(self, store_name: str) -> Optional[Dict]:
        """Load store profile from disk"""
        try:
            profile_path = self.base_path / store_name / "profile.json"
            if not profile_path.exists():
                return None
            
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            # Update last accessed
            profile['last_accessed'] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            
            # Calculate monthly revenue
            profile['total_revenue'] = self.get_monthly_revenue(store_name)
            
            # Save updated profile
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=4)
            
            return profile
        except Exception:
            return None
    
    def store_exists(self, store_name: str) -> bool:
        """Check if store already exists"""
        store_path = self.base_path / store_name
        return store_path.exists() and (store_path / "profile.json").exists()
    
    def update_store_stats(self, store_name: str, sales_amount: float, revenue: float) -> bool:
        """Update total sales and revenue"""
        try:
            profile = self.load_store_profile(store_name)
            if profile:
                profile['total_sales'] = profile.get('total_sales', 0) + sales_amount
                profile['total_revenue'] = profile.get('total_revenue', 0) + revenue
                
                profile_path = self.base_path / store_name / "profile.json"
                with open(profile_path, 'w') as f:
                    json.dump(profile, f, indent=4)
                return True
        except Exception:
            pass
        return False
    
    def get_monthly_revenue(self, store_name: str, month: int = None, year: int = None) -> float:
        """
        Get revenue for a specific month
        
        Args:
            store_name: Name of the store
            month: Month number (1-12), if None uses current month
            year: Year, if None uses current year
        
        Returns:
            Total revenue for the month
        """
        try:
            from datetime import datetime
            import csv
            
            if month is None:
                month = datetime.now().month
            if year is None:
                year = datetime.now().year
            
            store_path = self.base_path / store_name
            sales_file = store_path / "sales.csv"
            
            if not sales_file.exists():
                return 0.0
            
            total_revenue = 0.0
            with open(sales_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        date_str = row.get('Date', '')
                        # Handle both DD-MM-YYYY and DD-MM-YYYY HH:MM:SS formats
                        if ' ' in date_str:
                            date_str = date_str.split(' ')[0]  # Extract just the date part
                        
                        # Parse date in DD-MM-YYYY format
                        sale_date = datetime.strptime(date_str, "%d-%m-%Y")
                        
                        if sale_date.month == month and sale_date.year == year:
                            revenue = float(row.get('Revenue', 0))
                            total_revenue += revenue
                    except (ValueError, KeyError):
                        continue
            
            return total_revenue
        except Exception:
            return 0.0
    
    def get_store_path(self, store_name: str) -> Path:
        """Get store directory path"""
        return self.base_path / store_name
    
    def _initialize_csv_files(self, store_path: Path) -> None:
        """Initialize empty CSV files for new store"""
        
        # Products CSV
        products_csv = store_path / "products.csv"
        if not products_csv.exists():
            with open(products_csv, 'w') as f:
                f.write("Product_ID,Product_Name,Category,Stock_Quantity,Unit_Price,Last_Updated\n")
        
        # Sales CSV
        sales_csv = store_path / "sales.csv"
        if not sales_csv.exists():
            with open(sales_csv, 'w') as f:
                f.write("Sale_ID,Date,Product_Name,Units_Sold,Unit_Price,Discount,Revenue,Promo,Holiday,Shop_Closed\n")
        
        # Purchases CSV
        purchases_csv = store_path / "purchases.csv"
        if not purchases_csv.exists():
            with open(purchases_csv, 'w') as f:
                f.write("Purchase_ID,Date,Product_Name,Units_Purchased,Cost_Per_Unit,Total_Cost,Supplier\n")
        
        # Dataset CSV (for ML training)
        dataset_csv = store_path / "dataset.csv"
        if not dataset_csv.exists():
            with open(dataset_csv, 'w') as f:
                f.write("Date,Product_Name,Stock_Quantity,Units_Sold,Revenue,Discount,Promo,Holiday,Shop_Closed,Category\n")
    
    def get_investment_category(self, investment: int) -> str:
        """Categorize investment amount"""
        if investment < 50000:
            return "Budget"
        elif investment < 150000:
            return "Moderate"
        elif investment < 500000:
            return "Premium"
        else:
            return "Enterprise"


class StoreInfo:
    """Data class for store information"""
    
    def __init__(self, name: str, location: str, store_type: str, investment: int):
        self.store_name = name
        self.location = location
        self.store_type = store_type
        self.investment = investment
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'store_name': self.store_name,
            'location': self.location,
            'store_type': self.store_type,
            'investment': self.investment
        }
    
    def __str__(self) -> str:
        return f"""
📦 Store Profile:
   Name: {self.store_name}
   Location: {self.location}
   Type: {self.store_type}
   Investment: ₹{self.investment:,.0f}
        """

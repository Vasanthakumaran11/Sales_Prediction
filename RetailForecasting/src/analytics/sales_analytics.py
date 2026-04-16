"""
Sales Analytics and Insights
Analyzes sales data and generates insights
"""

import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


class SalesAnalytics:
    """Analyzes sales data and provides insights"""
    
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        self.sales_file = self.store_path / "sales.csv"
    
    def load_sales_data(self) -> List[Dict]:
        """Load all sales records"""
        sales = []
        if not self.sales_file.exists():
            return sales
        
        try:
            with open(self.sales_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sales.append(row)
        except Exception:
            pass
        
        return sales
    
    def get_total_sales(self) -> float:
        """Get total units sold"""
        sales = self.load_sales_data()
        return sum(float(s.get('Units_Sold', 0)) for s in sales)
    
    def get_total_revenue(self) -> float:
        """Get total revenue generated"""
        sales = self.load_sales_data()
        return sum(float(s.get('Revenue', 0)) for s in sales)
    
    def get_best_selling_product(self) -> Tuple[str, int]:
        """Get product with highest sales volume"""
        sales = self.load_sales_data()
        if not sales:
            return "N/A", 0
        
        product_sales = defaultdict(int)
        for sale in sales:
            product = sale.get('Product_Name', 'Unknown')
            units = int(sale.get('Units_Sold', 0))
            product_sales[product] += units
        
        if product_sales:
            best_product = max(product_sales, key=product_sales.get)
            return best_product, product_sales[best_product]
        
        return "N/A", 0
    
    def get_lowest_selling_product(self) -> Tuple[str, int]:
        """Get product with lowest sales volume"""
        sales = self.load_sales_data()
        if not sales:
            return "N/A", 0
        
        product_sales = defaultdict(int)
        for sale in sales:
            product = sale.get('Product_Name', 'Unknown')
            units = int(sale.get('Units_Sold', 0))
            product_sales[product] += units
        
        if product_sales:
            lowest_product = min(product_sales, key=product_sales.get)
            return lowest_product, product_sales[lowest_product]
        
        return "N/A", 0
    
    def get_product_performance(self) -> Dict[str, Dict]:
        """Get performance metrics for each product"""
        sales = self.load_sales_data()
        performance = defaultdict(lambda: {
            'units_sold': 0,
            'revenue': 0,
            'transactions': 0,
            'avg_price': 0,
            'max_units': 0,
            'promo_sales': 0
        })
        
        for sale in sales:
            product = sale.get('Product_Name', 'Unknown')
            units = int(sale.get('Units_Sold', 0))
            revenue = float(sale.get('Revenue', 0))
            price = float(sale.get('Unit_Price', 0))
            promo = int(sale.get('Promo', 0))
            
            performance[product]['units_sold'] += units
            performance[product]['revenue'] += revenue
            performance[product]['transactions'] += 1
            performance[product]['max_units'] = max(performance[product]['max_units'], units)
            if promo:
                performance[product]['promo_sales'] += 1
        
        # Calculate averages
        for product, data in performance.items():
            if data['transactions'] > 0:
                data['avg_price'] = data['revenue'] / data['units_sold'] if data['units_sold'] > 0 else 0
                data['avg_units_per_transaction'] = data['units_sold'] / data['transactions']
        
        return dict(performance)
    
    def get_daily_average(self) -> Dict[str, float]:
        """Get average sales per day"""
        sales = self.load_sales_data()
        if not sales:
            return {}
        
        dates = set(s.get('Date') for s in sales)
        num_days = len(dates)
        
        if num_days == 0:
            return {}
        
        total_revenue = self.get_total_revenue()
        total_units = self.get_total_sales()
        
        return {
            'avg_daily_revenue': total_revenue / num_days,
            'avg_daily_units': total_units / num_days,
            'total_transaction_days': num_days
        }
    
    def get_monthly_summary(self) -> Dict[str, Dict]:
        """Get sales summary by month"""
        sales = self.load_sales_data()
        monthly = defaultdict(lambda: {
            'units_sold': 0,
            'revenue': 0,
            'transactions': 0
        })
        
        for sale in sales:
            try:
                date_str = sale.get('Date', '')
                # Parse date DD-MM-YYYY
                parts = date_str.split('-')
                if len(parts) == 3:
                    month_year = f"{parts[2]}-{parts[1]}"  # YYYY-MM
                    
                    units = int(sale.get('Units_Sold', 0))
                    revenue = float(sale.get('Revenue', 0))
                    
                    monthly[month_year]['units_sold'] += units
                    monthly[month_year]['revenue'] += revenue
                    monthly[month_year]['transactions'] += 1
            except Exception:
                pass
        
        return dict(monthly)
    
    def get_promotion_impact(self) -> Dict:
        """Analyze impact of promotional sales"""
        sales = self.load_sales_data()
        
        promo_sales = []
        regular_sales = []
        
        for sale in sales:
            promo = int(sale.get('Promo', 0))
            units = int(sale.get('Units_Sold', 0))
            revenue = float(sale.get('Revenue', 0))
            
            if promo:
                promo_sales.append({'units': units, 'revenue': revenue})
            else:
                regular_sales.append({'units': units, 'revenue': revenue})
        
        promo_avg_units = np.mean([s['units'] for s in promo_sales]) if promo_sales else 0
        regular_avg_units = np.mean([s['units'] for s in regular_sales]) if regular_sales else 0
        
        promo_boost = ((promo_avg_units - regular_avg_units) / regular_avg_units * 100) if regular_avg_units > 0 else 0
        
        return {
            'total_promo_transactions': len(promo_sales),
            'total_regular_transactions': len(regular_sales),
            'avg_units_promo': promo_avg_units,
            'avg_units_regular': regular_avg_units,
            'promotion_boost_percent': promo_boost
        }
    
    def get_stockout_analysis(self) -> Dict:
        """Analyze stockout occurrences"""
        sales = self.load_sales_data()
        
        stockout_days = []
        shop_closed_days = []
        holiday_sales = []
        
        for sale in sales:
            date = sale.get('Date', '')
            holiday = int(sale.get('Holiday', 0))
            shop_closed = int(sale.get('Shop_Closed', 0))
            units = int(sale.get('Units_Sold', 0))
            
            if shop_closed:
                shop_closed_days.append(date)
            
            if holiday:
                holiday_sales.append(units)
        
        return {
            'shop_closed_count': len(shop_closed_days),
            'holiday_transactions': len(holiday_sales),
            'avg_holiday_units': np.mean(holiday_sales) if holiday_sales else 0
        }
    
    def get_insights(self) -> Dict:
        """Generate comprehensive insights"""
        performance = self.get_product_performance()
        daily_avg = self.get_daily_average()
        promo_impact = self.get_promotion_impact()
        best_product, best_units = self.get_best_selling_product()
        
        insights = {
            'total_products': len(performance),
            'total_revenue': self.get_total_revenue(),
            'total_units_sold': self.get_total_sales(),
            'daily_averages': daily_avg,
            'best_selling_product': best_product,
            'best_selling_units': best_units,
            'promotion_impact': promo_impact,
            'product_performance': performance
        }
        
        return insights
    
    @staticmethod
    def display_analytics(analytics_data: Dict) -> None:
        """Display formatted analytics"""
        from .dashboard import Dashboard
        
        Dashboard.print_section("📊 SALES ANALYTICS SUMMARY")
        
        # Overall metrics
        print(f"{Dashboard.COLORS['BOLD']}Overall Metrics:{Dashboard.COLORS['END']}")
        print(f"  Total Revenue: ₹{analytics_data.get('total_revenue', 0):,.2f}")
        print(f"  Total Units Sold: {int(analytics_data.get('total_units_sold', 0))} units")
        print(f"  Total Products: {analytics_data.get('total_products', 0)}")
        
        # Daily averages
        daily_avg = analytics_data.get('daily_averages', {})
        if daily_avg:
            print(f"\n{Dashboard.COLORS['BOLD']}Daily Averages:{Dashboard.COLORS['END']}")
            print(f"  Avg Daily Revenue: ₹{daily_avg.get('avg_daily_revenue', 0):,.2f}")
            print(f"  Avg Daily Units: {daily_avg.get('avg_daily_units', 0):.1f} units")
            print(f"  Active Days: {int(daily_avg.get('total_transaction_days', 0))} days")
        
        # Best performer
        best_product = analytics_data.get('best_selling_product', 'N/A')
        best_units = analytics_data.get('best_selling_units', 0)
        print(f"\n{Dashboard.COLORS['BOLD']}Best Performer:{Dashboard.COLORS['END']}")
        print(f"  Product: {best_product}")
        print(f"  Units Sold: {best_units} units")
        
        # Promotion impact
        promo = analytics_data.get('promotion_impact', {})
        print(f"\n{Dashboard.COLORS['BOLD']}Promotional Impact:{Dashboard.COLORS['END']}")
        print(f"  Promo Boost: {promo.get('promotion_boost_percent', 0):.1f}%")
        print(f"  Promo Transactions: {promo.get('total_promo_transactions', 0)}")
        
        print()


class SalesAnalyticsExtended(SalesAnalytics):
    """Extended analytics with visualization support"""
    
    def get_trend_analysis(self) -> Dict:
        """Analyze sales trends over time"""
        sales = self.load_sales_data()
        if not sales:
            return {}
        
        # Group by date and calculate trends
        daily_sales = defaultdict(lambda: {'units': 0, 'revenue': 0})
        
        for sale in sales:
            date = sale.get('Date', '')
            units = int(sale.get('Units_Sold', 0))
            revenue = float(sale.get('Revenue', 0))
            
            daily_sales[date]['units'] += units
            daily_sales[date]['revenue'] += revenue
        
        return {
            'daily_sales': dict(daily_sales),
            'num_days': len(daily_sales)
        }
    
    def get_category_analysis(self) -> Dict:
        """Analyze sales by category"""
        sales = self.load_sales_data()
        category_sales = defaultdict(lambda: {'units': 0, 'revenue': 0})
        
        for sale in sales:
            # Would need category data from products file
            # For now, just use product name as placeholder
            units = int(sale.get('Units_Sold', 0))
            revenue = float(sale.get('Revenue', 0))
            
            category_sales['General']['units'] += units
            category_sales['General']['revenue'] += revenue
        
        return dict(category_sales)

"""
Sales Prediction and Month-based Analytics
Predicts sales for a specific month and recommends products based on investment
Uses historical data from base_processed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
from analytics.decision_intelligence import DecisionIntelligenceLayer
from analytics.market_realism import MarketRealismLayer

warnings.filterwarnings("ignore")


class SalesPredictor:
    """Predicts sales based on month and generates product recommendations"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize predictor with historical data
        
        Args:
            data_path: Path to base_processed.csv
        """
        if data_path is None:
            # Default path
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / "data" / "processed" / "base_processed.csv"
        
        self.data_path = data_path
        self.df = None
        self.category_stats = {}
        self.product_stats = {}
        self.month_multipliers = {}
        self.load_data()
    
    def load_data(self):
        """Load and process the base dataset"""
        try:
            if isinstance(self.data_path, str):
                self.data_path = Path(self.data_path)
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded data with {len(self.df)} records")
            
            # Parse date and extract month
            self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%Y')
            self.df['Month_Num'] = self.df['Date'].dt.month
            self.df['Month_Name'] = self.df['Date'].dt.strftime('%B')
            
            self._calculate_statistics()
            
        except Exception as e:
            print(f"⚠️  Error loading data: {str(e)}")
            self.df = None
    
    def _calculate_statistics(self):
        """Calculate statistics for each category and product"""
        if self.df is None:
            return
        
        # Category statistics
        for category in self.df['Category'].unique():
            cat_data = self.df[self.df['Category'] == category]
            
            self.category_stats[category] = {
                'avg_daily_units': cat_data.groupby(self.df['Date'].dt.date)['Units_Sold'].sum().mean(),
                'avg_daily_revenue': cat_data.groupby(self.df['Date'].dt.date)['Revenue'].sum().mean(),
                'avg_price': cat_data['Unit_Price'].mean(),
                'total_revenue': cat_data['Revenue'].sum(),
                'total_units': cat_data['Units_Sold'].sum(),
                'demand_level': cat_data['Demand_Level'].value_counts().idxmax() if len(cat_data) > 0 else 'Low'
            }
        
        # Product statistics
        for product in self.df['Item_Name'].unique():
            prod_data = self.df[self.df['Item_Name'] == product]
            category = prod_data['Category'].iloc[0] if len(prod_data) > 0 else 'Unknown'
            
            self.product_stats[product] = {
                'category': category,
                'avg_units_sold': prod_data['Units_Sold'].mean(),
                'avg_price': prod_data['Unit_Price'].mean(),
                'total_revenue': prod_data['Revenue'].sum(),
                'sell_through_ratio': prod_data['Sell_Through_Ratio'].mean(),
                'demand_level': prod_data['Demand_Level'].value_counts().idxmax() if len(prod_data) > 0 else 'Low'
            }
        
        # Month multipliers (seasonality)
        for month_num in range(1, 13):
            month_data = self.df[self.df['Month_Num'] == month_num]
            if len(month_data) > 0:
                avg_monthly_revenue = month_data['Revenue'].sum() / month_data['Date'].dt.date.nunique()
                overall_avg_revenue = self.df.groupby(self.df['Date'].dt.date)['Revenue'].sum().mean()
                self.month_multipliers[month_num] = max(0.8, avg_monthly_revenue / overall_avg_revenue)
            else:
                self.month_multipliers[month_num] = 1.0
    
    def get_month_name(self, month_num: int) -> str:
        """Get month name from month number"""
        months = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        return months.get(month_num, 'Unknown')
    
    def predict_category_sales(self, 
                               month_num: int, 
                               store_type: str = 'Medium', 
                               investment: Optional[float] = None,
                               historical_context: Optional[float] = None,
                               months_active: int = 12,
                               location_type: str = 'Urban') -> Dict:
        """
        Predict sales for each category in a specific month with business logic
        
        Args:
            month_num: Month number (1-12)
            store_type: Type of store (Small, Medium, Supermarket)
            investment: Actual store investment for scaling
            historical_context: Historical monthly average units
            months_active: Number of months the store has been active (for cold start adjustment)
            location_type: Location type ('Urban', 'Semi-Urban', 'Rural') for demand adjustment
        
        Returns:
            Dictionary with category predictions
        """
        if self.df is None:
            return {}
        
        # Store type multiplier
        store_multiplier = {
            'Small': 0.7,
            'Medium': 1.0,
            'Supermarket': 1.5
        }.get(store_type, 1.0)
        
        month_multiplier = self.month_multipliers.get(month_num, 1.0)
        
        predictions = {}
        for category, stats in self.category_stats.items():
            base_daily_units = stats['avg_daily_units']
            base_daily_revenue = stats['avg_daily_revenue']
            business_interpretation = ""
            
            # Integrated Decision Intelligence Logic
            if investment is not None:
                # Initialize layer for this store
                decision_layer = DecisionIntelligenceLayer(
                    store_size=store_type,
                    investment=investment,
                    margin_percent=15.0 # Default fallback margin
                )
                
                # Process the raw prediction through business logic
                cat_history = historical_context / len(self.category_stats) if historical_context else None
                
                business_aware = decision_layer.process_prediction(
                    raw_daily_units=base_daily_units * store_multiplier * month_multiplier,
                    avg_unit_price=stats['avg_price'],
                    month_num=month_num,
                    historical_avg_units=cat_history
                )
                
                # Update with business-aware metrics
                predicted_daily_units = business_aware['adjusted_daily_units']
                predicted_daily_revenue = business_aware['monthly_revenue'] / 30
                predicted_monthly_units = business_aware['monthly_units']
                predicted_monthly_revenue = business_aware['monthly_revenue']
                business_interpretation = business_aware['business_interpretation']
                
                # Apply Market Realism Layer
                market_layer = MarketRealismLayer(months_active=months_active, location_type=location_type)
                market_realism = market_layer.apply_market_realism(predicted_monthly_revenue)
                
                # Update with market-realistic metrics
                realistic_monthly_revenue = market_realism['realistic_revenue']
                realistic_daily_revenue = realistic_monthly_revenue / 30
                realistic_monthly_units = predicted_monthly_units * market_realism['total_adjustment_factor']
                realistic_daily_units = realistic_monthly_units / 30
                
                # Update business interpretation
                business_interpretation += " " + market_realism['business_explanation']
                
            else:
                # Standard multiplier logic (Fallback)
                predicted_daily_units = base_daily_units * store_multiplier * month_multiplier
                predicted_daily_revenue = base_daily_revenue * store_multiplier * month_multiplier
                predicted_monthly_units = predicted_daily_units * 30
                predicted_monthly_revenue = predicted_daily_revenue * 30
                
                # Apply basic market realism even without investment
                market_layer = MarketRealismLayer(months_active=months_active, location_type=location_type)
                market_realism = market_layer.apply_market_realism(predicted_monthly_revenue)
                
                realistic_monthly_revenue = market_realism['realistic_revenue']
                realistic_daily_revenue = realistic_monthly_revenue / 30
                realistic_monthly_units = predicted_monthly_units * market_realism['total_adjustment_factor']
                realistic_daily_units = realistic_monthly_units / 30
                business_interpretation = market_realism['business_explanation']
            
            predictions[category] = {
                'predicted_monthly_units': round(predicted_monthly_units, 2),
                'predicted_monthly_revenue': round(predicted_monthly_revenue, 2),
                'predicted_daily_units': round(predicted_daily_units, 2),
                'predicted_daily_revenue': round(predicted_daily_revenue, 2),
                'realistic_monthly_units': round(realistic_monthly_units, 2),
                'realistic_monthly_revenue': round(realistic_monthly_revenue, 2),
                'realistic_daily_units': round(realistic_daily_units, 2),
                'realistic_daily_revenue': round(realistic_daily_revenue, 2),
                'avg_price': round(stats['avg_price'], 2),
                'demand_level': stats['demand_level'],
                'interpretation': business_interpretation,
                'market_adjustments': {
                    'cold_start_factor': market_realism['cold_start_factor'],
                    'location_factor': market_realism['location_factor'],
                    'total_adjustment': market_realism['total_adjustment_factor']
                }
            }
        
        return predictions
    
    def recommend_products(self, 
                          investment: float, 
                          month_num: int, 
                          store_type: str = 'Medium',
                          months_active: int = 12,
                          location_type: str = 'Urban') -> List[Dict]:
        """
        Generate product recommendations based on investment and predicted sales
        
        Args:
            investment: Total investment amount (Rs.)
            month_num: Month number when shop opens (1-12)
            store_type: Type of store (Small, Medium, Supermarket)
            months_active: Number of months the store has been active
            location_type: Location type for demand adjustment
        
        Returns:
            List of product recommendations with quantities and prices
        """
        if self.df is None:
            return []
        
        # Get category predictions (Scaled by investment and market realism)
        category_predictions = self.predict_category_sales(
            month_num, store_type, investment=investment,
            months_active=months_active, location_type=location_type
        )
        
        # Calculate investment allocation per category based on REALISTIC revenue
        total_realistic_revenue = sum(pred['realistic_monthly_revenue'] 
                                     for pred in category_predictions.values())
        
        recommendations = []
        allocated_investment = 0
        
        # Sort categories by realistic revenue (descending)
        sorted_categories = sorted(category_predictions.items(), 
                                  key=lambda x: x[1]['realistic_monthly_revenue'], 
                                  reverse=True)
        
        for category, pred in sorted_categories:
            # Allocate investment proportionally to REALISTIC revenue
            if total_realistic_revenue > 0:
                category_investment = (pred['realistic_monthly_revenue'] / total_realistic_revenue) * investment
            else:
                category_investment = investment / len(category_predictions)
            
            # Ensure we don't exceed total investment
            if allocated_investment + category_investment > investment * 0.9:  # Leave 10% for operations
                category_investment = (investment * 0.9) - allocated_investment
            
            # Get all products in this category
            category_products = self.df[self.df['Category'] == category]['Item_Name'].unique()
            
            # Distribute category investment across products
            num_products = min(len(category_products), 3)  # Show top 3 products per category
            investment_per_product = category_investment / num_products
            
            for product in list(category_products)[:num_products]:
                prod_data = self.df[self.df['Item_Name'] == product]
                if len(prod_data) == 0:
                    continue
                
                avg_price = prod_data['Unit_Price'].mean()
                avg_daily_demand = prod_data['Units_Sold'].mean()
                total_revenue = prod_data['Revenue'].sum()
                
                # Calculate recommended quantity based on REALISTIC demand
                realistic_monthly_units = pred['realistic_monthly_units']
                if realistic_monthly_units > 0 and num_products > 0:
                    # Allocate proportionally but ensure reasonable quantities
                    demand_share = realistic_monthly_units / num_products
                    # Buy 1.3x of predicted demand for safety stock
                    demand_based_quantity = int(demand_share * 1.3 / avg_daily_demand)
                    
                    # Use whichever is larger
                    recommended_quantity = max(
                        int(investment_per_product / avg_price),
                        int(demand_based_quantity * 0.5)
                    )
                
                cost = recommended_quantity * avg_price
                
                # Only include if cost is reasonable and we haven't exceeded budget
                if cost > 500 and allocated_investment + cost <= investment * 0.95:
                    product_demand_share = (total_revenue / self.df[self.df['Category'] == category]['Revenue'].sum() * 100) if len(prod_data) > 0 else 0
                    
                    recommendations.append({
                        'product_name': product,
                        'category': category,
                        'quantity': recommended_quantity,
                        'unit_price': round(avg_price, 2),
                        'total_cost': round(cost, 2),
                        'percentage_investment': round((cost / investment) * 100, 2),
                        'predicted_monthly_units': round(pred['realistic_monthly_units'] / num_products, 2),
                        'demand_level': pred['demand_level'],
                        'demand_percentage': round(product_demand_share, 1)
                    })
                    
                    allocated_investment += cost
        
        # Sort by total cost (investment amount)
        recommendations.sort(key=lambda x: x['total_cost'], reverse=True)
        
        return recommendations
    
    def display_sales_prediction(self, 
                                month_num: int, 
                                store_type: str = 'Medium', 
                                investment: Optional[float] = None,
                                months_active: int = 12,
                                location_type: str = 'Urban'):
        """Display formatted sales prediction for a month with business insights"""
        month_name = self.get_month_name(month_num)
        predictions = self.predict_category_sales(
            month_num, store_type, investment=investment, 
            months_active=months_active, location_type=location_type
        )
        
        print(f"\n{'='*100}")
        print(f"📊 SALES PREDICTION FOR {month_name.upper()} ({store_type} Store - {location_type})")
        print(f"{'='*100}\n")
        
        total_predicted_units = 0
        total_predicted_revenue = 0
        total_realistic_units = 0
        total_realistic_revenue = 0
        
        for category, pred in predictions.items():
            print(f"📦 {category}")
            print(f"   Realistic Revenue (Market-Adjusted):")
            print(f"     Daily: {pred['realistic_daily_units']:.0f} units | Rs.{pred['realistic_daily_revenue']:,.0f}")
            print(f"     Monthly: {pred['realistic_monthly_units']:.0f} units | Rs.{pred['realistic_monthly_revenue']:,.0f}")
            print(f"   Avg Price: Rs.{pred['avg_price']:.2f} | Demand: {pred['demand_level']}")
            if 'market_adjustments' in pred:
                adj = pred['market_adjustments']
                print(f"   Adjustments: Cold Start {adj['cold_start_factor']:.1f}x | Location {adj['location_factor']:.1f}x | Total {adj['total_adjustment']:.2f}x")
            print()
            
            total_predicted_units += pred['predicted_monthly_units']
            total_predicted_revenue += pred['predicted_monthly_revenue']
            total_realistic_units += pred['realistic_monthly_units']
            total_realistic_revenue += pred['realistic_monthly_revenue']
        
        print(f"{'='*100}")
        print(f"📈 TOTAL PREDICTION FOR {month_name.upper()}")
        print(f"   Realistic Monthly Units: {total_realistic_units:,.0f}")
        print(f"   Realistic Monthly Revenue: Rs.{total_realistic_revenue:,.0f}")
        print(f"   Daily Average Revenue: Rs.{total_realistic_revenue/30:,.0f}")
        
        # Show market context
        market_layer = MarketRealismLayer(months_active=months_active, location_type=location_type)
        context = market_layer.get_market_context()
        print(f"\n🏪 MARKET CONTEXT:")
        print(f"   Store Maturity: {context['maturity_description']}")
        print(f"   Location Impact: {context['location_description']}")
        print(f"   Cold Start Factor: {context['cold_start_factor']:.1f}x")
        print(f"   Location Factor: {context['location_factor']:.1f}x")
        
        # Business explanation
        if total_realistic_revenue < total_predicted_revenue:
            adjustment_pct = ((total_predicted_revenue - total_realistic_revenue) / total_predicted_revenue) * 100
            print(f"   Revenue Adjustment: {adjustment_pct:.1f}% reduction for market realism")
        
        print(f"{'='*100}\n")
    
    def display_product_recommendations(self, 
                                       investment: float, 
                                       month_num: int, 
                                       store_type: str = 'Medium',
                                       months_active: int = 12,
                                       location_type: str = 'Urban'):
        """Display formatted product recommendations"""
        month_name = self.get_month_name(month_num)
        recommendations = self.recommend_products(
            investment, month_num, store_type, months_active, location_type
        )
        
        print(f"\n{'='*110}")
        print(f"🛒 SMART PRODUCT RECOMMENDATIONS FOR {month_name.upper()}")
        print(f"{'='*110}")
        print(f"📊 Store Type: {store_type} | 💰 Total Investment: Rs.{investment:,.0f}")
        print(f"🏪 Store Age: {months_active} months | 📍 Location: {location_type}")
        print(f"{'='*110}\n")
        
        if not recommendations:
            print("❌ No recommendations available")
            return
        
        total_cost = 0
        print(f"{'Product':<25} {'Qty':>6} {'Unit':>10} {'Total Cost':>12} {'% Invest':>10} {'Demand':<10}")
        print(f"{'─'*25} {'─'*6} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
        
        for idx, rec in enumerate(recommendations, 1):
            # Format: 20 - milk products (100 Rs) / 50 biscuits (1000 Rs)
            product_display = f"{rec['quantity']} - {rec['product_name'].lower()}"
            if len(product_display) > 23:
                product_display = product_display[:20] + "..."
            
            print(f"{product_display:<25} {rec['quantity']:>6} Rs.{rec['unit_price']:>8.2f} "
                  f"Rs.{rec['total_cost']:>10,.0f} {rec['percentage_investment']:>9.1f}% {rec['demand_level']:<10}")
            
            total_cost += rec['total_cost']
        
        print(f"{'─'*25} {'─'*6} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")
        
        remaining_investment = investment - total_cost
        allocated_percentage = (total_cost / investment) * 100
        
        print(f"\n{'='*110}")
        print(f"💰 INVESTMENT ALLOCATION (Based on Realistic Revenue Projections)")
        print(f"{'='*110}")
        print(f"  Total Investment      : Rs.{investment:>15,.2f}")
        print(f"  Product Inventory     : Rs.{total_cost:>15,.2f} ({allocated_percentage:.1f}%)")
        print(f"  Operations/Buffer     : Rs.{remaining_investment:>15,.2f} ({100-allocated_percentage:.1f}%)")
        print(f"{'='*110}\n")
        
        # Show expected revenue (Scaled by investment)
        category_predictions = self.predict_category_sales(month_num, store_type, investment=investment)
        total_monthly_revenue = sum(pred['predicted_monthly_revenue'] for pred in category_predictions.values())
        total_monthly_units = sum(pred['predicted_monthly_units'] for pred in category_predictions.values())
        
        print(f"{'='*90}")
        print(f"📈 EXPECTED MONTHLY PERFORMANCE")
        print(f"{'='*90}")
        print(f"  Predicted Units       : {total_monthly_units:>15,.0f}")
        print(f"  Predicted Revenue     : Rs.{total_monthly_revenue:>15,.0f}")
        daily_revenue = total_monthly_revenue / 30
        print(f"  Daily Average Revenue : Rs.{daily_revenue:>15,.0f}")
        roi_percentage = ((total_monthly_revenue - total_cost) / total_cost) * 100
        print(f"  Initial ROI (Monthly) : {roi_percentage:>15.1f}%")
        print(f"{'='*90}\n")

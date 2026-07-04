"""
Inventory Management and Optimization
Calculates safety stock, reorder points, and risk levels
"""

import math
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class InventoryManager:
    """Handles inventory calculations and recommendations"""
    
    # Z-score for 95% service level (1.96 for 95%)
    Z_SCORE = 1.96
    
    # Lead time in days
    LEAD_TIME_DAYS = 2
    
    # Risk level thresholds (as percentage of average daily demand)
    RISK_THRESHOLDS = {
        'LOW': (100, float('inf')),         # > 100 days
        'MEDIUM': (30, 100),                # 30-100 days
        'HIGH': (10, 30),                   # 10-30 days
        'CRITICAL': (0, 10)                 # < 10 days
    }
    
    @staticmethod
    def calculate_safety_stock(std_demand: float, lead_time: int = LEAD_TIME_DAYS) -> float:
        """
        Calculate safety stock using formula:
        Safety Stock = Z × σ × √LeadTime
        
        Args:
            std_demand: Standard deviation of daily demand
            lead_time: Lead time in days
        
        Returns:
            Safety stock quantity
        """
        if std_demand <= 0:
            return 0
        
        safety_stock = InventoryManager.Z_SCORE * std_demand * math.sqrt(lead_time)
        return max(safety_stock, 1)  # At least 1 unit
    
    @staticmethod
    def calculate_reorder_point(avg_demand: float, safety_stock: float, 
                               lead_time: int = LEAD_TIME_DAYS) -> float:
        """
        Calculate reorder point:
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        
        Args:
            avg_demand: Average daily demand
            safety_stock: Calculated safety stock
            lead_time: Lead time in days
        
        Returns:
            Reorder point quantity
        """
        rop = (avg_demand * lead_time) + safety_stock
        return max(rop, 1)
    
    @staticmethod
    def calculate_economic_order_quantity(annual_demand: float, 
                                         ordering_cost: float = 100,
                                         holding_cost_per_unit: float = 5) -> float:
        """
        Calculate economic order quantity (EOQ):
        EOQ = √(2DS/H)
        
        Where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
        
        Args:
            annual_demand: Expected annual demand
            ordering_cost: Cost per order
            holding_cost_per_unit: Holding cost per unit
        
        Returns:
            EOQ quantity
        """
        if annual_demand <= 0 or holding_cost_per_unit <= 0:
            return 0
        
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        return max(eoq, 1)
    
    @staticmethod
    def determine_risk_level(current_stock: int, avg_daily_demand: float) -> str:
        """
        Determine inventory risk level based on days of supply
        
        Args:
            current_stock: Current inventory quantity
            avg_daily_demand: Average daily demand
        
        Returns:
            Risk level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
        """
        if avg_daily_demand <= 0:
            return 'LOW'
        
        days_of_supply = current_stock / avg_daily_demand
        
        for risk_level, (min_days, max_days) in InventoryManager.RISK_THRESHOLDS.items():
            if min_days <= days_of_supply <= max_days:
                return risk_level
        
        return 'CRITICAL'
    
    @staticmethod
    def get_risk_level_emoji(risk_level: str) -> str:
        """Get emoji for risk level"""
        emojis = {
            'LOW': '🟢',
            'MEDIUM': '🟡',
            'HIGH': '🔴',
            'CRITICAL': '🔴🔴'
        }
        return emojis.get(risk_level, '⚪')
    
    @staticmethod
    def get_inventory_recommendation(product_name: str,
                                    current_stock: int,
                                    demand_history: List[int],
                                    unit_cost: float,
                                    unit_price: float) -> Dict:
        """
        Get comprehensive inventory recommendation
        
        Args:
            product_name: Product name
            current_stock: Current inventory
            demand_history: List of historical demand values
            unit_cost: Cost per unit
            unit_price: Selling price per unit
        
        Returns:
            Dictionary with recommendations
        """
        if not demand_history or len(demand_history) == 0:
            demand_history = [10]  # Default for new products
        
        # Calculate statistics
        avg_demand = np.mean(demand_history)
        std_demand = np.std(demand_history) if len(demand_history) > 1 else 0
        
        # Calculate inventory metrics
        safety_stock = InventoryManager.calculate_safety_stock(std_demand)
        reorder_point = InventoryManager.calculate_reorder_point(avg_demand, safety_stock)
        eoq = InventoryManager.calculate_economic_order_quantity(
            avg_demand * 365,  # Annualize
            ordering_cost=100,
            holding_cost_per_unit=unit_cost * 0.25  # Assume 25% holding cost
        )
        
        # Risk assessment
        risk_level = InventoryManager.determine_risk_level(current_stock, avg_demand)
        
        # Order recommendation
        if current_stock <= reorder_point:
            order_quantity = max(eoq, reorder_point - current_stock + safety_stock)
            order_recommendation = "URGENT REORDER"
        elif current_stock <= safety_stock:
            order_quantity = eoq
            order_recommendation = "PRIORITY REORDER"
        else:
            order_quantity = 0
            order_recommendation = "NO ACTION NEEDED"
        
        # Financial metrics
        holding_cost_value = current_stock * unit_cost * 0.25 / 365  # Daily cost
        stockout_risk = 1.0 - (current_stock / reorder_point) if reorder_point > 0 else 0
        stockout_risk = max(0, min(1.0, stockout_risk))  # Clamp to [0,1]
        
        return {
            'product_name': product_name,
            'current_stock': current_stock,
            'avg_daily_demand': avg_demand,
            'std_daily_demand': std_demand,
            'safety_stock': int(safety_stock),
            'reorder_point': int(reorder_point),
            'economic_order_qty': int(eoq),
            'risk_level': risk_level,
            'order_recommendation': order_recommendation,
            'order_quantity': int(order_quantity),
            'days_of_supply': current_stock / avg_demand if avg_demand > 0 else 0,
            'holding_cost_daily': holding_cost_value,
            'stockout_risk_percent': stockout_risk * 100
        }
    
    @staticmethod
    def display_inventory_recommendation(recommendation: Dict) -> None:
        """Display formatted inventory recommendation"""
        from interface.dashboard import Dashboard
        
        Dashboard.print_section(f"📦 INVENTORY RECOMMENDATION: {recommendation['product_name']}")
        
        # Current status
        print(f"{Dashboard.COLORS['BOLD']}Current Status:{Dashboard.COLORS['END']}")
        print(f"  Current Stock: {int(recommendation['current_stock'])} units")
        print(f"  Avg Daily Demand: {recommendation['avg_daily_demand']:.2f} units/day")
        print(f"  Days of Supply: {recommendation['days_of_supply']:.1f} days")
        
        # Risk level
        risk_emoji = InventoryManager.get_risk_level_emoji(recommendation['risk_level'])
        print(f"\n{Dashboard.COLORS['BOLD']}Risk Level: {risk_emoji} {recommendation['risk_level']}{Dashboard.COLORS['END']}")
        print(f"  Stockout Risk: {recommendation['stockout_risk_percent']:.1f}%")
        
        # Recommendations
        print(f"\n{Dashboard.COLORS['BOLD']}Recommendations:{Dashboard.COLORS['END']}")
        print(f"  Safety Stock: {recommendation['safety_stock']} units")
        print(f"  Reorder Point: {recommendation['reorder_point']} units")
        print(f"  Economic Order Qty: {recommendation['economic_order_qty']} units")
        
        # Order action
        print(f"\n{Dashboard.COLORS['BOLD']}Order Action:{Dashboard.COLORS['END']}")
        if 'CRITICAL' in recommendation['risk_level'] or 'URGENT' in recommendation['order_recommendation']:
            print(f"  {Dashboard.COLORS['RED']}🚨 {recommendation['order_recommendation']}{Dashboard.COLORS['END']}")
        elif 'PRIORITY' in recommendation['order_recommendation']:
            print(f"  {Dashboard.COLORS['YELLOW']}⚠️  {recommendation['order_recommendation']}{Dashboard.COLORS['END']}")
        else:
            print(f"  {Dashboard.COLORS['GREEN']}✅ {recommendation['order_recommendation']}{Dashboard.COLORS['END']}")
        
        if recommendation['order_quantity'] > 0:
            print(f"  Suggested Order Quantity: {recommendation['order_quantity']} units")
        
        # Financial metrics
        print(f"\n{Dashboard.COLORS['BOLD']}Financial Metrics:{Dashboard.COLORS['END']}")
        print(f"  Daily Holding Cost: ₹{recommendation['holding_cost_daily']:.2f}")
        
        print()

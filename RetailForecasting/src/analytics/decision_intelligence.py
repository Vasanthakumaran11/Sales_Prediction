"""
Decision Intelligence Layer
Processes raw ML predictions into business-aware actionable decisions.
Ensures realism through investment scaling, historical prioritization, and seasonality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class DecisionIntelligenceLayer:
    """
    Transforms raw sales predictions into business-aligned results.
    Prevents unrealistic projections by applying budget constraints and context.
    """

    # Base Capital Targets (Default)
    BASE_CAPITAL = {
        'Small': 200000.0,
        'Medium': 1000000.0,
        'Large': 5000000.0
    }

    # Daily Unit Capacity Constraints
    CAPACITY_LIMITS = {
        'Small': 800,
        'Medium': 3500,
        'Large': 15000
    }

    # Seasonality Lookup Table (Event Multipliers)
    SEASONAL_MULTIPLIERS = {
        1: 1.30,  # January: Pongal / New Year
        3: 1.10,  # March: Early Summer
        4: 1.25,  # April: Ramzan (Approx/Variable)
        5: 1.15,  # May: Summer peak
        10: 1.40, # October: Diwali / Festive season
        11: 1.30, # November: Post-Festive / Pre-Winter
        12: 1.35  # December: Christmas / Year-end
    }

    def __init__(self, store_size: str, investment: float, margin_percent: float):
        self.store_size = store_size if store_size in self.BASE_CAPITAL else 'Small'
        self.investment = investment
        self.margin_percent = margin_percent
        self.base_capital = self.BASE_CAPITAL.get(self.store_size, 200000.0)
        self.capacity_limit = self.CAPACITY_LIMITS.get(self.store_size, 800)

    def apply_budget_constraint(self, monthly_revenue: float, investment: float, month_num: Optional[int] = None) -> float:
        """
        Ensures monthly revenue projections are realistic relative to investment.
        Uses a dynamic soft-cap mechanism:
        - Regular month cap: ~1.5x investment
        - Festive month cap: ~2.5x investment
        """
        # Get seasonal multiplier for this month
        multiplier = self.SEASONAL_MULTIPLIERS.get(month_num, 1.0) if month_num else 1.0
        
        # Calculate dynamic caps based on season
        # Festivals allow for significantly higher turnover
        soft_cap_multiplier = 1.5 * multiplier
        hard_max_multiplier = 3.0 * multiplier
        
        soft_cap = investment * soft_cap_multiplier
        hard_max = investment * hard_max_multiplier

        if monthly_revenue <= soft_cap:
            return monthly_revenue
        
        # Apply logarithmic dampen for revenue above soft cap
        excess = monthly_revenue - soft_cap
        dampened_excess = soft_cap * np.log1p(excess / soft_cap)
        
        bounded_revenue = soft_cap + dampened_excess
        return min(bounded_revenue, hard_max)

    def apply_seasonal_multiplier(self, base_val: float, month_num: int) -> float:
        """Applies seasonal boost or drop based on the month."""
        multiplier = self.SEASONAL_MULTIPLIERS.get(month_num, 1.0)
        return base_val * multiplier

    def apply_historical_baseline(self, prediction: float, historical_avg: float, weight: float = 0.7) -> float:
        """
        Blends new prediction with historical store performance.
        Default weight (0.7) favors historical data over global model baseline.
        """
        if historical_avg <= 0:
            return prediction
        
        return (historical_avg * weight) + (prediction * (1 - weight))

    def process_prediction(self, 
                           raw_daily_units: float, 
                           avg_unit_price: float = 100.0, 
                           month_num: Optional[int] = None,
                           historical_avg_units: Optional[float] = None) -> Dict:
        """
        Enhanced processing logic for realistic business metrics.
        """
        # 1. Start with raw daily prediction
        current_units = raw_daily_units

        # 2. Apply Historical Baseline (if available)
        if historical_avg_units is not None:
            current_units = self.apply_historical_baseline(current_units, historical_avg_units)

        # 3. Apply Seasonal Multiplier
        if month_num is not None:
            current_units = self.apply_seasonal_multiplier(current_units, month_num)

        # 4. Apply Scaling based on investment vs base capital
        scaling_factor = min(1.2, self.investment / self.base_capital)
        current_units = current_units * scaling_factor

        # 5. Enforce Capacity Capping
        adjusted_daily_units = min(current_units, self.capacity_limit)
        
        # 6. Financial Calculations
        raw_monthly_revenue = adjusted_daily_units * avg_unit_price * 30
        
        # 7. Apply Budget Constraint (The "Realism" Filter)
        # This prevents unrealistic revenue compared to capital
        adjusted_monthly_revenue = self.apply_budget_constraint(raw_monthly_revenue, self.investment, month_num)
        
        # Adjust daily units and monthly units to match the budget-constrained revenue
        revenue_dampen_factor = adjusted_monthly_revenue / raw_monthly_revenue if raw_monthly_revenue > 0 else 1.0
        final_daily_units = adjusted_daily_units * revenue_dampen_factor
        final_monthly_units = final_daily_units * 30
        
        # Profit Calculation
        monthly_profit = adjusted_monthly_revenue * (self.margin_percent / 100.0)

        # Interpretation Details
        interpretation = f"Business metrics aligned with {self.store_size} store constraints."
        if revenue_dampen_factor < 0.95:
             interpretation += f" Projections capped based on Rs.{self.investment:,.0f} investment realism."
        
        events = {
            1: "Pongal/New Year Boost",
            5: "Summer Season Adjustment",
            10: "Diwali Festive Boost",
            12: "Christmas/Year-end Boost"
        }
        if month_num in events:
            interpretation += f" Includes {events[month_num]}."

        return {
            'raw_prediction': raw_daily_units,
            'adjusted_daily_units': final_daily_units,
            'monthly_units': final_monthly_units,
            'monthly_revenue': adjusted_monthly_revenue,
            'expected_profit': monthly_profit,
            'scaling_factor': scaling_factor,
            'budget_utilization': revenue_dampen_factor,
            'business_interpretation': interpretation
        }

    def optimize_inventory(self, 
                           predicted_daily_demand: float, 
                           current_stock: int, 
                           std_dev_demand: Optional[float] = None, 
                           lead_time_days: int = 3, 
                           z_score: float = 1.65) -> Dict:
        """
        Calculate inventory optimization metrics (Safety Stock, ROP, Risk).
        """
        if std_dev_demand is None:
            std_dev_demand = predicted_daily_demand * 0.15  # Default 15% variability
            
        safety_stock = z_score * std_dev_demand * np.sqrt(lead_time_days)
        reorder_point = (predicted_daily_demand * lead_time_days) + safety_stock
        
        reorder_needed = current_stock < reorder_point
        recommended_qty = max(0, int(reorder_point + safety_stock - current_stock)) if reorder_needed else 0
        
        days_remaining = current_stock / predicted_daily_demand if predicted_daily_demand > 0 else 99
        
        risk_level = 'HIGH'
        if days_remaining > 20: risk_level = 'LOW'
        elif days_remaining > 7: risk_level = 'MEDIUM'
        
        return {
            'safety_stock': int(safety_stock),
            'reorder_point': int(reorder_point),
            'reorder_needed': reorder_needed,
            'recommended_reorder_qty': recommended_qty,
            'days_of_supply': round(days_remaining, 1),
            'risk_level': risk_level
        }

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analytics.decision_intelligence import DecisionIntelligenceLayer

def test_decision_intelligence():
    print("--- Testing Decision Intelligence Layer ---")
    
    # Store settings: Small store, 1.5L investment (base is 2L), 15% margin
    layer = DecisionIntelligenceLayer(store_size='Small', investment=150000, margin_percent=15)
    
    # Case 1: High demand exceeding capacity
    # Raw demand: 1000 units/day (Small capacity is 800)
    raw_demand = 1000
    results = layer.process_prediction(raw_demand)
    
    print(f"\nCase 1: High Demand Overflow")
    print(f"  Raw Units: {results['raw_prediction']}")
    print(f"  Scaling Factor (1.5L/2L): {results['scaling_factor']:.2f}")
    # Expected scaling: 0.75 * 1000 = 750 (still below 800 cap)
    print(f"  Adjusted Daily Units: {results['adjusted_daily_units']}")
    print(f"  Capacity Hit: {results['capacity_hit']}")
    print(f"  Monthly Revenue: Rs.{results['monthly_revenue']:,.2f}")
    
    # Case 2: Extreme demand hitting capacity
    # Raw demand: 2000 units/day
    layer_large_investment = DecisionIntelligenceLayer(store_size='Small', investment=300000, margin_percent=15)
    results_hit = layer_large_investment.process_prediction(2000)
    
    print(f"\nCase 2: Capacity Capping")
    print(f"  Raw Units: 2000")
    print(f"  Scaling Factor (3L/2L): {results_hit['scaling_factor']:.2f}") # capped at 1.2
    # 2000 * 1.2 = 2400 -> capped at 800
    print(f"  Adjusted Daily Units (Capped): {results_hit['adjusted_daily_units']}")
    print(f"  Capacity Hit: {results_hit['capacity_hit']}")

    # Case 3: Inventory Optimization
    print(f"\nCase 3: Inventory Optimization")
    # Demand: 500 units/day, Current Stock: 1000 units (2 days supply)
    inv_results = layer.optimize_inventory(
        predicted_daily_demand=500,
        current_stock=1000,
        std_dev_demand=50, # 10% variability
        lead_time_days=3
    )
    
    print(f"  Safety Stock: {inv_results['safety_stock']}")
    print(f"  Reorder Point: {inv_results['reorder_point']}")
    print(f"  Risk Level: {inv_results['risk_level']} ({inv_results['days_of_supply']} days)")
    print(f"  Reorder Needed: {inv_results['reorder_needed']}")
    print(f"  Recommended Qty: {inv_results['recommended_reorder_qty']}")

if __name__ == "__main__":
    test_decision_intelligence()

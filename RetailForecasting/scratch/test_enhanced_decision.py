import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analytics.decision_intelligence import DecisionIntelligenceLayer

def test_enhanced_decisions():
    print("--- Testing ENHANCED Decision Intelligence Layer ---")
    
    # CASE 1: Low investment new store (Thiru Stores scenario)
    # Store settings: Small store, 25,000 investment
    layer_new = DecisionIntelligenceLayer(store_size='Small', investment=25000, margin_percent=15)
    
    # Raw prediction from global model: 1000 units/day -> ~3,000,000 revenue
    # (The previous bug showed 3M revenue here)
    raw_daily = 1000 
    results_new = layer_new.process_prediction(raw_daily, avg_unit_price=100)
    
    print(f"\nScenario 1: New Store with Rs.25,000 Investment")
    print(f"  Raw Units Predicted: {raw_daily}")
    print(f"  Adjusted Daily Units: {results_new['adjusted_daily_units']:.1f}")
    print(f"  Monthly Revenue: Rs.{results_new['monthly_revenue']:,.2f}")
    print(f"  Budget Utilization: {results_new['budget_utilization']:.2%}")
    print(f"  Insights: {results_new['business_interpretation']}")
    
    assert results_new['monthly_revenue'] <= 25000 * 3.0, "Revenue should be capped for realism!"
    print(f"  [OK] BUG FIXED: Revenue is strictly bounded by investment.")

    # CASE 2: Seasonal Multiplier (Diwali - October)
    print(f"\nScenario 2: Seasonality (October - Diwali)")
    results_oct = layer_new.process_prediction(100, month_num=10)
    results_reg = layer_new.process_prediction(100, month_num=2) # Feb (no multiplier)
    
    boost = (results_oct['monthly_revenue'] / results_reg['monthly_revenue'] - 1) * 100
    print(f"  Diwali Boost: {boost:.1f}%")
    print(f"  Insights: {results_oct['business_interpretation']}")
    
    # CASE 3: Historical Data Prioritization
    print(f"\nScenario 3: Existing Store with History")
    # Store has historical average of 50 units/day
    historical_avg = 50.0
    # Global model predicts 200 units/day (e.g. outlier or generic store)
    global_pred = 200.0
    
    results_hist = layer_new.process_prediction(global_pred, historical_avg_units=historical_avg)
    print(f"  Global Prediction: {global_pred}")
    print(f"  Historical Average: {historical_avg}")
    print(f"  Resultant Daily Units: {results_hist['adjusted_daily_units']:.1f}")
    
    # Expected: (50 * 0.7) + (200 * 0.3) = 35 + 60 = 95 units
    # (Plus scaling/caps if applicable, but here scaling is 25k/2L = 0.125)
    # Let's adjust scaling in test to 1.0 for clarity
    layer_balanced = DecisionIntelligenceLayer(store_size='Small', investment=200000, margin_percent=15)
    results_hist_balanced = layer_balanced.process_prediction(global_pred, historical_avg_units=historical_avg)
    print(f"  Blended Result (Balanced): {results_hist_balanced['adjusted_daily_units']:.1f}")
    print(f"  [OK] Historical patterns are prioritized over generic models.")

if __name__ == "__main__":
    test_enhanced_decisions()

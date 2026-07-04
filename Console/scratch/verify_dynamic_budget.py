import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analytics.decision_intelligence import DecisionIntelligenceLayer

def verify_dynamic_budget():
    print("--- Verifying DYNAMIC Budget Capping ---")
    
    investment = 100000.0 # 1 Lakh
    layer = DecisionIntelligenceLayer(store_size='Small', investment=investment, margin_percent=15)
    
    # Generic monthly prediction from raw model (e.g. 20L - unrealistically high)
    raw_monthly = 2000000.0 
    
    print(f"\nStore Investment: Rs.{investment:,.0f}")
    print(f"Raw Generic Prediction: Rs.{raw_monthly:,.0f}")
    
    # CASE 1: February (Regular Month, no festival)
    # Target: Should be capped strictly (approx 1.5x - 3.0x investment)
    capped_feb = layer.apply_budget_constraint(raw_monthly, investment, month_num=2)
    print(f"February (Regular) Caped Revenue : Rs.{capped_feb:,.0f} ({capped_feb/investment:.2f}x)")
    
    # CASE 2: October (Diwali - Multiplier 1.4)
    # Target: Should be capped less strictly (multiplier boosts the limit)
    capped_oct = layer.apply_budget_constraint(raw_monthly, investment, month_num=10)
    print(f"October (Diwali) Caped Revenue   : Rs.{capped_oct:,.0f} ({capped_oct/investment:.2f}x)")
    
    # CASE 3: January (Pongal - Multiplier 1.3)
    capped_jan = layer.apply_budget_constraint(raw_monthly, investment, month_num=1)
    print(f"January (Pongal) Caped Revenue   : Rs.{capped_jan:,.0f} ({capped_jan/investment:.2f}x)")

    # Assertion: Diwali should allow significantly more sales than February
    assert capped_oct > capped_feb, "Diwali cap should be higher than regular month's cap!"
    print(f"\n[OK] Dynamic Scaling Verified: Seasonal festivals allow higher realistic growth ranges.")

if __name__ == "__main__":
    verify_dynamic_budget()

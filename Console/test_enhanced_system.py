#!/usr/bin/env python3
"""
Test script for the enhanced Retail Forecasting System with Market Realism Layer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analytics.sales_predictor import SalesPredictor
from analytics.market_realism import MarketRealismLayer

def test_market_realism_layer():
    """Test the Market Realism Layer independently"""
    print("=" * 60)
    print("TESTING MARKET REALISM LAYER")
    print("=" * 60)

    # Test different scenarios
    scenarios = [
        (1, 'Urban'),      # New store, urban
        (1, 'Rural'),      # New store, rural
        (2, 'Semi-Urban'), # 2-month old store, semi-urban
        (12, 'Urban')      # Mature store, urban
    ]

    for months_active, location in scenarios:
        layer = MarketRealismLayer(months_active=months_active, location_type=location)
        result = layer.apply_market_realism(100000)  # Test with 100k predicted revenue

        print(f"\n📍 {months_active} months active, {location} location:")
        print(f"   Raw Revenue: ₹{result['raw_predicted_revenue']:,.0f}")
        print(f"   Realistic Revenue: ₹{result['realistic_revenue']:,.0f}")
        print(f"   Cold Start Factor: {result['cold_start_factor']:.1f}x")
        print(f"   Location Factor: {result['location_factor']:.1f}x")
        print(f"   Total Adjustment: {result['total_adjustment_factor']:.2f}x")
        print(f"   Business Explanation: {result['business_explanation']}")

def test_full_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE PREDICTION PIPELINE")
    print("=" * 60)

    try:
        predictor = SalesPredictor()

        # Test scenarios
        test_cases = [
            {
                'month_num': 1,
                'store_type': 'Medium',
                'investment': 500000,
                'months_active': 1,
                'location_type': 'Rural',
                'description': 'New Rural Store (High Risk)'
            },
            {
                'month_num': 10,
                'store_type': 'Medium',
                'investment': 750000,
                'months_active': 6,
                'location_type': 'Urban',
                'description': 'Mature Urban Store (Low Risk)'
            }
        ]

        for case in test_cases:
            desc = case.pop('description')  # Remove description from kwargs
            print(f"\n🏪 {desc}")
            print("-" * 50)

            predictions = predictor.predict_category_sales(**case)

            total_realistic = sum(pred['realistic_monthly_revenue'] for pred in predictions.values())

            print(f"Total Realistic Revenue: ₹{total_realistic:,.0f}")

            # Show top 3 categories
            sorted_cats = sorted(predictions.items(), key=lambda x: x[1]['realistic_monthly_revenue'], reverse=True)
            print("\nTop 3 Categories (Realistic Revenue):")
            for cat, pred in sorted_cats[:3]:
                adj = pred['market_adjustments']
                print(f"  {cat}: ₹{pred['realistic_monthly_revenue']:,.0f} "
                      f"(Cold Start: {adj['cold_start_factor']:.1f}x, Location: {adj['location_factor']:.1f}x)")

    except Exception as e:
        print(f"❌ Error in prediction pipeline: {str(e)}")

def main():
    """Run all tests"""
    print("🧪 TESTING ENHANCED RETAIL FORECASTING SYSTEM")
    print("Including Market Realism Layer for real-world adjustments")

    test_market_realism_layer()
    test_full_prediction_pipeline()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)
    print("\n🎯 Key Enhancements Implemented:")
    print("   • Market Realism Layer after Decision Intelligence")
    print("   • Cold start adjustments (0.4x for month 1, 0.7x for months 2-3)")
    print("   • Location-based demand factors (Urban: 1.0x, Semi-Urban: 0.8x, Rural: 0.6x)")
    print("   • realistic_revenue = predicted_revenue * cold_start_factor * location_factor")
    print("   • Clear distinction between raw predictions and adjusted realistic revenue")
    print("   • Business explanations for market adjustments")

if __name__ == "__main__":
    main()
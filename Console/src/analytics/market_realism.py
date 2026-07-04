"""
Market Realism Layer
Adjusts predictions for real-world practical values by incorporating store maturity and location-based demand behavior.
"""

from typing import Dict, Optional


class MarketRealismLayer:
    """
    Transforms business-aware predictions into market-realistic results.
    Accounts for store maturity (cold start) and location-based demand patterns.
    """

    # Cold Start Scaling Factors based on store age
    COLD_START_FACTORS = {
        1: 0.4,   # First month: 40% of predicted sales
        2: 0.7,   # Months 2-3: 70% of predicted sales
        3: 0.7,
        'mature': 1.0  # 4+ months: 100% of predicted sales
    }

    # Location-based Demand Factors
    LOCATION_FACTORS = {
        'Urban': 1.0,      # High footfall, high purchasing power
        'Semi-Urban': 0.8, # Moderate footfall and purchasing power
        'Rural': 0.6       # Lower footfall, lower purchasing power
    }

    def __init__(self, months_active: int, location_type: str):
        """
        Initialize Market Realism Layer

        Args:
            months_active: Number of months the store has been active
            location_type: Type of location ('Urban', 'Semi-Urban', 'Rural')
        """
        self.months_active = months_active
        self.location_type = location_type

        # Validate inputs
        if location_type not in self.LOCATION_FACTORS:
            raise ValueError(f"Invalid location_type: {location_type}. Must be one of {list(self.LOCATION_FACTORS.keys())}")

        if months_active < 1:
            raise ValueError("months_active must be at least 1")

    def get_cold_start_factor(self) -> float:
        """
        Get the cold start adjustment factor based on store age

        Returns:
            Scaling factor between 0.4 and 1.0
        """
        if self.months_active <= 3:
            return self.COLD_START_FACTORS.get(self.months_active, 0.7)
        else:
            return self.COLD_START_FACTORS['mature']

    def get_location_factor(self) -> float:
        """
        Get the location-based demand adjustment factor

        Returns:
            Scaling factor between 0.6 and 1.0
        """
        return self.LOCATION_FACTORS[self.location_type]

    def apply_market_realism(self, predicted_revenue: float) -> Dict:
        """
        Apply market realism adjustments to predicted revenue

        Args:
            predicted_revenue: Revenue after Decision Intelligence Layer processing

        Returns:
            Dictionary with raw and adjusted revenue metrics
        """
        cold_start_factor = self.get_cold_start_factor()
        location_factor = self.get_location_factor()

        # Calculate realistic revenue
        realistic_revenue = predicted_revenue * cold_start_factor * location_factor

        # Generate business explanation
        explanation_parts = []

        if cold_start_factor < 1.0:
            if self.months_active == 1:
                explanation_parts.append("new store stabilization period (first month)")
            elif self.months_active <= 3:
                explanation_parts.append(f"early-stage store adjustment (month {self.months_active})")
            else:
                explanation_parts.append("store maturity considerations")

        if location_factor < 1.0:
            location_desc = {
                'Semi-Urban': 'moderate customer footfall and purchasing power',
                'Rural': 'lower customer footfall and purchasing power'
            }
            explanation_parts.append(location_desc.get(self.location_type, f"{self.location_type.lower()} location demand conditions"))

        business_explanation = "Revenue adjusted for " + " and ".join(explanation_parts) + "." if explanation_parts else "No market realism adjustments applied."

        return {
            'raw_predicted_revenue': predicted_revenue,
            'realistic_revenue': realistic_revenue,
            'cold_start_factor': cold_start_factor,
            'location_factor': location_factor,
            'total_adjustment_factor': cold_start_factor * location_factor,
            'business_explanation': business_explanation,
            'store_maturity': self._get_maturity_description(),
            'location_type': self.location_type
        }

    def _get_maturity_description(self) -> str:
        """Get human-readable description of store maturity"""
        if self.months_active == 1:
            return "New Store (Month 1)"
        elif self.months_active <= 3:
            return f"Early Stage (Month {self.months_active})"
        else:
            return "Mature Store"

    def get_market_context(self) -> Dict:
        """
        Get current market context information

        Returns:
            Dictionary with market factors and descriptions
        """
        return {
            'months_active': self.months_active,
            'location_type': self.location_type,
            'cold_start_factor': self.get_cold_start_factor(),
            'location_factor': self.get_location_factor(),
            'maturity_description': self._get_maturity_description(),
            'location_description': self._get_location_description()
        }

    def _get_location_description(self) -> str:
        """Get human-readable description of location impact"""
        descriptions = {
            'Urban': 'High customer footfall and purchasing power',
            'Semi-Urban': 'Moderate customer footfall and purchasing power',
            'Rural': 'Lower customer footfall and purchasing power'
        }
        return descriptions.get(self.location_type, self.location_type)
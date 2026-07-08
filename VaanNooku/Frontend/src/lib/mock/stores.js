export const STORE_TYPES = ["Small", "Medium", "Supermarket"];
export const LOCATION_TYPES = ["Urban", "Semi-Urban", "Rural"];
export const OPENING_MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

export const LOCATION_MULTIPLIERS = {
  Urban: 1.0,
  "Semi-Urban": 0.8,
  Rural: 0.6,
};

export const CAPACITY_LIMITS = {
  Small: 400,
  Medium: 800,
  Supermarket: 2000,
};

export const COLD_START_FACTORS = [
  { monthsSinceOpening: 1, factor: 0.4, label: "Month 1" },
  { monthsSinceOpening: 3, factor: 0.7, label: "Months 2-3" },
  { monthsSinceOpening: Infinity, factor: 1.0, label: "Month 4+" },
];

export function getColdStartFactor(monthsActive) {
  return (
    COLD_START_FACTORS.find((f) => monthsActive <= f.monthsSinceOpening) ??
    COLD_START_FACTORS[COLD_START_FACTORS.length - 1]
  );
}

export const STORE_PROFILES = [
  {
    id: "balaji-store",
    name: "Balaji Store",
    type: "Supermarket",
    location: "Urban",
    investment: 850000,
    openingMonth: "October",
    monthsActive: 9,
    metrics: {
      forecastR2: 0.932,
      wasteMargin: 0.024,
      stockouts: 2,
      leakingMargin: 0.048,
      deficitCount: 3,
      revenue: 294200,
      inventoryValue: 185000,
    },
  },
  {
    id: "shiva-stores",
    name: "Shiva Stores",
    type: "Medium",
    location: "Semi-Urban",
    investment: 300000,
    openingMonth: "January",
    monthsActive: 6,
    metrics: {
      forecastR2: 0.918,
      wasteMargin: 0.041,
      stockouts: 8,
      leakingMargin: 0.072,
      deficitCount: 6,
      revenue: 120500,
      inventoryValue: 92000,
    },
  },
  {
    id: "surya-markets",
    name: "Surya Markets",
    type: "Small",
    location: "Rural",
    investment: 90000,
    openingMonth: "June",
    monthsActive: 2,
    metrics: {
      forecastR2: 0.895,
      wasteMargin: 0.085,
      stockouts: 14,
      leakingMargin: 0.113,
      deficitCount: 9,
      revenue: 41200,
      inventoryValue: 28400,
    },
  },
];

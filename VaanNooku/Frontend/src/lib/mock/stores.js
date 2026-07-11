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

export const STORE_PROFILES = [];

import { resolveData } from "./client";
import { DEMAND_BASELINE } from "@/lib/mock/forecast";
import { CAPACITY_LIMITS, LOCATION_MULTIPLIERS, getColdStartFactor } from "@/lib/mock/stores";
import { getFestivalForMonth, FESTIVALS } from "@/lib/mock/catalog";

function buildForecast(store) {
  const capacity = CAPACITY_LIMITS[store?.type] ?? CAPACITY_LIMITS.Supermarket;
  const locationMultiplier = LOCATION_MULTIPLIERS[store?.location] ?? 1;
  const coldStart = getColdStartFactor(store?.monthsActive ?? 12);
  const festival = getFestivalForMonth(store?.openingMonth ?? "");
  const seasonalMultiplier = festival ? festival.modifier : 1;

  const adjusted = DEMAND_BASELINE.rawSignal.map((val) => {
    const scaled = val * locationMultiplier * coldStart.factor * seasonalMultiplier;
    return Math.round(Math.min(scaled, capacity));
  });

  return {
    days: DEMAND_BASELINE.days,
    raw: DEMAND_BASELINE.rawSignal,
    actual: DEMAND_BASELINE.actualSales,
    adjusted,
    capacity,
    locationMultiplier,
    coldStart,
    activeFestival: festival,
    upcomingFestivals: FESTIVALS,
  };
}

// GET /api/stores/:storeId/forecast
export async function getDemandForecast(store) {
  return resolveData(`/api/stores/${store?.id ?? "executive"}/forecast`, () => buildForecast(store));
}

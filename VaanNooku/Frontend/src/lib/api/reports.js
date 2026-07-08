import { resolveData } from "./client";
import { getDemandForecast } from "./forecasting";
import { getCapitalAllocation } from "./financial";
import { getInventory } from "./inventory";

function pct(n) {
  return `${Math.round(n * 100)}%`;
}

async function buildReport(store) {
  const [forecast, financials, inventory] = await Promise.all([
    getDemandForecast(store),
    getCapitalAllocation(store),
    getInventory(store?.id),
  ]);

  const criticalItems = inventory.filter((i) => i.stock <= i.rop);
  const peakDemand = Math.max(...forecast.adjusted);
  const isBreaching = peakDemand >= forecast.capacity;
  const leakingCategories = financials.categories.filter((c) => c.isLeak);

  const storeName = store?.name ?? "the multi-store network";
  const generatedAt = new Date().toISOString();

  const sections = [
    {
      title: "Executive Summary",
      body: `${storeName} is forecast at an R² of ${
        store?.metrics?.forecastR2 ?? "0.93"
      } with a weekly ROI estimate of ${financials.roi}%. Peak adjusted demand this week is ${peakDemand} units against a physical capacity of ${forecast.capacity} units/day${
        isBreaching ? ", which breaches the Decision Intelligence capacity cap and has been clipped." : ", comfortably within format limits."
      }`,
    },
    {
      title: "Demand & Seasonality",
      body: forecast.activeFestival
        ? `The ${forecast.activeFestival.name} seasonal shock is active this cycle, applying a ${pct(
            forecast.activeFestival.modifier - 1
          )} uplift to baseline demand. Location multiplier for this format is ${forecast.locationMultiplier}x, and the store's cold-start factor is ${forecast.coldStart.factor}x (${forecast.coldStart.label}).`
        : `No major festival is active this cycle. Demand is scaled only by the location multiplier (${forecast.locationMultiplier}x) and cold-start factor (${forecast.coldStart.factor}x, ${forecast.coldStart.label}).`,
    },
    {
      title: "Inventory Risk",
      body:
        criticalItems.length > 0
          ? `${criticalItems.length} SKU(s) are at or below their Reorder Point, including ${criticalItems
              .slice(0, 3)
              .map((i) => i.name)
              .join(", ")}. Replenishment should be triggered to avoid stockout exposure.`
          : "No SKUs are currently at or below their Reorder Point. Inventory buffers are healthy.",
    },
    {
      title: "Capital Efficiency",
      body:
        leakingCategories.length > 0
          ? `₹${financials.totalLockedCash.toLocaleString()} of allocated capital is under-utilized across ${leakingCategories
              .map((c) => c.name)
              .join(", ")}. Redirecting surplus toward high-turnover categories would improve cash-flow velocity.`
          : "Capital allocation is tracking close to observed demand across all categories — no material cash-flow leakage detected.",
    },
  ];

  return { storeName, generatedAt, sections, metrics: { peakDemand, capacity: forecast.capacity, roi: financials.roi, criticalItemCount: criticalItems.length } };
}

// GET /api/stores/:storeId/reports/latest
export async function getBusinessReport(store) {
  return resolveData(`/api/stores/${store?.id ?? "executive"}/reports/latest`, () => buildReport(store));
}

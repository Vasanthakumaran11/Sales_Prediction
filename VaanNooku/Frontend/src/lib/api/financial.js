import { resolveData } from "./client";
import { CAPITAL_ALLOCATION_TEMPLATE, AVERAGE_NET_MARGIN, WEEKLY_DEMAND_RATIO } from "@/lib/mock/financial";

function buildFinancials(store) {
  const investment = store?.investment ?? 150000;

  const categories = CAPITAL_ALLOCATION_TEMPLATE.map((cat) => {
    const allocated = Math.round(investment * cat.pctAllocated);
    const demandVal = Math.round(investment * cat.pctDemand);
    const pct = Math.min(100, Math.round((demandVal / allocated) * 100));
    const isLeak = pct < 85;
    const efficiency = pct >= 90 ? "Optimal" : pct >= 60 ? "Over-allocated" : "Critical Surplus";
    return { name: cat.name, allocated, demandVal, pct, isLeak, efficiency, colorVar: cat.colorVar };
  });

  const totalLockedCash = categories
    .filter((c) => c.isLeak)
    .reduce((sum, c) => sum + Math.max(0, c.allocated - c.demandVal), 0);

  const weeklySales = Math.round(investment * WEEKLY_DEMAND_RATIO);
  const weeklyProfit = Math.round(weeklySales * AVERAGE_NET_MARGIN);
  const weeklyCost = weeklySales - weeklyProfit;
  const roi = weeklyCost > 0 ? Number(((weeklyProfit / weeklyCost) * 100).toFixed(1)) : 0;

  return {
    categories,
    totalLockedCash,
    weeklySales,
    weeklyProfit,
    weeklyCost,
    roi,
    averageMargin: AVERAGE_NET_MARGIN,
  };
}

// GET /api/stores/:storeId/financials
export async function getCapitalAllocation(store) {
  return resolveData(`/api/stores/${store?.id ?? "executive"}/financials`, () => buildFinancials(store));
}

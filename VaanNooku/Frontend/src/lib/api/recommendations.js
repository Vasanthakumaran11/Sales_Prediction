import { resolveData } from "./client";
import { BASELINE_RECOMMENDATIONS, FESTIVAL_RECOMMENDATIONS } from "@/lib/mock/recommendations";
import { getFestivalForMonth } from "@/lib/mock/catalog";
import { flattenCatalog } from "@/lib/mock/catalog";

function buildRecommendations(store) {
  const festival = getFestivalForMonth(store?.openingMonth ?? "");
  const seasonal = festival ? FESTIVAL_RECOMMENDATIONS[festival.name] ?? [] : [];
  const investment = store?.investment ?? 150000;

  const investmentBlueprint = [
    { category: "Staples & Grains", pct: 0.3 },
    { category: "Beverages", pct: 0.2 },
    { category: "Snacks & Biscuits", pct: 0.2 },
    { category: "Perishables", pct: 0.18 },
    { category: "Personal Care", pct: 0.12 },
  ].map((row) => ({ ...row, amount: Math.round(investment * row.pct) }));

  return {
    activeFestival: festival,
    products: [...seasonal, ...BASELINE_RECOMMENDATIONS],
    investmentBlueprint,
    catalogSize: flattenCatalog().length,
  };
}

// GET /api/stores/:storeId/recommendations
export async function getSkuRecommendations(store) {
  return resolveData(`/api/stores/${store?.id ?? "executive"}/recommendations`, () => buildRecommendations(store));
}

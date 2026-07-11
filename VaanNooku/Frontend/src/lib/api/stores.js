import { resolveData } from "./client";
import { STORE_PROFILES } from "@/lib/mock/stores";

let sessionStores = null;
function getSessionStores() {
  if (!sessionStores) sessionStores = STORE_PROFILES.map((s) => ({ ...s }));
  return sessionStores;
}

// GET /api/stores
export async function listStores() {
  return resolveData("/api/stores", () => getSessionStores());
}

// GET /api/stores/:storeId
export async function getStore(storeId) {
  return resolveData(`/api/stores/${storeId}`, () =>
    getSessionStores().find((s) => s.id === storeId) ?? null
  );
}

// POST /api/stores/register
export async function registerStore(payload) {
  return resolveData(
    "/api/stores/register",
    () => {
      const store = {
        id: `store-${Date.now()}`,
        name: payload.storeName,
        type: payload.storeType,
        location: payload.locationType,
        investment: Number(payload.investment) || 0,
        openingMonth: payload.openingMonth,
        monthsActive: 0,
        metrics: {
          forecastR2: 0.9,
          wasteMargin: 0.05,
          stockouts: 0,
          leakingMargin: 0.05,
          deficitCount: 0,
          revenue: 0,
          inventoryValue: 0,
        },
      };
      getSessionStores().unshift(store);
      return store;
    },
    { method: "POST", body: JSON.stringify(payload) }
  );
}

// POST /api/auth/login
export async function loginStore(credentials) {
  return resolveData(
    "/api/auth/login",
    () => null, // fallback to null to trigger mock matching in gateway if backend is offline
    { method: "POST", body: JSON.stringify(credentials) }
  );
}


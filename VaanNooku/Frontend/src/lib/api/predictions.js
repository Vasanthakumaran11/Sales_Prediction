import { request } from "./client";

// GET /api/stores/:storeId/predictions/stock-summary
export async function getStockSummary(storeId) {
  return request(`/api/stores/${storeId}/predictions/stock-summary`);
}

// POST /api/stores/:storeId/predictions/forecast
export async function getForecast(storeId) {
  return request(`/api/stores/${storeId}/predictions/forecast`, { method: "POST" });
}

// GET /api/predictions/next-month/:storeId/:itemId
export async function getNextMonthPrediction(storeId, itemId) {
  return request(`/api/predictions/next-month/${storeId}/${itemId}`);
}

import { request } from "./client";

// POST /api/stores/:storeId/daily-log
export async function submitDailyLog(storeId, payload) {
  return request(`/api/stores/${storeId}/daily-log`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// GET /api/stores/:storeId/daily-logs
export async function listDailyLogs(storeId) {
  return request(`/api/stores/${storeId}/daily-logs`);
}

import { resolveData } from "./client";
import { SEED_TRANSACTIONS } from "@/lib/mock/transactions";

let sessionLog = null;
function getSessionLog() {
  if (!sessionLog) sessionLog = SEED_TRANSACTIONS.map((t) => ({ ...t }));
  return sessionLog;
}

// GET /api/stores/:storeId/transactions
export async function listTransactions(storeId) {
  return resolveData(`/api/stores/${storeId ?? "executive"}/transactions`, () => getSessionLog());
}

// POST /api/stores/:storeId/transactions
export async function createTransaction(storeId, payload) {
  return resolveData(
    `/api/stores/${storeId ?? "executive"}/transactions`,
    () => {
      const entry = {
        id: Math.floor(Math.random() * 100000),
        timestamp: new Date().toTimeString().split(" ")[0],
        syncState: "syncing",
        ...payload,
      };
      getSessionLog().unshift(entry);
      return entry;
    },
    { method: "POST", body: JSON.stringify(payload) }
  );
}

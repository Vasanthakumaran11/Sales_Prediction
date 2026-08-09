import { request } from "./client";

// GET /api/suppliers/suggestions
export async function getSupplierSuggestions() {
  return request("/api/suppliers/suggestions");
}

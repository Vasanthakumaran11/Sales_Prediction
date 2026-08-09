import { request } from "./client";

// GET /api/stores/:storeId/products
export async function getProducts(storeId, { category, search } = {}) {
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (search) params.set("search", search);
  const qs = params.toString();
  return request(`/api/stores/${storeId}/products${qs ? `?${qs}` : ""}`);
}

// POST /api/stores/:storeId/products
export async function addProduct(storeId, payload) {
  return request(`/api/stores/${storeId}/products`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

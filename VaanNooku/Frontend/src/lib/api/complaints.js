import { request } from "./client";

// POST /api/admin/complaints
export async function fileComplaint(storeId, subject, description) {
  return request("/api/admin/complaints", {
    method: "POST",
    body: JSON.stringify({ storeId, subject, description }),
  });
}

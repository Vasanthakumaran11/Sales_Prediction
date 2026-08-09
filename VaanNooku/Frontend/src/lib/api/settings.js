import { request } from "./client";

// PUT /api/users/password
export async function updatePassword(storeId, currentPassword, newPassword) {
  return request("/api/users/password", {
    method: "PUT",
    body: JSON.stringify({ storeId, currentPassword, newPassword }),
  });
}

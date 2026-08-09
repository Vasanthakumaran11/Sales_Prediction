import { request, getAdminToken } from "./client";

// POST /api/admin/auth/login
export async function loginAdmin(email, password) {
  return request("/api/admin/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

// GET /api/admin/models/status
export async function getModelStatus() {
  return request("/api/admin/models/status");
}

// POST /api/admin/models/reload
export async function reloadModels() {
  return request("/api/admin/models/reload", { method: "POST" });
}

// GET /api/admin/datasets
export async function listDatasets() {
  return request("/api/admin/datasets");
}

// POST /api/admin/datasets/upload (multipart — bypasses the JSON request() helper)
export async function uploadDataset(file) {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
  const token = getAdminToken();
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE_URL}/api/admin/datasets/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// POST /api/admin/retraining/trigger
export async function triggerRetraining(datasetFilename) {
  return request("/api/admin/retraining/trigger", {
    method: "POST",
    body: JSON.stringify({ datasetFilename: datasetFilename || null }),
  });
}

// GET /api/admin/retraining/status/:jobId
export async function getRetrainingStatus(jobId) {
  return request(`/api/admin/retraining/status/${jobId}`);
}

// GET /api/admin/complaints
export async function listComplaints() {
  return request("/api/admin/complaints");
}

// PUT /api/admin/complaints/:ticketId
export async function updateComplaintStatus(ticketId, status) {
  return request(`/api/admin/complaints/${ticketId}`, {
    method: "PUT",
    body: JSON.stringify({ status }),
  });
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const TOKEN_STORAGE_KEY = "vaannooku_admin_token";

export function isLiveBackendConfigured() {
  return Boolean(API_BASE_URL);
}

export function setAdminToken(token) {
  if (typeof window === "undefined") return;
  if (token) sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  else sessionStorage.removeItem(TOKEN_STORAGE_KEY);
}

export function getAdminToken() {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(TOKEN_STORAGE_KEY);
}

export async function request(path, options = {}) {
  const token = getAdminToken();
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // response wasn't JSON — keep the status text
    }
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

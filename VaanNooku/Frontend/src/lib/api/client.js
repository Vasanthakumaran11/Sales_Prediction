const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const MOCK_LATENCY_MS = 420;
const TOKEN_STORAGE_KEY = "vaannooku_auth_token";

export function isLiveBackendConfigured() {
  return Boolean(API_BASE_URL);
}

function delay(ms = MOCK_LATENCY_MS) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Persist (or clear, if token is falsy) the JWT issued at login/registration. */
export function setAuthToken(token) {
  if (typeof window === "undefined") return;
  if (token) sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  else sessionStorage.removeItem(TOKEN_STORAGE_KEY);
}

function getAuthToken() {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(TOKEN_STORAGE_KEY);
}

/**
 * Direct call to the live backend — throws on failure instead of falling
 * back to mock data. Use this (not resolveData) for real, DB-backed store
 * data where a silent fallback to fabricated numbers would be misleading.
 */
export async function request(path, options = {}) {
  const token = getAuthToken();
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) {
    throw new Error(`API ${options.method || "GET"} ${path} failed: ${res.status} ${res.statusText}`);
  }
  if (res.status === 204) return null;
  return res.json();
}

/**
 * Every domain module in lib/api/ calls this instead of fetch() directly.
 * When NEXT_PUBLIC_API_BASE_URL is unset (default for this demo build), it
 * resolves the supplied mock value after a small simulated network delay.
 * Once a real backend is deployed, set the env var and every call site
 * switches to live data with no component changes required.
 */
export async function resolveData(path, mockValue, options) {
  if (isLiveBackendConfigured()) {
    try {
      return await request(path, options);
    } catch (err) {
      console.error(`[api] live request to ${path} failed, falling back to mock data:`, err);
    }
  }
  await delay();
  return typeof mockValue === "function" ? mockValue() : mockValue;
}

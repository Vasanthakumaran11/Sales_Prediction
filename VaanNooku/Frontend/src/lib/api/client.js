const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const MOCK_LATENCY_MS = 420;

export function isLiveBackendConfigured() {
  return Boolean(API_BASE_URL);
}

function delay(ms = MOCK_LATENCY_MS) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
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

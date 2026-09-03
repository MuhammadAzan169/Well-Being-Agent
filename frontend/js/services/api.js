// API client for the WellBeing Agent backend.
// All requests are built from window.APP_CONFIG.API_BASE_URL (see js/config.js),
// so the same static bundle works against localhost or a deployed Render API.

const API_BASE = (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) || "";

function apiUrl(path) {
  // Ensure exactly one slash between base and path.
  const base = API_BASE.replace(/\/+$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}

async function request(path, { method = "GET", body, signal } = {}) {
  const opts = { method, signal, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let resp;
  try {
    resp = await fetch(apiUrl(path), opts);
  } catch (cause) {
    // fetch() rejects with a TypeError and no status for DNS failures, dropped
    // connections, and CORS rejections. A sleeping Render free instance looks
    // exactly like this, so flag it as retryable rather than losing the reason.
    const err = new Error("Could not reach the server. Please check your connection.");
    err.status = 0;
    err.isNetworkError = true;
    err.cause = cause;
    throw err;
  }
  if (!resp.ok) {
    let detail = `Server error ${resp.status}`;
    try {
      const data = await resp.json();
      detail = data.detail || detail;
    } catch (_) {
      /* non-JSON error body */
    }
    const err = new Error(detail);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

export const api = {
  baseUrl: API_BASE,

  askQuery(message, language) {
    return request("/api/ask-query", { method: "POST", body: { message, language } });
  },

  predefinedQuestions(language) {
    return request(`/api/predefined-questions?language=${encodeURIComponent(language)}`);
  },

  health() {
    return request("/health");
  },
};

// build.mjs — generates js/config.js from the API_BASE_URL environment variable.
//
// vercel.json cannot read environment variables, so instead of proxying /api/*
// through a rewrite with a hardcoded backend host, the browser calls the backend
// directly at the URL baked in here. The backend's CORS config (ALLOWED_ORIGINS)
// permits the Vercel origin, so cross-origin requests work without a proxy —
// and changing backends is a dashboard change, not a code change.
//
// Resolution order:
//   1. process.env.API_BASE_URL   (Vercel dashboard var, or shell export)
//   2. frontend/.env              (local development)
//   3. http://localhost:8000      (the local `python app.py` backend)
import { writeFileSync, readFileSync, existsSync } from "node:fs";

// Minimal .env loader (no dependency). process.env always wins.
function loadEnv() {
  const url = new URL("./.env", import.meta.url);
  if (!existsSync(url)) return;
  for (const line of readFileSync(url, "utf8").split(/\r?\n/)) {
    if (/^\s*#/.test(line)) continue;
    const m = line.match(/^\s*([\w.-]+)\s*=\s*(.*?)\s*$/);
    if (!m) continue;
    const val = m[2].replace(/^['"]|['"]$/g, "");
    if (process.env[m[1]] === undefined) process.env[m[1]] = val;
  }
}
// Capture whether a *real* environment variable was provided before the .env
// fallback is layered in, so the Vercel warning below can't be masked by a
// stray local .env file.
const hadRealEnvVar = process.env.API_BASE_URL !== undefined;
loadEnv();

const raw = process.env.API_BASE_URL ?? "http://localhost:8000";
const apiBase = raw.trim().replace(/\/+$/, "");

if (process.env.VERCEL && !hadRealEnvVar) {
  console.warn(
    "[build] ⚠️  API_BASE_URL is not set in the Vercel project settings.\n" +
    "[build]     The deployed site will try to reach http://localhost:8000 and fail.\n" +
    "[build]     Set it to your Render URL, e.g. https://<service>.onrender.com"
  );
}

const contents = `// AUTO-GENERATED at build time by build.mjs — do not edit by hand.
window.APP_CONFIG = {
  API_BASE_URL: ${JSON.stringify(apiBase)},
};
`;

writeFileSync(new URL("./js/config.js", import.meta.url), contents);
console.log(`[build] Wrote js/config.js with API_BASE_URL="${apiBase || "(same origin)"}"`);

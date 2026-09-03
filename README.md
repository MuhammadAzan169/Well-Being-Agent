# WellBeing Agent

A bilingual (English / Urdu / Roman Urdu) RAG assistant that provides
well-being support for breast cancer patients — grounded in a curated dataset,
with crisis detection, emotional-tone awareness, and source citations.

The repository holds two independently deployable applications plus a
one-command local runner:

```
Well-Being-Agent/
├── app.py              # Local runner — starts backend + frontend together
├── backend/            # FastAPI + RAG API      → deploy to Render (free plan)
│   ├── app/            # Application package
│   ├── data/           # Dataset + pre-built vector index (committed)
│   ├── scripts/        # build_index.py, warm_cache.py
│   ├── Dockerfile      # Optional container deploy
│   ├── render.yaml     # Render blueprint
│   └── requirements.txt
├── frontend/           # Static landing + chat UI → deploy to Vercel
│   ├── assets/  js/    # CSS and ES modules
│   ├── build.mjs       # Generates js/config.js from API_BASE_URL
│   └── vercel.json     # Vercel configuration
└── README.md
```

---

## Table of contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Local installation](#local-installation)
4. [Environment variables](#environment-variables)
5. [Running locally with `app.py`](#running-locally-with-apppy)
6. [Local models vs. API-based models](#local-models-vs-api-based-models)
7. [Deploying the backend to Render](#deploying-the-backend-to-render-free-plan)
8. [Deploying the frontend to Vercel](#deploying-the-frontend-to-vercel)
9. [Render free-plan limitations](#render-free-plan-limitations)
10. [Local vs. production behaviour](#local-vs-production-behaviour)
11. [API reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

---

## Architecture

```
                Browser
                   │
     ┌─────────────┴──────────────┐
     │                            │
 Static site                  JSON API
 (Vercel / local :3000)  →  (Render / local :8000)
                                  │
                    ┌─────────────┼──────────────┐
                    │             │              │
              Vector index   Safety filter   OpenRouter
              (fastembed,    (crisis /       (LLM answer
               local ONNX)    off-topic)      generation)
```

The frontend is fully static. It reads one build-time value —
`window.APP_CONFIG.API_BASE_URL`, generated into `frontend/js/config.js` — and
calls the backend directly over CORS. Nothing else about the deployment is
baked into the code, so pointing the UI at a different backend is a single
environment-variable change.

**Request pipeline:** detect language → safety check → cache lookup →
retrieve context → build prompt → LLM call (with key + model rotation) →
post-process → cache → respond with sources.

---

## Prerequisites

| Requirement | Version | Needed for |
|---|---|---|
| Python | 3.10 – 3.12 | Backend and `app.py` |
| pip | recent | Installing dependencies |
| Node.js | 18+ | Frontend build (**optional** locally — `app.py` does not need it) |
| OpenRouter API key | — | Generating answers ([openrouter.ai/keys](https://openrouter.ai/keys)) |

Free OpenRouter models are sufficient. Without a key the app still runs and
retrieval still works, but every answer falls back to an apology message.

---

## Local installation

```bash
git clone <your-repo-url>
cd Well-Being-Agent

# 1. Virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Backend dependencies
pip install -r backend/requirements.txt

# 3. Optional: local Whisper voice transcription (LOCAL ONLY, ~500 MB)
pip install -r backend/requirements-voice.txt

# 4. Configuration
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
#   → open backend/.env and add your OpenRouter API key
```

The vector index is committed under `backend/data/cancer_index_store`, so
there is no index-building step on a fresh clone. The embedding model (~88 MB
ONNX) downloads automatically the first time you run the app.

---

## Environment variables

Secrets live only in `.env` files (gitignored) or in the hosting platform's
dashboard. **Nothing sensitive is ever committed.**

### `backend/.env`

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | `development` or `production` |
| `PORT` | `8000` | Port to bind |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins, no trailing slash |
| `OPENROUTER_API_KEY1`, `…2`, `…3` | — | API keys; extra numbered keys rotate on rate limits |
| `LLM_API_KEY` | — | Single-key fallback when no numbered keys exist |
| `OPENROUTER_MODEL_1`, `…2`, … | — | Ordered model fallback chain |
| `LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Fallback when no numbered models exist |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint |
| `LLM_MAX_TOKENS` | `1500` | Response length cap |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `LLM_TOTAL_TIMEOUT_SECONDS` | `120` | Cap on the whole key/model retry sweep |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | **Must match the model the index was built with** |
| `INDEX_PATH` | `data/cancer_index_store` | Persisted vector index |
| `DATASET_PATH` | `data/DataSet/breast_cancer_comprehensive.json` | Source dataset |
| `SIMILARITY_TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Indexing parameters |
| `CACHE_TTL_HOURS` | `24` | Cached-response lifetime |
| `CACHE_SIMILARITY_THRESHOLD` | `0.85` | Fuzzy cache-hit threshold |
| `CACHE_MAX_ENTRIES` | `500` | Cache size cap |
| `CONVERSATION_LOG_MAX_ENTRIES` | `1000` | Conversation-log size cap |
| `ENABLE_VOICE` | `false` | Enable server-side Whisper (**local only**) |
| `WHISPER_MODEL_ID` | `small` | `base` (~150 MB) or `small` (~500 MB) |
| `MAX_AUDIO_SIZE_MB` | `10` | Upload limit for voice queries |
| `MAX_QUERY_LENGTH` | `2000` | Max characters per question |
| `RUNTIME_DIR` | `var` | Writable dir for cache, logs, recordings |

Key rotation: the backend discovers `OPENROUTER_API_KEY1`, `OPENROUTER_API_KEY2`,
… in order and tries every key for each model before moving to the next model
in the `OPENROUTER_MODEL_<N>` chain. A rate-limited or invalid key is skipped
automatically; a model that returns 400/404 is dropped without burning the
remaining keys against it. The entire sweep is bounded by
`LLM_TOTAL_TIMEOUT_SECONDS`, so a provider outage returns the fallback
message promptly rather than hanging.

Model ids on OpenRouter are retired regularly. If answers suddenly become
the fallback message, check your chain against the live list:
`curl -H "Authorization: Bearer $KEY" https://openrouter.ai/api/v1/models`.

### `frontend/.env`

| Variable | Local value | Purpose |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Backend base URL, **no trailing slash** |

---

## Running locally with `app.py`

One command starts everything:

```bash
python app.py
```

It will:

1. Verify the layout, dependencies, and `backend/.env`.
2. Build the vector index if it is missing.
3. Generate `frontend/js/config.js` pointing at the local backend.
4. Widen the backend's CORS policy to cover the frontend port actually used.
5. Start the API on `:8000` and the static site on `:3000`.
6. Wait until the index is loaded, then open your browser.

```
Frontend : http://localhost:3000
API      : http://localhost:8000
API docs : http://localhost:8000/docs
```

If a port is busy, the next free one is chosen automatically.

### Options

```bash
python app.py --no-browser         # don't open a browser
python app.py --reload             # auto-reload the backend on edits
python app.py --build-index        # rebuild the vector index first
python app.py --backend-only       # API only
python app.py --frontend-only      # static site only
python app.py --backend-port 8080 --frontend-port 4000
```

### Running the parts separately

```bash
# Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Frontend (needs Node)
cd frontend
npm run dev            # builds config.js, serves on :5500
```

### Rebuilding the index

Required only after editing the dataset or changing `EMBEDDING_MODEL`:

```bash
cd backend
python -m scripts.build_index
```

---

## Local models vs. API-based models

The project deliberately splits its AI workloads so the *same code* runs in
both environments, with the heavy pieces degrading gracefully:

| Capability | Local | Render free plan |
|---|---|---|
| **Embeddings** | Local ONNX model via `fastembed` | **Identical** — same local model |
| **Answer generation** | OpenRouter API | OpenRouter API |
| **Speech-to-text** | Local Whisper (`faster-whisper`) when installed | Browser Web Speech API |

**Embeddings are local in both environments.** `fastembed` runs the embedding
model through ONNX Runtime rather than PyTorch — roughly 50 MB of RAM and an
88 MB download, instead of the ~250 MB+ that `sentence-transformers` needs.
That is what makes a genuine local embedding model viable inside Render's free
512 MB limit, so retrieval quality is the same in production as on your
machine. `llama-index-core` is pinned deliberately: the full `llama-index`
meta-package pulls in PyTorch and would not fit.

**Speech-to-text is the one capability that switches.** Whisper does not fit
the free plan, so `ENABLE_VOICE=false` there and the microphone button uses the
browser's built-in Web Speech API, which needs no backend at all. Voice input
therefore works in production too — it is just transcribed client-side. Set
`ENABLE_VOICE=true` locally (after installing `requirements-voice.txt`) for
higher-accuracy server-side transcription, especially for Urdu. When voice is
disabled the `/api/voice-query` endpoint returns a friendly "please type your
question instead" message rather than an error.

**Answer generation is always API-based**, so no LLM weights are ever loaded
and switching models is an environment-variable change.

---

## Deploying the backend to Render (free plan)

1. Push this repository to GitHub.
2. In Render: **New → Web Service** → connect the repo.
3. Configure:

   | Setting | Value |
   |---|---|
   | **Root Directory** | `backend` |
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt && python -m scripts.warm_cache` |
   | **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
   | **Health Check Path** | `/health` |
   | **Instance Type** | Free |

4. Add environment variables (**Environment** tab):

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.12` |
   | `ENVIRONMENT` | `production` |
   | `ENABLE_VOICE` | `false` |
   | `ALLOWED_ORIGINS` | `https://<your-project>.vercel.app` |
   | `OPENROUTER_API_KEY1` | *your key* (add `…2`, `…3` for rotation) |
   | `OPENROUTER_MODEL_1` | *e.g.* `openai/gpt-oss-120b` (optional chain) |
   | `LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` |
   | `LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
   | `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
   | `FASTEMBED_CACHE_PATH` | `/opt/render/project/src/backend/var/fastembed-cache` |
   | `OMP_NUM_THREADS` | `1` |
   | `MKL_NUM_THREADS` | `1` |
   | `TOKENIZERS_PARALLELISM` | `false` |

5. Deploy, then verify: `https://<service>.onrender.com/health` should return
   `{"status":"healthy","rag_loaded":true,"whisper_available":false}`.

`backend/render.yaml` encodes all of the above. To deploy it as a Blueprint
instead of filling in the dashboard by hand, **copy it to the repository root
first** — Render only reads `render.yaml` from the repo root — then use
**New → Blueprint** and set only the `sync: false` secrets. Its `rootDir:
backend` already points the service at the backend directory.

The `warm_cache` build step matters: it downloads the embedding model *during
the build* so cold starts never pay for it or time out the health check.
`OMP_NUM_THREADS=1` prevents ONNX from spawning thread pools that would push a
single-core instance past its memory limit.

### Docker alternative

`backend/Dockerfile` is a working alternative (Render: Runtime → Docker, root
directory `backend`). The native Python runtime is lighter on the free plan, so
prefer it unless you specifically want a container.

---

## Deploying the frontend to Vercel

1. In Vercel: **Add New → Project** → import the repository.
2. Configure:

   | Setting | Value |
   |---|---|
   | **Root Directory** | `frontend` |
   | **Framework Preset** | Other |
   | **Build Command** | `node build.mjs` *(from `vercel.json`)* |
   | **Output Directory** | `.` *(from `vercel.json`)* |

3. Add one environment variable, for **all** environments:

   | Key | Value |
   |---|---|
   | `API_BASE_URL` | `https://<your-service>.onrender.com` — **no trailing slash** |

4. Deploy.

`build.mjs` bakes that URL into `js/config.js` at build time. If the variable
is missing, the build log prints a loud warning and the site falls back to
`http://localhost:8000` (which will fail in production) — so check the build
log if the deployed chat cannot reach the API.

**After deploying, set `ALLOWED_ORIGINS` on Render to your Vercel URL** and
redeploy the backend, otherwise the browser blocks every request via CORS.

> `js/config.js` is generated, gitignored, and must not be committed —
> committing it would freeze one environment's URL into every deployment.

---

## Render free-plan limitations

| Limit | Effect | Mitigation |
|---|---|---|
| **Sleeps after ~15 min idle** | First request afterwards takes ~50 s | Health-check ping, or accept the delay; the UI shows a loading state |
| **512 MB RAM** | No PyTorch, no server-side Whisper | ONNX embeddings (~350 MB peak measured); voice runs in the browser |
| **0.1 CPU** | Answers take a few seconds | Responses are cached with fuzzy matching |
| **Ephemeral disk** | `var/` is wiped on restart | Only cache, logs, and recordings live there — all disposable |
| **Build minutes capped** | — | Index is committed, so builds only install deps |
| **No persistent storage** | Conversation log is not durable | Capped at 1000 entries; use a database if you need durability |

Measured peak memory in a clean production simulation: **~353 MB** after
loading the index and serving queries — comfortably inside the 512 MB limit.

---

## Local vs. production behaviour

| | Local (`python app.py`) | Production (Vercel + Render) |
|---|---|---|
| Frontend | Python static server, `:3000` | Vercel CDN |
| Backend | uvicorn, `:8000` | Render web service |
| API URL | `http://localhost:8000` (auto-generated) | `API_BASE_URL` env var |
| Embeddings | Local ONNX | Local ONNX (same model) |
| LLM | OpenRouter API | OpenRouter API |
| Voice | Local Whisper *(if installed)* + Web Speech API | Web Speech API |
| Secrets | `backend/.env` | Render dashboard |
| CORS | Auto-widened to the local port | `ALLOWED_ORIGINS` |
| Cold start | None | ~50 s after idle |
| Data persistence | Survives restarts | Ephemeral |

Local never depends on the deployed backend, and production never depends on
anything on your machine.

---

## API reference

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health` | `{status, rag_loaded, whisper_available}` |
| `GET` | `/info` | Capabilities and version |
| `POST` | `/api/ask-query` | `{message, language?}` → `{answer, sources[], language}` |
| `POST` | `/api/voice-query` | `{audio_data}` (base64) → answer + `transcribed_text` |
| `GET` | `/api/predefined-questions?language=` | Starter questions |

`/health` responds immediately while the index loads in a background thread,
so Render detects the open port right away. API endpoints return **503** until
`rag_loaded` is `true`.

---

## Troubleshooting

**Chat says "unable to respond right now"**
No usable API key or every model failed. Confirm `OPENROUTER_API_KEY1` is set,
has credit, and that the models in your `OPENROUTER_MODEL_<N>` chain still
exist on OpenRouter (retired model ids return 400/404 and are skipped).

**CORS error in the browser console**
`ALLOWED_ORIGINS` on Render must contain your exact Vercel origin — scheme
included, no trailing slash, comma-separated for several. Redeploy the backend
after changing it. `*` allows everything (this API sends no credentials).

**Deployed chat calls `localhost:8000`**
`API_BASE_URL` was not set in Vercel when the build ran. Set it and
**redeploy** — it is baked in at build time, so changing the variable alone is
not enough.

**Render deploy fails: `ModuleNotFoundError: No module named 'app'`**
Root Directory is not set to `backend`.

**Render build succeeds, service returns 503 forever**
The index failed to load. Check that `backend/data/cancer_index_store/` was
committed (`git ls-files backend/data | head`) and that `EMBEDDING_MODEL`
matches `index_metadata.json`.

**Answers are irrelevant / retrieval scores are low**
`EMBEDDING_MODEL` does not match the model the index was built with. The app
detects this, logs a warning, and uses the index's own model — but the real fix
is to rebuild: `cd backend && python -m scripts.build_index`.

**First request after idle times out**
Expected on the free plan (~50 s cold start). Retry, or keep the service warm
with a periodic `/health` ping.

**Out of memory on Render**
Ensure `ENABLE_VOICE=false`, `OMP_NUM_THREADS=1`, and that nothing has added
`torch`, `transformers`, or `sentence-transformers` to `requirements.txt`.

**`python app.py` reports missing packages**
`pip install -r backend/requirements.txt` inside your activated virtualenv.

**Voice button does nothing**
The Web Speech API needs Chrome or Edge and an HTTPS (or `localhost`) origin.
For server-side transcription, install `backend/requirements-voice.txt` and set
`ENABLE_VOICE=true`.

**Port already in use**
`app.py` picks the next free port automatically, or pass `--backend-port` /
`--frontend-port`.

---

## Safety

This is an informational support tool, **not** a medical service. Responses
carry a medical disclaimer, off-topic questions are declined, and crisis
language triggers a safety response directing users to emergency help. It does
not diagnose, prescribe, or replace professional medical advice.

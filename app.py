#!/usr/bin/env python3
"""app.py - one-command local runner for the WellBeing Agent.

Starts the complete application on your machine:

  - Backend  (FastAPI + uvicorn)   -> http://localhost:8000
  - Frontend (static file server)  -> http://localhost:3000

It mirrors the production topology (Vercel frontend + Render backend) so what
you test locally is what you ship, but with the *local* feature set: local ONNX
embeddings, and local Whisper voice transcription when it is installed.

Usage:
    python app.py                  # run everything
    python app.py --no-browser     # do not open a browser window
    python app.py --build-index    # rebuild the vector index, then run
    python app.py --backend-only   # API only
    python app.py --frontend-only  # static site only
"""

from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import os
import socket
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000

# ES modules must be served with the correct MIME type or the browser refuses to
# execute them. The Windows registry value for .js is frequently wrong.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


# -- Console helpers --------------------------------------------------------
def info(msg: str) -> None:
    print(f"  {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  [!] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  [x] {msg}", flush=True)


def banner(title: str) -> None:
    line = "-" * 62
    print(f"\n{line}\n  {title}\n{line}", flush=True)


# -- Preflight --------------------------------------------------------------
def check_layout() -> bool:
    ok = True
    for path, label in ((BACKEND_DIR, "backend/"), (FRONTEND_DIR, "frontend/")):
        if not path.is_dir():
            fail(f"Missing {label} directory (expected at {path})")
            ok = False
    return ok


def check_dependencies() -> bool:
    """Verify the backend's Python dependencies are importable."""
    import importlib.util

    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic_settings": "pydantic-settings",
        "dotenv": "python-dotenv",
        "httpx": "httpx",
        "langdetect": "langdetect",
        "llama_index.core": "llama-index-core",
        "llama_index.embeddings.fastembed": "llama-index-embeddings-fastembed",
    }
    missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
    if missing:
        fail("Missing Python packages: " + ", ".join(missing))
        info("Install them with:  pip install -r backend/requirements.txt")
        return False
    return True


def _env_lines(env_file: Path):
    """Yield (key, value) pairs from a .env file, skipping comments and blanks."""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        yield key.strip(), value.strip()


def check_env() -> bool:
    """Warn (but do not block) about a missing or key-less backend/.env."""
    env_file = BACKEND_DIR / ".env"
    if not env_file.exists():
        warn("backend/.env not found.")
        info("Create it by copying backend/.env.example to backend/.env")
        info("Then add your OpenRouter API key. Without a key the app still runs,")
        info("but every answer falls back to an apology message.")
        return False

    has_key = any(
        value and (key.startswith("OPENROUTER_API_KEY") or key == "LLM_API_KEY")
        for key, value in _env_lines(env_file)
    )
    if not has_key:
        warn("No LLM API key in backend/.env (OPENROUTER_API_KEY1 or LLM_API_KEY).")
        info("Retrieval will work, but answers will be the fallback message.")
    return has_key


def check_index() -> bool:
    index_dir = BACKEND_DIR / "data" / "cancer_index_store"
    required = ("docstore.json", "default__vector_store.json", "index_store.json")
    return all((index_dir / name).exists() for name in required)


def build_index() -> bool:
    banner("Building the vector index (downloads the embedding model once)")
    sys.path.insert(0, str(BACKEND_DIR))
    cwd = os.getcwd()
    try:
        os.chdir(BACKEND_DIR)
        from app.services.rag.index import build_index as _build  # type: ignore

        return bool(_build())
    except Exception as exc:
        fail(f"Index build failed: {exc}")
        return False
    finally:
        os.chdir(cwd)


def voice_status() -> str:
    """Describe local voice support. faster-whisper is optional and local-only."""
    import importlib.util

    installed = importlib.util.find_spec("faster_whisper") is not None
    enabled = any(
        key == "ENABLE_VOICE" and value.lower() in ("1", "true", "yes")
        for key, value in _env_lines(BACKEND_DIR / ".env")
    )
    if installed and enabled:
        return "local Whisper (faster-whisper) + browser Web Speech API"
    if installed and not enabled:
        return "browser Web Speech API (set ENABLE_VOICE=true in backend/.env for local Whisper)"
    return "browser Web Speech API (pip install -r backend/requirements-voice.txt for local Whisper)"


# -- Port handling ----------------------------------------------------------
def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_port(preferred: int, label: str) -> int:
    if port_is_free(preferred):
        return preferred
    for candidate in range(preferred + 1, preferred + 20):
        if port_is_free(candidate):
            warn(f"Port {preferred} is busy - using {candidate} for the {label}.")
            return candidate
    raise RuntimeError(f"No free port available near {preferred} for the {label}.")


# -- Frontend config generation ---------------------------------------------
def write_frontend_config(backend_url: str) -> None:
    """Generate frontend/js/config.js so the browser talks to the local backend.

    This is the same file build.mjs produces on Vercel. Generating it here means
    local development needs neither Node nor the deployed Render backend.
    """
    config_path = FRONTEND_DIR / "js" / "config.js"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "// AUTO-GENERATED by app.py for local development - do not edit by hand.\n"
        "window.APP_CONFIG = {\n"
        f"  API_BASE_URL: {json.dumps(backend_url)},\n"
        "};\n",
        encoding="utf-8",
    )
    info(f"Frontend will call the API at {backend_url}")


def allow_frontend_origin(port: int) -> None:
    """Ensure the backend's CORS policy covers the port we actually bound to.

    backend/.env lists a fixed set of origins, but pick_port() may fall back to
    a different port when the default is busy - which would make every browser
    request fail preflight. Process environment variables take priority over
    .env in pydantic-settings, so setting this before the backend is imported
    reliably widens the policy for the local run only.
    """
    existing = [
        value
        for key, value in _env_lines(BACKEND_DIR / ".env")
        if key == "ALLOWED_ORIGINS"
    ]
    origins: list[str] = []
    if existing and existing[0].strip() == "*":
        return  # already permissive
    for value in existing:
        origins.extend(o.strip() for o in value.split(",") if o.strip())
    for host in ("localhost", "127.0.0.1"):
        origin = f"http://{host}:{port}"
        if origin not in origins:
            origins.append(origin)
    os.environ["ALLOWED_ORIGINS"] = ",".join(origins)


# -- Static frontend server -------------------------------------------------
class FrontendHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler reproducing the Vercel cleanUrls + rewrite behaviour."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def translate_path(self, path: str) -> str:
        clean = unquote(urlparse(path).path)

        # Mirror vercel.json: "/" -> index.html, "/chat" -> chat.html, and any
        # extension-less URL resolves to its .html file (cleanUrls: true).
        if clean in ("", "/"):
            clean = "/index.html"
        elif not Path(clean).suffix:
            candidate = FRONTEND_DIR / clean.strip("/")
            if candidate.with_suffix(".html").is_file():
                clean = clean.rstrip("/") + ".html"
            elif candidate.is_dir() and (candidate / "index.html").is_file():
                clean = clean.rstrip("/") + "/index.html"

        return super().translate_path(clean)

    def end_headers(self) -> None:
        # Never cache during development so edits show up on reload.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        # Suppress per-request noise; real errors still surface via log_error.
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_frontend(port: int) -> ThreadedHTTPServer:
    server = ThreadedHTTPServer(("0.0.0.0", port), FrontendHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# -- Backend server ---------------------------------------------------------
def run_backend(port: int, reload: bool = False) -> None:
    """Run uvicorn in this process (blocking). Requires cwd == backend/."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


def wait_for_backend(url: str, timeout: float = 180.0) -> bool:
    """Poll /health until the RAG system reports it has finished loading."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    reported_up = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if not reported_up:
                info("Backend is up - loading the vector index...")
                reported_up = True
            if data.get("rag_loaded"):
                return True
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            pass
        time.sleep(1.0)
    return False


# -- Main -------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete WellBeing Agent locally.")
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--reload", action="store_true", help="auto-reload the backend on edits")
    parser.add_argument("--build-index", action="store_true", help="rebuild the vector index first")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()

    banner("WellBeing Agent - local development")

    if not check_layout():
        return 1

    run_backend_part = not args.frontend_only
    run_frontend_part = not args.backend_only

    if run_backend_part:
        if not check_dependencies():
            return 1
        check_env()

        if args.build_index or not check_index():
            if not check_index():
                warn("No vector index found - building it now.")
            if not build_index():
                fail("Could not build the vector index. Aborting.")
                return 1

        info(f"Voice input: {voice_status()}")

    backend_port = pick_port(args.backend_port, "backend") if run_backend_part else args.backend_port
    backend_url = f"http://localhost:{backend_port}"

    frontend_url = None
    if run_frontend_part:
        frontend_port = pick_port(args.frontend_port, "frontend")
        # Point the browser at the local backend when we are running it; otherwise
        # honour whatever API_BASE_URL the environment specifies.
        write_frontend_config(
            backend_url
            if run_backend_part
            else os.environ.get("API_BASE_URL", backend_url).rstrip("/")
        )
        if run_backend_part:
            allow_frontend_origin(frontend_port)
        serve_frontend(frontend_port)
        frontend_url = f"http://localhost:{frontend_port}"

    if run_backend_part:
        # The backend imports as `app.main:app` from inside backend/ and reads
        # backend/.env - both require the working directory to be backend/.
        os.chdir(BACKEND_DIR)
        sys.path.insert(0, str(BACKEND_DIR))

        def announce() -> None:
            if wait_for_backend(backend_url):
                banner("Ready")
                info(f"Frontend : {frontend_url}" if frontend_url else "Frontend : (not started)")
                info(f"API      : {backend_url}")
                info(f"API docs : {backend_url}/docs")
                print("\n  Press Ctrl+C to stop.\n", flush=True)
                if frontend_url and not args.no_browser:
                    webbrowser.open(frontend_url)
            else:
                warn("Backend did not finish loading within the timeout.")
                info(f"Check the logs above; the API may still come up at {backend_url}")

        threading.Thread(target=announce, daemon=True).start()

        try:
            run_backend(backend_port, reload=args.reload)
        except KeyboardInterrupt:
            pass
    else:
        banner("Ready (frontend only)")
        info(f"Frontend : {frontend_url}")
        print("\n  Press Ctrl+C to stop.\n", flush=True)
        if frontend_url and not args.no_browser:
            webbrowser.open(frontend_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    print("\n  Stopped.\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

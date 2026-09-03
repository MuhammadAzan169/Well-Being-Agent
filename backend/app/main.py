"""FastAPI application entry point for the WellBeing Agent API.

Run locally:  uvicorn app.main:app --reload
Run for prod: uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, system
from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger("WellBeingAgent.Server")


def _init_rag(app: FastAPI) -> None:
    """Load the RAG system in a background thread so the HTTP server starts immediately."""
    try:
        from app.services.rag.system import create_rag_system

        app.state.rag = create_rag_system()
        logger.info("✅ RAG system ready")
    except Exception as exc:
        logger.error(f"❌ RAG init failed: {exc}")
        import traceback

        traceback.print_exc()
        app.state.rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the HTTP server immediately; load heavy resources in the background."""
    logger.info("🚀 Starting WellBeing Agent API…")

    # rag=None means the /api endpoints return 503 until the model finishes loading.
    # The /health endpoint is available immediately so Render can detect the port.
    app.state.rag = None

    thread = threading.Thread(target=_init_rag, args=(app,), daemon=True)
    thread.start()

    try:
        from app.services.audio.processor import is_whisper_available

        app.state.whisper_available = is_whisper_available()
        if app.state.whisper_available:
            logger.info("✅ Whisper available")
        elif settings.ENABLE_VOICE:
            logger.warning("⚠️  Voice enabled but Whisper unavailable — voice will error")
        else:
            logger.info("ℹ️  Voice disabled (ENABLE_VOICE=false)")
    except Exception as exc:
        logger.warning(f"⚠️  Whisper init error: {exc}")
        app.state.whisper_available = False

    yield
    logger.info("🛑 Server shutting down")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(chat.router, prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=False)

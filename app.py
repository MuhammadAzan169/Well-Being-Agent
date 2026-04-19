"""app.py — FastAPI Server for WellBeing Agent

Main entry point. Serves the web interface, text queries, and voice queries.
Integrates the RAG system and Whisper speech-to-text pipeline.
"""

import os
import re
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("WellBeingAgent.Server")

MAX_QUERY_LENGTH = 2000
MAX_AUDIO_SIZE_MB = 10


# ── Lifespan (import RAG system + Whisper on startup) ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting WellBeing Agent server…")

    # Load RAG system
    try:
        from backend.agent import rag_system
        app.state.rag = rag_system
        logger.info("✅ RAG system loaded")
    except Exception as exc:
        logger.error(f"❌ RAG init failed: {exc}")
        import traceback
        traceback.print_exc()
        app.state.rag = None

    # Pre-load Whisper (in background so startup isn't blocked)
    try:
        from backend.audio_processor import is_whisper_available
        app.state.whisper_available = is_whisper_available()
        if app.state.whisper_available:
            logger.info("✅ Whisper Large v3 available")
        else:
            logger.warning("⚠️  Whisper not available — voice queries will return errors")
    except Exception as exc:
        logger.warning(f"⚠️  Whisper init error: {exc}")
        app.state.whisper_available = False

    yield
    logger.info("🛑 Server shutting down")


app = FastAPI(title="Well Being Agent", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files — serve frontend assets
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# ── Request / Response Models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    message: str
    language: str = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Message must be under {MAX_QUERY_LENGTH} characters")
        return v


class VoiceRequest(BaseModel):
    audio_data: str  # base64


# ══════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════

# ── Serve project home page at root ──────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("frontend/home.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="home.html not found")


# ── Serve chat app ───────────────────────────────────────────────────────
@app.get("/chat", response_class=HTMLResponse)
async def chat():
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")


# ── Text query ───────────────────────────────────────────────────────────
@app.post("/ask-query")
async def ask_query(req: QueryRequest):
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        language = req.language or rag.detect_language(message)
        result = await asyncio.to_thread(
            rag.get_enhanced_answer_with_sources, message, language, "text"
        )
        return JSONResponse(content={
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "language": result.get("language", language),
        })
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


# ── Voice query (Whisper Large v3 transcription) ─────────────────────────
@app.post("/voice-query")
async def voice_query(req: VoiceRequest):
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")

    import base64
    from backend.audio_processor import transcribe_audio
    from backend.language_utils import map_whisper_lang_to_system

    try:
        audio_bytes = base64.b64decode(req.audio_data)

        # Transcribe using Whisper Large v3
        transcription = await asyncio.to_thread(transcribe_audio, audio_bytes)

        if not transcription["success"]:
            return JSONResponse(content={
                "answer": transcription["error"],
                "sources": [],
                "language": "english",
                "transcribed_text": "",
            })

        transcribed_text = transcription["text"]
        detected_lang = transcription["language"]

        # Map Whisper language to system language ('urdu' or 'english')
        language = map_whisper_lang_to_system(detected_lang)

        # For voice input, only override to Urdu if Urdu SCRIPT characters
        # are present in the transcription. Do NOT use Roman Urdu detection
        # here because Whisper transcribes English speech as English text
        # which can false-positive on Roman Urdu word lists.
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', transcribed_text):
            language = "urdu"

        # Get answer from RAG system
        result = await asyncio.to_thread(
            rag.get_enhanced_answer_with_sources, transcribed_text, language, "voice"
        )

        return JSONResponse(content={
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "language": result.get("language", language),
            "transcribed_text": transcribed_text,
        })

    except Exception as exc:
        logger.error(f"Voice query error: {exc}")
        raise HTTPException(status_code=500, detail="Voice processing failed. Please try again.")


# ── Predefined questions ─────────────────────────────────────────────────
@app.get("/predefined-questions")
async def predefined_questions(language: str = "english"):
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")
    return JSONResponse(content={"questions": rag.get_predefined_questions(language)})


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    rag = getattr(app.state, "rag", None)
    whisper = getattr(app.state, "whisper_available", False)
    return JSONResponse(content={
        "status": "healthy" if rag else "degraded",
        "rag_loaded": rag is not None,
        "whisper_available": whisper,
    })


# ── System info ──────────────────────────────────────────────────────────
@app.get("/info")
async def info():
    return JSONResponse(content={
        "name": "WellBeing Agent",
        "description": "RAG-based Breast Cancer Well-Being Support System",
        "version": "3.0.0",
        "features": [
            "Bilingual support (English / Urdu / Roman Urdu)",
            "RAG-powered well-being answers",
            "Whisper Large v3 voice transcription",
            "Emotional support detection",
            "Crisis / safety filtering",
            "Response caching with similarity matching",
            "Source citations",
        ],
    })


# ── Cleanup helper ───────────────────────────────────────────────────────
def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

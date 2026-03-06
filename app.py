# app.py - FastAPI server for Well Being Agent
import os
import asyncio
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan (import RAG system once on startup) ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Well Being Agent server…")
    try:
        from Agent_v2 import rag_system
        app.state.rag = rag_system
        logger.info("✅ RAG system loaded")
    except Exception as e:
        logger.error(f"❌ RAG init failed: {e}")
        import traceback
        traceback.print_exc()
        app.state.rag = None
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

# Static files
os.makedirs("static/audio", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request / Response Models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    message: str
    language: str = None


class VoiceRequest(BaseModel):
    audio_data: str  # base64


# ══════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════

# ── Serve index.html at root ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
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
        raise HTTPException(status_code=500, detail=str(e))


# ── Voice query ──────────────────────────────────────────────────────────
@app.post("/voice-query")
async def voice_query(req: VoiceRequest):
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")

    import base64

    try:
        audio_bytes = base64.b64decode(req.audio_data)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir="static/audio")
        tmp.write(audio_bytes)
        tmp.close()

        # Placeholder: real STT integration goes here.
        # For now, return a helpful message.
        transcribed = "Voice query received. Please type your question for now."
        language = rag.detect_language(transcribed)
        result = await asyncio.to_thread(
            rag.get_enhanced_answer_with_sources, transcribed, language, "voice"
        )

        # Cleanup temp file in background
        asyncio.get_event_loop().call_later(60, _safe_remove, tmp.name)

        return JSONResponse(content={
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "language": result.get("language", language),
            "transcribed_text": transcribed,
        })
    except Exception as e:
        logger.error(f"Voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    return JSONResponse(content={
        "status": "healthy" if rag else "degraded",
        "rag_loaded": rag is not None,
    })


# ── System info ──────────────────────────────────────────────────────────
@app.get("/info")
async def info():
    return JSONResponse(content={
        "name": "Well Being Agent",
        "description": "Breast Cancer Support System",
        "version": "2.0.0",
        "features": [
            "Bilingual support (English / Urdu)",
            "RAG-powered medical answers",
            "Emotional support detection",
            "Crisis / safety filtering",
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

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

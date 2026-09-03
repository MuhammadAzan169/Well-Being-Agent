"""Chat endpoints: text query, voice query, and predefined questions."""

import asyncio
import base64

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    PredefinedQuestionsResponse,
    QueryRequest,
    QueryResponse,
    VoiceRequest,
    VoiceResponse,
)
from app.services.language.utils import has_urdu_script, map_whisper_lang_to_system
from app.services.rag.system import BreastCancerRAGSystem

logger = get_logger("WellBeingAgent.API.Chat")

router = APIRouter(tags=["chat"])


@router.post("/ask-query", response_model=QueryResponse)
async def ask_query(req: QueryRequest, rag: BreastCancerRAGSystem = Depends(get_rag)):
    """Answer a text query through the RAG pipeline."""
    try:
        language = req.language or rag.detect_language(req.message)
        result = await asyncio.to_thread(
            rag.get_enhanced_answer_with_sources, req.message, language, "text"
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            language=result.get("language", language),
        )
    except Exception as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


@router.post("/voice-query", response_model=VoiceResponse)
async def voice_query(req: VoiceRequest, rag: BreastCancerRAGSystem = Depends(get_rag)):
    """Transcribe audio via Whisper, then answer through the RAG pipeline."""
    from app.services.audio.processor import transcribe_audio

    try:
        audio_bytes = base64.b64decode(req.audio_data)

        max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if len(audio_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Audio exceeds the {settings.MAX_AUDIO_SIZE_MB} MB limit. "
                       f"Please record a shorter clip.",
            )

        transcription = await asyncio.to_thread(transcribe_audio, audio_bytes)

        if not transcription["success"]:
            return VoiceResponse(
                answer=transcription["error"], sources=[], language="english", transcribed_text=""
            )

        transcribed_text = transcription["text"]
        language = map_whisper_lang_to_system(transcription["language"])
        # Only force Urdu when the transcription actually contains Urdu script.
        if has_urdu_script(transcribed_text):
            language = "urdu"

        result = await asyncio.to_thread(
            rag.get_enhanced_answer_with_sources, transcribed_text, language, "voice"
        )
        return VoiceResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            language=result.get("language", language),
            transcribed_text=transcribed_text,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Voice query error: {exc}")
        raise HTTPException(status_code=500, detail="Voice processing failed. Please try again.")


@router.get("/predefined-questions", response_model=PredefinedQuestionsResponse)
async def predefined_questions(
    language: str = "english", rag: BreastCancerRAGSystem = Depends(get_rag)
):
    """Return FAQ-style starter questions for the requested language."""
    return PredefinedQuestionsResponse(questions=rag.get_predefined_questions(language))

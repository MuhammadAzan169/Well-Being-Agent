"""System endpoints: root, health check, and service info."""

from fastapi import APIRouter, Request

from app.core.config import settings
from app.models.schemas import HealthResponse, InfoResponse

router = APIRouter(tags=["system"])


@router.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Lightweight root payload describing the API.

    HEAD is included because platform probes (Render's port detection among
    them) issue `HEAD /`, which a GET-only route answers with a 405.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Report whether the RAG system and Whisper are available."""
    rag = getattr(request.app.state, "rag", None)
    whisper = getattr(request.app.state, "whisper_available", False)
    return HealthResponse(
        status="healthy" if rag else "degraded",
        rag_loaded=rag is not None,
        whisper_available=bool(whisper),
    )


@router.get("/info", response_model=InfoResponse)
async def info():
    """Describe the service and its capabilities."""
    return InfoResponse(
        name=settings.APP_NAME,
        description="RAG-based Breast Cancer Well-Being Support System",
        version=settings.APP_VERSION,
        features=[
            "Bilingual support (English / Urdu / Roman Urdu)",
            "RAG-powered well-being answers",
            "Whisper Large v3 voice transcription",
            "Emotional support detection",
            "Crisis / safety filtering",
            "Response caching with similarity matching",
            "Source citations",
        ],
    )

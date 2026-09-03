"""Shared API dependencies."""

from fastapi import HTTPException, Request

from app.services.rag.system import BreastCancerRAGSystem


def get_rag(request: Request) -> BreastCancerRAGSystem:
    """Return the RAG system initialized at startup, or 503 if unavailable."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")
    return rag

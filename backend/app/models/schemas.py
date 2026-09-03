"""Pydantic request/response models for the API."""

from typing import List, Optional

from pydantic import BaseModel, field_validator

from app.core.config import settings


class QueryRequest(BaseModel):
    message: str
    language: Optional[str] = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Message is required")
        if len(v) > settings.MAX_QUERY_LENGTH:
            raise ValueError(f"Message must be under {settings.MAX_QUERY_LENGTH} characters")
        return v


class VoiceRequest(BaseModel):
    audio_data: str  # base64-encoded audio


class Source(BaseModel):
    topic: str
    category: str = ""
    source: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    language: str


class VoiceResponse(QueryResponse):
    transcribed_text: str = ""


class PredefinedQuestion(BaseModel):
    question: str
    category: str = "general"
    icon: str = "fas fa-question-circle"


class PredefinedQuestionsResponse(BaseModel):
    questions: List[PredefinedQuestion]


class HealthResponse(BaseModel):
    status: str
    rag_loaded: bool
    whisper_available: bool


class InfoResponse(BaseModel):
    name: str
    description: str
    version: str
    features: List[str]

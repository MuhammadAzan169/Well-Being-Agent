"""RAG pipeline orchestration for the WellBeing Agent.

Pipeline: detect language → safety check → cache lookup → retrieve →
build prompt → LLM query → post-process → cache store → respond.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.services.language.utils import clean_urdu_text, detect_language
from app.services.rag.cache import ConversationLogger, ResponseCache
from app.services.rag.emotional import EmotionalAnalyzer
from app.services.rag.index import load_index
from app.services.rag.llm_client import (
    FALLBACK_MESSAGE,
    FALLBACK_MESSAGE_URDU,
    LLMClient,
)
from app.services.rag.prompts import PromptBuilder
from app.services.safety.validator import SafetyValidator

logger = get_logger("WellBeingAgent.RAG")


PREDEFINED_QUESTIONS_EN: List[Dict] = [
    {"question": "What are the earliest warning signs of breast cancer?",
     "category": "symptoms", "icon": "fas fa-search"},
    {"question": "How do I deal with anxiety about my next treatment?",
     "category": "emotional", "icon": "fas fa-heart"},
    {"question": "Will my hair grow back after chemotherapy?",
     "category": "appearance", "icon": "fas fa-user"},
    {"question": "Can I breastfeed after breast cancer surgery?",
     "category": "lactation", "icon": "fas fa-baby"},
    {"question": "What exercises are safe during treatment?",
     "category": "exercise", "icon": "fas fa-walking"},
    {"question": "What foods should I eat during chemotherapy?",
     "category": "nutrition", "icon": "fas fa-apple-alt"},
]

PREDEFINED_QUESTIONS_UR: List[Dict] = [
    {"question": "بریسٹ کینسر کی ابتدائی علامات کیا ہیں؟",
     "category": "symptoms", "icon": "fas fa-search"},
    {"question": "کیموتھراپی کے دوران پریشانی کیسے کم کریں؟",
     "category": "emotional", "icon": "fas fa-heart"},
    {"question": "کیا علاج کے بعد بال واپس آئیں گے؟",
     "category": "appearance", "icon": "fas fa-user"},
    {"question": "کیا سرجری کے بعد بچے کو دودھ پلا سکتی ہوں؟",
     "category": "lactation", "icon": "fas fa-baby"},
    {"question": "علاج کے دوران کون سی ورزشیں محفوظ ہیں؟",
     "category": "exercise", "icon": "fas fa-walking"},
    {"question": "کیموتھراپی کے دوران کیا کھانا چاہیے؟",
     "category": "nutrition", "icon": "fas fa-apple-alt"},
]


class BreastCancerRAGSystem:
    """Full RAG pipeline for breast-cancer well-being support."""

    def __init__(self, index: Any, retriever: Any) -> None:
        self.index = index
        self.retriever = retriever
        self.llm = LLMClient()
        self.cache = ResponseCache()
        self.conv_logger = ConversationLogger()
        self.conversation_history: List[Dict] = []

    # ── Language detection ─────────────────────────────────────────────────
    def detect_language(self, text: str) -> str:
        return detect_language(text)

    # ── Predefined questions ───────────────────────────────────────────────
    def get_predefined_questions(self, language: str = "english") -> List[Dict]:
        return PREDEFINED_QUESTIONS_UR if language == "urdu" else PREDEFINED_QUESTIONS_EN

    # ── Retrieval ──────────────────────────────────────────────────────────
    def _retrieve(self, query: str) -> Tuple[List[Any], List[Dict]]:
        if not self.retriever:
            logger.warning("Retriever not available — skipping retrieval")
            return [], []
        try:
            results = self.retriever.retrieve(query)
            top_score = max((r.score for r in results if hasattr(r, "score")), default=0)
            if len(results) < 3 or top_score < 0.4:
                simplified = re.sub(
                    r"\b(what|how|can|will|do|does|is|are|should|could|would|"
                    r"when|where|why|tell me about|please|i want to know)\b",
                    "",
                    query.lower(),
                ).strip()
                if simplified and len(simplified) > 10:
                    extra = self.retriever.retrieve(simplified)
                    seen = {hash(r.text[:100]) for r in results if hasattr(r, "text")}
                    for r in extra:
                        if hasattr(r, "text") and hash(r.text[:100]) not in seen:
                            results.append(r)
                            seen.add(hash(r.text[:100]))

            good = [r for r in results if hasattr(r, "score") and r.score >= 0.2]
            if not good:
                good = results[:3]
            good.sort(key=lambda r: getattr(r, "score", 0), reverse=True)

            sources: List[Dict] = []
            seen_topics: set = set()
            for r in good:
                meta = getattr(r, "metadata", {})
                topic = meta.get("topic", "General")
                if topic not in seen_topics:
                    seen_topics.add(topic)
                    sources.append({
                        "topic": topic,
                        "category": meta.get("category", ""),
                        "source": meta.get("source", ""),
                        "score": round(getattr(r, "score", 0), 3),
                    })
            logger.info(f"✅ Retrieved {len(good)} chunks, {len(sources)} unique sources")
            return good[:5], sources
        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            return [], []

    # ── Post-processing ────────────────────────────────────────────────────
    def _postprocess(self, text: str, language: str) -> str:
        if not text or text in (FALLBACK_MESSAGE, FALLBACK_MESSAGE_URDU):
            return text
        if language == "urdu":
            text = clean_urdu_text(text)
        text = SafetyValidator.validate_response(text, language)
        text = SafetyValidator.add_medical_disclaimer(text, language)
        return text.strip()

    # ── Main pipeline ──────────────────────────────────────────────────────
    def get_enhanced_answer_with_sources(
        self,
        user_query: str,
        language: Optional[str] = None,
        response_type: str = "text",
    ) -> Dict[str, Any]:
        if language is None:
            language = self.detect_language(user_query)

        safety = SafetyValidator.validate_query(user_query, language)
        if safety["is_crisis"] or not safety["is_on_topic"]:
            self.conv_logger.log(user_query, safety["response"], language, response_type)
            return {"answer": safety["response"], "sources": [], "language": language}

        cached = self.cache.get(user_query)
        if cached:
            cached_resp = cached.get("response", "")
            is_error = cached.get("is_error", False)
            is_fallback = cached_resp in (FALLBACK_MESSAGE, FALLBACK_MESSAGE_URDU)
            if not is_error and not is_fallback:
                logger.info("Returning cached response")
                self.conv_logger.log(
                    user_query, cached_resp, language, response_type, cached.get("sources", [])
                )
                return {
                    "answer": cached_resp,
                    "sources": cached.get("sources", []),
                    "language": cached.get("language", language),
                }
            logger.info("Skipping cached error/fallback — will retry LLM")

        chunks, sources = self._retrieve(user_query)
        emotional = EmotionalAnalyzer.analyze(user_query, language)
        prompt = PromptBuilder.build(user_query, chunks, language, emotional)

        raw_response = self.llm.complete(prompt, language)
        is_fallback = raw_response in (FALLBACK_MESSAGE, FALLBACK_MESSAGE_URDU)
        final_response = self._postprocess(raw_response, language)

        self.cache.put(user_query, final_response, language, sources, is_error=is_fallback)
        self.conv_logger.log(user_query, final_response, language, response_type, sources)
        self.conversation_history.append({
            "query": user_query,
            "answer": final_response,
            "language": language,
            "sources": sources,
            "timestamp": time.time(),
        })

        return {"answer": final_response, "sources": sources, "language": language}

    def get_enhanced_answer(
        self, user_query: str, language: Optional[str] = None, response_type: str = "text"
    ) -> str:
        return self.get_enhanced_answer_with_sources(user_query, language, response_type)["answer"]


def create_rag_system() -> BreastCancerRAGSystem:
    """Build the RAG system (loads the vector index). Call once at startup."""
    logger.info("🚀 Initializing WellBeing Agent RAG system…")
    start = time.time()
    index, retriever = load_index()
    system = BreastCancerRAGSystem(index, retriever)
    logger.info(f"✅ RAG system ready in {time.time() - start:.1f}s")
    return system

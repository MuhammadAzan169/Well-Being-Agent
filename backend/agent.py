"""
Agent.py — RAG-based Breast Cancer Well-Being Support Agent

This module implements the core RAG (Retrieval-Augmented Generation) pipeline
for the WellBeing Agent. It provides supportive, empathetic responses to breast
cancer patients using a curated knowledge base and LLM generation.

⚠️ IMPORTANT: This agent does NOT prescribe treatments or medications.
It provides supportive guidance, education, and reassurance only.
"""

import os
import json
import time
import hashlib
import re
import logging
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime
from difflib import SequenceMatcher

import httpx
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

from backend.safety import SafetyValidator
from backend.language_utils import (
    detect_language,
    clean_urdu_text,
    detect_roman_urdu,
    format_response,
)

# Ensure deterministic language detection
DetectorFactory.seed = 0

# ── Load environment variables ───────────────────────────────────────────
load_dotenv()

# ── Logging Configuration ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("WellBeingAgent")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — loaded entirely from environment variables
# ═══════════════════════════════════════════════════════════════════════════
class AgentConfig:
    """Centralized configuration loaded from environment variables (.env)."""

    def __init__(self) -> None:
        # LLM settings
        self.LLM_PROVIDER: str = self._require("LLM_PROVIDER")
        self.LLM_MODEL: str = self._require("LLM_MODEL")
        self.LLM_BASE_URL: str = self._require("LLM_BASE_URL")
        self.LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))
        self.LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

        # Embedding / Index settings
        self.EMBEDDING_MODEL: str = self._require("EMBEDDING_MODEL")
        self.INDEX_PATH: str = self._require("INDEX_PATH")
        self.DATASET_PATH: str = self._require("DATASET_PATH")
        self.SIMILARITY_TOP_K: int = int(os.getenv("SIMILARITY_TOP_K", "5"))

        # Cache settings
        self.CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
        self.CACHE_SIMILARITY_THRESHOLD: float = float(
            os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
        )

        # Fallback message when LLM is unavailable
        self.FALLBACK_MESSAGE: str = (
            "I'm sorry, I'm unable to respond right now. "
            "Please try again shortly or consult your healthcare provider."
        )
        self.FALLBACK_MESSAGE_URDU: str = (
            "معذرت، میں ابھی جواب دینے سے قاصر ہوں۔ "
            "براہ کرم دوبارہ کوشش کریں یا اپنے ڈاکٹر سے رابطہ کریں۔"
        )

        # API key rotation
        self.api_keys: List[str] = []
        self.current_key_index: int = 0
        self._load_api_keys()
        self.api_key: str = self._current_key()

        self._log_config()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _require(name: str) -> str:
        """Retrieve a required environment variable or raise an error."""
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"Required environment variable '{name}' is not set in .env")
        return value

    def _load_api_keys(self) -> None:
        """Load all available API keys for rotation."""
        for name in ("LLM_API_KEY", "LLM_API_KEY_2", "LLM_API_KEY_3"):
            val = os.getenv(name, "").strip()
            if val:
                self.api_keys.append(val)
        if not self.api_keys:
            logger.warning("⚠️  No API keys found in .env — LLM calls will fail")

    def _current_key(self) -> str:
        if self.api_keys and self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return ""

    def rotate_key(self) -> bool:
        """Rotate to the next API key. Returns True if successful."""
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.api_key = self._current_key()
            logger.info(f"🔄 Rotated to API key #{self.current_key_index + 1}")
            return True
        return False

    def get_fallback(self, language: str = "english") -> str:
        """Return the appropriate fallback message for the given language."""
        return self.FALLBACK_MESSAGE_URDU if language == "urdu" else self.FALLBACK_MESSAGE

    def _log_config(self) -> None:
        logger.info(f"📋 Provider : {self.LLM_PROVIDER} | Model: {self.LLM_MODEL}")
        logger.info(f"📋 Index    : {self.INDEX_PATH} | Embed: {self.EMBEDDING_MODEL}")
        logger.info(f"📋 API keys : {len(self.api_keys)} loaded")


config = AgentConfig()


# ═══════════════════════════════════════════════════════════════════════════
# Response Cache — JSON-based with similarity matching
# ═══════════════════════════════════════════════════════════════════════════
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.json")


class ResponseCache:
    """
    Persistent response cache with:
      - Exact-match lookup (MD5 key)
      - Fuzzy / similarity-based lookup for near-duplicate queries
      - TTL-based expiry
      - JSON storage (human-readable, not pickle)
      - Error-response caching for resilience
    """

    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ── Key Generation ────────────────────────────────────────────────────

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize a query for consistent hashing."""
        return re.sub(r"\s+", " ", query.lower().strip())

    def make_key(self, query: str) -> str:
        """Generate a cache key from a normalized query."""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    # ── Exact Lookup ──────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Look up a cached response by exact match first, then by similarity.
        Returns the full cache entry dict or None.
        """
        key = self.make_key(query)

        # 1) Exact match
        if key in self.store:
            entry = self.store[key]
            if self._is_valid(entry):
                logger.info("✅ Cache HIT (exact)")
                return entry
            else:
                del self.store[key]

        # 2) Similarity match
        return self._find_similar(query)

    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        """Check whether a cache entry is still within TTL."""
        ts = entry.get("timestamp", 0)
        return (time.time() - ts) < config.CACHE_TTL_HOURS * 3600

    def _find_similar(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a similar cached query using SequenceMatcher."""
        normalized = self._normalize_query(query)
        best_score = 0.0
        best_entry: Optional[Dict[str, Any]] = None

        for entry in self.store.values():
            if not self._is_valid(entry):
                continue
            cached_query = entry.get("query_normalized", "")
            score = SequenceMatcher(None, normalized, cached_query).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= config.CACHE_SIMILARITY_THRESHOLD and best_entry:
            logger.info(f"✅ Cache HIT (similarity={best_score:.2f})")
            return best_entry

        return None

    # ── Store ─────────────────────────────────────────────────────────────

    def put(
        self,
        query: str,
        response: str,
        language: str,
        sources: Optional[List[Dict]] = None,
        is_error: bool = False,
    ) -> None:
        """Store a response in the cache."""
        key = self.make_key(query)
        self.store[key] = {
            "query": query,
            "query_normalized": self._normalize_query(query),
            "response": response,
            "language": language,
            "sources": sources or [],
            "timestamp": time.time(),
            "is_error": is_error,
        }
        self._save()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.store, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Cache save error: {exc}")

    def _load(self) -> None:
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
                # Prune expired entries on load
                self._prune_expired()
                logger.info(f"✅ Cache loaded: {len(self.store)} entries")
        except Exception:
            logger.warning("Cache file corrupt or missing — starting fresh")
            self.store = {}

    def _prune_expired(self) -> None:
        """Remove expired entries from the store."""
        expired_keys = [k for k, v in self.store.items() if not self._is_valid(v)]
        for k in expired_keys:
            del self.store[k]
        if expired_keys:
            logger.info(f"🗑️  Pruned {len(expired_keys)} expired cache entries")
            self._save()

    def clear(self) -> None:
        """Clear the entire cache."""
        self.store = {}
        self._save()
        logger.info("🗑️  Cache cleared")


cache = ResponseCache()


# ═══════════════════════════════════════════════════════════════════════════
# Conversation Logger
# ═══════════════════════════════════════════════════════════════════════════
class ConversationLogger:
    """Logs every conversation turn to a JSON file for audit / analysis."""

    def __init__(self, path: str = "conversations.json") -> None:
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log(
        self,
        user_input: str,
        response: str,
        language: str,
        response_type: str = "text",
        sources: Optional[List[Dict]] = None,
    ) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        data.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "response": response,
                "language": language,
                "response_type": response_type,
                "sources": sources or [],
            }
        )
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Conversation log write error: {exc}")


conv_logger = ConversationLogger()


# ═══════════════════════════════════════════════════════════════════════════
# Vector Index Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_index() -> Tuple[Any, Any]:
    """Load the persisted vector index and return (index, retriever)."""
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        if not os.path.exists(config.INDEX_PATH):
            logger.error(f"❌ Index directory not found: {config.INDEX_PATH}")
            return None, None

        embed = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        ctx = StorageContext.from_defaults(persist_dir=config.INDEX_PATH)
        index = VectorStoreIndex.from_documents(
            [], storage_context=ctx, embed_model=embed
        )
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        logger.info("✅ Vector index loaded successfully")
        return index, retriever

    except Exception as exc:
        logger.error(f"❌ Index load failed: {exc}")
        import traceback
        traceback.print_exc()
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Emotional Support Analyzer
# ═══════════════════════════════════════════════════════════════════════════
class EmotionalAnalyzer:
    """Detects emotional needs in patient queries for tone calibration."""

    # Explicit emotional distress keywords
    EXPLICIT_EN = [
        "scared", "afraid", "worried", "anxious", "fear", "nervous",
        "stressed", "overwhelmed", "depressed", "sad", "lonely", "alone",
        "hopeless", "can't cope", "struggling", "suffering", "terrified",
        "panic", "crying", "devastated", "angry", "frustrated", "exhausted",
        "broken", "lost", "helpless", "despair", "grief", "give up",
        "can't sleep", "why me", "feel like a burden", "no one understands",
    ]
    EXPLICIT_UR = [
        "خوف", "ڈر", "پریشانی", "فکر", "تنہائی", "اداسی", "مایوسی",
        "تکلیف", "گھبراہٹ", "بے چینی", "بے بسی", "رونا", "آنسو",
        "دکھ", "غم", "تھکاوٹ", "ٹوٹ", "اکیلا", "اکیلی", "مشکل",
        "نیند نہیں", "بھوک نہیں", "کیوں میں", "بوجھ", "برداشت",
    ]

    # Implicit emotional topics
    IMPLICIT_EN = [
        "hair loss", "bald", "body image", "husband", "wife", "partner",
        "children", "family", "tell my", "pregnant", "baby", "fertility",
        "breastfeed", "mastectomy", "die", "dying", "death", "survival rate",
        "recurrence", "stage 4", "metastatic", "chemo", "side effects",
        "pain", "nausea", "fatigue", "work", "money", "normal life",
    ]
    IMPLICIT_UR = [
        "بال", "شکل", "شوہر", "بیوی", "بچے", "خاندان",
        "حمل", "دودھ", "سرجری", "موت", "زندگی", "واپس آنا",
        "کیمو", "علاج", "درد", "متلی", "نوکری", "پیسے",
    ]

    # Vulnerability patterns
    VULN_EN = ["will i", "can i still", "am i going to", "what if", "how do i cope"]
    VULN_UR = ["کیا میں", "کب تک", "کیسے", "ممکن ہے", "اگر"]

    @classmethod
    def analyze(cls, query: str, language: str) -> Dict[str, Any]:
        """Return emotional analysis dict with needs_support flag and score."""
        q = query.lower()
        is_urdu = language == "urdu"

        explicit = cls.EXPLICIT_UR if is_urdu else cls.EXPLICIT_EN
        implicit = cls.IMPLICIT_UR if is_urdu else cls.IMPLICIT_EN
        vulns = cls.VULN_UR if is_urdu else cls.VULN_EN

        e_score = sum(2 for t in explicit if t in q)
        i_score = sum(1 for t in implicit if t in q)
        v_score = sum(1 for p in vulns if p in q)
        total = e_score + i_score + v_score

        return {"needs_emotional_support": total > 0, "score": total}


# ═══════════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════════
class PromptBuilder:
    """Constructs carefully engineered prompts for the LLM."""

    @staticmethod
    def build(
        query: str,
        chunks: List[Any],
        language: str,
        emotional: Dict[str, Any],
    ) -> str:
        """Build the full prompt with context, emotional guidance, and rules."""
        context = PromptBuilder._format_context(chunks)

        if language == "urdu":
            return PromptBuilder._urdu_prompt(query, context, emotional)
        return PromptBuilder._english_prompt(query, context, emotional)

    @staticmethod
    def _format_context(chunks: List[Any]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks[:5]):
            text = getattr(c, "text", str(c))
            meta = getattr(c, "metadata", {})
            # Limit to ~200 words per chunk
            words = " ".join(text.split()[:200])
            topic = meta.get("topic", "General")
            source = meta.get("source", "Knowledge Base")
            parts.append(f"[Source {i + 1}: {topic} — {source}]\n{words}")
        return "\n\n".join(parts)

    @staticmethod
    def _emotional_guidance_en(score: int) -> str:
        if score >= 3:
            return (
                "CRITICAL: This patient is in significant emotional distress. "
                "Lead with 2–3 sentences of deep empathy and validation BEFORE "
                "providing any medical information. Acknowledge their feelings first."
            )
        if score >= 1:
            return (
                "This patient may need emotional support alongside information. "
                "Open with a warm, validating sentence and end on a hopeful note."
            )
        return "Be warm and friendly — like a caring, knowledgeable friend."

    @staticmethod
    def _emotional_guidance_ur(score: int) -> str:
        if score >= 3:
            return (
                "اہم: یہ مریض بہت زیادہ جذباتی تکلیف میں ہے۔ "
                "پہلے 2-3 جملے صرف ہمدردی اور تسلی کے ہوں، پھر معلومات دیں۔"
            )
        if score >= 1:
            return "مریض کو جذباتی مدد کی ضرورت ہے۔ گرمجوشی سے شروع کریں، امید پر ختم کریں۔"
        return "دوستانہ اور گرمجوش انداز میں بات کریں۔"

    @staticmethod
    def _english_prompt(query: str, context: str, emo: Dict) -> str:
        guide = PromptBuilder._emotional_guidance_en(emo["score"])
        no_info_instruction = (
            "If the retrieved context does not contain relevant information, "
            'respond with: "I\'m sorry, I don\'t have enough information to answer '
            'that right now. Please consult your doctor or healthcare provider."'
        )
        context_block = context or "No specific context retrieved."

        return f"""# WELLBEING AGENT — Breast Cancer Support Assistant

## YOUR IDENTITY
You are a warm, knowledgeable breast cancer well-being support companion.
You speak in a friendly, conversational tone while being medically accurate.
You are NOT a doctor and must NEVER prescribe medications or treatments.
Always encourage patients to consult their healthcare team for medical decisions.

## PATIENT'S QUESTION
"{query}"

## EMOTIONAL GUIDANCE
{guide}

## RETRIEVED KNOWLEDGE BASE CONTEXT
{context_block}

## RESPONSE RULES (MUST FOLLOW)
1. Use the retrieved context as your PRIMARY source of information.
   Supplement with general breast cancer knowledge ONLY when the context is relevant but incomplete.
2. {no_info_instruction}
3. Write like you are talking to someone you care about — warm, human, supportive.
4. Provide specific, actionable information: exercises, foods, timelines, coping strategies.
5. NEVER prescribe medications, dosages, or specific treatments.
   Instead say: "Your doctor may suggest..." or "Many patients find... helpful."
6. NEVER suggest stopping any treatment or medication.
7. NEVER give false hope about survival rates or outcomes.
8. Validate feelings naturally and authentically.
9. End with genuine warmth and encouragement. 💛
10. Keep responses 5–10 sentences. Be concise but thorough.
11. If you cite medical facts, note the source conversationally:
    "According to cancer research..." or "Studies have shown..."

## YOUR RESPONSE:"""

    @staticmethod
    def _urdu_prompt(query: str, context: str, emo: Dict) -> str:
        guide = PromptBuilder._emotional_guidance_ur(emo["score"])
        context_block = context or "عمومی بریسٹ کینسر کی معلومات سے جواب دیں۔"

        return f"""# ویل بینگ ایجنٹ — بریسٹ کینسر سپورٹ اسسٹنٹ

## آپ کا کردار
آپ ایک شفیق اور باعلم بریسٹ کینسر سپورٹ ساتھی ہیں۔
آپ ڈاکٹر نہیں ہیں — کبھی بھی دوائیں یا علاج تجویز نہ کریں۔
ہمیشہ مریض کو اپنی طبی ٹیم سے مشورے کی تاکید کریں۔

## مریض کا سوال
"{query}"

## جذباتی رہنمائی
{guide}

## طبی سیاق و سباق (نالج بیس)
{context_block}

## ⚠️ زبان کے سخت اصول (سب سے اہم)
- صرف اور صرف اردو/عربی رسم الخط (ا-ی، ء-ے) استعمال کریں۔
- ہندی (अ-ह)، چینی، ویتنامی، فرانسیسی، یا کسی بھی غیر اردو حروف بالکل نہ لکھیں۔
- انگریزی الفاظ صرف طبی اصطلاحات کے لیے قابل قبول ہیں (جیسے: cancer, chemo, DNA)۔
- ❌ ممنوع: आकार, difficile, vấn, 亲, सबसे — یہ حروف کبھی استعمال نہ کریں۔

## جواب کے اصول (لازمی)
1. سیاق و سباق کو بنیادی ذریعے کے طور پر استعمال کریں۔
2. اگر متعلقہ معلومات نہ ملیں تو کہیں: "معذرت، اس بارے میں مجھے کافی معلومات نہیں ہیں۔ براہ کرم اپنے ڈاکٹر سے مشورہ کریں۔"
3. دوستانہ لہجہ — جیسے کسی عزیز سے بات کر رہے ہوں۔
4. مخصوص اور عملی معلومات دیں (ورزشیں، غذائیں، طریقے)۔
5. کبھی دوائیں یا خوراکیں تجویز نہ کریں۔ کہیں: "آپ کا ڈاکٹر تجویز کر سکتا ہے..."
6. کبھی علاج بند کرنے کا مشورہ نہ دیں۔
7. جھوٹی امید نہ دیں۔
8. جذبات کی تصدیق کریں۔
9. آخر میں گرمجوشی اور حوصلہ افزائی۔ 💛
10. مختصر جواب دیں — زیادہ سے زیادہ 5-8 جملے۔ غیر ضروری تفصیل سے بچیں۔

## ہجوں کے اصول
✅ "مجھے" ❌ "مجہے" | ✅ "کینسر" ❌ "کہےنسر" | ✅ "ڈاکٹر" ❌ "ڈڈاکٹر"
✅ "ہے" ❌ "ہےہ" | ✅ "میں" ❌ "مہےں" | ✅ "کے لیے" ❌ "کا ے لہےے"

## آپ کا اردو جواب (مختصر، صرف اردو رسم الخط میں):"""


# ═══════════════════════════════════════════════════════════════════════════
# System Prompts
# ═══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT_EN = (
    "You are a warm, caring breast cancer well-being support companion. "
    "You combine medically accurate information with genuine emotional warmth. "
    "Use conversational, supportive language. "
    "NEVER prescribe medications, dosages, or specific treatments. "
    "NEVER suggest stopping treatment. "
    "NEVER give false medical promises about outcomes. "
    "Always recommend that patients discuss concerns with their healthcare team."
)

SYSTEM_PROMPT_UR = (
    "آپ ایک شفیق بریسٹ کینسر سپورٹ ساتھی ہیں۔ "
    "صرف اور صرف اردو/عربی رسم الخط (ا-ی) میں جواب دیں۔ "
    "ہندی (Devanagari)، چینی، ویتنامی، فرانسیسی، یا کوئی بھی غیر اردو حروف بالکل استعمال نہ کریں۔ "
    "طبی معلومات کے ساتھ جذباتی مدد شامل کریں۔ "
    "کبھی دوائیں یا علاج تجویز نہ کریں۔ "
    "ہمیشہ طبی ٹیم سے مشورے کی تاکید کریں۔ "
    "جواب مختصر رکھیں — زیادہ سے زیادہ 5-8 جملے۔"
)


# ═══════════════════════════════════════════════════════════════════════════
# RAG System — Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════
class BreastCancerRAGSystem:
    """
    Full RAG pipeline:
      detect language → safety check → cache lookup → retrieve →
      build prompt → LLM query → post-process → cache store → respond
    """

    def __init__(self, index: Any, retriever: Any) -> None:
        self.index = index
        self.retriever = retriever
        self.conversation_history: List[Dict] = []
        self.http_client = httpx.Client(timeout=60.0)

    # ── Language Detection ────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """
        Detect language with Roman Urdu support.
        Returns 'urdu' or 'english'.
        """
        # 1) Urdu script check
        if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", text):
            return "urdu"

        # 2) Roman Urdu check (common transliterated Urdu words)
        if detect_roman_urdu(text):
            return "urdu"

        # 3) langdetect fallback
        try:
            lang = detect(text)
            return "urdu" if lang == "ur" else "english"
        except Exception:
            return "english"

    # ── Predefined Questions ──────────────────────────────────────────────

    def get_predefined_questions(self, language: str = "english") -> List[Dict]:
        """Return a list of predefined FAQ-style questions."""
        english = [
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
        urdu = [
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
        return urdu if language == "urdu" else english

    # ── Retrieval with Sources ────────────────────────────────────────────

    def _retrieve(self, query: str) -> Tuple[List[Any], List[Dict]]:
        """Retrieve relevant chunks from the vector index."""
        if not self.retriever:
            logger.warning("Retriever not available — skipping retrieval")
            return [], []

        try:
            results = self.retriever.retrieve(query)

            # If initial results are poor, try a simplified query for better recall
            top_score = max(
                (r.score for r in results if hasattr(r, "score")), default=0
            )
            if len(results) < 3 or top_score < 0.4:
                simplified = re.sub(
                    r"\b(what|how|can|will|do|does|is|are|should|could|would|"
                    r"when|where|why|tell me about|please|i want to know)\b",
                    "",
                    query.lower(),
                ).strip()
                if simplified and len(simplified) > 10:
                    extra = self.retriever.retrieve(simplified)
                    seen = {
                        hash(r.text[:100])
                        for r in results
                        if hasattr(r, "text")
                    }
                    for r in extra:
                        if hasattr(r, "text") and hash(r.text[:100]) not in seen:
                            results.append(r)
                            seen.add(hash(r.text[:100]))

            # Filter by minimum relevance score
            good = [r for r in results if hasattr(r, "score") and r.score >= 0.2]
            if not good:
                good = results[:3]
            good.sort(key=lambda r: getattr(r, "score", 0), reverse=True)

            # Extract source metadata (deduplicated by topic)
            sources: List[Dict] = []
            seen_topics: set = set()
            for r in good:
                meta = getattr(r, "metadata", {})
                topic = meta.get("topic", "General")
                if topic not in seen_topics:
                    seen_topics.add(topic)
                    sources.append(
                        {
                            "topic": topic,
                            "category": meta.get("category", ""),
                            "source": meta.get("source", ""),
                            "score": round(getattr(r, "score", 0), 3),
                        }
                    )

            logger.info(
                f"✅ Retrieved {len(good)} chunks, {len(sources)} unique sources"
            )
            return good[:5], sources

        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            return [], []

    # ── LLM Query ─────────────────────────────────────────────────────────

    def _query_llm(self, prompt: str, language: str, retries: int = 3) -> str:
        """Send the prompt to the LLM with retry + key rotation logic."""
        if not config.api_key:
            logger.error("No API key available — cannot query LLM")
            return config.get_fallback(language)

        system_msg = SYSTEM_PROMPT_UR if language == "urdu" else SYSTEM_PROMPT_EN

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
        }

        for attempt in range(retries):
            try:
                url = f"{config.LLM_BASE_URL}/chat/completions"
                resp = self.http_client.post(url, headers=headers, json=payload)

                # Rate limit or server error → rotate key & retry
                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning(
                        f"API {resp.status_code} on attempt {attempt + 1}/{retries}"
                    )
                    if config.rotate_key():
                        headers["Authorization"] = f"Bearer {config.api_key}"
                    # Longer backoff for free-tier rate limits
                    time.sleep(3 + 2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()

                if not text:
                    logger.warning("LLM returned empty response")
                    return config.get_fallback(language)

                logger.info("✅ LLM response received")
                return text

            except httpx.TimeoutException:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{retries})")
                time.sleep(2 ** attempt)
            except (KeyError, IndexError) as exc:
                logger.error(f"Unexpected API response format: {exc}")
                return config.get_fallback(language)
            except Exception as exc:
                logger.error(f"LLM error (attempt {attempt + 1}): {exc}")
                if config.rotate_key():
                    headers["Authorization"] = f"Bearer {config.api_key}"
                time.sleep(2 ** attempt)

        logger.error("All LLM retries exhausted")
        return config.get_fallback(language)

    # ── Post-Processing ───────────────────────────────────────────────────

    def _postprocess(self, text: str, language: str) -> str:
        """Clean, validate, and add disclaimers to the response."""
        if not text or text == config.get_fallback(language):
            return text

        # Clean Urdu text if needed
        if language == "urdu":
            text = clean_urdu_text(text)

        # Safety validation (check for dangerous advice)
        text = SafetyValidator.validate_response(text, language)

        # Add medical disclaimer
        text = SafetyValidator.add_medical_disclaimer(text, language)

        return text.strip()

    # ── Main Pipeline (with sources) ──────────────────────────────────────

    def get_enhanced_answer_with_sources(
        self,
        user_query: str,
        language: Optional[str] = None,
        response_type: str = "text",
    ) -> Dict[str, Any]:
        """
        Full pipeline: language detect → safety → cache → retrieve → LLM → format.
        Returns dict with keys: answer, sources, language.
        """
        # 1) Detect language
        if language is None:
            language = self.detect_language(user_query)

        # 2) Safety check
        safety = SafetyValidator.validate_query(user_query, language)
        if safety["is_crisis"]:
            conv_logger.log(user_query, safety["response"], language, response_type)
            return {
                "answer": safety["response"],
                "sources": [],
                "language": language,
            }
        if not safety["is_on_topic"]:
            conv_logger.log(user_query, safety["response"], language, response_type)
            return {
                "answer": safety["response"],
                "sources": [],
                "language": language,
            }

        # 3) Check cache (skip error/fallback responses so we retry the LLM)
        cached = cache.get(user_query)
        if cached:
            cached_resp = cached.get("response", "")
            is_error = cached.get("is_error", False)
            is_fallback = cached_resp in (
                config.FALLBACK_MESSAGE,
                config.FALLBACK_MESSAGE_URDU,
            )
            if not is_error and not is_fallback:
                logger.info("Returning cached response")
                conv_logger.log(
                    user_query,
                    cached_resp,
                    language,
                    response_type,
                    cached.get("sources", []),
                )
                return {
                    "answer": cached_resp,
                    "sources": cached.get("sources", []),
                    "language": cached.get("language", language),
                }
            else:
                logger.info("Skipping cached error/fallback — will retry LLM")

        # 4) Retrieve from knowledge base
        chunks, sources = self._retrieve(user_query)

        # 5) Analyze emotional needs
        emotional = EmotionalAnalyzer.analyze(user_query, language)

        # 6) Build prompt
        prompt = PromptBuilder.build(user_query, chunks, language, emotional)

        # 7) Query LLM
        raw_response = self._query_llm(prompt, language)
        is_fallback = raw_response in (
            config.FALLBACK_MESSAGE,
            config.FALLBACK_MESSAGE_URDU,
        )

        # 8) Post-process
        final_response = self._postprocess(raw_response, language)

        # 9) Cache the response (even errors, for resilience)
        cache.put(
            user_query,
            final_response,
            language,
            sources,
            is_error=is_fallback,
        )

        # 10) Log conversation
        conv_logger.log(user_query, final_response, language, response_type, sources)

        # 11) Update conversation history
        self.conversation_history.append(
            {
                "query": user_query,
                "answer": final_response,
                "language": language,
                "sources": sources,
                "timestamp": time.time(),
            }
        )

        return {
            "answer": final_response,
            "sources": sources,
            "language": language,
        }

    def get_enhanced_answer(
        self,
        user_query: str,
        language: Optional[str] = None,
        response_type: str = "text",
    ) -> str:
        """Convenience method that returns just the answer string."""
        result = self.get_enhanced_answer_with_sources(
            user_query, language, response_type
        )
        return result["answer"]


# ═══════════════════════════════════════════════════════════════════════════
# Module-level Initialization (runs on import)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("🚀 Initializing WellBeing Agent…")
_start = time.time()
index, retriever = load_index()
logger.info(f"✅ System ready in {time.time() - _start:.1f}s")

rag_system = BreastCancerRAGSystem(index, retriever)


# ═══════════════════════════════════════════════════════════════════════════
# CLI Mode — for direct testing
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n💬 WellBeing Agent — Breast Cancer Support")
    print("=" * 55)
    print("Type 'quit' to exit.  Supports English & Urdu.\n")

    while True:
        try:
            q = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue

        t0 = time.time()
        result = rag_system.get_enhanced_answer_with_sources(q)
        print(f"\n💡 {result['answer']}")
        if result["sources"]:
            print(
                "📚 Sources:",
                ", ".join(s["topic"] for s in result["sources"][:3]),
            )
        print(f"⏱️  {time.time() - t0:.1f}s\n")

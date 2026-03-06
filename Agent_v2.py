# Agent_v2.py - RAG-based Breast Cancer Well-Being Support Agent
# All configuration from .env — no config.json dependency
import os
import json
import time
import pickle
import hashlib
import re
import logging
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
from datetime import datetime

import httpx
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

from safety import SafetyValidator

# ── Load .env from project root ─────────────────────────────────────────
load_dotenv()  # loads .env from cwd

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — 100 % from environment variables
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    def __init__(self):
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER")
        if not self.LLM_PROVIDER:
            raise ValueError("LLM_PROVIDER not set in .env")
        self.LLM_MODEL = os.getenv("LLM_MODEL")
        if not self.LLM_MODEL:
            raise ValueError("LLM_MODEL not set in .env")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL")
        if not self.LLM_BASE_URL:
            raise ValueError("LLM_BASE_URL not set in .env")
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        if not self.EMBEDDING_MODEL:
            raise ValueError("EMBEDDING_MODEL not set in .env")
        self.INDEX_PATH = os.getenv("INDEX_PATH")
        if not self.INDEX_PATH:
            raise ValueError("INDEX_PATH not set in .env")
        self.DATASET_PATH = os.getenv("DATASET_PATH")
        if not self.DATASET_PATH:
            raise ValueError("DATASET_PATH not set in .env")
        self.SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
        self.CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))

        self.FALLBACK_MESSAGE = (
            "I'm sorry, I'm unable to respond right now. Please try again shortly."
        )

        # API key rotation
        self.api_keys: List[str] = []
        self.current_key_index = 0
        self._load_api_keys()
        self.api_key = self._current_key()
        self._log_config()

    def _load_api_keys(self):
        for name in ["LLM_API_KEY", "LLM_API_KEY_2", "LLM_API_KEY_3"]:
            val = os.getenv(name, "").strip()
            if val:
                self.api_keys.append(val)
        if not self.api_keys:
            logger.warning("⚠️  No API keys found in .env")

    def _current_key(self) -> str:
        if self.api_keys and self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return ""

    def rotate_key(self) -> bool:
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.api_key = self._current_key()
            logger.info(f"🔄 Rotated to API key {self.current_key_index + 1}")
            return True
        return False

    def _log_config(self):
        logger.info(f"📋 Provider: {self.LLM_PROVIDER} | Model: {self.LLM_MODEL}")
        logger.info(f"📋 Index: {self.INDEX_PATH} | Embedding: {self.EMBEDDING_MODEL}")
        logger.info(f"📋 API keys loaded: {len(self.api_keys)}")


config = Config()


# ═══════════════════════════════════════════════════════════════════════════
# Response Cache
# ═══════════════════════════════════════════════════════════════════════════
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.pkl")


class ResponseCache:
    def __init__(self):
        self.store: Dict[str, tuple] = {}
        self._load()

    def make_key(self, query: str, chunks: List[Any]) -> str:
        q_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
        ctx = ""
        if chunks:
            ctx = hashlib.md5(
                "".join(c.text for c in chunks if hasattr(c, "text")).encode()
            ).hexdigest()
        return f"{q_hash}_{ctx or 'empty'}"

    def get(self, key: str) -> Optional[str]:
        if key in self.store:
            ts, resp = self.store[key]
            if time.time() - ts < config.CACHE_TTL_HOURS * 3600:
                return resp
            del self.store[key]
        return None

    def put(self, key: str, resp: str):
        self.store[key] = (time.time(), resp)
        self._save()

    def _save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.store, f)
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def _load(self):
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "rb") as f:
                    self.store = pickle.load(f)
                logger.info(f"✅ Cache: {len(self.store)} entries loaded")
        except Exception:
            self.store = {}


cache = ResponseCache()


# ═══════════════════════════════════════════════════════════════════════════
# Conversation Logger
# ═══════════════════════════════════════════════════════════════════════════
class ConversationLogger:
    def __init__(self, path: str = "conversations.json"):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log(self, user_input: str, response: str, language: str,
            response_type: str = "text", sources: List[Dict] = None):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.append({
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "response": response,
                "language": language,
                "response_type": response_type,
                "sources": sources or [],
            })
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Log error: {e}")


conv_logger = ConversationLogger()


# ═══════════════════════════════════════════════════════════════════════════
# Vector Index Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_index():
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        if not os.path.exists(config.INDEX_PATH):
            logger.error(f"❌ Index not found: {config.INDEX_PATH}")
            return None, None

        embed = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        ctx = StorageContext.from_defaults(persist_dir=config.INDEX_PATH)
        index = VectorStoreIndex.from_documents([], storage_context=ctx, embed_model=embed)
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        logger.info("✅ Vector index loaded")
        return index, retriever
    except Exception as e:
        logger.error(f"❌ Index load failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ═══════════════════════════════════════════════════════════════════════════
# RAG System
# ═══════════════════════════════════════════════════════════════════════════
class BreastCancerRAGSystem:
    """
    Full pipeline:
      detect language → safety check → retrieve → prompt → LLM → clean → cache
    """

    def __init__(self, index, retriever):
        self.index = index
        self.retriever = retriever
        self.conversation_history: List[Dict] = []
        self.http_client = httpx.Client(timeout=60.0)

    # ── Language Detection ───────────────────────────────────────────────
    def detect_language(self, text: str) -> str:
        try:
            if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", text):
                return "urdu"
            return "urdu" if detect(text) == "ur" else "english"
        except Exception:
            return "english"

    # ── Predefined Questions ─────────────────────────────────────────────
    def get_predefined_questions(self, language: str = "english") -> List[dict]:
        en = [
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
        ]
        ur = [
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
        ]
        return ur if language == "urdu" else en

    # ── Urdu Text Cleaning ───────────────────────────────────────────────
    def _clean_urdu(self, text: str) -> str:
        if not text:
            return text
        fixes = {
            "مجہے": "مجھے", "پروگرہوں": "پروگرام", "کہےنسر": "کینسر",
            "ڈڈاکٹر": "ڈاکٹر", "ہےہ": "ہے", "مہےں": "میں",
            "ہےں": "ہیں", "ھے": "ہے", "ھوں": "ہوں", "ھیں": "ہیں",
            "ےے": "ے", "ںں": "ں", "ہہ": "ہ", "یی": "ی",
            "ے لہےے": "کے لیے", "کا ے لہےے": "کے لیے",
            "و ہےہ": "کو", "نہہےں": "نہیں", "بارے مہےں": "بارے میں",
            "کرہےں": "کریں", "بہترہےن": "بہترین",
            "برہےسٹ": "بریسٹ", "کہےموتھراپہے": "کیموتھراپی",
        }
        for wrong, right in fixes.items():
            text = text.replace(wrong, right)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"۔۔+", "۔", text)
        return text.strip()

    # ── Emotional Need Detection ─────────────────────────────────────────
    def _detect_emotional_needs(self, query: str, language: str) -> dict:
        q = query.lower()

        explicit_en = [
            "scared", "afraid", "worried", "anxious", "fear", "nervous",
            "stressed", "overwhelmed", "depressed", "sad", "lonely", "alone",
            "hopeless", "can't cope", "struggling", "suffering", "terrified",
            "panic", "crying", "devastated", "angry", "frustrated", "exhausted",
            "broken", "lost", "helpless", "despair", "grief", "give up",
            "can't sleep", "why me", "feel like a burden", "no one understands",
        ]
        explicit_ur = [
            "خوف", "ڈر", "پریشانی", "فکر", "تنہائی", "اداسی", "مایوسی",
            "تکلیف", "گھبراہٹ", "بے چینی", "بے بسی", "رونا", "آنسو",
            "دکھ", "غم", "تھکاوٹ", "ٹوٹ", "اکیلا", "اکیلی", "مشکل",
            "نیند نہیں", "بھوک نہیں", "کیوں میں", "بوجھ", "برداشت",
        ]

        implicit_en = [
            "hair loss", "bald", "body image", "husband", "wife", "partner",
            "children", "family", "tell my", "pregnant", "baby", "fertility",
            "breastfeed", "mastectomy", "die", "dying", "death", "survival rate",
            "recurrence", "stage 4", "metastatic", "chemo", "side effects",
            "pain", "nausea", "fatigue", "work", "money", "normal life",
        ]
        implicit_ur = [
            "بال", "شکل", "شوہر", "بیوی", "بچے", "خاندان",
            "حمل", "دودھ", "سرجری", "موت", "زندگی", "واپس آنا",
            "کیمو", "علاج", "درد", "متلی", "نوکری", "پیسے",
        ]

        expl = explicit_ur if language == "urdu" else explicit_en
        impl = implicit_ur if language == "urdu" else implicit_en

        e_score = sum(2 for t in expl if t in q)
        i_score = sum(1 for t in impl if t in q)

        vuln_en = ["will i", "can i still", "am i going to", "what if", "how do i cope"]
        vuln_ur = ["کیا میں", "کب تک", "کیسے", "ممکن ہے", "اگر"]
        vulns = vuln_ur if language == "urdu" else vuln_en
        v_score = sum(1 for p in vulns if p in q)

        total = e_score + i_score + v_score
        return {"needs_emotional_support": total > 0, "score": total}

    # ── Retrieval with Sources ───────────────────────────────────────────
    def _retrieve(self, query: str) -> tuple:
        if not self.retriever:
            return [], []
        try:
            results = self.retriever.retrieve(query)

            # Boost recall with simplified query
            top_score = max((r.score for r in results if hasattr(r, "score")), default=0)
            if len(results) < 3 or top_score < 0.4:
                simplified = re.sub(
                    r"\b(what|how|can|will|do|does|is|are|should|could|would|when|where|why|tell me about)\b",
                    "", query.lower(),
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

            sources, seen_topics = [], set()
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

            logger.info(f"✅ Retrieved {len(good)} chunks, {len(sources)} sources")
            return good[:5], sources
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return [], []

    # ── Prompt Building ──────────────────────────────────────────────────
    def _build_prompt(self, query: str, chunks: List[Any], language: str) -> str:
        ctx = ""
        if chunks:
            parts = []
            for i, c in enumerate(chunks[:5]):
                text = getattr(c, "text", str(c))
                meta = getattr(c, "metadata", {})
                words = " ".join(text.split()[:200])
                parts.append(
                    f"[Source {i+1}: {meta.get('topic','General')} — "
                    f"{meta.get('source','Database')}]\n{words}"
                )
            ctx = "\n\n".join(parts)

        emo = self._detect_emotional_needs(query, language)

        if language == "urdu":
            return self._urdu_prompt(query, ctx, emo)
        return self._english_prompt(query, ctx, emo)

    def _english_prompt(self, query: str, context: str, emo: dict) -> str:
        score = emo["score"]
        if score >= 3:
            guide = (
                "CRITICAL: This person is in significant emotional distress. "
                "Lead with deep empathy — at least 2-3 sentences of pure emotional "
                "support before any medical information."
            )
        elif score >= 1:
            guide = (
                "This person needs emotional support alongside information. "
                "Open with a warm, validating sentence. End on a hopeful note."
            )
        else:
            guide = "Be warm and friendly — like a caring, knowledgeable friend."

        return f"""# WELL BEING AGENT — Breast Cancer Support

## YOUR IDENTITY
You are a warm, knowledgeable breast cancer support companion. You speak in a
friendly, conversational tone while being medically accurate. You are NOT a
doctor — always encourage consulting their healthcare team.

## PATIENT'S QUESTION
"{query}"

## EMOTIONAL GUIDANCE
{guide}

## RETRIEVED MEDICAL CONTEXT
{context or "No specific retrieval — use your general breast cancer knowledge."}

## RESPONSE RULES
1. Use retrieved context as PRIMARY source. Supplement with your own knowledge.
   Never say "I don't have information."
2. Write like you're talking to someone you care about.
3. Give specific, actionable information — exercises, foods, timelines, medications.
4. Validate feelings naturally and authentically.
5. Cite sources conversationally: "According to the American Cancer Society..."
6. End with genuine warmth and encouragement. 💛
7. Length: 5-10 sentences.
8. NEVER suggest stopping treatment or give false hope about outcomes.

## YOUR RESPONSE:"""

    def _urdu_prompt(self, query: str, context: str, emo: dict) -> str:
        score = emo["score"]
        if score >= 3:
            guide = (
                "اہم: یہ شخص بہت زیادہ جذباتی تکلیف میں ہے۔ "
                "پہلے 2-3 جملے صرف ہمدردی اور تسلی کے ہوں۔"
            )
        elif score >= 1:
            guide = "مریض کو جذباتی مدد کی ضرورت ہے۔ گرمجوشی سے شروع کریں۔"
        else:
            guide = "دوستانہ اور گرمجوش انداز۔"

        return f"""# ویل بینگ ایجنٹ — بریسٹ کینسر سپورٹ

## آپ کا کردار
آپ ایک شفیق سہیلی ہیں جو بریسٹ کینسر کے بارے میں گہری معلومات رکھتی ہیں۔
آپ ڈاکٹر نہیں ہیں — ہمیشہ طبی ٹیم سے مشورے کی تاکید کریں۔

## مریض کا سوال
"{query}"

## جذباتی رہنمائی
{guide}

## طبی سیاق و سباق
{context or "عمومی بریسٹ کینسر کی معلومات سے جواب دیں۔"}

## جواب کے اصول
1. سیاق و سباق بنیادی ذریعہ ہے۔ ضرورت ہو تو اپنی معلومات سے اضافہ کریں۔
2. دوستانہ لہجہ — جیسے کسی عزیز سے بات کر رہے ہوں۔
3. مخصوص اور عملی معلومات دیں۔
4. جذبات کی تصدیق کریں۔
5. آخر میں گرمجوشی۔ 💛
6. 5-10 جملے۔ صرف اردو — درست ہجے۔

## ہجوں کے اصول
✅ "مجھے" ❌ "مجہے" | ✅ "کینسر" ❌ "کہےنسر" | ✅ "ڈاکٹر" ❌ "ڈڈاکٹر"
✅ "ہے" ❌ "ہےہ" | ✅ "میں" ❌ "مہےں" | ✅ "کے لیے" ❌ "کا ے لہےے"

## آپ کا اردو جواب:"""

    # ── LLM Query ────────────────────────────────────────────────────────
    def _query_llm(self, prompt: str, language: str, retries: int = 3) -> str:
        if not config.api_key:
            logger.error("No API key available")
            return config.FALLBACK_MESSAGE

        sys_msg = (
            "آپ ایک شفیق بریسٹ کینسر سپورٹ ساتھی ہیں۔ صرف درست اردو میں جواب دیں۔ "
            "طبی معلومات کے ساتھ جذباتی مدد شامل کریں۔ ہمیشہ طبی ٹیم سے مشورے کی تاکید کریں۔"
            if language == "urdu"
            else
            "You are a warm, caring breast cancer support companion. Combine medical "
            "accuracy with genuine emotional warmth. Use conversational language. "
            "Always recommend discussing with their healthcare team. Never suggest "
            "stopping treatment or give false medical promises."
        )

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
        }

        for attempt in range(retries):
            try:
                url = f"{config.LLM_BASE_URL}/chat/completions"
                resp = self.http_client.post(url, headers=headers, json=payload)

                if resp.status_code == 429 or resp.status_code >= 500:
                    logger.warning(f"API returned {resp.status_code}, attempt {attempt + 1}")
                    if config.rotate_key():
                        headers["Authorization"] = f"Bearer {config.api_key}"
                    time.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                logger.info("✅ LLM response received")
                return text

            except httpx.TimeoutException:
                logger.warning(f"LLM timeout (attempt {attempt + 1}/{retries})")
                time.sleep(2 ** attempt)
            except (KeyError, IndexError) as e:
                logger.error(f"Unexpected API response format: {e}")
                return config.FALLBACK_MESSAGE
            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")
                if config.rotate_key():
                    headers["Authorization"] = f"Bearer {config.api_key}"
                time.sleep(2 ** attempt)

        return config.FALLBACK_MESSAGE

    # ── Post-Processing ──────────────────────────────────────────────────
    def _format(self, text: str, language: str) -> str:
        text = text.strip()
        if not text:
            text = config.FALLBACK_MESSAGE
        text = SafetyValidator.validate_response(text, language)
        text = SafetyValidator.add_medical_disclaimer(text, language)
        return text

    # ── Main Pipeline ────────────────────────────────────────────────────
    def get_enhanced_answer(self, user_query: str, language: str = None,
                            response_type: str = "text") -> str:
        if language is None:
            language = self.detect_language(user_query)

        safety = SafetyValidator.validate_query(user_query, language)
        if safety["is_crisis"]:
            return safety["response"]
        if not safety["is_on_topic"]:
            return safety["response"]

        chunks, sources = self._retrieve(user_query)

        ck = cache.make_key(user_query, chunks)
        cached = cache.get(ck)
        if cached:
            conv_logger.log(user_query, cached, language, response_type, sources)
            return cached

        prompt = self._build_prompt(user_query, chunks, language)
        raw = self._query_llm(prompt, language)
        final = self._format(raw, language)

        cache.put(ck, final)
        conv_logger.log(user_query, final, language, response_type, sources)
        self.conversation_history.append({
            "query": user_query, "answer": final,
            "language": language, "sources": sources, "timestamp": time.time(),
        })
        return final

    def get_enhanced_answer_with_sources(self, user_query: str, language: str = None,
                                         response_type: str = "text") -> Dict:
        if language is None:
            language = self.detect_language(user_query)

        safety = SafetyValidator.validate_query(user_query, language)
        if safety["is_crisis"]:
            return {"answer": safety["response"], "sources": [], "language": language}
        if not safety["is_on_topic"]:
            return {"answer": safety["response"], "sources": [], "language": language}

        chunks, sources = self._retrieve(user_query)

        ck = cache.make_key(user_query, chunks)
        cached = cache.get(ck)
        if cached:
            return {"answer": cached, "sources": sources, "language": language}

        prompt = self._build_prompt(user_query, chunks, language)
        raw = self._query_llm(prompt, language)
        final = self._format(raw, language)

        cache.put(ck, final)
        conv_logger.log(user_query, final, language, response_type, sources)
        self.conversation_history.append({
            "query": user_query, "answer": final,
            "language": language, "sources": sources, "timestamp": time.time(),
        })
        return {"answer": final, "sources": sources, "language": language}


# ═══════════════════════════════════════════════════════════════════════════
# Module-level initialization (runs on import)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("🚀 Starting Well Being Agent…")
_t = time.time()
index, retriever = load_index()
logger.info(f"✅ System ready in {time.time() - _t:.1f}s")

rag_system = BreastCancerRAGSystem(index, retriever)


# ── CLI mode ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("💬 Well Being Agent — Breast Cancer Support")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    while True:
        q = input("❓ Your question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        t0 = time.time()
        r = rag_system.get_enhanced_answer_with_sources(q)
        print(f"\n💡 {r['answer']}")
        if r["sources"]:
            print("📚 Sources:", ", ".join(s["topic"] for s in r["sources"][:3]))
        print(f"⏱️  {time.time() - t0:.1f}s\n")

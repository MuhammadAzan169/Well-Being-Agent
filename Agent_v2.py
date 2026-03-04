# Agent.py - Refactored RAG-based Breast Cancer Support Agent
# Clean architecture with safety validation, citations, and empathetic responses
import os
import json
import time
import pickle
import hashlib
import random
import re
import logging
from typing import List, Optional, Any, Dict
from dotenv import load_dotenv
from datetime import datetime

# === Language Detection ===
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# === OpenAI Client ===
from openai import OpenAI

# === Safety Module ===
from safety import SafetyValidator

# === Environment Detection ===
IS_HUGGING_FACE = os.path.exists('/.dockerenv') or 'SPACE_ID' in os.environ
if IS_HUGGING_FACE:
    print("🚀 Hugging Face Space detected")
    os.environ['FORCE_FREE_MODEL'] = 'true'

# Load environment variables
if not IS_HUGGING_FACE:
    load_dotenv()  # Load from .env in current directory
    print("✅ .env file loaded")
else:
    print("✅ Hugging Face environment - using repository secrets")

# === Cache Configuration ===
CACHE_DIR = "cache"
RESPONSE_CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.pkl")


class ResponseCache:
    """Simple response cache to avoid repeated LLM calls."""

    def __init__(self):
        self.cache = {}
        self.load_cache()

    def get_cache_key(self, query: str, context_chunks: List[Any]) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if context_chunks:
            context_text = "".join(
                [chunk.text for chunk in context_chunks if hasattr(chunk, 'text')]
            )
            context_hash = hashlib.md5(context_text.encode()).hexdigest()
        else:
            context_hash = "no_context"
        return f"{query_hash}_{context_hash}"

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            cached_time, response = self.cache[key]
            if time.time() - cached_time < 24 * 3600:
                return response
            else:
                del self.cache[key]
        return None

    def set(self, key: str, response: str):
        self.cache[key] = (time.time(), response)
        self.save_cache()

    def save_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(RESPONSE_CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")

    def load_cache(self):
        try:
            if os.path.exists(RESPONSE_CACHE_FILE):
                with open(RESPONSE_CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✅ Loaded {len(self.cache)} cached responses")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
            self.cache = {}


response_cache = ResponseCache()


# === Conversation Logger ===
class ConversationLogger:
    """JSON-based conversation logging."""

    def __init__(self, log_file="conversations.json"):
        self.log_file = log_file
        self.ensure_log_file()

    def ensure_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)

    def log_conversation(self, user_input: str, llm_response: str, language: str,
                         response_type: str, sources: List[Dict] = None):
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)

            entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "llm_response": llm_response,
                "language": language,
                "response_type": response_type,
                "sources": sources or [],
            }
            conversations.append(entry)

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Logging error: {e}")


conversation_logger = ConversationLogger()


# === Configuration ===
class Config:
    """Centralized configuration from config/config.json."""

    def __init__(self):
        self.api_keys = []
        self.current_key_index = 0
        self.settings = self._load_config_file()
        self._validate_and_correct_paths()

        self.SUPPORTED_LANGUAGES = ["english", "urdu"]
        self.DEFAULT_LANGUAGE = "english"
        self.MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")  # fallback for compatibility
        self.MODEL_ID = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")  # fallback
        self.API_KEYS_FOLDER = self.settings["api_keys_folder"]
        self.INDEX_PATH = self.settings["index_path"]
        self.DATASET_PATH = self.settings["dataset_path"]
        self.SIMILARITY_TOP_K = self.settings.get("similarity_top_k", 5)
        self.TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", self.settings.get("temperature", 0.2)))
        self.MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", self.settings.get("max_tokens", 500)))
        self.FALLBACK_MESSAGE = self.settings["fallback_message"]
        self.EMBEDDING_MODEL = self.settings.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.api_keys = self._load_api_keys()
        self.api_key = self._get_current_api_key()
        self._validate_config()

    def _load_config_file(self):
        config_file = os.path.join("config", "config.json")
        default_config = {
            "api_keys_folder": "config",
            "index_path": "cancer_index_store",
            "dataset_path": "breast_cancer_comprehensive.json",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_top_k": 5,
            "temperature": 0.2,
            "max_tokens": 500,
            "fallback_message": "Sorry, I don't know the answer.",
        }
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                return {**default_config, **loaded_config}
            else:
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            print(f"❌ Config error: {e}")
            return default_config

    def _validate_and_correct_paths(self):
        original = self.settings.get("dataset_path", "")
        if not os.path.exists(original):
            for path in [f"DataSet/{original}", "DataSet/breast_cancer_comprehensive.json",
                         "DataSet/breast_cancer.json", original]:
                if os.path.exists(path):
                    self.settings["dataset_path"] = path
                    return

    def _load_api_keys(self) -> List[str]:
        api_keys = []
        key_value = os.getenv("LLM_API_KEY")
        if key_value and key_value.strip():
            api_keys.append(key_value.strip())
            print("✅ Found LLM_API_KEY")
        return api_keys

    def _get_current_api_key(self) -> str:
        if self.api_keys and self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return ""

    def rotate_to_next_key(self) -> bool:
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.api_key = self._get_current_api_key()
            print(f"🔄 Rotated to API key {self.current_key_index + 1}")
            return True
        return False

    def _validate_config(self):
        if not self.api_keys:
            print("❌ No API keys found")
        else:
            print(f"✅ Found {len(self.api_keys)} API key(s)")
        print(f"📋 Model: {self.MODEL_ID} | Index: {self.INDEX_PATH}")


config = Config()

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] + (
        [logging.FileHandler('rag_system.log')] if not IS_HUGGING_FACE else []
    ),
)


# === Index Loading ===
def load_index():
    """Load persisted vector index with consistent embedding model."""
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        print(f"🔍 Loading index from: {config.INDEX_PATH}")
        if not os.path.exists(config.INDEX_PATH):
            print(f"❌ Index path doesn't exist: {config.INDEX_PATH}")
            return None, None

        # Use the SAME embedding model as the indexing pipeline
        embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=config.INDEX_PATH)
        index = VectorStoreIndex.from_documents(
            [], storage_context=storage_context, embed_model=embed_model
        )
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        print("✅ Index loaded successfully")
        return index, retriever
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# === Enhanced RAG System ===
class BreastCancerRAGSystem:
    """
    RAG system for breast cancer support with:
    - Safety validation & guardrails
    - Source citations in responses
    - Empathetic, medically-grounded answers
    - Emotional need detection
    """

    def __init__(self, index, retriever):
        self.index = index
        self.retriever = retriever
        self.conversation_history = []
        if not config.api_keys:
            logging.error("🚫 No API key — LLM features will not work")

    # --- Predefined Questions ---
    def get_predefined_questions(self, language: str = "english") -> List[dict]:
        english_questions = [
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
        urdu_questions = [
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
        return urdu_questions if language == "urdu" else english_questions

    # --- Language Detection ---
    def detect_language(self, text: str) -> str:
        try:
            urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
            if urdu_pattern.search(text):
                return 'urdu'
            detected_lang = detect(text)
            return 'urdu' if detected_lang == 'ur' else 'english'
        except Exception:
            return 'english'

    # --- Urdu Text Cleaning ---
    def _clean_urdu_text(self, text: str) -> str:
        if not text or not text.strip():
            return text
        corrections = {
            'مجہے': 'مجھے', 'پروگرہوں': 'پروگرام', 'کہےنسر': 'کینسر',
            'ڈڈاکٹر': 'ڈاکٹر', 'ہےہ': 'ہے', 'مہےں': 'میں',
            'ہےں': 'ہیں', 'ھے': 'ہے', 'ھوں': 'ہوں', 'ھیں': 'ہیں',
            'ےے': 'ے', 'ںں': 'ں', 'ہہ': 'ہ', 'یی': 'ی',
            'ے لہےے': 'کے لیے', 'کا ے لہےے': 'کے لیے',
            'و ہےہ': 'کو', 'نہہےں': 'نہیں', 'بارے مہےں': 'بارے میں',
            'کرہےں': 'کریں', 'بہترہےن': 'بہترین',
            'برہےسٹ': 'بریسٹ', 'کہےموتھراپہے': 'کیموتھراپی',
        }
        cleaned = text
        for wrong, correct in corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'۔۔', '۔', cleaned)
        return cleaned.strip()

    # --- Emotional Need Detection ---
    def _detect_emotional_needs(self, query: str, language: str = "english") -> dict:
        query_lower = query.lower()

        # === Explicit emotional keywords (direct expressions of distress) ===
        explicit_en = [
            "scared", "afraid", "worried", "anxious", "fear", "nervous", "stressed",
            "overwhelmed", "depressed", "sad", "lonely", "alone", "hopeless",
            "can't cope", "struggling", "difficult", "hard time", "suffering",
            "terrified", "panic", "crying", "misery", "heartbroken", "devastated",
            "angry", "frustrated", "exhausted", "broken", "lost", "helpless",
            "numb", "despair", "grief", "mourn", "hate my body", "ugly",
            "don't want to", "give up", "can't sleep", "can't eat", "can't stop",
            "why me", "not fair", "feel like a burden", "no one understands",
        ]
        explicit_ur = [
            "خوف", "ڈر", "پریشانی", "فکر", "تنہائی", "اداسی", "مایوسی",
            "تکلیف", "گھبراہٹ", "بے چینی", "بے بسی", "رونا", "آنسو",
            "دکھ", "غم", "ہمت", "طاقت", "حوصلہ", "تھکاوٹ", "ٹوٹ",
            "اکیلا", "اکیلی", "نیند نہیں", "بھوک نہیں", "کیوں میں",
            "بوجھ", "برداشت", "صبر", "آزمائش", "مشکل",
        ]

        # === Implicit emotional patterns (situations that imply emotional need) ===
        implicit_en = [
            "hair loss", "hair fall", "bald", "losing my hair", "hair grow back",
            "body image", "look different", "mirror", "ugly", "attractive",
            "husband", "wife", "partner", "marriage", "relationship", "intimacy",
            "children", "kids", "family", "tell my", "how to tell", "break the news",
            "pregnant", "pregnancy", "baby", "fertility", "can i have",
            "breastfeed", "breast removal", "mastectomy", "losing a breast",
            "die", "dying", "death", "survive", "survival rate", "prognosis",
            "recurrence", "come back", "stage 4", "stage iv", "metastatic",
            "chemo", "chemotherapy", "first treatment", "next treatment",
            "side effects", "pain", "nausea", "vomiting", "tired", "fatigue",
            "work", "job", "career", "money", "afford", "insurance", "financial",
            "normal life", "new normal", "will i ever", "go back to",
        ]
        implicit_ur = [
            "بال", "گنجا", "بال گرنا", "بال واپس", "شکل", "خوبصورتی",
            "شوہر", "بیوی", "رشتہ", "بچے", "خاندان", "کیسے بتاؤں",
            "حمل", "بچہ", "دودھ", "ماں", "سرجری", "چھاتی",
            "موت", "مرنا", "زندگی", "بچنا", "واپس آنا", "مرحلہ",
            "کیمو", "علاج", "درد", "متلی", "تھکاوٹ",
            "نوکری", "پیسے", "خرچہ", "عام زندگی",
        ]

        if language == "urdu":
            explicit_triggers = explicit_ur
            implicit_triggers = implicit_ur
        else:
            explicit_triggers = explicit_en
            implicit_triggers = implicit_en

        # Score: explicit emotions count double
        explicit_score = sum(2 for t in explicit_triggers if t in query_lower)
        implicit_score = sum(1 for t in implicit_triggers if t in query_lower)

        # Question-mark patterns that suggest vulnerability
        vulnerability_phrases_en = [
            "will i", "can i still", "am i going to", "is it possible",
            "what if", "how long", "how do i cope", "how do i deal",
        ]
        vulnerability_phrases_ur = [
            "کیا میں", "کب تک", "کیسے", "ممکن ہے", "اگر",
        ]
        vuln_phrases = vulnerability_phrases_ur if language == "urdu" else vulnerability_phrases_en
        vuln_score = sum(1 for p in vuln_phrases if p in query_lower)

        total_score = explicit_score + implicit_score + vuln_score

        return {
            "needs_emotional_support": total_score > 0,
            "emotional_score": total_score,
            "explicit_distress": explicit_score > 0,
            "implicit_concern": implicit_score > 0,
            "vulnerability": vuln_score > 0,
        }

    # --- Retrieval with Source Extraction ---
    def retrieve_with_sources(self, query: str, language: str = "english") -> tuple:
        """
        Retrieve relevant chunks and extract structured source citations.
        Uses the original query + a simplified version for better recall.
        Returns: (chunks, sources_list)
        """
        if not self.retriever:
            return [], []

        try:
            results = self.retriever.retrieve(query)

            # If low results, also try a simplified query for better recall
            if len(results) < 3 or (results and max(r.score for r in results if hasattr(r, 'score')) < 0.4):
                # Strip question words for a keyword-focused retrieval
                simplified = re.sub(
                    r'\b(what|how|can|will|do|does|is|are|should|could|would|when|where|why|tell me about|i want to know)\b',
                    '', query.lower()
                ).strip()
                if simplified and len(simplified) > 10:
                    extra_results = self.retriever.retrieve(simplified)
                    # Merge, avoiding duplicates by text hash
                    seen = {hash(r.text[:100]) for r in results if hasattr(r, 'text')}
                    for r in extra_results:
                        if hasattr(r, 'text') and hash(r.text[:100]) not in seen:
                            results.append(r)
                            seen.add(hash(r.text[:100]))

            # Filter by quality threshold — 0.2 is generous to avoid missing relevant content
            quality_threshold = 0.2
            high_quality = [
                r for r in results
                if hasattr(r, 'score') and r.score >= quality_threshold
            ]

            if not high_quality and results:
                high_quality = results[:3]

            # Sort by score descending
            high_quality.sort(key=lambda r: r.score if hasattr(r, 'score') else 0, reverse=True)

            # Extract source citations
            sources = []
            seen_topics = set()
            for r in high_quality:
                meta = r.metadata if hasattr(r, 'metadata') else {}
                topic = meta.get('topic', 'General')
                if topic in seen_topics:
                    continue
                seen_topics.add(topic)
                sources.append({
                    "topic": topic,
                    "category": meta.get('category', ''),
                    "source": meta.get('source', ''),
                    "score": round(r.score, 3) if hasattr(r, 'score') else 0,
                    "tags": meta.get('tags', ''),
                })

            print(f"✅ Retrieved {len(high_quality)} chunks, {len(sources)} unique sources")
            return high_quality[:5], sources

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return [], []

    # --- Prompt Building ---
    def build_prompt(self, query: str, chunks: List[Any],
                     language: str = "english") -> str:
        """Build a focused, empathetic prompt with RAG context and citation instructions."""

        # Build context from retrieved chunks — use up to 5 for richer grounding
        context_text = ""
        if chunks:
            parts = []
            for i, chunk in enumerate(chunks[:5]):
                text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
                source = meta.get('source', 'Database')
                topic = meta.get('topic', 'General')
                # Use up to 200 words per chunk for thorough context
                words = text.split()[:200]
                parts.append(f"[Source {i+1}: {topic} — {source}]\n{' '.join(words)}")
            context_text = "\n\n".join(parts)

        emotional = self._detect_emotional_needs(query, language)

        if language == "urdu":
            return self._build_urdu_prompt(query, context_text, emotional)
        return self._build_english_prompt(query, context_text, emotional)

    def _build_english_prompt(self, query: str, context: str, emotional: dict) -> str:
        emotional_score = emotional.get('emotional_score', 0)
        # Determine emotional guidance level
        if emotional_score >= 3:
            emotional_instruction = (
                "CRITICAL: This person is in significant emotional distress. "
                "Lead with deep empathy and validation FIRST — at least 2-3 sentences of pure emotional support "
                "before any medical information. Use phrases like 'I hear you', 'What you're feeling is completely valid', "
                "'You are not alone in this'. Make them feel genuinely seen and cared for."
            )
        elif emotional_score >= 1:
            emotional_instruction = (
                "This person needs emotional support alongside information. "
                "Open with a warm, validating sentence that acknowledges their feelings. "
                "Weave encouragement naturally throughout your response. End on a hopeful note."
            )
        else:
            emotional_instruction = (
                "Be warm and friendly — like a caring, knowledgeable friend. "
                "Start by acknowledging their question warmly. Keep the tone supportive throughout."
            )

        return f"""# WELL BEING AGENT — Breast Cancer Support

## YOUR IDENTITY
You are a warm, knowledgeable breast cancer support companion — think of yourself as a caring friend 
who also happens to have deep medical knowledge. You speak in a friendly, conversational tone while 
being medically accurate. You are NOT a doctor — you always encourage consulting their healthcare team.

## PATIENT'S QUESTION
"{query}"

## EMOTIONAL GUIDANCE
{emotional_instruction}

## RETRIEVED MEDICAL CONTEXT (ground your answer in this when relevant)
{context if context else "No specific retrieval results — use your general breast cancer knowledge to give a helpful, accurate answer."}

## HOW TO RESPOND — CRITICAL RULES
1. **Blend RAG + Knowledge**: Use the retrieved context above as your PRIMARY source. If the context doesn't 
   fully answer the question, SUPPLEMENT with your own medical knowledge about breast cancer. Never say 
   "I don't have information" — between the context and your knowledge, provide a thorough answer.
2. **Be a caring friend**: Write like you're talking to someone you care about. Use "you" and "your" naturally.
   Avoid clinical/robotic language. Say "That's a really great question" not "Your query has been received."
3. **Be specific and actionable**: Give real, concrete information — specific exercises, specific foods, 
   actual statistics, named medications, clear timelines. No vague "talk to your doctor" as the whole answer.
4. **Validate feelings naturally**: If someone is scared, don't just say "I understand." Say something like 
   "It's completely natural to feel scared right now — so many people in your position feel the same way, 
   and that takes nothing away from how strong you are."
5. **Cite sources conversationally**: Say "Research from the American Cancer Society shows..." or 
   "According to Mayo Clinic..." — not "[Source 1] states."
6. **End with genuine warmth**: Close with something that makes them feel supported — not a generic disclaimer. 
   Something like "You're doing an amazing job taking care of yourself by asking these questions. 
   Your medical team can help tailor this to your specific situation. 💛"
7. **Length**: 5-10 sentences. Thorough enough to be truly helpful, warm enough to feel like a hug.
8. **NEVER**: Suggest stopping treatment, replace medical advice, or give false hope about outcomes.

## YOUR WARM, HELPFUL RESPONSE:"""

    def _build_urdu_prompt(self, query: str, context: str, emotional: dict) -> str:
        emotional_score = emotional.get('emotional_score', 0)
        if emotional_score >= 3:
            emotional_instruction = (
                "اہم: یہ شخص بہت زیادہ جذباتی تکلیف میں ہے۔ پہلے 2-3 جملے صرف ہمدردی اور تسلی کے ہوں۔ "
                "'میں آپ کے ساتھ ہوں'، 'آپ کے جذبات بالکل فطری ہیں'، 'آپ اکیلے نہیں ہیں' جیسے الفاظ استعمال کریں۔"
            )
        elif emotional_score >= 1:
            emotional_instruction = (
                "مریض کو جذباتی مدد کی ضرورت ہے۔ گرمجوشی سے شروع کریں، حوصلہ افزائی شامل کریں، امید بخش انداز میں ختم کریں۔"
            )
        else:
            emotional_instruction = (
                "دوستانہ اور گرمجوش انداز — جیسے ایک خیال رکھنے والی سہیلی بات کر رہی ہو۔"
            )

        return f"""# ویل بینگ ایجنٹ — بریسٹ کینسر سپورٹ

## آپ کا کردار
آپ ایک شفیق، جاننے والی سہیلی ہیں جو بریسٹ کینسر کے بارے میں گہری معلومات رکھتی ہیں۔ 
آپ کا لہجہ دوستانہ، محبت بھرا اور حوصلہ افزا ہے۔ آپ ڈاکٹر نہیں ہیں — ہمیشہ طبی ٹیم سے مشورے کی تاکید کریں۔

## مریض کا سوال
"{query}"

## جذباتی رہنمائی
{emotional_instruction}

## طبی سیاق و سباق (اپنے جواب کی بنیاد اس پر رکھیں)
{context if context else "کوئی مخصوص سیاق و سباق دستیاب نہیں — اپنی عمومی بریسٹ کینسر کی معلومات سے مکمل جواب دیں۔"}

## جواب کے لازمی اصول
1. **RAG + علم ملائیں**: اوپر والا سیاق و سباق بنیادی ذریعہ ہے۔ اگر مکمل جواب نہ ملے تو اپنی معلومات سے اضافہ کریں۔ "مجھے معلومات نہیں" کبھی نہ کہیں۔
2. **دوستانہ لہجہ**: ایسے لکھیں جیسے کسی عزیز سے بات کر رہے ہوں۔ "آپ" اور "آپ کا" قدرتی طور پر استعمال کریں۔
3. **مخصوص اور عملی**: حقیقی معلومات دیں — مخصوص ورزشیں، غذائیں، اعداد و شمار، واضح وقت کی حد۔
4. **جذبات کی تصدیق**: اگر خوف ہے تو کہیں "خوف محسوس کرنا بالکل فطری ہے — آپ بہادری سے اس کا سامنا کر رہے ہیں۔"
5. **آخر میں حقیقی گرمجوشی**: ایسا اختتام جو محسوس ہو — عمومی فقرہ نہیں بلکہ دل سے لکھا ہوا۔ 💛
6. **5-10 جملے**: مکمل، مفید اور محبت بھرا جواب۔
7. صرف اردو — درست ہجے اور قواعد۔

## درست ہجوں کے اصول
✅ "مجھے" ❌ "مجہے" | ✅ "کینسر" ❌ "کہےنسر" | ✅ "ڈاکٹر" ❌ "ڈڈاکٹر"
✅ "ہے" ❌ "ہےہ" | ✅ "میں" ❌ "مہےں" | ✅ "کے لیے" ❌ "کا ے لہےے"

## آپ کا گرمجوش، مکمل اردو جواب:"""

    # --- LLM Query ---
    def query_llm(self, prompt: str, language: str = "english", max_retries: int = 3) -> str:
        if not config.api_key:
            return config.FALLBACK_MESSAGE

        if language == "urdu":
            system_msg = (
                "آپ ایک شفیق اور جاننے والی بریسٹ کینسر سپورٹ ساتھی ہیں۔ "
                "آپ دوستانہ، محبت بھرے اور حوصلہ افزا انداز میں بات کرتی ہیں۔ "
                "صرف درست اردو میں جواب دیں — مکمل جملے، درست ہجے۔ "
                "طبی معلومات دیتے ہوئے ہمیشہ جذباتی مدد بھی شامل کریں۔ "
                "اگر مریض پریشان ہو تو پہلے تسلی دیں پھر معلومات دیں۔ "
                "ہمیشہ طبی ٹیم سے مشورے کی تاکید کریں۔"
            )
        else:
            system_msg = (
                "You are a warm, caring breast cancer support companion — like a knowledgeable best friend. "
                "You combine medical accuracy with genuine emotional warmth. "
                "Use conversational, friendly language — never robotic or clinical. "
                "When someone is scared or emotional, lead with empathy and validation before information. "
                "Use both the provided context AND your medical knowledge to give thorough, helpful answers. "
                "Always recommend discussing with their healthcare team, but never make that the entire answer. "
                "Never suggest stopping treatment or give false medical promises."
            )

        for attempt in range(max_retries):
            try:
                client = OpenAI(
                    base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                    api_key=config.api_key,
                )

                temperature = 0.35 if language == "urdu" else 0.45
                max_tokens = 700 if language == "urdu" else 650

                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://huggingface.co",
                        "X-Title": "Well Being Agent",
                    },
                    model=config.MODEL_ID,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                response = completion.choices[0].message.content
                print("✅ LLM response received")
                return response

            except Exception as e:
                print(f"❌ LLM request failed (attempt {attempt + 1}): {e}")
                if "429" in str(e):
                    time.sleep(2 ** attempt)
                    continue
                elif "401" in str(e) or "402" in str(e):
                    if config.rotate_to_next_key():
                        continue
                    return config.FALLBACK_MESSAGE
                if attempt == max_retries - 1:
                    if config.rotate_to_next_key():
                        return self.query_llm(prompt, language, max_retries)
                    return config.FALLBACK_MESSAGE
                time.sleep(1)

        return config.FALLBACK_MESSAGE

    # --- Post-Processing ---
    def format_response(self, response: str, language: str = "english") -> str:
        """Clean, validate, and add disclaimers to the response."""
        cleaned = response.strip()

        # Urdu cleaning
        if language == "urdu":
            cleaned = self._clean_urdu_text(cleaned)

        # Safety validation
        cleaned = SafetyValidator.validate_response(cleaned, language)

        # Add medical disclaimer
        cleaned = SafetyValidator.add_medical_disclaimer(cleaned, language)

        return cleaned

    # --- Main Entry Point ---
    def get_enhanced_answer(self, user_query: str, language: str = None,
                            response_type: str = "text") -> str:
        """
        Main RAG pipeline:
        1. Safety check → 2. Retrieve → 3. Build prompt → 4. LLM → 5. Validate → 6. Return
        """
        print(f"🔍 Processing: '{user_query[:60]}...' (type={response_type})")

        # Step 0: Detect language
        if language is None:
            language = self.detect_language(user_query)
            print(f"🌐 Detected language: {language}")

        # Step 1: Safety validation
        safety_result = SafetyValidator.validate_query(user_query, language)
        if safety_result["is_crisis"]:
            print("⚠️ Crisis response triggered")
            return safety_result["response"]
        if not safety_result["is_on_topic"]:
            print("📋 Off-topic response triggered")
            return safety_result["response"]

        # Step 2: Retrieve with sources
        chunks, sources = self.retrieve_with_sources(user_query, language)

        # Step 3: Check cache
        cache_key = response_cache.get_cache_key(user_query, chunks)
        cached = response_cache.get(cache_key)
        if cached:
            print("✅ Using cached response")
            # Log even cached responses
            conversation_logger.log_conversation(
                user_query, cached, language, response_type, sources
            )
            return cached

        # Step 4: Build prompt and query LLM
        prompt = self.build_prompt(user_query, chunks, language)
        raw_answer = self.query_llm(prompt, language)

        # Step 5: Format and validate
        final_answer = self.format_response(raw_answer, language)

        # Step 6: Cache and log
        response_cache.set(cache_key, final_answer)

        conversation_logger.log_conversation(
            user_query, final_answer, language, response_type, sources
        )

        self.conversation_history.append({
            "query": user_query,
            "answer": final_answer,
            "language": language,
            "sources": sources,
            "timestamp": time.time(),
        })

        return final_answer

    def get_enhanced_answer_with_sources(self, user_query: str, language: str = None,
                                         response_type: str = "text") -> Dict:
        """
        Same as get_enhanced_answer but returns sources alongside the answer.
        Used by the API to pass citations to the frontend.
        """
        print(f"🔍 Processing with sources: '{user_query[:60]}...'")

        if language is None:
            language = self.detect_language(user_query)

        # Safety check
        safety_result = SafetyValidator.validate_query(user_query, language)
        if safety_result["is_crisis"]:
            return {"answer": safety_result["response"], "sources": [], "language": language}
        if not safety_result["is_on_topic"]:
            return {"answer": safety_result["response"], "sources": [], "language": language}

        # Retrieve
        chunks, sources = self.retrieve_with_sources(user_query, language)

        # Cache check
        cache_key = response_cache.get_cache_key(user_query, chunks)
        cached = response_cache.get(cache_key)
        if cached:
            return {"answer": cached, "sources": sources, "language": language}

        # Build and query
        prompt = self.build_prompt(user_query, chunks, language)
        raw_answer = self.query_llm(prompt, language)
        final_answer = self.format_response(raw_answer, language)

        # Cache
        response_cache.set(cache_key, final_answer)

        conversation_logger.log_conversation(
            user_query, final_answer, language, response_type, sources
        )

        self.conversation_history.append({
            "query": user_query,
            "answer": final_answer,
            "language": language,
            "sources": sources,
            "timestamp": time.time(),
        })

        return {"answer": final_answer, "sources": sources, "language": language}


# === Module-level Initialization ===
print("🚀 Starting Well Being Agent...")
_start_time = time.time()

print("🔄 Loading vector index...")
index, retriever = load_index()

_load_time = time.time() - _start_time
print(f"✅ System ready in {_load_time:.2f} seconds")

rag_system = BreastCancerRAGSystem(index, retriever)


# === Interactive CLI ===
def interactive_chat():
    print("💬 Well Being Agent - Breast Cancer Support")
    print("=" * 50)
    print("Type 'quit' to exit")
    print("=" * 50)

    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue

        print("🤔 Thinking...")
        start = time.time()
        result = rag_system.get_enhanced_answer_with_sources(user_input)
        elapsed = time.time() - start

        print(f"\n💡 {result['answer']}")
        if result['sources']:
            print(f"\n📚 Sources:")
            for s in result['sources'][:3]:
                print(f"   • {s['topic']} ({s['source']}) — relevance: {s['score']}")
        print(f"⏱️  {elapsed:.2f}s")


def main():
    print("🏥 Well Being Agent - Breast Cancer Support System")
    print(f"📋 Model: {config.MODEL_ID} | Index: {config.INDEX_PATH}")

    if not config.api_keys:
        print("❌ API keys not configured.")
        return

    interactive_chat()


if __name__ == "__main__":
    main()

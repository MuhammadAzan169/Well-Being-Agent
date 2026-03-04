import os
import json
import requests
import time
import pickle
import hashlib
import random
import re
from typing import List, Optional, Any
import logging
from dotenv import load_dotenv
from datetime import datetime

# === LANGUAGE DETECTION IMPORTS ===
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
# ==================================

# === OPENAI CLIENT IMPORT ===
from openai import OpenAI
# ============================

# === HUGGING FACE DETECTION ===
IS_HUGGING_FACE = os.path.exists('/.dockerenv') or 'SPACE_ID' in os.environ

if IS_HUGGING_FACE:
    print("🚀 Hugging Face Space detected")
    os.environ['FORCE_FREE_MODEL'] = 'true'

# Load environment variables
if not IS_HUGGING_FACE:
<<<<<<< HEAD
    load_dotenv()  # Load from .env in current directory
    print("✅ .env file loaded successfully")
=======
    env_path = ".env"
    print(f"🔍 Looking for .env file at: {env_path}")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("✅ .env file loaded successfully")
    else:
        print(f"❌ .env file not found at: {env_path}")
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
else:
    print("✅ Hugging Face environment - using repository secrets")

# === CACHE CONFIGURATION ===
CACHE_DIR = "cache"
RESPONSE_CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.pkl")

class ResponseCache:
    """Simple response cache to avoid repeated LLM calls"""
    
    def __init__(self):
        self.cache = {}
        self.load_cache()
    
    def get_cache_key(self, query: str, context_chunks: List[Any]) -> str:
        """Generate unique cache key from query and context"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if context_chunks:
            # Use full context for better uniqueness
            context_text = "".join([chunk.text for chunk in context_chunks if hasattr(chunk, 'text')])
            context_hash = hashlib.md5(context_text.encode()).hexdigest()
        else:
            context_hash = "no_context"
        return f"{query_hash}_{context_hash}"
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response if exists and not expired"""
        if key in self.cache:
            cached_time, response = self.cache[key]
            if time.time() - cached_time < 24 * 3600:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: str):
        """Cache response with timestamp"""
        self.cache[key] = (time.time(), response)
        self.save_cache()
    
    def save_cache(self):
        """Save cache to disk"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(RESPONSE_CACHE_FILE, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def load_cache(self):
        """Load cache from disk"""
        try:
            if os.path.exists(RESPONSE_CACHE_FILE):
                with open(RESPONSE_CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"✅ Loaded response cache with {len(self.cache)} entries")
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
            self.cache = {}

# Initialize cache globally
response_cache = ResponseCache()

# === Conversation Logger ===
class ConversationLogger:
    """JSON-based conversation logging system"""
    
    def __init__(self, log_file="conversations.json"):
        self.log_file = log_file
        self.ensure_log_file()
    
    def ensure_log_file(self):
        """Ensure log file exists with proper structure"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
    
    def log_conversation(self, user_input: str, llm_response: str, language: str, response_type: str):
        """Log conversation to JSON file"""
        try:
            # Read existing data
            with open(self.log_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            # Add new conversation
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "llm_response": llm_response,
                "language": language,
                "response_type": response_type
            }
            
            conversations.append(conversation_entry)
            
            # Write back to file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Conversation logged to {self.log_file}")
            
        except Exception as e:
            print(f"❌ Error logging conversation: {e}")

# Initialize global logger
conversation_logger = ConversationLogger()

# === Centralized Configuration System ===
class Config:
<<<<<<< HEAD
    """Centralized configuration - loads ONLY from config/config.json for all environments"""
=======
    """Centralized configuration - loads ONLY from environment variables for all environments"""
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
    
    def __init__(self):
        self.api_keys = []
        self.current_key_index = 0
<<<<<<< HEAD
        self.settings = self._load_config_file()
=======
        self.settings = self._load_config_from_env()
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
        self._validate_and_correct_paths()
        
        self.SUPPORTED_LANGUAGES = ["english", "urdu"]
        self.DEFAULT_LANGUAGE = "english"
        
        # Apply settings
<<<<<<< HEAD
        self.MODEL_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter")
        self.MODEL_ID = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")
        self.API_KEYS_FOLDER = self.settings["api_keys_folder"]
        self.INDEX_PATH = self.settings["index_path"]
        self.DATASET_PATH = self.settings["dataset_path"]
        self.SIMILARITY_TOP_K = self.settings["similarity_top_k"]
        self.TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", self.settings["temperature"]))
        self.MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", self.settings["max_tokens"]))
=======
        self.MODEL_PROVIDER = self.settings["model_provider"]
        self.MODEL_ID = self.settings["model_id"]
        self.INDEX_PATH = self.settings["index_path"]
        self.DATASET_PATH = self.settings["dataset_path"]
        self.SIMILARITY_TOP_K = self.settings["similarity_top_k"]
        self.TEMPERATURE = self.settings["temperature"]
        self.MAX_TOKENS = self.settings["max_tokens"]
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
        self.FALLBACK_MESSAGE = self.settings["fallback_message"]
        
        self.api_keys = self._load_api_keys()
        self.api_key = self._get_current_api_key()
        
        self._validate_config()

<<<<<<< HEAD
    def _load_config_file(self):
        """Load configuration ONLY from config/config.json file for ALL environments"""
        config_file = os.path.join("config", "config.json")
        
        # Default configuration as fallback
        default_config = {
            "api_keys_folder": "config",
            "index_path": "cancer_index_store",
            "dataset_path": "breast_cancer.json",
            "similarity_top_k": 5,
            "temperature": 0.2,
            "max_tokens": 350,
            "fallback_message": "Sorry, I don't know the answer."
        }
    
        try:
            if os.path.exists(config_file):
                print(f"✅ Loading configuration from: {config_file}")
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults for missing keys
                final_config = {**default_config, **loaded_config}
                
                print("📋 Configuration loaded successfully from config.json")
                return final_config
            else:
                # Create directory and config file if it doesn't exist
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                print(f"📁 Created default config file at: {config_file}")
                return default_config
                
        except Exception as e:
            print(f"❌ Error loading config from {config_file}: {e}")
            print("🔄 Using default configuration as fallback")
            return default_config
=======
    def _load_config_from_env(self):
        """Load configuration from environment variables with defaults for non-sensitive settings"""
        # Required environment variables (no defaults)
        required_vars = {
            "LLM_PROVIDER": "model_provider",
            "LLM_MODEL": "model_id",
            "LLM_TEMPERATURE": "temperature",
            "LLM_MAX_TOKENS": "max_tokens"
        }
        
        # Optional environment variables with defaults
        optional_vars = {
            "INDEX_PATH": ("index_path", "cancer_index_store"),
            "DATASET_PATH": ("dataset_path", "breast_cancer.json"),
            "SIMILARITY_TOP_K": ("similarity_top_k", 5),
            "COMBINE_SOURCES": ("combine_sources", True),
            "FALLBACK_MESSAGE": ("fallback_message", "Sorry, I don't know the answer."),
            "STRICT_BREAST_CANCER_ONLY": ("strict_breast_cancer_only", True)
        }
        
        config = {}
        missing_vars = []
        
        # Load required vars
        for env_var, config_key in required_vars.items():
            value = os.getenv(env_var)
            if value is None:
                missing_vars.append(env_var)
            else:
                # Convert types appropriately
                if config_key in ["max_tokens"]:
                    config[config_key] = int(value)
                elif config_key in ["temperature"]:
                    config[config_key] = float(value)
                else:
                    config[config_key] = value
        
        # Load optional vars with defaults
        for env_var, (config_key, default_value) in optional_vars.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert types appropriately
                if config_key in ["similarity_top_k", "max_tokens"]:
                    config[config_key] = int(value)
                elif config_key in ["temperature"]:
                    config[config_key] = float(value)
                elif config_key in ["combine_sources", "strict_breast_cancer_only"]:
                    config[config_key] = value.lower() == "true"
                else:
                    config[config_key] = value
            else:
                config[config_key] = default_value
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}. Please set them in .env file.")
        
        print("📋 Configuration loaded successfully from environment variables")
        return config
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325

    def _validate_and_correct_paths(self):
        """Validate and correct file paths"""
        # Correct dataset path if needed
        original_dataset_path = self.settings["dataset_path"]
        if not os.path.exists(original_dataset_path):
            possible_paths = [
                original_dataset_path,
                f"DataSet/{original_dataset_path}",
                f"data/{original_dataset_path}",
                "DataSet/breast_cancer.json",
                "breast_cancer.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if path != original_dataset_path:
                        print(f"🔄 Using dataset at: {path}")
                        self.settings["dataset_path"] = path
                    else:
                        print(f"✅ Dataset found at: {path}")
                    return
            
            print(f"❌ Dataset not found in any location")
        else:
            print(f"✅ Dataset found at: {original_dataset_path}")

    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variables"""
        api_keys = []
        print("🔍 Checking for API keys in environment variables...")
        
<<<<<<< HEAD
        key_value = os.getenv("LLM_API_KEY")
        if key_value and key_value.strip():
            api_keys.append(key_value.strip())
            print("✅ Found LLM_API_KEY")
=======
        keys_to_check = ["LLM_API_KEY", "LLM_API_KEY_2", "LLM_API_KEY_3", "LLM_API_KEY_4", "LLM_API_KEY_5"]
        
        for key_name in keys_to_check:
            key_value = os.getenv(key_name)
            if key_value and key_value.strip():
                api_keys.append(key_value.strip())
                print(f"✅ Found {key_name}")
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
        
        return api_keys

    def _get_current_api_key(self) -> str:
        """Get current active API key"""
        if self.api_keys and self.current_key_index < len(self.api_keys):
            return self.api_keys[self.current_key_index]
        return ""

    def rotate_to_next_key(self) -> bool:
        """Rotate to next API key if available"""
        if self.current_key_index < len(self.api_keys) - 1:
            self.current_key_index += 1
            self.api_key = self._get_current_api_key()
            print(f"🔄 Rotated to API key {self.current_key_index + 1}")
            return True
        else:
            print("❌ No more API keys available")
            return False

    def _validate_config(self):
        """Validate configuration"""
        if not self.api_keys:
            print("❌ No API keys found in environment variables")
            if IS_HUGGING_FACE:
                print("💡 Please add API keys in Hugging Face Space Settings → Repository secrets")
        else:
            print(f"✅ Found {len(self.api_keys)} API key(s)")
            
        # Print current configuration
<<<<<<< HEAD
        print("📋 Current Configuration (from config.json):")
=======
        print("📋 Current Configuration (from environment variables):")
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
        print(f"   Model Provider: {self.MODEL_PROVIDER}")
        print(f"   Model ID: {self.MODEL_ID}")
        print(f"   Index Path: {self.INDEX_PATH}")
        print(f"   Dataset Path: {self.DATASET_PATH}")
        print(f"   Similarity Top K: {self.SIMILARITY_TOP_K}")
        print(f"   Temperature: {self.TEMPERATURE}")
        print(f"   Max Tokens: {self.MAX_TOKENS}")

# Initialize configuration
config = Config()

# === Setup Logging ===
if IS_HUGGING_FACE:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
else:
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_system.log')
        ]
    )

# === FAST INDEX LOADING ===
def load_index_fast():
    """Fast index loading by reusing cached embeddings"""
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print(f"🔍 Loading index from: {config.INDEX_PATH}")
        if not os.path.exists(config.INDEX_PATH):
            print(f"❌ Index path doesn't exist: {config.INDEX_PATH}")
            return None, None
        
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        storage_context = StorageContext.from_defaults(persist_dir=config.INDEX_PATH)
        index = VectorStoreIndex.from_documents(
            [], 
            storage_context=storage_context, 
            embed_model=embed_model
        )
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        print("✅ Index loaded successfully")
        return index, retriever
        
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_index():
    return load_index_fast()

# === Enhanced RAG System Class ===
class BreastCancerRAGSystem:
    """Enhanced RAG system for breast cancer information with emotional support"""
    
    def __init__(self, index, retriever):
        self.index = index
        self.retriever = retriever
        self.conversation_history = []
        
        if not config.api_keys:
            logging.error("🚫 System initialized without API key - LLM features will not work")

    def get_predefined_questions(self, language: str = "english") -> List[dict]:
        """Get predefined daily routine questions for breast cancer patients"""
        
        english_questions = [
            {
                "question": "What are some gentle exercises I can do during recovery?",
                "category": "exercise", 
                "icon": "fas fa-walking"
            },
            {
                "question": "How do I deal with anxiety about my next treatment?",
                "category": "emotional",
                "icon": "fas fa-heart"
            },
            {
                "question": "When can I expect my hair to grow back after treatment?",
                "category": "appearance",
                "icon": "fas fa-user"
            },
            {
                "question": "How do I talk to my family about my diagnosis?",
                "category": "emotional",
                "icon": "fas fa-users"
            },
            {
                "question": "What are the signs of infection I should watch for?",
                "category": "symptoms",
                "icon": "fas fa-exclamation-triangle"
            }
        ]
        
        urdu_questions = [
            {
                "question": "کیموتھراپی کے دوران تھکاوٹ کیسے کم کریں؟",
                "category": "symptoms",
                "icon": "fas fa-bed"
            },
            {
                "question": "ریکوری کے دوران ہلکی پھلکی ورزشیں کون سی ہیں؟",
                "category": "exercise",
                "icon": "fas fa-walking"
            },
            {
                "question": "اگلے علاج کے بارے میں پریشانی کیسے دور کریں؟",
                "category": "emotional",
                "icon": "fas fa-heart"
            },
            {
                "question": "کیموتھراپی کے بعد متلی کے لیے کون سی غذائیں مفید ہیں؟",
                "category": "nutrition", 
                "icon": "fas fa-apple-alt"
            },
            {
                "question": "ماسٹکٹومی کے بعد درد کیسے منظم کریں؟",
                "category": "pain",
                "icon": "fas fa-hand-holding-heart"
            },
        ]
        
        return urdu_questions if language == "urdu" else english_questions
    
    def detect_language(self, text: str) -> str:
        """Detect language of user query"""
        try:
            urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
            if urdu_pattern.search(text):
                return 'urdu'
            detected_lang = detect(text)
            return 'urdu' if detected_lang == 'ur' else 'english'
        except:
            return 'english'

    def _clean_urdu_text(self, text: str) -> str:
        """Advanced cleaning for Urdu text with comprehensive spelling correction"""
        if not text or not text.strip():
            return text
            
        # Comprehensive spelling correction dictionary
        spelling_corrections = {
            # Character repetition fixes
            'مجہے': 'مجھے',
            'پروگرہوں': 'پروگرام',
            'کہےنسر': 'کینسر',
            'ڈڈاکٹر': 'ڈاکٹر',
            'ہےہ': 'ہے',
            'مہےں': 'میں',
            'ہےں': 'ہیں',
            'ھے': 'ہے',
            'ھوں': 'ہوں',
            'ھیں': 'ہیں',
            'ےے': 'ے',
            'ںں': 'ں',
            'ہہ': 'ہ',
            'یی': 'ی',
            
            # Common phrase corrections
            'ے لہےے': 'کے لیے',
            'کا ے لہےے': 'کے لیے',
            'و ہےہ': 'کو',
            'ہےقہےن': 'یقین',
            'اکہےلے': 'اکیلے',
            'نہہےں': 'نہیں',
            'ہہےں': 'ہیں',
            'کا ے': 'کے',
            'ساتھ ہہےں': 'ساتھ ہیں',
            'تجوہےز': 'تجویز',
            'ضرورہے': 'ضروری',
            'بارے مہےں': 'بارے میں',
            'کرہےں': 'کریں',
            'بہترہےن': 'بہترین',
            'ہے مدد': 'کی مدد',
            'خوشہے': 'خوشی',
            'ترجہےح': 'ترجیح',
            'جسے سے': 'جس سے',
            
            # Medical term corrections
            'برہےسٹ': 'بریسٹ',
            'کہےموتھراپہے': 'کیموتھراپی',
            'متلہے': 'متلی',
            'غذائہےں': 'غذائیں',
            'چربہے': 'چربی',
            'ہلکے': 'ہلکی',
            'آسانہے': 'آسانی',
            'ہائہےڈرہےٹنگ': 'ہائیڈریٹنگ',
            'ہائہےڈرہےٹڈ': 'ہائیڈریٹڈ',
            
            # Grammar and structure fixes
            'کرنےے': 'کرنے',
            'ہونےے': 'ہونے',
            'سکتےے': 'سکتے',
            'سکتیی': 'سکتی',
            'والےے': 'والے',
            'والیی': 'والی',
            'کہے': 'کے',
            'ہےے': 'ہے',
            
            # Common word fixes
            'ام ': 'ہوں ',
            'می ': 'میں ',
            'آپ ک': 'آپ کا ',
            'دوران ': 'دوران ',
            'عام ': 'عام ',
            'مسئل ': 'مسئلہ ',
            'اس ': 'اس ',
            'کو ': 'کو ',
            'کرن ': 'کرنے ',
            'س ': 'سے ',
            'طریق ': 'طریقے ',
            'بتا ': 'بتا ',
            'سکتی ': 'سکتی ',
            'اکٹر': 'ڈاکٹر',
            'اکیل': 'اکیلے',
            'میش': 'میں',
            'وتی': 'ہوتی',
            'لکی': 'ہلکی',
            'بتر': 'بہتر',
            'محفوظ ر': 'محفوظ رکھتی ہے',
            'رشت': 'رشتہ داروں',
        }
        
        # Apply spelling corrections iteratively
        cleaned_text = text
        for wrong, correct in spelling_corrections.items():
            cleaned_text = cleaned_text.replace(wrong, correct)
        
        # Fix common grammatical patterns using regex for better coverage
        import re
        
        # Fix character repetition patterns
        repetition_patterns = [
            (r'ہہ', 'ہ'),
            (r'یی', 'ی'),
            (r'ےے', 'ے'),
            (r'ںں', 'ں'),
            (r'کک', 'ک'),
            (r'گگ', 'گ'),
        ]
        
        for pattern, replacement in repetition_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Fix common grammatical patterns
        grammatical_fixes = [
            ('ک دوران', 'کے دوران'),
            ('ک بار', 'کے بارے'),
            ('ک بعد', 'کے بعد'),
            ('ک لی', 'کے لیے'),
            ('ک ساتھ', 'کے ساتھ'),
            ('ک طور', 'کے طور'),
            ('ک ذریع', 'کے ذریعے'),
            ('ک مطابق', 'کے مطابق'),
        ]
        
        for wrong, correct in grammatical_fixes:
            cleaned_text = cleaned_text.replace(wrong, correct)
        
        # Fix spacing and punctuation issues
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single space
        cleaned_text = re.sub(r' \.', '.', cleaned_text)  # Space before period
        cleaned_text = re.sub(r' ،', '،', cleaned_text)   # Space before comma
        cleaned_text = re.sub(r'  ', ' ', cleaned_text)   # Double spaces
        cleaned_text = re.sub(r'۔۔', '۔', cleaned_text)   # Double periods
        
        # Ensure sentence completion and structure
        sentences = cleaned_text.split('۔')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 2:  # At least 2 words
                # Ensure sentence starts properly (no hanging characters)
                if sentence and sentence[0] in [' ', '،', '۔']:
                    sentence = sentence[1:].strip()
                if sentence:
                    cleaned_sentences.append(sentence)
        
        # Reconstruct text with proper punctuation
        if cleaned_sentences:
            cleaned_text = '۔ '.join(cleaned_sentences) + '۔'
        else:
            cleaned_text = cleaned_text.strip()
        
        # Final normalization
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def _detect_emotional_needs(self, user_query: str, language: str = "english") -> dict:
        """Enhanced emotional need detection with better Urdu support"""
        query_lower = user_query.lower()
        
        # Emotional triggers in both languages
        emotional_triggers_english = [
            "scared", "afraid", "worried", "anxious", "fear", "nervous", "stressed",
            "overwhelmed", "depressed", "sad", "lonely", "alone", "hopeless",
            "can't cope", "struggling", "difficult", "hard time", "suffering",
            "terrified", "panic", "breakdown", "crying", "tears", "misery"
        ]
        
        emotional_triggers_urdu = [
            "خوف", "ڈر", "پریشانی", "فکر", "تنہائی", "اداسی", "مایوسی", "تکلیف",
            "گھبراہٹ", "بے چینی", "بے بسی", "رونا", "آنسو", "دکھ", "غم",
            "ہمت", "طاقت", "حوصلہ", "پرسکون", "سکون", "چین"
        ]
        
        # Information triggers
        info_triggers_english = [
            "what", "how", "when", "where", "which", "why", 
            "treatment", "medication", "exercise", "diet", "symptoms", 
            "pain", "side effects", "recovery", "diagnosis", "procedure"
        ]
        
        info_triggers_urdu = [
            "کیا", "کیسے", "کب", "کہاں", "کون سا", "کیوں", "کس طرح",
            "علاج", "دوا", "ورزش", "غذا", "علامات", "درد", "مراحل",
            "طریقہ", "عمل", "تفصیل", "معلومات"
        ]
        
        if language == "urdu":
            emotional_triggers = emotional_triggers_urdu
            info_triggers = info_triggers_urdu
        else:
            emotional_triggers = emotional_triggers_english
            info_triggers = info_triggers_english
        
        # More sophisticated emotional detection
        emotional_score = 0
        for trigger in emotional_triggers:
            if trigger in query_lower:
                emotional_score += 1
        
        # Context-aware emotional detection
        negative_context_words = ["not", "don't", "no", "never", "n't"]
        has_negative_context = any(word in query_lower for word in negative_context_words)
        
        info_score = sum(1 for trigger in info_triggers if trigger in query_lower)
        
        return {
            "needs_emotional_support": emotional_score > 0 and not has_negative_context,
            "needs_information": info_score > 0,
            "emotional_score": emotional_score,
            "info_score": info_score
        }

    def _add_emotional_support(self, response: str, user_query: str, language: str = "english") -> str:
        """Add natural emotional support integrated into the response"""
        emotional_needs = self._detect_emotional_needs(user_query, language)
        
        # Always add some level of emotional support, but more if detected
        if language == "urdu":
            if emotional_needs["needs_emotional_support"]:
                # Strong emotional support phrases
                support_phrases = [
                    "آپ کی طاقت قابلِ تعریف ہے، اور میں آپ کے ساتھ ہوں۔",
                    "یہ مشکل وقت ہے، لیکن آپ اکیلے نہیں ہیں۔ ہم مل کر اس کا سامنا کریں گے۔",
                    "آپ کی ہمت اور صبر کو سلام، بہتر دن ضرور آئیں گے۔",
                ]
            else:
                # Gentle emotional support phrases
                support_phrases = [
                    "آپ کی صحت اور خوشی ہماری پہلی ترجیح ہے۔",
                    "یقین رکھیں، ہر طوفان کے بعد سکون ضرور آتا ہے۔",
                    "آپ جیسے بہادر لوگ ہی دنیا کو روشن کرتے ہیں۔",
                ]
        else:
            if emotional_needs["needs_emotional_support"]:
                # Strong emotional support phrases
                support_phrases = [
                    "Your strength is truly admirable, and I'm here with you every step of the way.",
                    "This is a challenging time, but you're not alone. We'll face this together.",
                    "I want you to know how much courage you're showing, and better days will come.",
                ]
            else:
                # Gentle emotional support phrases
                support_phrases = [
                    "Your wellbeing and happiness are my top priority right now.",
                    "Please remember that after every storm comes calm.",
                    "People like you, with such resilience, truly light up the world.",
                ]
        
        # Choose a support phrase that fits naturally
        support_text = random.choice(support_phrases)
        
        # Integrate support naturally - for Urdu, place at beginning for impact
        if language == "urdu":
            if support_text not in response:
                # Check if response already has emotional content
                if not any(phrase in response for phrase in ['طاقت', 'ہمت', 'حوصلہ', 'سکون', 'خوشی']):
                    return f"{support_text}\n\n{response}"
        else:
            if support_text not in response:
                # Check if response already has emotional content
                if not any(phrase in response for phrase in ['strength', 'courage', 'hope', 'together', 'proud']):
                    return f"{support_text}\n\n{response}"
        
        return response
    
    def retrieve_relevant_chunks(self, user_query: str, language: str = "english") -> List[Any]:
        """Retrieve relevant chunks with language-specific prioritization"""
        if not hasattr(self, 'retriever') or self.retriever is None:
            print("❌ Retriever not available")
            return []
        
        try:
            if language == "urdu":
                print("🔍 Prioritizing Urdu content for Urdu query...")
                from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
                urdu_filter = MetadataFilter(key="language", value="urdu")
                urdu_results = self.retriever.retrieve(
                    user_query, 
                    filters=MetadataFilters(filters=[urdu_filter])
                )
                
                quality_threshold = 0.5
                high_quality_urdu = [
                    result for result in urdu_results 
                    if hasattr(result, 'score') and result.score >= quality_threshold
                ]
                
                if high_quality_urdu:
                    print(f"✅ Found {len(high_quality_urdu)} high-quality Urdu chunks")
                    return high_quality_urdu[:5]
                elif urdu_results:
                    print(f"⚠️ Using {len(urdu_results)} lower-confidence Urdu chunks")
                    return urdu_results[:3]
                
                print("🔍 No Urdu content found, searching all content...")
            
            retrieval_results = self.retriever.retrieve(user_query)
            quality_threshold = 0.5
            high_quality_results = [
                result for result in retrieval_results 
                if hasattr(result, 'score') and result.score >= quality_threshold
            ]
            
            if not high_quality_results and retrieval_results:
                print("⚠️ Using lower confidence results")
                return retrieval_results[:3]
            
            print(f"✅ Retrieved {len(high_quality_results)} relevant chunks")
            return high_quality_results[:5]
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []
    
    def build_concise_prompt(self, user_query: str, context_chunks: List[Any], language: str = "english") -> str:
        """Build prompt for concise, targeted responses with emotional intelligence"""
        
        context_text = ""
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks[:2]):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                key_points = " ".join(chunk_text.split()[:100])
                context_parts.append(f"CONTEXT {i+1}: {key_points}")
            context_text = "\n".join(context_parts)

        # Analyze emotional and information needs
        needs_analysis = self._detect_emotional_needs(user_query, language)
        
        if language == "urdu":
            prompt = f"""
# WELL BEING AGENT - BREAST CANCER SUPPORT
# CRITICAL: RESPOND ONLY IN URDU LANGUAGE USING CORRECT URDU SPELLING AND GRAMMAR
# ABSOLUTELY NO HINDI, ARABIC, OR OTHER LANGUAGES - PURE URDU ONLY

## PATIENT'S QUERY:
"{user_query}"

## EMOTIONAL ANALYSIS:
- Needs Emotional Support: {'YES' if needs_analysis['needs_emotional_support'] else 'NO'}
- Needs Information: {'YES' if needs_analysis['needs_information'] else 'NO'}

## CONTEXT (USE IF RELEVANT):
{context_text if context_text else "General breast cancer knowledge"}

## CRITICAL SPELLING RULES - MUST FOLLOW:
1. ✅ "مجھے" ❌ "مجہے"
2. ✅ "پروگرام" ❌ "پروگرہوں"  
3. ✅ "کینسر" ❌ "کہےنسر"
4. ✅ "ڈاکٹر" ❌ "ڈڈاکٹر"
5. ✅ "ہے" ❌ "ہےہ"
6. ✅ "میں" ❌ "مہےں"
7. ✅ "کے لیے" ❌ "کا ے لہےے"
8. ✅ "جس سے" ❌ "جسے سے"

## RESPONSE REQUIREMENTS - URDU:
1. **LANGUAGE:** صرف اردو میں جواب دیں، درست ہجے اور قواعد کا استعمال کریں
2. **EMOTIONAL TONE:** ہمدردانہ، گرمجوش، اور امید بخش انداز اپنائیں
3. **CONTENT:** اگر معلومات درکار ہوں تو واضح، درست معلومات دیں
4. **SUPPORT:** جذباتی مدد قدرتی طور پر پیش کریں، الگ سے ذکر نہ کریں
5. **LENGTH:** 4-6 جملے، مختصر مگر جامع
6. **SPELLING:** درست اردو ہجے استعمال کریں، غلط ہجے سے پرہیز کریں
7. **COMPLETENESS:** مکمل جملے لکھیں، ادھورے جملے نہ چھوڑیں

## آپ کا گرمجوش، درست اردو میں اور مکمل جواب:
"""
        else:
            prompt = f"""
# WELL BEING AGENT - BREAST CANCER SUPPORT

## PATIENT'S QUERY:
"{user_query}"

## EMOTIONAL ANALYSIS:
- Needs Emotional Support: {'YES' if needs_analysis['needs_emotional_support'] else 'NO'}
- Needs Information: {'YES' if needs_analysis['needs_information'] else 'NO'}

## CONTEXT (USE IF RELEVANT):
{context_text if context_text else "General breast cancer knowledge"}

## RESPONSE REQUIREMENTS:

1. **TONE:** Warm, compassionate, and hopeful
2. **CONTENT:** Provide accurate information if needed
3. **SUPPORT:** Integrate emotional support naturally without explicitly stating it
4. **LENGTH:** 4-6 sentences, concise but comprehensive
5. **FOCUS:** Be caring and present with the patient
6. **COMPLETENESS:** Write complete sentences, no incomplete thoughts

## YOUR COMPASSIONATE RESPONSE:
"""
        
        return prompt.strip()
    
    def build_urdu_prompt(self, user_query: str, context_chunks: List[Any]) -> str:
        """Build detailed prompt for Urdu responses with strong language enforcement"""
        context_text = ""
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks[:3]):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                source_topic = chunk.metadata.get('topic', 'General Information') if hasattr(chunk, 'metadata') else 'General Information'
                context_parts.append(f"SOURCE {i+1} - {source_topic}:\n{chunk_text}")
            context_text = "\n\n".join(context_parts)
        
        urdu_prompt = f"""
# WELL BEING AGENT - BREAST CANCER SUPPORT
# CRITICAL: RESPOND ONLY IN URDU LANGUAGE WITH PERFECT SPELLING
# ABSOLUTELY NO HINDI, ARABIC, OR ENGLISH - PURE URDU ONLY

## YOUR ROLE IN URDU:
آپ بریسٹ کینسر کی سپیشلائزڈ ویل بینگ ایجنٹ ہیں۔ آپ مریضوں کو نہ صرف طبی معلومات بلکہ قدرتی طور پر جذباتی مدد اور ہمت بھی فراہم کرتی ہیں۔

## AVAILABLE CONTEXT:
{context_text if context_text else "General breast cancer knowledge"}

## USER'S QUESTION (IN URDU):
"{user_query}"

## CRITICAL SPELLING RULES - MUST FOLLOW:
1. ✅ "مجھے" ❌ "مجہے"
2. ✅ "پروگرام" ❌ "پروگرہوں"  
3. ✅ "کینسر" ❌ "کہےنسر"
4. ✅ "ڈاکٹر" ❌ "ڈڈاکٹر"
5. ✅ "ہے" ❌ "ہےہ"
6. ✅ "میں" ❌ "مہےں"
7. ✅ "کے لیے" ❌ "کا ے لہےے"
8. ✅ "جس سے" ❌ "جسے سے"

## RESPONSE REQUIREMENTS - URDU:
1. **LANGUAGE ENFORCEMENT:** صرف اور صرف اردو میں جواب دیں
2. **SPELLING ACCURACY:** درست اردو ہجے استعمال کریں، عام غلطیوں سے پرہیز کریں
3. **EMOTIONAL INTEGRATION:** جذباتی مدد کو قدرتی انداز میں پیش کریں
4. **COMPASSIONATE TONE:** گرمجوش، ہمدردانہ، اور امید بخش انداز
5. **INFORMATION ACCURACY:** سیاق و سباق کے مطابق درست معلومات دیں
6. **COMPLETE SENTENCES:** مکمل جملے لکھیں، ادھورے جملے نہ چھوڑیں

## EXAMPLES OF CORRECT URDU:
- ✅ "بریسٹ کینسر کے بارے میں معلومات حاصل کرنا ایک اہم قدم ہے۔"
- ✅ "میں آپ کو درست معلومات فراہم کرنے کی کوشش کروں گی۔"
- ✅ "آپ کے سوال کا جواب دینے میں مجھے خوشی ہو رہی ہے۔"

## آپ کا درست ہجے، مکمل جملوں اور ہمدردانہ انداز میں جواب:
"""
        return urdu_prompt.strip()
    
    def build_enhanced_prompt(self, user_query: str, context_chunks: List[Any]) -> str:
        """Build prompt for English responses with emotional intelligence"""
        context_text = ""
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks[:3]):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                source_topic = chunk.metadata.get('topic', 'General Information') if hasattr(chunk, 'metadata') else 'General Information'
                context_parts.append(f"SOURCE {i+1} - {source_topic}:\n{chunk_text}")
            context_text = "\n\n".join(context_parts)

        # Analyze emotional needs
        needs_analysis = self._detect_emotional_needs(user_query, "english")

        prompt = f"""
# WELL BEING AGENT - BREAST CANCER SUPPORT

## YOUR ROLE
You are a compassionate Well Being Agent specializing in breast cancer support. You provide supportive information, emotional comfort, and evidence-based guidance.

## EMOTIONAL ANALYSIS:
- Patient Needs Emotional Support: {'YES' if needs_analysis['needs_emotional_support'] else 'NO'}
- Patient Needs Information: {'YES' if needs_analysis['needs_information'] else 'NO'}

## RESPONSE GUIDELINES
- **Tone**: Warm, supportive, compassionate, and hopeful
- **Emotional Integration**: Naturally incorporate emotional support without explicitly stating it
- **Information**: Provide evidence-based guidance when needed
- **Presence**: Be fully present and caring with the patient
- **Completeness**: Write complete sentences, no incomplete thoughts

## AVAILABLE CONTEXT
{context_text if context_text else "General breast cancer knowledge"}

## USER'S QUESTION
"{user_query}"

## RESPONSE REQUIREMENTS
1. If emotional support is needed: Integrate comfort and hope naturally into your response
2. If information is needed: Provide clear, accurate guidance
3. Always acknowledge the patient's strength implicitly
4. Maintain a caring, present tone throughout
5. Keep response concise but comprehensive (4-6 complete sentences)

## YOUR COMPASSIONATE RESPONSE:
"""
        return prompt.strip()
    
    def query_llm_with_retry(self, prompt: str, language: str = "english", max_retries: int = 3) -> str:
        """Enhanced LLM query using OpenAI client format"""
        if not config.api_key:
            print("❌ No API key available")
            return config.FALLBACK_MESSAGE
        
        # Enhanced system message with Urdu-specific instructions
        if language == "urdu":
            system_message = """آپ بریسٹ کینسر کی سپیشلائزڈ ویل بینگ ایجنٹ ہیں۔ 

CRITICAL URDU LANGUAGE RULES:
1. صرف اور صرف اردو میں جواب دیں
2. ہر لفظ کے ہجے درست ہوں
3. مکمل اور واضح جملے استعمال کریں
4. غلط ہجے اور ادھورے جملوں سے پرہیز کریں
5. طبی معلومات درست اور واضح ہوں

مثال کے طور پر:
✅ "بریسٹ کینسر کے علاج کے مختلف طریقے ہیں۔"
❌ "برہےسٹ کہےنسر کا علاچ کہے طرح ہےہ۔"

جذباتی مدد قدرتی طور پر پیش کریں اور مریض کی طاقت کو تسلیم کریں۔"""
        else:
            system_message = """You are a compassionate Well Being Agent for breast cancer support. Provide direct, helpful information while naturally integrating emotional support. Always maintain a warm, hopeful, and caring tone. Ensure complete sentences and clear information."""
    
        for attempt in range(max_retries):
            try:
                # Initialize OpenAI client with OpenRouter configuration
                client = OpenAI(
<<<<<<< HEAD
                    base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
=======
                    base_url="https://openrouter.ai/api/v1",
>>>>>>> a94e781ef00522de046b38098b30cce04a40e325
                    api_key=config.api_key,
                )

                # Adjust parameters for better Urdu quality
                temperature = 0.2 if language == "urdu" else 0.3
                max_tokens = 500 if language == "urdu" else config.MAX_TOKENS
                
                print(f"🔄 Sending request to {config.MODEL_PROVIDER} (attempt {attempt + 1})")
                
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://huggingface.co",
                        "X-Title": "Well Being Agent",
                    },
                    extra_body={},
                    model=config.MODEL_ID,
                    messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                response_text = completion.choices[0].message.content
                print("✅ LLM response received")
                
                # For Urdu, do immediate quality check
                if language == "urdu":
                    if self._is_urdu_response_corrupted(response_text):
                        print("⚠️ Urdu response appears corrupted, applying enhanced cleaning")
                        
                return response_text
                
            except Exception as e:
                print(f"❌ Request failed: {e}")
                if "429" in str(e):
                    wait_time = 2 ** attempt
                    print(f"⏳ Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                elif "401" in str(e) or "402" in str(e):
                    print(f"❌ API key issue")
                    if config.rotate_to_next_key():
                        continue
                    else:
                        return config.FALLBACK_MESSAGE
                
                if attempt == max_retries - 1:
                    if config.rotate_to_next_key():
                        return self.query_llm_with_retry(prompt, language, max_retries)
                    return config.FALLBACK_MESSAGE
                time.sleep(1)
        
        return config.FALLBACK_MESSAGE

    def _is_urdu_response_corrupted(self, text: str) -> bool:
        """Check if Urdu response has common corruption patterns"""
        corruption_indicators = [
            'ہےہ', 'مہےں', 'کہے', 'پروگرہوں', 'ڈڈاکٹر', 'کا ے لہےے', 'جسے سے'
        ]
        
        for indicator in corruption_indicators:
            if indicator in text:
                return True
        
        # Check for excessive character repetition
        import re
        if re.search(r'(.)\1\1', text):  # Three repeated characters
            return True
        
        return False
    
    def _verify_language_compliance(self, text: str, expected_language: str) -> str:
        """Verify and correct language compliance"""
        if expected_language == "urdu":
            # Check for common incorrect language patterns
            hindi_pattern = re.compile(r'[\u0900-\u097F]+')  # Hindi characters
            arabic_pattern = re.compile(r'[\uFE70-\uFEFF]+')  # Arabic specific characters
            
            if hindi_pattern.search(text):
                print("⚠️ Hindi detected in Urdu response, applying correction...")
                # Add Urdu language reminder
                return text + "\n\nبراہ کرم صرف اردو میں جواب دیں۔"
            
            if arabic_pattern.search(text):
                print("⚠️ Arabic detected in Urdu response, applying correction...")
                # Add Urdu language reminder
                return text + "\n\nبراہ کرم صرف اردو میں جواب دیں۔"
                
        return text
    
    def format_final_response(self, llm_answer: str, language: str = "english") -> str:
        cleaned_answer = llm_answer.strip()
        
        # Enhanced Urdu text cleaning
        if language == 'urdu':
            print("🧹 Applying advanced Urdu text cleaning...")
            cleaned_answer = self._clean_urdu_text(cleaned_answer)
        
        # Verify language compliance
        cleaned_answer = self._verify_language_compliance(cleaned_answer, language)
        
        if language == 'urdu':
            gentle_reminder = "\n\nاپنی صحت کی دیکھ بھال ٹیم سے اپنے خدشات پر بات کرنا یاد رکھیں۔"
        else:
            gentle_reminder = "\n\nRemember to discuss any concerns with your healthcare team."
        
        if gentle_reminder not in cleaned_answer:
            cleaned_answer += gentle_reminder
        
        return cleaned_answer.strip()
    
    def get_enhanced_answer(self, user_query: str, language: str = None, response_type: str = "text") -> str:
        print(f"🔍 Processing query: '{user_query}' (Type: {response_type})")
        
        if language is None:
            language = self.detect_language(user_query)
            print(f"🌐 Detected language: {language}")
        
        # Special handling for problematic Urdu queries
        if language == "urdu":
            problematic_patterns = ['اوج ایک انسر', 'اصلاح ملکم', 'نعم']
            if any(pattern in user_query for pattern in problematic_patterns):
                print("⚠️ Detected problematic query pattern, applying enhanced Urdu handling")
        
        chunks = self.retrieve_relevant_chunks(user_query, language)
        
        cache_key = response_cache.get_cache_key(user_query, chunks)
        cached_response = response_cache.get(cache_key)
        
        if cached_response:
            print("✅ Using cached response")
            final_answer = cached_response
        else:
            # Enhanced prompt selection with quality focus
            query_lower = user_query.lower()
            wants_details = any(phrase in query_lower for phrase in [
                "give details", "more detail", "explain more", "tell me more", 
                "elaborate", "in detail", "detailed", "comprehensive"
            ])
            
            if language == 'urdu':
                if wants_details:
                    prompt = self.build_urdu_prompt(user_query, chunks)
                else:
                    prompt = self.build_concise_prompt(user_query, chunks, language)
            else:
                if wants_details:
                    prompt = self.build_enhanced_prompt(user_query, chunks)
                else:
                    prompt = self.build_concise_prompt(user_query, chunks, language)
            
            llm_answer = self.query_llm_with_retry(prompt, language)
            
            # Enhanced cleaning and validation for Urdu
            if language == 'urdu':
                original_length = len(llm_answer.strip().split())
                llm_answer = self.format_final_response(llm_answer, language)
                cleaned_length = len(llm_answer.strip().split())
                
                if cleaned_length < 5:  # Too short
                    print("⚠️ Urdu response too short, may be incomplete")
                elif cleaned_length < original_length * 0.7:  # Significant reduction
                    print("⚠️ Significant text reduction during cleaning")
            
            final_answer = self.format_final_response(llm_answer, language)
            
            # Always add emotional support naturally
            final_answer = self._add_emotional_support(final_answer, user_query, language)
            
            response_cache.set(cache_key, final_answer)
            print("💾 Response cached for future use")
        
        # Log conversation to JSON
        conversation_logger.log_conversation(
            user_input=user_query,
            llm_response=final_answer,
            language=language,
            response_type=response_type
        )
        
        self.conversation_history.append({
            "query": user_query, 
            "answer": final_answer, 
            "language": language,
            "response_type": response_type,
            "timestamp": time.time()
        })
        
        return final_answer

# === Pre-load Index at Module Level ===
print("🚀 Starting Well Being Agent with optimized loading...")
_start_time = time.time()

print("🔄 Loading vector index...")
index, retriever = load_index_fast()

_load_time = time.time() - _start_time
print(f"✅ System ready in {_load_time:.2f} seconds")

# Create global RAG system instance
rag_system = BreastCancerRAGSystem(index, retriever)

# === Interactive Chat Mode ===
def interactive_chat():
    print("💬 Well Being Agent - Breast Cancer Support")
    print("=" * 50)
    print("Type 'quit' to exit, 'topics' to see available topics, 'cache' for cache stats")
    print("=" * 50)
    
    global rag_system
    
    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'topics':
            print("\n📚 Available topics: Fertility, Treatment, Symptoms, Diagnosis, etc.")
            continue
        elif user_input.lower() == 'cache':
            print(f"\n📊 Cache stats: {len(response_cache.cache)} cached responses")
            continue
        elif not user_input:
            continue
        
        print("🤔 Thinking...")
        start_time = time.time()
        answer = rag_system.get_enhanced_answer(user_input)
        response_time = time.time() - start_time
        print(f"\n💡 {answer}")
        print(f"⏱️  Response time: {response_time:.2f} seconds")

# === Main Function ===
def main():
    print("🏥 Well Being Agent - Breast Cancer Support System")
    print("=" * 50)
    print(f"📋 Current Configuration:")
    print(f"   Model: {config.MODEL_ID}")
    print(f"   Provider: {config.MODEL_PROVIDER}")
    print(f"   Index: {config.INDEX_PATH}")
    print(f"   Cache: {len(response_cache.cache)} responses loaded")
    print("=" * 50)
    
    if not config.api_keys:
        print("❌ API keys not configured.")
        if IS_HUGGING_FACE:
            print("💡 Add API keys in Space Settings → Repository secrets")
        return
    
    interactive_chat()

if __name__ == "__main__":
    main()
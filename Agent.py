import os
import json
import requests
import time
from typing import List, Optional, Any
from pathlib import Path
import logging

# === Centralized Configuration System ===
class Config:
    """Centralized configuration - loads from config.json"""
    
    def __init__(self):
        # Load settings from config.json
        self.settings = self._load_config_file()
        
        # Apply settings
        self.MODEL_PROVIDER = self.settings["model_provider"]
        self.MODEL_ID = self.settings["model_id"]
        self.API_KEYS_FOLDER = self.settings["api_keys_folder"]
        self.INDEX_PATH = self.settings["index_path"]
        self.DATASET_PATH = self.settings["dataset_path"]
        self.SIMILARITY_TOP_K = self.settings["similarity_top_k"]
        self.TEMPERATURE = self.settings["temperature"]
        self.MAX_TOKENS = self.settings["max_tokens"]
        self.COMBINE_SOURCES = self.settings["combine_sources"]
        self.FALLBACK_MESSAGE = self.settings["fallback_message"]
        self.STRICT_BREAST_CANCER_ONLY = self.settings["strict_breast_cancer_only"]
        
        # Set API paths
        self.API_KEY_FILE = os.path.join(self.API_KEYS_FOLDER, "api_key.txt")
        self.API_URL = self._get_api_url()
        
        # Load API key
        self.api_key = self._load_api_key()
        self._validate_config()
    
    def _load_config_file(self):
        """Load configuration from config/config.json file"""
        config_file = os.path.join("config", "config.json")  # Updated path
        default_config = {
        "model_provider": "openrouter",
        "model_id": "meta-llama/llama-3.3-70b-instruct:free",
        "api_keys_folder": "config",
        "index_path": "cancer_index_store",
        "dataset_path": "breast_cancer.json",
        "similarity_top_k": 5,
        "temperature": 0.2,
        "max_tokens": 350,
        "combine_sources": True,
        "fallback_message": "Sorry, I don't know the answer.",
        "strict_breast_cancer_only": True
        }
    
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                merged_config = {**default_config, **loaded_config}
                logging.info("✅ Configuration loaded from config/config.json")
                return merged_config
            else:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                logging.info("📁 Created default config/config.json file")
                return default_config
        except Exception as e:
            logging.error(f"❌ Error loading config/config.json: {e}")
            logging.info("🔄 Using default configuration")
            return default_config
    def _get_api_url(self):
        """Auto-set API URL based on provider"""
        urls = {
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions", 
            "google": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        }
        return urls.get(self.MODEL_PROVIDER, urls["openrouter"])
    
    def _load_api_key(self) -> str:
        """Load API key from secure folder"""
        try:
            os.makedirs(self.API_KEYS_FOLDER, exist_ok=True)
            if not os.path.exists(self.API_KEY_FILE):
                logging.error(f"❌ API key file not found: {self.API_KEY_FILE}")
                return ""
            with open(self.API_KEY_FILE, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key:
                logging.error("❌ API key file is empty.")
                return ""
            logging.info("✅ API key loaded successfully")
            return api_key
        except Exception as e:
            logging.error(f"❌ Failed to load API key: {e}")
            return ""
    
    def _validate_config(self):
        """Validate configuration"""
        if not self.api_key:
            logging.error(f"""
❌ API KEY NOT CONFIGURED

To fix this:
1. Create folder: {self.API_KEYS_FOLDER}/
2. Add your API key to: {self.API_KEY_FILE}
3. Get a free API key from: https://openrouter.ai/keys

File should contain only your API key.
""")
        else:
            logging.info(f"✅ Configuration loaded successfully")
            logging.info(f"   Model: {self.MODEL_ID}")
            logging.info(f"   Index Path: {self.INDEX_PATH}")
            logging.info(f"   Strict Breast Cancer Only: {self.STRICT_BREAST_CANCER_ONLY}")

# Initialize configuration
config = Config()

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)

# === RAG System Class ===
class BreastCancerRAGSystem:
    """Enhanced RAG system for breast cancer information"""
    
    def __init__(self, index, retriever):
        self.index = index
        self.retriever = retriever
        self.conversation_history = []
        if not config.api_key:
            logging.error("🚫 System initialized without API key - LLM features will not work")
    
    def retrieve_relevant_chunks(self, user_query: str, topic_filter: Optional[str] = None) -> List[Any]:
        if not hasattr(self, 'retriever') or self.retriever is None:
            logging.error("❌ Retriever not initialized")
            return []
        try:
            if topic_filter:
                from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
                metadata_filter = MetadataFilter(key="topic", value=topic_filter)
                retrieval_results = self.retriever.retrieve(
                    user_query, 
                    filters=MetadataFilters(filters=[metadata_filter])
                )
            else:
                retrieval_results = self.retriever.retrieve(user_query)
            
            quality_threshold = 0.7
            high_quality_results = [
                result for result in retrieval_results 
                if hasattr(result, 'score') and result.score >= quality_threshold
            ]
            
            if not high_quality_results and retrieval_results:
                logging.warning("⚠️ No high-confidence results found, using lower confidence results")
                return retrieval_results[:3]
            
            logging.info(f"✅ Retrieved {len(high_quality_results)} relevant chunks")
            return high_quality_results[:5]
        except Exception as e:
            logging.error(f"❌ Retrieval error: {e}")
            return []
    
    def build_enhanced_prompt(self, user_query: str, context_chunks: List[Any]) -> str:
        context_text = ""
        if context_chunks:
            context_parts = []
            for i, chunk in enumerate(context_chunks[:3]):
                chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                source_topic = chunk.metadata.get('topic', 'General Information') if hasattr(chunk, 'metadata') else 'General Information'
                context_parts.append(f"SOURCE {i+1} - {source_topic}:\n{chunk_text}")
            context_text = "\n\n".join(context_parts)
        
        user_query_lower = user_query.lower()
        detailed_triggers = ["explain in detail", "detailed", "more information", "tell me everything",
                             "comprehensive", "elaborate", "break down", "in depth", "thorough"]
        urgent_triggers = ["serious", "critical", "urgent", "emergency", "dangerous", "life threatening"]
        
        is_detailed = any(trigger in user_query_lower for trigger in detailed_triggers)
        is_urgent = any(trigger in user_query_lower for trigger in urgent_triggers)
        
        if is_urgent:
            word_limit = "200-250 words (urgent matters need clarity)"
            detail_instruction = "Provide clear, actionable information for serious concerns"
        elif is_detailed:
            word_limit = "300-400 words"
            detail_instruction = "Provide comprehensive, detailed explanation"
        else:
            word_limit = "120-170 words"
            detail_instruction = "Provide concise, focused answer"

        breast_cancer_focus = ""
        if config.STRICT_BREAST_CANCER_ONLY:
            breast_cancer_focus = """
STRICT SCOPE BOUNDARIES:
🔒 ONLY answer questions related to breast cancer topics
🔒 If question is about other cancers or medical conditions, politely decline
🔒 Redirect to appropriate resources for non-breast-cancer topics
"""

        prompt = f"""
# WELL BEING AGENT - BREAST CANCER SUPPORT

## YOUR ROLE
You are a Well Being Agent specializing in breast cancer support. You provide supportive information and guidance to help patients understand their journey.

## RESPONSE PRIORITY
1. FIRST provide direct, helpful advice based on the available context and general knowledge
2. THEN if appropriate, gently suggest discussing with healthcare providers
3. NEVER lead with "consult your doctor" - lead with helpful information

## RESPONSE GUIDELINES
- **Length**: {word_limit}
- **Focus**: {detail_instruction}
- **Tone**: Warm, supportive, and clear
- **Scope**: Breast cancer education and emotional support
- **Priority**: Give concrete information first, suggestions second

## INFORMATION APPROACH
- Provide specific advice about pregnancy timing, fertility, breastfeeding capability
- Share what's generally known about mastectomy and single-breast breastfeeding
- Discuss typical recovery timelines and considerations
- Only THEN mention that individual cases may vary

## AVAILABLE CONTEXT
{context_text if context_text else "General breast cancer knowledge"}

## USER'S QUESTION
"{user_query}"

## RESPONSE STRUCTURE
1. Direct, helpful answer with specific information
2. Evidence-based guidance from medical knowledge
3. Practical considerations and typical experiences
4. Gentle reminder that individual situations may vary

## PROHIBITED PHRASES (DO NOT USE):
- "Consult your healthcare team" as the main advice
- "Discuss with your doctor" in the first sentence
- Repetitive disclaimers
- Vague non-answers

## YOUR RESPONSE:
"""
        return prompt.strip()
    
    def query_llm_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        if not config.api_key:
            return config.FALLBACK_MESSAGE
        
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Well Being Agent"
        }
        
        payload = {
            "model": config.MODEL_ID,
            "messages": [
                {"role": "system", "content": "You are a Well Being Agent for breast cancer support. Provide direct, helpful information first. Give specific advice about pregnancy, fertility, breastfeeding, and recovery. Only gently suggest professional consultation after providing substantial guidance. Avoid repetitive disclaimers."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": config.MAX_TOKENS,
            "top_p": 0.8,
        }
        
        for attempt in range(max_retries):
            try:
                logging.info(f"🔄 Sending request to {config.MODEL_PROVIDER} (attempt {attempt + 1}/{max_retries})")
                response = requests.post(config.API_URL, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    logging.warning(f"⏳ Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                logging.info("✅ LLM response received successfully")
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return config.FALLBACK_MESSAGE
        return config.FALLBACK_MESSAGE
    
    def format_final_response(self, llm_answer: str, context_chunks: List[Any]) -> str:
        cleaned_answer = llm_answer.strip()
        disclaimer_patterns = [
            "**Note**: This information is based on general knowledge.",
            "Please consult healthcare professionals for personalized medical advice.",
            "**Important Disclaimer:**",
            "This information is for educational purposes only and is not medical advice.",
            "Always consult qualified healthcare providers for medical decisions."
        ]
        for pattern in disclaimer_patterns:
            cleaned_answer = cleaned_answer.replace(pattern, "")
        
        gentle_reminder = "\n\nRemember to discuss any concerns with your healthcare team."
        if gentle_reminder not in cleaned_answer:
            cleaned_answer += gentle_reminder
        
        return cleaned_answer.strip()
    
    def get_enhanced_answer(self, user_query: str, topic_filter: Optional[str] = None) -> str:
        logging.info(f"🔍 Processing query: '{user_query}' with filter: {topic_filter}")
        chunks = self.retrieve_relevant_chunks(user_query, topic_filter)
        prompt = self.build_enhanced_prompt(user_query, chunks)
        llm_answer = self.query_llm_with_retry(prompt)
        final_answer = self.format_final_response(llm_answer, chunks)
        self.conversation_history.append({"query": user_query, "answer": final_answer, "timestamp": time.time()})
        return final_answer

# === Load Index Function ===
def load_index():
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        print(f"🔍 Loading index from: {config.INDEX_PATH}")
        if not os.path.exists(config.INDEX_PATH):
            print(f"❌ Index path doesn't exist: {config.INDEX_PATH}")
            return None, None
        
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        storage_context = StorageContext.from_defaults(persist_dir=config.INDEX_PATH)
        index = VectorStoreIndex.from_documents([], storage_context=storage_context, embed_model=embed_model)
        retriever = index.as_retriever(similarity_top_k=config.SIMILARITY_TOP_K)
        print("✅ Index loaded successfully with local embeddings")
        return index, retriever
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# === Interactive Chat Mode ===
def interactive_chat():
    print("💬 Well Being Agent - Breast Cancer Support")
    print("=" * 50)
    print("Type 'quit' to exit, 'topics' to see available topics")
    print("=" * 50)
    
    index, retriever = load_index()
    if index is None or retriever is None:
        print("❌ Failed to load index. Please check your index path.")
        return
    
    rag_system = BreastCancerRAGSystem(index, retriever)
    
    while True:
        user_input = input("\n❓ Your question: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'topics':
            print("\n📚 Available topics: Fertility, Treatment, Symptoms, Diagnosis, etc.")
            continue
        elif not user_input:
            continue
        
        print("🤔 Thinking...")
        answer = rag_system.get_enhanced_answer(user_input)
        print(f"\n💡 {answer}")

# === Main Function ===
def main():
    print("🏥 Well Being Agent - Breast Cancer Support System")
    print("=" * 50)
    print(f"📋 Current Configuration:")
    print(f"   Model: {config.MODEL_ID}")
    print(f"   Provider: {config.MODEL_PROVIDER}")
    print(f"   Index: {config.INDEX_PATH}")
    print("=" * 50)
    
    if not config.api_key:
        print("❌ API key not configured. Please add it in the config folder.")
        return
    
    interactive_chat()

if __name__ == "__main__":
    main()
# Index.py - Multilingual version (English + Urdu) for creating embeddings and vector index
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Centralized Configuration System ===
class Config:
    """Centralized configuration - loads from environment variables"""
    
    def __init__(self):
        # Load settings from environment variables
        self.settings = self._load_config_from_env()
        
        # Apply settings
        self.INDEX_PATH = self.settings["index_path"]
        self.DATASET_PATH = self._find_dataset_path()
        self._validate_config()
    
    def _load_config_from_env(self):
        """Load configuration from environment variables with defaults for non-sensitive settings"""
        # Optional environment variables with defaults (Index.py doesn't need LLM settings)
        optional_vars = {
            "INDEX_PATH": ("index_path", "cancer_index_store"),
            "DATASET_PATH": ("dataset_path", "breast_cancer.json"),
            "SIMILARITY_TOP_K": ("similarity_top_k", 5),
            "LLM_TEMPERATURE": ("temperature", 0.3),
            "LLM_MAX_TOKENS": ("max_tokens", 1500),
            "COMBINE_SOURCES": ("combine_sources", True),
            "FALLBACK_MESSAGE": ("fallback_message", "Sorry, I don't know the answer."),
            "STRICT_BREAST_CANCER_ONLY": ("strict_breast_cancer_only", True)
        }
        
        config = {}
        
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
        
        logging.info("✅ Configuration loaded from environment variables")
        return config
    
    def _find_dataset_path(self):
        """Find the correct dataset path"""
        original_path = self.settings["dataset_path"]
        possible_paths = [
            original_path,
            f"DataSet/{original_path}",
            f"data/{original_path}",
            "DataSet/breast_cancer.json",
            "breast_cancer.json",
            "../DataSet/breast_cancer.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                if path != original_path:
                    logging.info(f"🔄 Using dataset at: {path}")
                else:
                    logging.info(f"✅ Dataset found at: {path}")
                return path
        
        logging.error(f"❌ Dataset not found in any location")
        return original_path  # Return original even if not found for error handling
    
    def _validate_config(self):
        """Validate configuration"""
        if not os.path.exists(self.DATASET_PATH):
            logging.error(f"❌ Dataset file not found: {self.DATASET_PATH}")
        else:
            logging.info(f"✅ Dataset found: {self.DATASET_PATH}")
        logging.info(f"✅ Index will be stored at: {self.INDEX_PATH}")

# Initialize configuration
config = Config()

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def create_vector_index():
    """
    Creates embeddings and builds vector index from dataset
    Supports both English and Urdu text.
    """
    try:
        from llama_index.core import VectorStoreIndex, Document, StorageContext
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SimpleNodeParser
        
        print("🚀 Starting Multilingual Vector Index Creation...")
        print("=" * 60)
        
        # Check dataset
        if not os.path.exists(config.DATASET_PATH):
            print(f"❌ Dataset not found: {config.DATASET_PATH}")
            return False
        
        # Load dataset
        print(f"📖 Loading dataset from: {config.DATASET_PATH}")
        with open(config.DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✅ Loaded dataset with {len(dataset)} entries")
        
        # Normalize dataset for multilingual consistency
        documents = []
        for item in dataset:
            if isinstance(item, dict):
                text = item.get('content') or item.get('text') or item.get('answer') or str(item)
                if not text or len(text.strip()) < 10:
                    continue  # skip empty
                metadata = {k: v for k, v in item.items() if k not in ['content', 'text', 'answer']}
                documents.append(Document(text=text.strip(), metadata=metadata))
            else:
                documents.append(Document(text=str(item)))
        
        print(f"✅ Created {len(documents)} documents for embedding")

        # === Multilingual embedding model ===
        # Supports 50+ languages including Urdu + English
        multilingual_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"🔧 Loading embedding model: {multilingual_model}")
        embed_model = HuggingFaceEmbedding(model_name=multilingual_model)
        
        # Create node parser
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        
        # Parse documents
        print("🔨 Parsing documents into nodes...")
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"✅ Created {len(nodes)} nodes")
        
        # Build index
        print("🏗️ Building multilingual vector index...")
        index = VectorStoreIndex(nodes=nodes, embed_model=embed_model, show_progress=True)
        
        # Persist
        os.makedirs(config.INDEX_PATH, exist_ok=True)
        print(f"💾 Saving index to: {config.INDEX_PATH}")
        index.storage_context.persist(persist_dir=config.INDEX_PATH)
        
        print("✅ Multilingual vector index created successfully!")
        print(f"📁 Index location: {config.INDEX_PATH}")
        print(f"📊 Total nodes embedded: {len(nodes)}")

        # Test retrieval in both languages
        retriever = index.as_retriever(similarity_top_k=2)
        print("🔍 Testing bilingual retrieval:")
        en_test = retriever.retrieve("What are the symptoms of breast cancer?")
        ur_test = retriever.retrieve("بریسٹ کینسر کی علامات کیا ہیں؟")
        print(f"✅ English test retrieved {len(en_test)} results")
        print(f"✅ Urdu test retrieved {len(ur_test)} results")
        
        print("\n🎉 Multilingual index ready for RAG pipeline!")
        return True

    except Exception as e:
        print(f"❌ Failed to create multilingual vector index: {e}")
        import traceback; traceback.print_exc()
        return False

def check_index_exists():
    if os.path.exists(config.INDEX_PATH):
        print(f"✅ Index already exists at: {config.INDEX_PATH}")
        return True
    else:
        print(f"❌ Index not found at: {config.INDEX_PATH}")
        return False

def main():
    print("🏥 Well Being Agent - Multilingual Index Creator")
    print("=" * 60)
    
    if check_index_exists():
        response = input("Index already exists. Recreate? (y/n): ").strip().lower()
        if response != 'y':
            print("Operation cancelled.")
            return
    
    success = create_vector_index()
    if success:
        print("\n🎯 Next steps:")
        print("1️⃣ Run Agent.py for RAG operations")
        print("2️⃣ Run app.py for web interface")
    else:
        print("\n💥 Index creation failed!")

if __name__ == "__main__":
    main()
# Index.py - Enhanced RAG Indexing Pipeline with structured metadata
import os
import json
import logging
from pathlib import Path

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# === Centralized Configuration System ===
class Config:
    """Centralized configuration - loads from config.json"""

    def __init__(self):
        self.settings = self._load_config_file()
        self.INDEX_PATH = self.settings["index_path"]
        self.DATASET_PATH = self._find_dataset_path()
        self.EMBEDDING_MODEL = self.settings.get(
            "embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.CHUNK_SIZE = self.settings.get("chunk_size", 512)
        self.CHUNK_OVERLAP = self.settings.get("chunk_overlap", 64)
        self.SIMILARITY_TOP_K = self.settings.get("similarity_top_k", 5)
        self._validate_config()

    def _load_config_file(self):
        config_file = os.path.join("config", "config.json")
        default_config = {
            "model_provider": "openrouter",
            "model_id": "deepseek/deepseek-chat-v3.1:free",
            "api_keys_folder": "config",
            "index_path": "cancer_index_store",
            "dataset_path": "breast_cancer_comprehensive.json",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "similarity_top_k": 5,
            "temperature": 0.2,
            "max_tokens": 500,
            "combine_sources": True,
            "fallback_message": "Sorry, I don't know the answer.",
            "strict_breast_cancer_only": True,
        }
        try:
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                merged_config = {**default_config, **loaded_config}
                logger.info("[OK] Configuration loaded from config/config.json")
                return merged_config
            else:
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=4)
                logger.info("[OK] Created default config/config.json file")
                return default_config
        except Exception as e:
            logger.error(f"[FAIL] Error loading config: {e}")
            return default_config

    def _find_dataset_path(self):
        original_path = self.settings.get("dataset_path", "breast_cancer_comprehensive.json")
        possible_paths = [
            original_path,
            f"DataSet/{original_path}",
            f"data/{original_path}",
            "DataSet/breast_cancer_comprehensive.json",
            "DataSet/breast_cancer.json",
            "breast_cancer_comprehensive.json",
            "breast_cancer.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"[OK] Dataset found at: {path}")
                return path
        logger.error("[FAIL] Dataset not found in any location")
        return original_path

    def _validate_config(self):
        if not os.path.exists(self.DATASET_PATH):
            logger.error(f"[FAIL] Dataset file not found: {self.DATASET_PATH}")
        else:
            logger.info(f"[OK] Dataset: {self.DATASET_PATH}")
        logger.info(f"[OK] Index path: {self.INDEX_PATH}")
        logger.info(f"[OK] Embedding model: {self.EMBEDDING_MODEL}")
        logger.info(f"[OK] Chunk size: {self.CHUNK_SIZE}, overlap: {self.CHUNK_OVERLAP}")


config = Config()


def load_and_prepare_documents():
    """
    Load dataset and prepare documents with rich metadata for indexing.
    Supports both the new comprehensive format and legacy format.
    """
    from llama_index.core import Document

    if not os.path.exists(config.DATASET_PATH):
        logger.error(f"[FAIL] Dataset not found: {config.DATASET_PATH}")
        return []

    logger.info(f"[*] Loading dataset from: {config.DATASET_PATH}")
    with open(config.DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"[OK] Loaded {len(dataset)} entries")

    documents = []
    seen_texts = set()

    for item in dataset:
        if not isinstance(item, dict):
            continue

        # Build the document text (combine question + answer for better retrieval)
        question = item.get("question", "")
        answer = item.get("answer") or item.get("content") or item.get("text") or ""
        topic = item.get("topic", "General")

        if not answer or len(answer.strip()) < 20:
            continue

        # Combine for richer semantic indexing
        if question:
            text = f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}"
        else:
            text = f"Topic: {topic}\n{answer}"

        # Deduplicate
        text_hash = hash(text.strip().lower()[:200])
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)

        # Build structured metadata
        metadata = {
            "id": item.get("id", ""),
            "topic": topic,
            "category": item.get("category", "general"),
            "subcategory": item.get("subcategory", ""),
            "source": item.get("source", ""),
            "language": item.get("language", "english"),
            "tags": ", ".join(item.get("tags", [])) if isinstance(item.get("tags"), list) else item.get("tags", ""),
            "has_question": bool(question),
        }

        documents.append(Document(text=text.strip(), metadata=metadata))

    # Also load legacy dataset if it exists and is different
    legacy_path = "DataSet/breast_cancer.json"
    if os.path.exists(legacy_path) and os.path.abspath(legacy_path) != os.path.abspath(config.DATASET_PATH):
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            logger.info(f"[*] Also loading legacy dataset: {legacy_path} ({len(legacy_data)} entries)")
            for item in legacy_data:
                if not isinstance(item, dict):
                    continue
                answer = item.get("answer") or item.get("content") or item.get("text") or ""
                question = item.get("question", "")
                topic = item.get("topic", "General")
                if not answer or len(answer.strip()) < 20:
                    continue
                if question:
                    text = f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}"
                else:
                    text = f"Topic: {topic}\n{answer}"
                text_hash = hash(text.strip().lower()[:200])
                if text_hash in seen_texts:
                    continue
                seen_texts.add(text_hash)
                metadata = {
                    "id": "",
                    "topic": topic,
                    "category": item.get("category", "general"),
                    "subcategory": "",
                    "source": "Legacy Dataset",
                    "language": "english",
                    "tags": "",
                    "has_question": bool(question),
                }
                documents.append(Document(text=text.strip(), metadata=metadata))
        except Exception as e:
            logger.warning(f"[WARN] Could not load legacy dataset: {e}")

    logger.info(f"[OK] Prepared {len(documents)} unique documents for indexing")
    return documents


def create_vector_index():
    """
    Create vector index with HuggingFace embeddings and structured metadata.
    Uses sentence-aware chunking for better retrieval quality.
    """
    try:
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SentenceSplitter

        print("[*] Starting Enhanced Vector Index Creation...")
        print("=" * 60)

        # Load documents
        documents = load_and_prepare_documents()
        if not documents:
            print("[FAIL] No documents to index")
            return False

        # Initialize embedding model (consistent across index & query)
        print(f"[*] Loading embedding model: {config.EMBEDDING_MODEL}")
        embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)

        # Create node parser with sentence-aware splitting
        node_parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

        # Parse documents into nodes
        print("[*] Parsing documents into nodes...")
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"[OK] Created {len(nodes)} nodes from {len(documents)} documents")

        # Build vector index
        print("[*] Building vector index...")
        index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            show_progress=True,
        )
        print("[OK] Vector index created successfully!")

        # Persist index
        os.makedirs(config.INDEX_PATH, exist_ok=True)
        print(f"[*] Saving index to: {config.INDEX_PATH}")
        index.storage_context.persist(persist_dir=config.INDEX_PATH)

        # Save index metadata
        index_meta = {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "num_documents": len(documents),
            "num_nodes": len(nodes),
            "dataset_path": config.DATASET_PATH,
        }
        meta_path = os.path.join(config.INDEX_PATH, "index_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(index_meta, f, indent=2)

        print(f"\n{'=' * 60}")
        print("[OK] Vector index created successfully!")
        print(f"  Index location: {config.INDEX_PATH}")
        print(f"  Total documents: {len(documents)}")
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Embedding model: {config.EMBEDDING_MODEL}")

        # Test retrieval
        print("\n[*] Testing retrieval...")
        retriever = index.as_retriever(similarity_top_k=3)

        test_queries = [
            "What are the symptoms of breast cancer?",
            "Can I breastfeed after mastectomy?",
            "How do I deal with chemotherapy anxiety?",
            "Will my hair grow back after treatment?",
        ]

        for query in test_queries:
            results = retriever.retrieve(query)
            scores = [f"{r.score:.3f}" for r in results if hasattr(r, "score")]
            print(f"  [OK] '{query[:50]}...' -> {len(results)} results, scores: {scores}")

        print("\n[SUCCESS] Index ready for RAG pipeline!")
        return True

    except Exception as e:
        print(f"[FAIL] Failed to create vector index: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_index_exists():
    if os.path.exists(config.INDEX_PATH):
        meta_path = os.path.join(config.INDEX_PATH, "index_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"[OK] Index exists at: {config.INDEX_PATH}")
            print(f"   Documents: {meta.get('num_documents', '?')}")
            print(f"   Nodes: {meta.get('num_nodes', '?')}")
            print(f"   Embedding: {meta.get('embedding_model', '?')}")
        else:
            print(f"[OK] Index exists at: {config.INDEX_PATH} (no metadata)")
        return True
    else:
        print(f"[!] Index not found at: {config.INDEX_PATH}")
        return False


def main():
    print("[+] Well Being Agent - Enhanced Index Creator")
    print("=" * 60)

    if check_index_exists():
        print("[!] Existing index found -- will recreate with latest dataset...")

    success = create_vector_index()
    if success:
        print("\n[OK] Next steps:")
        print("  1. Run Agent_v2.py for RAG operations")
        print("  2. Run app.py for web interface")
    else:
        print("\n[FAIL] Index creation failed!")


if __name__ == "__main__":
    main()
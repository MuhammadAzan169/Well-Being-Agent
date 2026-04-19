"""Index.py — Vector Index Builder for WellBeing Agent

Reads configuration from .env and builds a vector index from the breast cancer
dataset using HuggingFace embeddings and sentence-aware chunking.

Usage:
    python Index.py
"""

import os
import json
import logging
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("WellBeingAgent.Index")

# ── Configuration from .env ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_PATH = os.getenv("INDEX_PATH", "cancer_index_store")
DATASET_PATH = os.getenv("DATASET_PATH", "DataSet/breast_cancer_comprehensive.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))


def _resolve_dataset() -> str:
    """Find the dataset file, trying several common locations."""
    for p in [
        DATASET_PATH,
        f"DataSet/{DATASET_PATH}",
        "DataSet/breast_cancer_comprehensive.json",
        "DataSet/breast_cancer.json",
    ]:
        if os.path.exists(p):
            return p
    return DATASET_PATH


RESOLVED_DATASET = _resolve_dataset()

logger.info(f"📋 Dataset : {RESOLVED_DATASET}")
logger.info(f"📋 Index   : {INDEX_PATH}")
logger.info(f"📋 Embedding: {EMBEDDING_MODEL}")
logger.info(f"📋 Chunk   : {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}")


# ═══════════════════════════════════════════════════════════════════════════
# Document Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_documents():
    """Load JSON datasets and return a list of LlamaIndex Documents."""
    from llama_index.core import Document

    if not os.path.exists(RESOLVED_DATASET):
        logger.error(f"❌ Dataset not found: {RESOLVED_DATASET}")
        return []

    with open(RESOLVED_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} entries from {RESOLVED_DATASET}")

    docs: list = []
    seen: set = set()

    def _add(item: dict, source_label: str = ""):
        if not isinstance(item, dict):
            return
        answer = (
            item.get("answer")
            or item.get("content")
            or item.get("text")
            or ""
        )
        question = item.get("question", "")
        topic = item.get("topic", "General")

        if not answer or len(answer.strip()) < 20:
            return

        text = (
            f"Topic: {topic}\nQuestion: {question}\nAnswer: {answer}"
            if question
            else f"Topic: {topic}\n{answer}"
        )
        h = hash(text.strip().lower()[:200])
        if h in seen:
            return
        seen.add(h)

        tags = item.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)

        meta = {
            "id": item.get("id", ""),
            "topic": topic,
            "category": item.get("category", "general"),
            "subcategory": item.get("subcategory", ""),
            "source": item.get("source", source_label or ""),
            "language": item.get("language", "english"),
            "tags": tags or "",
        }
        docs.append(Document(text=text.strip(), metadata=meta))

    # Primary dataset
    for item in dataset:
        _add(item)

    # Also load legacy dataset if it differs from primary
    legacy = "DataSet/breast_cancer.json"
    if os.path.exists(legacy) and os.path.abspath(legacy) != os.path.abspath(RESOLVED_DATASET):
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            logger.info(f"Loading legacy dataset: {legacy} ({len(legacy_data)} entries)")
            for item in legacy_data:
                _add(item, source_label="Legacy Dataset")
        except Exception as e:
            logger.warning(f"Could not load legacy dataset: {e}")

    logger.info(f"✅ {len(docs)} unique documents prepared")
    return docs


# ═══════════════════════════════════════════════════════════════════════════
# Index Building
# ═══════════════════════════════════════════════════════════════════════════
def build_index() -> bool:
    """Build vector index with HuggingFace embeddings and sentence-aware chunking."""
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core.node_parser import SentenceSplitter

        print("=" * 60)
        print("  Building Vector Index")
        print("=" * 60)

        docs = load_documents()
        if not docs:
            print("❌ No documents to index")
            return False

        print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
        embed = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents(docs)
        print(f"✅ {len(nodes)} nodes from {len(docs)} documents")

        print("Building vector index…")
        index = VectorStoreIndex(nodes=nodes, embed_model=embed, show_progress=True)

        os.makedirs(INDEX_PATH, exist_ok=True)
        index.storage_context.persist(persist_dir=INDEX_PATH)

        # Save metadata
        meta = {
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_documents": len(docs),
            "num_nodes": len(nodes),
            "dataset_path": RESOLVED_DATASET,
        }
        with open(os.path.join(INDEX_PATH, "index_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n✅ Index saved to {INDEX_PATH}")
        print(f"   Documents: {len(docs)} | Nodes: {len(nodes)}")

        # Quick retrieval sanity-check
        print("\nTesting retrieval…")
        retriever = index.as_retriever(similarity_top_k=3)
        for q in [
            "breast cancer symptoms",
            "chemotherapy anxiety",
            "hair loss after treatment",
        ]:
            results = retriever.retrieve(q)
            scores = [f"{r.score:.3f}" for r in results if hasattr(r, "score")]
            print(f"  ✅ '{q}' → {len(results)} results, scores: {scores}")

        print("\n✅ Index ready!")
        return True

    except Exception as e:
        print(f"❌ Index build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_index_exists() -> bool:
    """Check whether a persisted index already exists."""
    required = ["docstore.json", "default__vector_store.json", "index_store.json"]
    return all(os.path.exists(os.path.join(INDEX_PATH, f)) for f in required)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Well Being Agent — Index Builder")
    print("=" * 60)
    if check_index_exists():
        print(f"Existing index found at {INDEX_PATH} — will rebuild…")
    build_index()

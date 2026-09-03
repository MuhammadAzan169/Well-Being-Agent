"""Vector index loading and building (LlamaIndex + HuggingFace embeddings)."""

import json
import os
from typing import Any, List, Tuple

# Limit thread spawning from PyTorch / OpenBLAS before any heavy import.
# This trims resident memory on single-core containers without affecting accuracy.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("WellBeingAgent.Index")


# ── Embedding model resolution ─────────────────────────────────────────────
def resolve_embedding_model() -> str:
    """Return the embedding model the persisted index was actually built with.

    A vector index is only meaningful when queries are embedded by the *same*
    model that produced the stored vectors. Two different 384-dim models
    (e.g. all-MiniLM-L6-v2 and bge-small-en-v1.5) load without error but return
    nonsense neighbours, so we trust ``index_metadata.json`` over the env var
    and warn loudly when they disagree.
    """
    configured = settings.EMBEDDING_MODEL
    meta_file = settings.index_dir / "index_metadata.json"
    if not meta_file.exists():
        return configured
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            built_with = json.load(f).get("embedding_model", "")
    except Exception:
        return configured
    if not built_with or built_with == configured:
        return configured
    logger.warning(
        f"⚠️  EMBEDDING_MODEL={configured!r} does not match the model the index "
        f"was built with ({built_with!r}). Using {built_with!r} to keep retrieval "
        f"correct. Run `python -m scripts.build_index` to rebuild with {configured!r}."
    )
    return built_with


# ── Index Loading ──────────────────────────────────────────────────────────
def load_index() -> Tuple[Any, Any]:
    """Load the persisted vector index and return (index, retriever)."""
    try:
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        index_dir = settings.index_dir
        if not index_dir.exists():
            logger.error(f"❌ Index directory not found: {index_dir}")
            return None, None

        model_name = resolve_embedding_model()
        logger.info(f"Loading embedding model: {model_name}")
        embed = FastEmbedEmbedding(model_name=model_name)
        ctx = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(ctx, embed_model=embed)
        retriever = index.as_retriever(similarity_top_k=settings.SIMILARITY_TOP_K)
        logger.info("✅ Vector index loaded successfully")
        return index, retriever
    except Exception as exc:
        logger.error(f"❌ Index load failed: {exc}")
        import traceback

        traceback.print_exc()
        return None, None


# ── Index Building ─────────────────────────────────────────────────────────
def _load_documents() -> List[Any]:
    """Load JSON datasets and return a list of LlamaIndex Documents."""
    from llama_index.core import Document

    dataset_file = settings.dataset_file
    if not dataset_file.exists():
        logger.error(f"❌ Dataset not found: {dataset_file}")
        return []

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} entries from {dataset_file}")

    docs: list = []
    seen: set = set()

    def _add(item: dict, source_label: str = "") -> None:
        if not isinstance(item, dict):
            return
        answer = item.get("answer") or item.get("content") or item.get("text") or ""
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
        docs.append(
            Document(
                text=text.strip(),
                metadata={
                    "id": item.get("id", ""),
                    "topic": topic,
                    "category": item.get("category", "general"),
                    "subcategory": item.get("subcategory", ""),
                    "source": item.get("source", source_label or ""),
                    "language": item.get("language", "english"),
                    "tags": tags or "",
                },
            )
        )

    for item in dataset:
        _add(item)

    # Also load a legacy dataset if present and distinct from the primary.
    legacy = settings.data_dir / "DataSet" / "breast_cancer.json"
    if legacy.exists() and legacy.resolve() != dataset_file.resolve():
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    _add(item, source_label="Legacy Dataset")
            logger.info(f"Loaded legacy dataset: {legacy}")
        except Exception as e:
            logger.warning(f"Could not load legacy dataset: {e}")

    logger.info(f"✅ {len(docs)} unique documents prepared")
    return docs


def build_index() -> bool:
    """Build the vector index from the dataset and persist it to disk."""
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        docs = _load_documents()
        if not docs:
            logger.error("❌ No documents to index")
            return False

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        embed = FastEmbedEmbedding(model_name=settings.EMBEDDING_MODEL)
        splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        nodes = splitter.get_nodes_from_documents(docs)
        logger.info(f"✅ {len(nodes)} nodes from {len(docs)} documents")

        index = VectorStoreIndex(nodes=nodes, embed_model=embed, show_progress=True)

        index_dir = settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(index_dir))

        meta = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "num_documents": len(docs),
            "num_nodes": len(nodes),
            "dataset_path": str(settings.dataset_file),
        }
        with open(os.path.join(index_dir, "index_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✅ Index saved to {index_dir} ({len(docs)} docs, {len(nodes)} nodes)")
        return True
    except Exception as e:
        logger.error(f"❌ Index build failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_index_exists() -> bool:
    """Return True if a persisted index already exists on disk."""
    required = ["docstore.json", "default__vector_store.json", "index_store.json"]
    return all((settings.index_dir / f).exists() for f in required)

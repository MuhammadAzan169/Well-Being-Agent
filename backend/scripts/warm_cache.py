"""Pre-download the fastembed ONNX embedding model at build time.

Render's free plan spins the service down after ~15 minutes of inactivity. If
the ~90 MB embedding model were downloaded on first use, every cold start would
pay for it (and risk timing out the health check). Running this during the build
puts the model on disk before the server ever starts.

Usage (from the backend/ directory):
    python -m scripts.warm_cache
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.services.rag.index import check_index_exists, resolve_embedding_model  # noqa: E402

logger = get_logger("WellBeingAgent.WarmCache")


def main() -> int:
    configure_logging()

    if not check_index_exists():
        logger.error(
            "❌ No persisted index found. Commit data/cancer_index_store, or run "
            "`python -m scripts.build_index` before deploying."
        )
        return 1

    model_name = resolve_embedding_model()
    logger.info(f"Downloading embedding model: {model_name}")

    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    embed = FastEmbedEmbedding(model_name=model_name)
    # Force an actual encode so the ONNX weights are fetched, not just referenced.
    vec = embed.get_text_embedding("warm up the embedding model cache")
    logger.info(f"✅ Embedding model ready ({len(vec)} dimensions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

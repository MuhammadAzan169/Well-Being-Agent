"""CLI to (re)build the vector index from the dataset.

Usage (from the backend/ directory):
    python -m scripts.build_index
"""

import sys
from pathlib import Path

# Ensure the backend root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging  # noqa: E402
from app.services.rag.index import build_index, check_index_exists  # noqa: E402


def main() -> int:
    configure_logging()
    if check_index_exists():
        print("Existing index found — rebuilding…")
    ok = build_index()
    print("✅ Index ready!" if ok else "❌ Index build failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

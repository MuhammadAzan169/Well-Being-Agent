"""Response cache and conversation logging.

The cache provides exact-match plus fuzzy similarity lookup with TTL expiry,
persisted as human-readable JSON. The conversation logger appends every turn
to a JSON file for audit/analysis. All paths come from settings so the service
behaves identically regardless of the working directory.
"""

import hashlib
import json
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("WellBeingAgent.Cache")


class ResponseCache:
    """Persistent response cache with exact + similarity lookup and TTL expiry."""

    def __init__(self, cache_file: Optional[Path] = None) -> None:
        self.cache_file: Path = cache_file or settings.cache_file
        self.store: Dict[str, Dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.lower().strip())

    def make_key(self, query: str) -> str:
        return hashlib.md5(self._normalize_query(query).encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self.make_key(query)
        if key in self.store:
            entry = self.store[key]
            if self._is_valid(entry):
                logger.info("✅ Cache HIT (exact)")
                return entry
            del self.store[key]
        return self._find_similar(query)

    def _is_valid(self, entry: Dict[str, Any]) -> bool:
        ts = entry.get("timestamp", 0)
        return (time.time() - ts) < settings.CACHE_TTL_HOURS * 3600

    def _find_similar(self, query: str) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_query(query)
        best_score = 0.0
        best_entry: Optional[Dict[str, Any]] = None
        for entry in self.store.values():
            if not self._is_valid(entry):
                continue
            cached_query = entry.get("query_normalized", "")
            score = SequenceMatcher(None, normalized, cached_query).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_score >= settings.CACHE_SIMILARITY_THRESHOLD and best_entry:
            logger.info(f"✅ Cache HIT (similarity={best_score:.2f})")
            return best_entry
        return None

    def put(
        self,
        query: str,
        response: str,
        language: str,
        sources: Optional[List[Dict]] = None,
        is_error: bool = False,
    ) -> None:
        self.store[self.make_key(query)] = {
            "query": query,
            "query_normalized": self._normalize_query(query),
            "response": response,
            "language": language,
            "sources": sources or [],
            "timestamp": time.time(),
            "is_error": is_error,
        }
        self._enforce_limit()
        self._save()

    def _save(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.store, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Cache save error: {exc}")

    def _load(self) -> None:
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
                self._prune_expired()
                logger.info(f"✅ Cache loaded: {len(self.store)} entries")
        except Exception:
            logger.warning("Cache file corrupt or missing — starting fresh")
            self.store = {}

    def _enforce_limit(self) -> None:
        """Drop the oldest entries once the cache exceeds CACHE_MAX_ENTRIES."""
        limit = settings.CACHE_MAX_ENTRIES
        if limit <= 0 or len(self.store) <= limit:
            return
        ordered = sorted(self.store.items(), key=lambda kv: kv[1].get("timestamp", 0))
        for key, _ in ordered[: len(self.store) - limit]:
            del self.store[key]
        logger.info(f"🗑️  Cache trimmed to the {limit} most recent entries")

    def _prune_expired(self) -> None:
        expired = [k for k, v in self.store.items() if not self._is_valid(v)]
        for k in expired:
            del self.store[k]
        if expired:
            logger.info(f"🗑️  Pruned {len(expired)} expired cache entries")
            self._save()

    def clear(self) -> None:
        self.store = {}
        self._save()
        logger.info("🗑️  Cache cleared")


class ConversationLogger:
    """Logs every conversation turn to a JSON file for audit / analysis."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = path or settings.conversations_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log(
        self,
        user_input: str,
        response: str,
        language: str,
        response_type: str = "text",
        sources: Optional[List[Dict]] = None,
    ) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        data.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "language": language,
            "response_type": response_type,
            "sources": sources or [],
        })

        # Keep only the most recent turns: this file is rewritten in full on
        # every request, so an unbounded log would slowly consume the memory
        # and request budget of a small instance.
        limit = settings.CONVERSATION_LOG_MAX_ENTRIES
        if limit > 0 and len(data) > limit:
            data = data[-limit:]

        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Conversation log write error: {exc}")

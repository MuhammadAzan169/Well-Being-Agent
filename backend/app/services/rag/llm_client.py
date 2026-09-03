"""LLM client with per-model key rotation and model-level fallback.

Strategy:
  For each model (in OPENROUTER_MODEL_1 … OPENROUTER_MODEL_N order):
    Try every key (OPENROUTER_API_KEY1 … OPENROUTER_API_KEY41).
    On success → return immediately.
    On permanent model error (400/404) → skip remaining keys, move to next model.
    On transient key error (429 / 5xx / timeout / auth) → try next key.
  If every (model, key) pair fails → return the language-appropriate fallback message.
"""

import time
from typing import List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.services.rag.prompts import system_prompt

logger = get_logger("WellBeingAgent.LLM")

FALLBACK_MESSAGE = (
    "I'm sorry, I'm unable to respond right now. "
    "Please try again shortly or consult your healthcare provider."
)
FALLBACK_MESSAGE_URDU = (
    "معذرت، میں ابھی جواب دینے سے قاصر ہوں۔ "
    "براہ کرم دوبارہ کوشش کریں یا اپنے ڈاکٹر سے رابطہ کریں۔"
)

# Failure categories returned by _call()
_OK = "ok"
_RATE_LIMIT = "rate_limit"      # 429 — try next key
_MODEL_ERROR = "model_error"    # 400/404 — skip this model entirely
_KEY_ERROR = "key_error"        # 401/403 — key invalid, try next key
_SERVER_ERROR = "server_error"  # 5xx — try next key
_TIMEOUT = "timeout"            # httpx timeout — try next key
_EMPTY = "empty"                # API returned blank text — try next key
_FORMAT_ERROR = "format_error"  # Unexpected JSON shape — skip model


def fallback_message(language: str = "english") -> str:
    return FALLBACK_MESSAGE_URDU if language == "urdu" else FALLBACK_MESSAGE


class LLMClient:
    """OpenAI-compatible client that exhausts all keys per model before switching models."""

    def __init__(self) -> None:
        self.api_keys: List[str] = settings.api_keys
        self.models: List[str] = settings.models
        self.http = httpx.Client(timeout=settings.LLM_TIMEOUT_SECONDS)
        if not self.api_keys:
            logger.warning("⚠️  No LLM API keys configured — LLM calls will fail")
        logger.info(
            f"📋 LLM provider={settings.LLM_PROVIDER} "
            f"models={len(self.models)} keys={len(self.api_keys)}"
        )

    def _call(
        self, model: str, key: str, prompt: str, language: str
    ) -> Tuple[Optional[str], str]:
        """
        Make one API request and categorise the outcome.

        Returns:
            (response_text, _OK) on success
            (None, <failure_category>) on any failure
        """
        url = f"{settings.LLM_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt(language)},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
        }
        try:
            resp = self.http.post(url, headers=headers, json=payload)

            if resp.status_code == 429:
                logger.warning(
                    f"Rate limited — model={model} key=...{key[-6:]}"
                )
                return None, _RATE_LIMIT

            # 404 = no such model. 400 = the model rejected the request shape
            # (commonly a retired or misspelled model id). Neither is fixable by
            # retrying with a different key, so skip the model outright instead
            # of burning every key against it.
            if resp.status_code in (400, 404):
                logger.warning(
                    f"Model unavailable ({resp.status_code}): {model} — skipping "
                    f"remaining keys for this model"
                )
                return None, _MODEL_ERROR

            if resp.status_code in (401, 403):
                logger.warning(
                    f"Auth failure {resp.status_code} — key=...{key[-6:]}"
                )
                return None, _KEY_ERROR

            if resp.status_code >= 500:
                logger.warning(
                    f"Server error {resp.status_code} — model={model} key=...{key[-6:]}"
                )
                return None, _SERVER_ERROR

            resp.raise_for_status()

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if not text:
                logger.warning("LLM returned empty response")
                return None, _EMPTY
            return text, _OK

        except httpx.TimeoutException:
            logger.warning(f"Timeout — model={model} key=...{key[-6:]}")
            return None, _TIMEOUT
        except (KeyError, IndexError) as exc:
            logger.error(f"Unexpected API response shape: {exc}")
            return None, _FORMAT_ERROR
        except Exception as exc:
            logger.error(f"Request error — model={model}: {exc}")
            return None, _SERVER_ERROR

    def complete(self, prompt: str, language: str) -> str:
        """
        Try every key for each model in order.

        Key failure (429 / 5xx / timeout / auth) → try next key.
        All keys fail for a model → try next model with all keys reset.
        Permanent model failure (400 / 404 / bad response shape) → skip to next model immediately.
        All models exhausted → return fallback message.
        """
        if not self.api_keys:
            logger.error("No API keys available")
            return fallback_message(language)

        for model_idx, model in enumerate(self.models):
            logger.info(
                f"🤖 Trying model {model_idx + 1}/{len(self.models)}: {model}"
            )
            skip_model = False

            for key_idx, key in enumerate(self.api_keys):
                text, status = self._call(model, key, prompt, language)

                if status == _OK:
                    logger.info(
                        f"✅ Success — model={model} key=#{key_idx + 1}"
                    )
                    return text  # type: ignore[return-value]

                if status in (_MODEL_ERROR, _FORMAT_ERROR):
                    skip_model = True
                    break

                # rate_limit / key_error / server_error / timeout / empty:
                # log and try the next key
                logger.debug(
                    f"Key #{key_idx + 1} → {status} (model={model}), trying next key"
                )

            if skip_model:
                logger.warning(
                    f"Permanent failure for model {model} — moving to next model"
                )
            else:
                logger.warning(
                    f"All {len(self.api_keys)} keys exhausted for model {model} "
                    f"— moving to next model"
                )

        logger.error("All models and all keys exhausted — returning fallback")
        return fallback_message(language)

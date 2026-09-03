"""Speech-to-text pipeline using faster-whisper (CTranslate2, CPU-friendly).

Loaded lazily only when ``ENABLE_VOICE`` is set; otherwise every call returns
a graceful error telling the user to type instead.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.services.language.utils import has_urdu_script

logger = get_logger("WellBeingAgent.Audio")

_whisper_model = None
_whisper_available: Optional[bool] = None  # None = not yet checked


def _load_whisper_pipeline():
    """Lazy-load the faster-whisper model; returns the model or None on failure."""
    global _whisper_model, _whisper_available

    if not settings.ENABLE_VOICE:
        _whisper_available = False
        return None
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel

        model_size = settings.WHISPER_MODEL_ID
        logger.info(f"Loading faster-whisper ({model_size}) on CPU…")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _whisper_available = True
        logger.info("✅ faster-whisper loaded successfully")
        return _whisper_model
    except ImportError as exc:
        logger.warning(f"faster-whisper not installed: {exc}")
        _whisper_available = False
        return None
    except Exception as exc:
        logger.error(f"Failed to load faster-whisper model: {exc}")
        _whisper_available = False
        return None


def is_whisper_available() -> bool:
    """Return True if Whisper is available (loaded or loadable)."""
    global _whisper_available
    if not settings.ENABLE_VOICE:
        return False
    if _whisper_model is not None:
        return True
    if _whisper_available is None:
        _load_whisper_pipeline()
    return bool(_whisper_available)


def transcribe_audio(audio_bytes: bytes, language_hint: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe audio bytes to text. Returns text/language/success/error dict."""
    model = _load_whisper_pipeline()
    if model is None:
        return {
            "text": "",
            "language": "english",
            "success": False,
            "error": "Speech-to-text service is not available. Please type your question instead.",
        }

    try:
        settings.recordings_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{timestamp}_{uuid.uuid4().hex[:8]}.webm"
        file_path = settings.recordings_dir / filename
        with open(file_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Saved audio to {file_path}")

        language: Optional[str] = None
        if language_hint:
            lang_map = {"urdu": "ur", "english": "en", "ur": "ur", "en": "en"}
            language = lang_map.get(language_hint.lower(), language_hint)

        t0 = time.time()
        segments, _info = model.transcribe(str(file_path), language=language, beam_size=5)
        text = "".join(segment.text for segment in segments).strip()
        elapsed = time.time() - t0

        if not text:
            return {
                "text": "",
                "language": "english",
                "success": False,
                "error": "No speech detected in the audio. Please try again.",
            }

        detected_lang = "urdu" if has_urdu_script(text) else "english"
        logger.info(f"✅ Transcribed ({elapsed:.1f}s): [{detected_lang}] {text[:80]}")
        return {"text": text, "language": detected_lang, "success": True, "error": None}
    except Exception as exc:
        logger.error(f"Transcription error: {exc}")
        return {
            "text": "",
            "language": "english",
            "success": False,
            "error": f"Transcription failed: {exc}. Please try again.",
        }

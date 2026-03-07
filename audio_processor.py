"""
audio_processor.py — Speech-to-Text Pipeline for WellBeing Agent

Uses Whisper Large v3 (via the `transformers` library from Hugging Face)
for high-quality, multilingual transcription with automatic language detection.

Supported flow:
  Audio bytes → save to temp file → transcribe with Whisper → detect language → return text + language
"""

import os
import re
import time
import tempfile
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("WellBeingAgent.Audio")

# ── Whisper Model State ───────────────────────────────────────────────────
_whisper_pipeline = None
_whisper_available: Optional[bool] = None  # None = not yet checked


def _load_whisper_pipeline():
    """
    Lazy-load Whisper Large v3 using Hugging Face transformers.
    Falls back gracefully if dependencies are missing.
    """
    global _whisper_pipeline, _whisper_available

    if _whisper_pipeline is not None:
        return _whisper_pipeline

    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model_id = "openai/whisper-large-v3"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading Whisper Large v3 on {device}…")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
        )
        _whisper_available = True
        logger.info("✅ Whisper Large v3 loaded successfully")
        return _whisper_pipeline

    except ImportError as exc:
        logger.warning(f"Whisper dependencies not installed: {exc}")
        logger.warning(
            "Install with: pip install transformers torch accelerate"
        )
        _whisper_available = False
        return None

    except Exception as exc:
        logger.error(f"Failed to load Whisper model: {exc}")
        _whisper_available = False
        return None


def is_whisper_available() -> bool:
    """Check whether Whisper is available (loaded or loadable)."""
    global _whisper_available
    if _whisper_pipeline is not None:
        return True
    if _whisper_available is None:
        _load_whisper_pipeline()
    return bool(_whisper_available)


def transcribe_audio(
    audio_bytes: bytes,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio bytes to text using Whisper Large v3.

    Args:
        audio_bytes: Raw audio data (supports webm, wav, mp3, etc.)
        language_hint: Optional language hint ('en', 'ur', 'english', 'urdu')

    Returns:
        Dict with keys:
            - text:     transcribed text
            - language: detected language ('urdu' or 'english')
            - success:  whether transcription succeeded
            - error:    error message if failed (None on success)
    """
    pipe = _load_whisper_pipeline()

    if pipe is None:
        logger.warning("Whisper not available — returning error")
        return {
            "text": "",
            "language": "english",
            "success": False,
            "error": (
                "Speech-to-text service is not available. "
                "Please type your question instead."
            ),
        }

    # Ensure audio output directory exists
    os.makedirs("static/audio", exist_ok=True)

    tmp_path: Optional[str] = None
    try:
        # Save bytes to a temporary file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(
            suffix=".webm", delete=False, dir="static/audio"
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Build generate_kwargs
        generate_kwargs: Dict[str, Any] = {}
        if language_hint:
            lang_map = {
                "urdu": "ur", "english": "en",
                "ur": "ur", "en": "en",
            }
            whisper_lang = lang_map.get(language_hint.lower(), language_hint)
            generate_kwargs["language"] = whisper_lang

        # Transcribe
        t0 = time.time()
        result = pipe(
            tmp_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )
        elapsed = time.time() - t0

        text = result.get("text", "").strip()
        if not text:
            return {
                "text": "",
                "language": "english",
                "success": False,
                "error": "No speech detected in the audio. Please try again.",
            }

        # ── Language Detection ─────────────────────────────────────────────
        # Priority 1: Use Whisper's own language prediction from chunk metadata.
        # Whisper embeds language tokens in the output — trust these over
        # text heuristics which false-positive on medical English ("cancer",
        # "chemo", "surgery" are in the Roman-Urdu word list).
        whisper_lang = _extract_whisper_language(result)

        if whisper_lang is not None:
            # Whisper told us the language
            if whisper_lang == "ur":
                detected_lang = "urdu"
            elif whisper_lang == "en":
                detected_lang = "english"
            else:
                # Non-English, non-Urdu — fall back to script check only
                detected_lang = _detect_audio_language_by_script(text)
        else:
            # Whisper gave no language hint — use script/text heuristics
            detected_lang = _detect_audio_language(text)

        logger.info(
            f"✅ Transcribed ({elapsed:.1f}s): [{detected_lang}] "
            f"{text[:80]}{'…' if len(text) > 80 else ''}"
        )

        return {
            "text": text,
            "language": detected_lang,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        logger.error(f"Transcription error: {exc}")
        return {
            "text": "",
            "language": "english",
            "success": False,
            "error": f"Transcription failed: {exc}. Please try again.",
        }

    finally:
        # Clean up temporary file
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _extract_whisper_language(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract Whisper's own language prediction from pipeline output.
    Whisper encodes the language in chunk metadata when return_timestamps=True.
    Returns a BCP-47 language code (e.g. 'en', 'ur') or None if unavailable.
    """
    # The transformers ASR pipeline stores per-chunk metadata in result["chunks"]
    chunks = result.get("chunks", [])
    if chunks:
        # Each chunk may have a "language" key set by Whisper's language token
        for chunk in chunks:
            lang = chunk.get("language")
            if lang:
                return lang

    # Some pipeline versions surface language at the top level
    lang = result.get("language")
    if lang:
        return lang

    return None


def _detect_audio_language_by_script(text: str) -> str:
    """
    Detect language using ONLY script (Unicode range) checks.
    Does NOT use Roman Urdu word matching — avoids false positives
    on English medical terms that overlap with the Roman Urdu word list.
    Returns 'urdu' or 'english'.
    """
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", text):
        return "urdu"
    return "english"


def _detect_audio_language(text: str) -> str:
    """
    Detect whether transcribed text is Urdu or English.
    Used only as a fallback when Whisper provides no language metadata.
    Returns 'urdu' or 'english'.
    """
    # 1) Check for Urdu script characters (most reliable)
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", text):
        return "urdu"

    # 2) Roman Urdu check — use ONLY grammar/pronoun words, NOT medical
    # terms like "cancer" / "chemo" that appear in English speech too.
    roman_urdu_grammar = {
        "mera", "meri", "mere", "mujhe", "apna", "apni", "apne",
        "bohat", "bohot", "kaise", "kyun", "kab",
        "hai", "hain", "tha", "thi",
        "nahi", "nahin", "haan", "bilkul",
        "batao", "batain", "chahiye", "sakta", "sakti",
        "shukria", "shukriya", "meharbani",
        "dard", "bukhar", "thakan", "kamzori",
        "ilaj", "ilaaj", "dawa", "dawai", "daktar",
    }
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if len(words & roman_urdu_grammar) >= 2:
        return "urdu"

    return "english"


def cleanup_old_audio_files(
    directory: str = "static/audio",
    max_age_seconds: int = 3600,
) -> int:
    """
    Remove audio files older than max_age_seconds.
    Returns the number of files removed.
    """
    if not os.path.isdir(directory):
        return 0

    removed = 0
    now = time.time()
    audio_extensions = (".webm", ".wav", ".mp3", ".ogg", ".m4a")

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath) and fname.lower().endswith(audio_extensions):
            try:
                if now - os.path.getmtime(fpath) > max_age_seconds:
                    os.unlink(fpath)
                    removed += 1
                    logger.debug(f"Cleaned old audio: {fname}")
            except Exception as exc:
                logger.warning(f"Failed to delete {fname}: {exc}")

    if removed > 0:
        logger.info(f"🗑️  Cleaned {removed} old audio file(s)")
    return removed

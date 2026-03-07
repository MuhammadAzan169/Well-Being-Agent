"""
language_utils.py — Language Detection and Text Utilities for WellBeing Agent

Supports:
  - Urdu script detection
  - Roman Urdu detection (Urdu written in Latin/English characters)
  - Language-aware text cleaning
  - Urdu spelling normalization
"""

import re
import logging
from typing import Optional

from langdetect import detect, DetectorFactory

# Ensure deterministic detection results
DetectorFactory.seed = 0

logger = logging.getLogger("WellBeingAgent.Language")

# ═══════════════════════════════════════════════════════════════════════════
# Roman Urdu Detection
# ═══════════════════════════════════════════════════════════════════════════

# Common Roman Urdu words / phrases used by Pakistani patients
ROMAN_URDU_WORDS = {
    # Greetings / common words
    "assalam", "walaikum", "salam", "khuda", "allah", "inshallah",
    "mashallah", "jazakallah", "shukria", "shukriya", "meharbani",
    # Medical / body
    "dard", "sar", "sir", "pet", "kamar", "seena", "hath", "pair",
    "jism", "khoon", "haddi", "dil", "jigar", "gurda", "sar dard",
    # Cancer related
    "cancer", "kenser", "kanser", "chemo", "chemotherapy", "radiation",
    "surgery", "operation", "biopsy", "tumor", "rasoli",
    "breast", "chhati", "seena", "mastectomy",
    # Symptoms
    "dard", "bukhar", "ulti", "matli", "thakan", "kamzori",
    "bhook", "neend", "sujan", "kharish", "baal", "girna",
    # Feelings
    "dar", "khauf", "fikar", "pareshani", "ghabrahat", "udasi",
    "mayoosi", "akela", "akeli", "mushkil", "takleef", "aziyat",
    # Actions / verbs
    "batao", "batain", "bataiye", "karo", "kijiye", "karein",
    "chahiye", "hona", "raha", "rahi", "sakta", "sakti",
    "kaise", "kya", "kyun", "kab", "kahan", "kaun",
    # Treatment
    "ilaj", "ilaaj", "dawa", "dawai", "doctor", "daktar",
    "hospital", "aspatal", "nurse", "operation",
    # Common Urdu phrases (romanized)
    "mujhe", "mera", "meri", "mere", "apna", "apni", "apne",
    "bohat", "bohot", "bahut", "acha", "achi", "theek",
    "haan", "nahi", "nahin", "bilkul", "zaroor", "zaruri",
    "ke baad", "ke doran", "ke liye", "ke bare", "ke sath",
    "ho raha", "ho rahi", "kar raha", "kar rahi",
    "kya hai", "kaise hai", "kyun hai",
    # Body/milk/breast specific
    "doodh", "dudh", "pilana", "bachcha", "bacche", "mamta",
}

# Threshold: if this many Roman Urdu words appear, classify as Urdu
ROMAN_URDU_THRESHOLD = 2


def detect_roman_urdu(text: str) -> bool:
    """
    Detect if text is Roman Urdu (Urdu written in English/Latin script).
    Uses a dictionary of common Roman Urdu words.

    Examples:
        "mera sir bohat dard kar raha hai" → True
        "I feel very tired after radiation" → False
    """
    if not text:
        return False

    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    matches = words & ROMAN_URDU_WORDS
    return len(matches) >= ROMAN_URDU_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════
# Language Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect if text is Urdu or English.
    Supports: Urdu script, Roman Urdu, and standard English.
    Returns 'urdu' or 'english'.
    """
    if not text or not text.strip():
        return "english"

    # 1) Quick check for Urdu script (Arabic-derived characters)
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+", text):
        return "urdu"

    # 2) Check for Roman Urdu
    if detect_roman_urdu(text):
        return "urdu"

    # 3) Use langdetect as fallback
    try:
        lang_code = detect(text)
        if lang_code in ("ur", "ar"):  # Arabic can sometimes match Urdu
            return "urdu"
        return "english"
    except Exception:
        return "english"


# ═══════════════════════════════════════════════════════════════════════════
# Urdu Text Cleaning / Normalization
# ═══════════════════════════════════════════════════════════════════════════

# Common Urdu spelling mistakes produced by LLMs
URDU_SPELLING_FIXES = {
    "مجہے": "مجھے",
    "کہےنسر": "کینسر",
    "ڈڈاکٹر": "ڈاکٹر",
    "ہےہ": "ہے",
    "مہےں": "میں",
    "ہےں": "ہیں",
    "ھے": "ہے",
    "ھوں": "ہوں",
    "ھیں": "ہیں",
    "ےے": "ے",
    "ںں": "ں",
    "ہہ": "ہ",
    "یی": "ی",
    "ے لہےے": "کے لیے",
    "کا ے لہےے": "کے لیے",
    "و ہےہ": "کو",
    "نہہےں": "نہیں",
    "بارے مہےں": "بارے میں",
    "کرہےں": "کریں",
    "بہترہےن": "بہترین",
    "برہےسٹ": "بریسٹ",
    "کہےموتھراپہے": "کیموتھراپی",
    "پروگرہوں": "پروگرام",
    "رکہیں": "رکھیں",
    "آرہوں": "آرام",
    "ہوں ": "ہوں ",
    "وتا": "ہوتا",
    "عہوں": "عام",
}


def clean_urdu_text(text: str) -> str:
    """
    Normalize common Urdu spelling mistakes and fix character issues.
    This is especially important for LLM-generated Urdu text.
    """
    if not text:
        return text

    for wrong, right in URDU_SPELLING_FIXES.items():
        text = text.replace(wrong, right)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove repeated Urdu punctuation
    text = re.sub(r"۔۔+", "۔", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Response Formatting
# ═══════════════════════════════════════════════════════════════════════════

def format_response(text: str, language: str = "english") -> str:
    """
    Apply language-specific formatting and cleaning to a response.
    """
    if not text:
        return text

    if language == "urdu":
        text = clean_urdu_text(text)

    return text.strip()


def map_whisper_lang_to_system(lang_code: str) -> str:
    """
    Map Whisper/ISO language codes to system language identifiers.
    Returns 'urdu' or 'english'.
    """
    urdu_codes = {"ur", "urdu"}
    return "urdu" if lang_code.lower() in urdu_codes else "english"

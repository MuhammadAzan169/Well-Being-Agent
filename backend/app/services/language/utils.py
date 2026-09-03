"""Language detection and text utilities.

Supports Urdu script detection, Roman Urdu detection (Urdu written in Latin
characters), language-aware text cleaning, and Urdu spelling normalization.
"""

import re

from langdetect import DetectorFactory, detect

from app.core.logging import get_logger

# Ensure deterministic detection results
DetectorFactory.seed = 0

logger = get_logger("WellBeingAgent.Language")

# ── Roman Urdu Detection ───────────────────────────────────────────────────
# Common Roman Urdu words used by Pakistani patients. Deliberately excludes
# English medical terms (cancer, chemo, surgery) that also appear in English.
ROMAN_URDU_WORDS = {
    "assalam", "walaikum", "salam", "khuda", "allah", "inshallah",
    "mashallah", "jazakallah", "shukria", "shukriya", "meharbani",
    "dard", "sar", "sir", "pet", "kamar", "seena", "hath", "pair",
    "jism", "khoon", "haddi", "dil", "jigar", "gurda", "sar dard",
    "kenser", "kanser", "rasoli", "chhati", "mastectomy",
    "bukhar", "ulti", "matli", "thakan", "kamzori",
    "bhook", "neend", "sujan", "kharish", "girna",
    "dar", "khauf", "fikar", "pareshani", "ghabrahat", "udasi",
    "mayoosi", "akela", "akeli", "mushkil", "takleef", "aziyat",
    "batao", "batain", "bataiye", "karo", "kijiye", "karein",
    "chahiye", "hona", "raha", "rahi", "sakta", "sakti",
    "kaise", "kya", "kyun", "kab", "kahan", "kaun",
    "ilaj", "ilaaj", "dawa", "dawai", "daktar", "aspatal",
    "mujhe", "mera", "meri", "mere", "apna", "apni", "apne",
    "bohat", "bohot", "bahut", "acha", "achi", "theek",
    "haan", "nahi", "nahin", "bilkul", "zaroor", "zaruri",
    "ke baad", "ke doran", "ke liye", "ke bare", "ke sath",
    "ho raha", "ho rahi", "kar raha", "kar rahi",
    "kya hai", "kaise hai", "kyun hai",
    "doodh", "dudh", "pilana", "bachcha", "bacche", "mamta",
}

ROMAN_URDU_THRESHOLD = 2

_URDU_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]+")


def has_urdu_script(text: str) -> bool:
    """Return True if the text contains Urdu/Arabic script characters."""
    return bool(_URDU_SCRIPT_RE.search(text or ""))


def detect_roman_urdu(text: str) -> bool:
    """Detect if text is Roman Urdu (Urdu written in Latin script)."""
    if not text:
        return False
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    return len(words & ROMAN_URDU_WORDS) >= ROMAN_URDU_THRESHOLD


def detect_language(text: str) -> str:
    """Detect whether text is 'urdu' or 'english'."""
    if not text or not text.strip():
        return "english"
    if has_urdu_script(text):
        return "urdu"
    if detect_roman_urdu(text):
        return "urdu"
    try:
        lang_code = detect(text)
        if lang_code in ("ur", "ar"):
            return "urdu"
        return "english"
    except Exception:
        return "english"


def map_whisper_lang_to_system(lang_code: str) -> str:
    """Map Whisper/ISO language codes to 'urdu' or 'english'."""
    return "urdu" if (lang_code or "").lower() in {"ur", "urdu"} else "english"


# ── Urdu Text Cleaning / Normalization ─────────────────────────────────────
# Common Urdu spelling mistakes produced by LLMs.
URDU_SPELLING_FIXES = {
    "مجہے": "مجھے", "کہےنسر": "کینسر", "ڈڈاکٹر": "ڈاکٹر",
    "ہےہ": "ہے", "مہےں": "میں", "ہےں": "ہیں",
    "ھے": "ہے", "ھوں": "ہوں", "ھیں": "ہیں",
    "ےے": "ے", "ںں": "ں", "ہہ": "ہ", "یی": "ی",
    "ے لہےے": "کے لیے", "کا ے لہےے": "کے لیے", "و ہےہ": "کو",
    "نہہےں": "نہیں", "بارے مہےں": "بارے میں", "کرہےں": "کریں",
    "بہترہےن": "بہترین", "برہےسٹ": "بریسٹ",
    "کہےموتھراپہے": "کیموتھراپی", "پروگرہوں": "پروگرام",
    "رکہیں": "رکھیں", "آرہوں": "آرام", "وتا": "ہوتا", "عہوں": "عام",
}


def clean_urdu_text(text: str) -> str:
    """Normalize Urdu spelling and strip foreign characters LLMs leak in."""
    if not text:
        return text
    for wrong, right in URDU_SPELLING_FIXES.items():
        text = text.replace(wrong, right)
    # Strip non-Urdu scripts (Devanagari, CJK, Vietnamese, combining marks).
    text = re.sub(r"[ऀ-ॿ]+", "", text)
    text = re.sub(r"[一-鿿]+", "", text)
    text = re.sub(r"[Ḁ-ỿ]+", "", text)
    text = re.sub(r"[̀-ͯ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"۔۔+", "۔", text)
    return text.strip()


def format_response(text: str, language: str = "english") -> str:
    """Apply language-specific formatting and cleaning to a response."""
    if not text:
        return text
    if language == "urdu":
        text = clean_urdu_text(text)
    return text.strip()

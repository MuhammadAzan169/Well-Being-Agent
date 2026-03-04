# language_utils.py
import re
from langdetect import detect, LangDetectError

def detect_query_language(text: str) -> str:
    """
    Detect if text is English or Urdu
    Returns: 'english' or 'urdu'
    """
    try:
        # First check for Urdu characters (more reliable)
        urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        if urdu_pattern.search(text):
            return 'urdu'
        
        # Then use langdetect for other cases
        detected_lang = detect(text)
        return 'urdu' if detected_lang == 'ur' else 'english'
        
    except LangDetectError:
        return 'english'
    except Exception:
        return 'english'

def is_urdu_text(text: str) -> bool:
    """Check if text contains Urdu characters"""
    urdu_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(urdu_pattern.search(text))
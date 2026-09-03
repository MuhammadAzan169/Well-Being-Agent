"""Emotional support analyzer — detects emotional needs for tone calibration."""

from typing import Any, Dict, List


class EmotionalAnalyzer:
    """Detects emotional needs in patient queries for tone calibration."""

    EXPLICIT_EN = [
        "scared", "afraid", "worried", "anxious", "fear", "nervous",
        "stressed", "overwhelmed", "depressed", "sad", "lonely", "alone",
        "hopeless", "can't cope", "struggling", "suffering", "terrified",
        "panic", "crying", "devastated", "angry", "frustrated", "exhausted",
        "broken", "lost", "helpless", "despair", "grief", "give up",
        "can't sleep", "why me", "feel like a burden", "no one understands",
    ]
    EXPLICIT_UR = [
        "خوف", "ڈر", "پریشانی", "فکر", "تنہائی", "اداسی", "مایوسی",
        "تکلیف", "گھبراہٹ", "بے چینی", "بے بسی", "رونا", "آنسو",
        "دکھ", "غم", "تھکاوٹ", "ٹوٹ", "اکیلا", "اکیلی", "مشکل",
        "نیند نہیں", "بھوک نہیں", "کیوں میں", "بوجھ", "برداشت",
    ]

    IMPLICIT_EN = [
        "hair loss", "bald", "body image", "husband", "wife", "partner",
        "children", "family", "tell my", "pregnant", "baby", "fertility",
        "breastfeed", "mastectomy", "die", "dying", "death", "survival rate",
        "recurrence", "stage 4", "metastatic", "chemo", "side effects",
        "pain", "nausea", "fatigue", "work", "money", "normal life",
    ]
    IMPLICIT_UR = [
        "بال", "شکل", "شوہر", "بیوی", "بچے", "خاندان",
        "حمل", "دودھ", "سرجری", "موت", "زندگی", "واپس آنا",
        "کیمو", "علاج", "درد", "متلی", "نوکری", "پیسے",
    ]

    VULN_EN = ["will i", "can i still", "am i going to", "what if", "how do i cope"]
    VULN_UR = ["کیا میں", "کب تک", "کیسے", "ممکن ہے", "اگر"]

    @classmethod
    def analyze(cls, query: str, language: str) -> Dict[str, Any]:
        """Return emotional analysis with needs_emotional_support flag and score."""
        q = query.lower()
        is_urdu = language == "urdu"

        explicit: List[str] = cls.EXPLICIT_UR if is_urdu else cls.EXPLICIT_EN
        implicit: List[str] = cls.IMPLICIT_UR if is_urdu else cls.IMPLICIT_EN
        vulns: List[str] = cls.VULN_UR if is_urdu else cls.VULN_EN

        e_score = sum(2 for t in explicit if t in q)
        i_score = sum(1 for t in implicit if t in q)
        v_score = sum(1 for p in vulns if p in q)
        total = e_score + i_score + v_score

        return {"needs_emotional_support": total > 0, "score": total}

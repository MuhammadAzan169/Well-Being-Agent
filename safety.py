"""
safety.py — Content Safety Validation and Guardrails for WellBeing Agent

Provides:
  - Crisis detection (suicidal ideation, self-harm)
  - Off-topic query filtering
  - Dangerous medical advice detection in responses
  - Medical disclaimer injection
  - Prescription / medication prescription prevention

⚠️ IMPORTANT: This module ensures the agent never provides medical prescriptions,
   always flags crisis situations, and stays within its breast cancer support scope.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger("WellBeingAgent.Safety")


class SafetyValidator:
    """Validates queries and responses for safety, relevance, and sensitivity."""

    # ── Allowed Topics (breast cancer & well-being related) ───────────────

    ALLOWED_TOPICS = [
        # Medical
        "breast cancer", "cancer", "tumor", "oncology", "chemotherapy", "radiation",
        "mastectomy", "lumpectomy", "biopsy", "mammogram", "screening", "diagnosis",
        "treatment", "surgery", "recovery", "survivorship", "recurrence", "metastatic",
        "fertility", "pregnancy", "breastfeeding", "lactation", "hormone therapy",
        "tamoxifen", "aromatase inhibitor", "targeted therapy", "immunotherapy",
        # Side effects
        "side effects", "fatigue", "nausea", "hair loss", "lymphedema", "neuropathy",
        "headache", "dizziness", "vomiting", "appetite", "weight",
        # Emotional
        "emotional support", "anxiety", "depression", "coping", "mental health",
        "body image", "intimacy", "relationships", "family", "children",
        # Lifestyle
        "nutrition", "diet", "exercise", "fitness", "sleep", "pain",
        "well-being", "self-care", "mindfulness", "meditation", "stress",
        # Support
        "genetics", "brca", "clinical trials", "support groups", "financial",
        "follow-up", "reconstruction", "prosthetics", "palliative", "hospice",
        "immune system", "bone health", "heart health", "cognitive", "chemo brain",
        "caregiver", "support system", "counseling", "therapy", "psychologist",
        # Emotional keywords
        "scared", "afraid", "worried", "overwhelmed", "hopeless", "lonely",
        "stressed", "anxious", "depressed", "sad", "angry", "frustrated",
        "help", "cope", "support", "comfort", "hope", "strength",
        # Body / milk related
        "milk", "breast milk", "doodh", "one breast", "removed",
        # Greetings / general
        "hello", "hi", "hey", "good morning", "good evening", "thank you",
        "thanks", "how are you", "what can you do", "who are you",
        # Urdu keywords
        "کینسر", "بریسٹ", "علاج", "کیموتھراپی", "درد", "تھکاوٹ",
        "سرجری", "دودھ", "بال", "خوف", "پریشانی",
    ]

    # ── Crisis Patterns ───────────────────────────────────────────────────

    CRISIS_PATTERNS = [
        r"\b(suicid|kill\s*my\s*self|end\s*my\s*life|want\s*to\s*die|self[\s-]*harm)\b",
        r"\b(overdose|hurt\s*myself|no\s*reason\s*to\s*live)\b",
        r"\b(jump off|hang myself|slit my|cut myself)\b",
        # Urdu crisis patterns
        r"(خودکشی|مرنا چاہ|زندگی ختم|خود کو مار)",
    ]

    # ── Off-Topic Patterns ────────────────────────────────────────────────

    OFF_TOPIC_PATTERNS = [
        r"\b(stock|crypto|bitcoin|invest|trading|forex)\b",
        r"\b(weather forecast|sports score|game result|movie review|music|recipe|cook)\b",
        r"\b(politics|election|president|government|war|military)\b",
        r"\b(write\s*code|programming|software|javascript|python|html|css)\b",
        r"\b(homework|essay|math\s*problem|calculate|algebra)\b",
        r"\b(dating|tinder|love letter|pickup line)\b",
    ]

    # ── Dangerous Advice Patterns (in responses) ──────────────────────────

    DANGEROUS_PATTERNS = [
        (
            r"\b(stop|discontinue|quit)\s+(?:taking\s+)?(?:all\s+)?(?:your\s+)?(?:medications?|medicine|treatment|chemo|radiation)",
            "Always consult your doctor before making any changes to your treatment plan.",
        ),
        (
            r"\bdon'?t\s+(?:see|visit|go\s+to)\s+(?:a\s+)?doctor",
            "Regular medical consultations are an important part of your care.",
        ),
        (
            r"\bcure[ds]?\s+(?:your\s+)?cancer\s+(?:with|using)\s+(?:herbs?|supplements?|essential\s+oils?|turmeric|garlic)",
            "Please rely on evidence-based treatments recommended by your oncology team.",
        ),
        (
            r"\btake\s+\d+\s*(?:mg|ml|tablets?|pills?|capsules?)\s+(?:of\s+)?\w+",
            "I cannot prescribe specific dosages. Please consult your doctor for medication guidance.",
        ),
        (
            r"\bi\s+(?:recommend|prescribe|suggest)\s+(?:you\s+)?(?:take|start|use)\s+\w+\s+\d+\s*(?:mg|ml|times)",
            "I cannot prescribe medications. Your healthcare team can advise on appropriate treatments.",
        ),
    ]

    # ── Crisis Responses ──────────────────────────────────────────────────

    CRISIS_RESPONSE_EN = (
        "💛 I can hear that you're going through an incredibly difficult time, "
        "and I want you to know that your life matters deeply.\n\n"
        "Please reach out to one of these resources right now:\n\n"
        "🆘 **988 Suicide & Crisis Lifeline**: Call or text **988** (US)\n"
        "📞 **Crisis Text Line**: Text **HOME** to **741741**\n"
        "📞 **Cancer Support Helpline**: **1-888-793-9355**\n"
        "🌐 **International Association for Suicide Prevention**: "
        "https://www.iasp.info/resources/Crisis_Centres/\n\n"
        "You are not alone, and trained counselors are available 24/7. "
        "Please reach out — you deserve support and care. 💛"
    )

    CRISIS_RESPONSE_UR = (
        "💛 میں سمجھ سکتی ہوں کہ آپ بہت مشکل وقت سے گزر رہے ہیں۔ "
        "آپ کی زندگی بہت اہم ہے۔\n\n"
        "براہ کرم ابھی مدد حاصل کریں:\n\n"
        "🆘 **ایمرجنسی ہیلپ لائن**: فوری طبی مدد کے لیے ہسپتال سے رابطہ کریں\n"
        "📞 **امنگ ہیلپ لائن**: 0311-7786264\n"
        "📞 **روزن ہیلپ لائن**: 0800-22-444\n\n"
        "آپ اکیلے نہیں ہیں۔ مدد دستیاب ہے۔ 💛"
    )

    # ── Off-Topic Responses ───────────────────────────────────────────────

    OFF_TOPIC_RESPONSE_EN = (
        "I appreciate your question, but I'm specifically designed to support "
        "breast cancer patients with well-being information, emotional support, "
        "and practical guidance.\n\n"
        "Here are things I can help with:\n"
        "🏥 Breast cancer symptoms, diagnosis, and understanding treatment options\n"
        "💊 Managing side effects and recovery guidance\n"
        "💛 Emotional support, anxiety, and coping strategies\n"
        "👶 Fertility, pregnancy, and breastfeeding concerns\n"
        "🏃 Exercise, nutrition, and lifestyle during treatment\n"
        "🧠 Mental health and body image support\n\n"
        "Is there anything related to breast cancer well-being I can help you with?"
    )

    OFF_TOPIC_RESPONSE_UR = (
        "آپ کے سوال کا شکریہ، لیکن میں خاص طور پر بریسٹ کینسر کے مریضوں کی "
        "فلاح و بہبود، جذباتی مدد، اور عملی رہنمائی کے لیے بنائی گئی ہوں۔\n\n"
        "میں ان موضوعات میں مدد کر سکتی ہوں:\n"
        "🏥 بریسٹ کینسر کی علامات، تشخیص اور علاج کو سمجھنا\n"
        "💊 ضمنی اثرات کا انتظام اور صحت یابی\n"
        "💛 جذباتی مدد اور حوصلہ افزائی\n"
        "👶 زرخیزی، حمل اور دودھ پلانے کے سوالات\n"
        "🏃 ورزش، غذائیت اور طرز زندگی\n\n"
        "کیا بریسٹ کینسر سے متعلق کوئی سوال ہے جس میں میں مدد کر سکتی ہوں؟"
    )

    # ══════════════════════════════════════════════════════════════════════
    # Query Validation
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def validate_query(cls, query: str, language: str = "english") -> Dict:
        """
        Validate a user query for safety, crisis detection, and topic relevance.

        Returns:
            dict with keys: is_safe, is_on_topic, is_crisis, response
            If response is not None, it should be returned directly to the user.
        """
        query_lower = query.lower().strip()

        # 1) Crisis detection (highest priority)
        for pattern in cls.CRISIS_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"⚠️  CRISIS detected: {query[:50]}…")
                return {
                    "is_safe": True,
                    "is_on_topic": True,
                    "is_crisis": True,
                    "response": (
                        cls.CRISIS_RESPONSE_UR
                        if language == "urdu"
                        else cls.CRISIS_RESPONSE_EN
                    ),
                }

        # 2) Off-topic detection
        off_score = sum(
            1 for p in cls.OFF_TOPIC_PATTERNS if re.search(p, query_lower)
        )
        on_score = sum(1 for t in cls.ALLOWED_TOPICS if t in query_lower)

        # Only flag as off-topic if clearly unrelated AND no on-topic keywords
        if off_score > 0 and on_score == 0 and len(query_lower.split()) > 3:
            logger.info(f"Off-topic query detected: {query[:50]}…")
            return {
                "is_safe": True,
                "is_on_topic": False,
                "is_crisis": False,
                "response": (
                    cls.OFF_TOPIC_RESPONSE_UR
                    if language == "urdu"
                    else cls.OFF_TOPIC_RESPONSE_EN
                ),
            }

        # 3) Query is safe and on-topic
        return {
            "is_safe": True,
            "is_on_topic": True,
            "is_crisis": False,
            "response": None,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Response Validation
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def validate_response(cls, response: str, language: str = "english") -> str:
        """
        Check an LLM response for dangerous medical advice.
        Appends warnings if dangerous patterns are found.
        """
        for pattern, warning_en in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning("⚠️  Dangerous advice detected — adding disclaimer")
                if language == "urdu":
                    response += (
                        "\n\n⚠️ اہم: براہ کرم کوئی بھی فیصلہ اپنے ڈاکٹر "
                        "سے مشورے کے بغیر نہ کریں۔"
                    )
                else:
                    response += f"\n\n⚠️ Important: {warning_en}"
        return response

    @classmethod
    def add_medical_disclaimer(cls, response: str, language: str = "english") -> str:
        """
        Append a gentle medical disclaimer to the response if not already present.
        """
        disclaimer_ur = (
            "\n\nاپنی صحت کی دیکھ بھال ٹیم سے اپنے خدشات پر بات کرنا یاد رکھیں۔"
        )
        disclaimer_en = (
            "\n\nRemember to discuss any concerns with your healthcare team."
        )
        disclaimer = disclaimer_ur if language == "urdu" else disclaimer_en

        if disclaimer.strip() not in response:
            response += disclaimer

        return response

# safety.py - Content safety validation and guardrails for breast cancer chatbot
import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class SafetyValidator:
    """Validates queries and responses for safety, relevance, and sensitivity."""

    # Topics the chatbot is qualified to discuss
    ALLOWED_TOPICS = [
        "breast cancer", "cancer", "tumor", "oncology", "chemotherapy", "radiation",
        "mastectomy", "lumpectomy", "biopsy", "mammogram", "screening", "diagnosis",
        "treatment", "surgery", "recovery", "survivorship", "recurrence", "metastatic",
        "fertility", "pregnancy", "breastfeeding", "lactation", "hormone therapy",
        "tamoxifen", "aromatase inhibitor", "targeted therapy", "immunotherapy",
        "side effects", "fatigue", "nausea", "hair loss", "lymphedema", "neuropathy",
        "emotional support", "anxiety", "depression", "coping", "mental health",
        "body image", "intimacy", "relationships", "family", "children",
        "nutrition", "diet", "exercise", "fitness", "sleep", "pain",
        "genetics", "brca", "clinical trials", "support groups", "financial",
        "follow-up", "reconstruction", "prosthetics", "palliative", "hospice",
        "immune system", "bone health", "heart health", "cognitive", "chemo brain",
        # Emotional keywords
        "scared", "afraid", "worried", "overwhelmed", "hopeless", "lonely",
        "stressed", "anxious", "depressed", "sad", "angry", "frustrated",
        "help", "cope", "support", "comfort", "hope", "strength",
        # Greetings and conversational
        "hello", "hi", "hey", "good morning", "good evening", "thank you",
        "thanks", "how are you", "what can you do", "who are you",
    ]

    # Dangerous patterns that require immediate safety responses
    CRISIS_PATTERNS = [
        r'\b(suicid|kill\s*my\s*self|end\s*my\s*life|want\s*to\s*die|self[\s-]*harm)\b',
        r'\b(overdose|hurt\s*myself|no\s*reason\s*to\s*live)\b',
    ]

    # Off-topic patterns
    OFF_TOPIC_PATTERNS = [
        r'\b(stock|crypto|bitcoin|invest|trading|forex)\b',
        r'\b(weather|sports|game|movie|music|recipe|cook)\b',
        r'\b(politics|election|president|government|war)\b',
        r'\b(write\s*code|programming|software|javascript|python)\b',
        r'\b(homework|essay|math\s*problem|calculate)\b',
    ]

    CRISIS_RESPONSE_EN = (
        "💛 I can hear that you're going through an incredibly difficult time, "
        "and I want you to know that your life matters deeply. "
        "Please reach out to one of these resources right now:\n\n"
        "🆘 **988 Suicide & Crisis Lifeline**: Call or text **988** (US)\n"
        "📞 **Crisis Text Line**: Text **HOME** to **741741**\n"
        "📞 **Cancer Support Helpline**: **1-888-793-9355**\n"
        "🌐 **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n"
        "You are not alone, and trained counselors are available 24/7. "
        "Please reach out — you deserve support and care. 💛"
    )

    CRISIS_RESPONSE_UR = (
        "💛 میں سمجھ سکتی ہوں کہ آپ بہت مشکل وقت سے گزر رہے ہیں۔ "
        "آپ کی زندگی بہت اہم ہے۔ براہ کرم ابھی مدد حاصل کریں:\n\n"
        "🆘 **ایمرجنسی ہیلپ لائن**: فوری طبی مدد کے لیے ہسپتال سے رابطہ کریں\n"
        "📞 **امنگ ہیلپ لائن**: 0311-7786264\n\n"
        "آپ اکیلے نہیں ہیں۔ مدد دستیاب ہے۔ 💛"
    )

    OFF_TOPIC_RESPONSE_EN = (
        "I appreciate your question, but I'm specifically designed to support "
        "breast cancer patients with medical information, emotional support, "
        "and practical guidance. I'm not the best resource for that particular topic.\n\n"
        "Here are some things I can help you with:\n"
        "🏥 Breast cancer symptoms, diagnosis, and treatment options\n"
        "💊 Side effects management and recovery guidance\n"
        "💛 Emotional support, anxiety, and coping strategies\n"
        "👶 Fertility, pregnancy, and breastfeeding after treatment\n"
        "🏃 Exercise, nutrition, and lifestyle during treatment\n\n"
        "Is there anything related to breast cancer I can help you with?"
    )

    OFF_TOPIC_RESPONSE_UR = (
        "آپ کے سوال کا شکریہ، لیکن میں خاص طور پر بریسٹ کینسر کے مریضوں کی "
        "طبی معلومات، جذباتی مدد، اور عملی رہنمائی کے لیے بنائی گئی ہوں۔\n\n"
        "میں ان موضوعات میں آپ کی مدد کر سکتی ہوں:\n"
        "🏥 بریسٹ کینسر کی علامات، تشخیص اور علاج\n"
        "💊 ضمنی اثرات کا انتظام اور صحت یابی\n"
        "💛 جذباتی مدد اور حوصلہ افزائی\n"
        "👶 زرخیزی، حمل اور دودھ پلانا\n\n"
        "کیا بریسٹ کینسر سے متعلق کوئی سوال ہے جس میں آپ کی مدد کر سکوں؟"
    )

    @classmethod
    def validate_query(cls, query: str, language: str = "english") -> Dict:
        """
        Validate user query for safety and relevance.
        Returns dict with: is_safe, is_on_topic, is_crisis, response (if applicable)
        """
        query_lower = query.lower().strip()

        # Check for crisis patterns first
        for pattern in cls.CRISIS_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"⚠️ CRISIS query detected: {query[:50]}...")
                return {
                    "is_safe": True,
                    "is_on_topic": True,
                    "is_crisis": True,
                    "response": cls.CRISIS_RESPONSE_UR if language == "urdu" else cls.CRISIS_RESPONSE_EN,
                }

        # Check for off-topic queries (only if clearly off-topic)
        off_topic_score = 0
        on_topic_score = 0

        for pattern in cls.OFF_TOPIC_PATTERNS:
            if re.search(pattern, query_lower):
                off_topic_score += 1

        for topic in cls.ALLOWED_TOPICS:
            if topic in query_lower:
                on_topic_score += 1

        # Only flag as off-topic if clearly unrelated and no on-topic keywords
        if off_topic_score > 0 and on_topic_score == 0 and len(query_lower.split()) > 3:
            logger.info(f"📋 Off-topic query detected: {query[:50]}...")
            return {
                "is_safe": True,
                "is_on_topic": False,
                "is_crisis": False,
                "response": cls.OFF_TOPIC_RESPONSE_UR if language == "urdu" else cls.OFF_TOPIC_RESPONSE_EN,
            }

        return {
            "is_safe": True,
            "is_on_topic": True,
            "is_crisis": False,
            "response": None,
        }

    @classmethod
    def validate_response(cls, response: str, language: str = "english") -> str:
        """
        Validate and sanitize the LLM response before sending to user.
        Ensures response doesn't contain harmful medical advice.
        """
        # Check for dangerous medical advice patterns
        dangerous_patterns = [
            (r'\bstop\s+(?:taking\s+)?(?:all\s+)?(?:your\s+)?(?:medications?|medicine|treatment|chemo)',
             "Always consult your doctor before making any changes to your treatment."),
            (r'\bdon\'?t\s+(?:see|visit|go\s+to)\s+(?:a\s+)?doctor',
             "Regular medical consultations are important for your care."),
            (r'\bcure[ds]?\s+(?:your\s+)?cancer\s+(?:with|using)\s+(?:herbs?|supplements?|essential\s+oils?)',
             "Please rely on evidence-based treatments recommended by your oncology team."),
        ]

        for pattern, warning in dangerous_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                logger.warning(f"⚠️ Dangerous advice detected in response, adding disclaimer")
                if language == "urdu":
                    response += "\n\n⚠️ اہم: براہ کرم کوئی بھی فیصلہ اپنے ڈاکٹر سے مشورے کے بغیر نہ کریں۔"
                else:
                    response += f"\n\n⚠️ Important: {warning}"

        return response

    @classmethod
    def add_medical_disclaimer(cls, response: str, language: str = "english") -> str:
        """Add a gentle medical disclaimer when providing clinical information."""
        disclaimer_en = "\n\nRemember to discuss any concerns with your healthcare team."
        disclaimer_ur = "\n\nاپنی صحت کی دیکھ بھال ٹیم سے اپنے خدشات پر بات کرنا یاد رکھیں۔"

        disclaimer = disclaimer_ur if language == "urdu" else disclaimer_en

        if disclaimer.strip() not in response:
            response += disclaimer

        return response

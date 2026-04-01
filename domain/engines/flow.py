from typing import Tuple
import logging

from backend.infrastructure.ml.bert_service import predict_intent_detailed
from backend.intents import (
    INTENT_ACCOUNT_BALANCE,
    INTENT_APPLY_LOAN,
    INTENT_CHECK_APPLICATION_STATUS,
    INTENT_GENERAL_QUERY,
    INTENT_REGISTER_COMPLAINT,
    INTENT_SCHEME_QUERY,
)
from backend.services.rag_service import retrieve_scheme

logger = logging.getLogger(__name__)

RESPONSES = {
    INTENT_ACCOUNT_BALANCE: {
        "en": {
            "confirmation": "Checking your account balance.",
            "explanation": "Your balance is 24,530 rupees.",
            "next_step": "Would you like recent transactions?",
        },
        "hi": {
            "confirmation": "आपका खाता बैलेंस देखा जा रहा है।",
            "explanation": "आपके खाते में 24,530 रुपये हैं।",
            "next_step": "क्या आप हाल की लेनदेन देखना चाहते हैं?",
        },
    },
    INTENT_APPLY_LOAN: {
        "en": {
            "confirmation": "I can provide loan application information.",
            "explanation": "You can apply for personal, home, or business loans through eligible channels.",
            "next_step": "Would you like the required documents list?",
        },
        "hi": {
            "confirmation": "मैं लोन आवेदन की जानकारी देने में मदद कर सकता हूँ।",
            "explanation": "आप व्यक्तिगत, गृह या व्यवसाय ऋण के लिए पात्र चैनलों से आवेदन कर सकते हैं।",
            "next_step": "क्या आप जरूरी दस्तावेजों की सूची जानना चाहते हैं?",
        },
    },
    INTENT_CHECK_APPLICATION_STATUS: {
        "en": {
            "confirmation": "I can help check your application status.",
            "explanation": "Please keep your application reference number ready for status tracking.",
            "next_step": "Would you like step-by-step tracking guidance?",
        },
        "hi": {
            "confirmation": "मैं आपके आवेदन की स्थिति देखने में मदद कर सकता हूँ।",
            "explanation": "कृपया स्टेटस ट्रैक करने के लिए अपना आवेदन संदर्भ नंबर तैयार रखें।",
            "next_step": "क्या आप स्टेटस जांचने के चरण जानना चाहते हैं?",
        },
    },
    INTENT_REGISTER_COMPLAINT: {
        "en": {
            "confirmation": "I can help you register a complaint.",
            "explanation": "Please share your issue details, department, and location to proceed.",
            "next_step": "Would you like a complaint registration checklist?",
        },
        "hi": {
            "confirmation": "मैं शिकायत दर्ज कराने में आपकी मदद कर सकता हूँ।",
            "explanation": "आगे बढ़ने के लिए कृपया समस्या, विभाग और स्थान की जानकारी दें।",
            "next_step": "क्या आप शिकायत दर्ज करने की चेकलिस्ट चाहते हैं?",
        },
    },
    INTENT_GENERAL_QUERY: {
        "en": {
            "confirmation": "I couldn't understand your request.",
            "explanation": "Please try asking again in a short sentence.",
            "next_step": "You can ask about government schemes, loans, or complaints.",
        },
        "hi": {
            "confirmation": "मैं आपकी बात समझ नहीं पाया।",
            "explanation": "कृपया एक छोटे वाक्य में फिर से पूछें।",
            "next_step": "आप सरकारी योजनाओं, लोन या शिकायत के बारे में पूछ सकते हैं।",
        },
    },
}


def generate_response(language: str, transcript: str) -> Tuple[dict, str, float]:
    lang = "hi" if (language or "").strip().lower() == "hi" else "en"

    # 1) RAG first
    logger.debug("Evaluating scheme retrieval")
    scheme = retrieve_scheme(transcript, lang)
    if scheme:
        intent = INTENT_SCHEME_QUERY
        confidence = 1.0
        logger.info("intent_detected intent=%s confidence=%.3f fallback_used=%s source=%s", intent, confidence, False, "rag")
        return scheme, intent, confidence

    # 2) Intent model only after RAG miss
    decision = predict_intent_detailed(transcript)
    raw_intent = decision.get("raw_intent", "")
    intent = decision["primary_intent"]
    confidence = float(decision["confidence"])
    used_fallback = bool(decision["fallback_used"])

    if intent not in RESPONSES:
        logger.warning("unrecognized_intent intent=%s confidence=%.3f", intent, confidence)
        intent = INTENT_GENERAL_QUERY
        used_fallback = True

    logger.info(
        "intent_detected intent=%s confidence=%.3f fallback_used=%s source=%s raw_intent=%s",
        intent,
        confidence,
        used_fallback,
        decision.get("source", "model_or_fallback"),
        raw_intent,
    )
    return RESPONSES[intent][lang], intent, confidence

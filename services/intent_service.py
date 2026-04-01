import time
import asyncio
import logging
import os
import re
import threading
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    pd = None

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    rapidfuzz_fuzz = None

from ..intents import (
    ACTION_INTENTS,
    INFO_INTENTS,
    INTENT_ACCOUNT_BALANCE,
    INTENT_APPLY_LOAN,
    INTENT_CHECK_APPLICATION_STATUS,
    INTENT_GENERAL_QUERY,
    INTENT_REGISTER_COMPLAINT,
    INTENT_SCHEME_QUERY,
    VALID_INTENTS,
    calibrate_confidence,
    get_flexible_intent_threshold,
    get_intent_threshold,
    keyword_intent_signal,
    migrate_intent,
    normalize_intent,
)
from backend.core.logger import log_event
from backend.core.intent_analytics import record_intent_event
from backend.core.metrics import record_fallback
from backend.shared.security.privacy import fingerprint_text
from ..text_normalizer import normalize_for_intent


logger = logging.getLogger(__name__)


INTENT_PROVIDE_INFO = "provide_information"
INTENT_FALLBACK = "fallback"
HIGH_CONFIDENCE_THRESHOLD = 0.70
MEDIUM_CONFIDENCE_THRESHOLD = 0.40

HYBRID_APPLICATION_KEYWORDS = {
    "apply",
    "apply process",
    "aply process",
    "application process",
    "form",
    "registration",
    "register",
    "kaise kare",
    "kaise apply kare",
    "kya karna hai",
    "what karna hai",
    "kaise banega",
}

HYBRID_BENEFIT_KEYWORDS = {
    "kitna milega",
    "benefit",
    "paisa",
    "amount",
}

HYBRID_DOCUMENT_KEYWORDS = {
    "documents kya chahiye",
    "document kya chahiye",
    "documents required",
    "required documents",
    "kaunse documents",
    "kagaz",
    "dastavej",
    "दस्तावेज",
}

HYBRID_SCHEME_INFO_KEYWORDS = {
    "kya hai",
    "yojna kya hai",
    "scheme kya hai",
}

HYBRID_SCHEME_KEYWORDS = {
    "yojna",
    "yojana",
    "scheme",
    "card",
    "loan",
    "pension",
    "gas",
}

SURFACE_BENEFIT_KEYWORDS = {
    "benefit",
    "benefits",
    "लाभ",
    "kitna milega",
    "milega",
    "free",
    "paisa",
    "amount",
    "subsidy",
}

SURFACE_DOCUMENT_KEYWORDS = {
    "document",
    "documents",
    "documents required",
    "required documents",
    "kagaz",
    "कागज",
    "dastavej",
    "दस्तावेज",
    "proof",
}

SURFACE_INTENT_APPLICATION_PROCESS = "application_process"
SURFACE_INTENT_BENEFITS = "benefits"
SURFACE_INTENT_DOCUMENTS_REQUIRED = "documents_required"
SURFACE_INTENT_CHECK_APPLICATION_STATUS = INTENT_CHECK_APPLICATION_STATUS
SURFACE_INTENT_SCHEME_QUERY = INTENT_SCHEME_QUERY
SURFACE_INTENT_GENERAL_QUERY = INTENT_GENERAL_QUERY

SURFACE_TO_CANONICAL_INTENT_MAP: Dict[str, str] = {
    SURFACE_INTENT_APPLICATION_PROCESS: INTENT_APPLY_LOAN,
    SURFACE_INTENT_BENEFITS: INTENT_SCHEME_QUERY,
    SURFACE_INTENT_DOCUMENTS_REQUIRED: INTENT_SCHEME_QUERY,
    SURFACE_INTENT_CHECK_APPLICATION_STATUS: INTENT_CHECK_APPLICATION_STATUS,
    SURFACE_INTENT_SCHEME_QUERY: INTENT_SCHEME_QUERY,
    SURFACE_INTENT_GENERAL_QUERY: INTENT_GENERAL_QUERY,
    INTENT_APPLY_LOAN: INTENT_APPLY_LOAN,
    INTENT_CHECK_APPLICATION_STATUS: INTENT_CHECK_APPLICATION_STATUS,
    INTENT_SCHEME_QUERY: INTENT_SCHEME_QUERY,
    INTENT_GENERAL_QUERY: INTENT_GENERAL_QUERY,
    INTENT_FALLBACK: INTENT_FALLBACK,
    INTENT_PROVIDE_INFO: INTENT_PROVIDE_INFO,
}

CANONICAL_TO_SURFACE_INTENT_MAP: Dict[str, str] = {
    INTENT_APPLY_LOAN: SURFACE_INTENT_APPLICATION_PROCESS,
    INTENT_CHECK_APPLICATION_STATUS: SURFACE_INTENT_CHECK_APPLICATION_STATUS,
    INTENT_SCHEME_QUERY: SURFACE_INTENT_SCHEME_QUERY,
    INTENT_GENERAL_QUERY: SURFACE_INTENT_GENERAL_QUERY,
    INTENT_FALLBACK: INTENT_FALLBACK,
    INTENT_PROVIDE_INFO: INTENT_PROVIDE_INFO,
}

# Order matters: more specific rules first, generic rules last.
STRONG_PATTERN_LOCKS: tuple[tuple[str, str], ...] = (
    ("documents kya chahiye", SURFACE_INTENT_DOCUMENTS_REQUIRED),
    ("documents what need", SURFACE_INTENT_DOCUMENTS_REQUIRED),
    ("kitna milega", SURFACE_INTENT_BENEFITS),
    ("kaise banega", SURFACE_INTENT_APPLICATION_PROCESS),
    ("kya hai", SURFACE_INTENT_SCHEME_QUERY),
)


CORRECTION_MARKERS = {
    "wrong",
    "change",
    "edit",
    "update",
    "not correct",
    "गलत",
    "बदल",
    "सुधार",
    "change it",
}

HINGLISH_REPLACEMENTS = {
    "apply karna": "apply",
    "apply karo": "apply",
    "yojana": "yojana",
    "yojna": "yojana",
    "kya": "what",
    "chahiye": "need",
}

KEYWORD_RULES: Dict[str, set[str]] = {
    INTENT_APPLY_LOAN: {
        "apply",
        "register",
        "application",
        "apply now",
        "आवेदन",
        "loan apply",
    },
    INTENT_SCHEME_QUERY: {
        "what",
        "scheme",
        "yojana",
        "योजना",
        "kisan",
        "किसान",
        "eligibility",
    },
    "correction": {"wrong", "गलत", "बदल", "सुधार", "change", "edit"},
    "greeting": {"hello", "hi", "namaste", "नमस्ते", "hey"},
}

ACTION_KEYWORDS = {
    "apply",
    "application",
    "start form",
    "form bharna",
    "form fill",
    "fill",
    "register",
    "loan chahiye",
    "loan chaiye",
    "submit",
    "status",
    "track",
    "complaint",
    "loan",
    "autofill",
    "step",
}

INFO_KEYWORDS = {
    "what",
    "kya hai",
    "kya hota hai",
    "which",
    "tell me",
    "explain",
    "details",
    "about",
    "eligibility",
    "benefits",
    "scheme",
    "yojana",
    "pm",
    "process",
    "documents",
    "pm kisan",
    "pmay",
    "ayushman",
    "yojana",
}

FOLLOWUP_INFO_KEYWORDS = {
    "eligibility",
    "benefits",
    "documents",
    "how to apply",
    "apply process",
    "details",
    "more info",
    "more information",
    "isme kya milta hai",
    "kaun apply kar sakta hai",
}

SCHEME_CATEGORY_KEYWORDS: Dict[str, set[str]] = {
    "farmer": {"farmer", "kisan", "krishi", "agri", "agriculture"},
    "student": {"student", "vidyarthi", "scholarship", "education", "edu"},
    "women": {"women", "woman", "mahila", "ladki", "female"},
}

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    INTENT_APPLY_LOAN: ["apply", "loan", "application", "apply now", "start application", "loan chahiye"],
    INTENT_CHECK_APPLICATION_STATUS: ["status", "track", "reference", "application status", "check status"],
    INTENT_REGISTER_COMPLAINT: ["complaint", "issue", "problem", "register complaint", "grievance"],
    INTENT_ACCOUNT_BALANCE: ["balance", "account balance", "bank balance", "saldo"],
    INTENT_SCHEME_QUERY: ["scheme", "yojana", "benefits", "eligibility", "documents", "information"],
}

ACTION_INTENT_HINTS: Dict[str, set[str]] = {
    INTENT_CHECK_APPLICATION_STATUS: {"status", "track", "application status", "check status", "pending", "nahi aya", "kab milega"},
    INTENT_REGISTER_COMPLAINT: {"complaint", "grievance", "issue", "problem", "register complaint", "shikayat"},
    INTENT_ACCOUNT_BALANCE: {"balance", "account balance", "bank balance", "passbook", "saldo"},
    INTENT_APPLY_LOAN: {"apply", "application", "register", "form", "loan apply", "start application"},
}

INTENT_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "final_voice_ready_dataset.csv"
_INTENT_DATASET_LOCK = threading.RLock()
_INTENT_DATASET_CACHE: Dict[str, Any] = {
    "rows": [],
    "exact_map": {},
    "first_token_index": {},
    "token_index": {},
    "intent_groups": {},
    "query_cache": {},
    "loaded": False,
    "row_count": 0,
    "path": str(INTENT_DATASET_PATH),
}

_RUNTIME_ANALYTICS_LOCK = threading.RLock()
MAX_TRACKED_RUNTIME_QUERIES = 256
CONSISTENCY_CHECK_RUNS = 3
FALLBACK_TREND_WINDOW_SIZE = 50
FALLBACK_SPIKE_MIN_RATE = 0.35
FALLBACK_SPIKE_DELTA = 0.20
UNUSUAL_CONFIDENCE_DROP_DELTA = 0.35
_RUNTIME_ANALYTICS: Dict[str, Any] = {
    "total_queries": 0,
    "fallback_queries": 0,
    "intent_frequency": {},
    "failed_queries": {},
    "consistency_mismatch_events": 0,
    "confidence_drop_events": 0,
    "fallback_rate_alerts": 0,
    "surface_pattern_drift_events": 0,
    "surface_canonical_drift_events": 0,
    "query_recent_intents": {},
    "query_last_confidence": {},
    "recent_fallback_window": [],
    "recent_fallback_window_rate": 0.0,
    "confidence_distribution": {
        "lt_0_40": 0,
        "0_40_to_0_60": 0,
        "0_60_to_0_80": 0,
        "gte_0_80": 0,
    },
}


def _tokenize_for_match(value: str) -> List[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", _normalize_query_key(value))
    return [token for token in normalized.split(" ") if token]


def _token_overlap_score(query_tokens: set[str], row_tokens: set[str]) -> float:
    if not query_tokens or not row_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(row_tokens))
    union_size = max(1, len(query_tokens.union(row_tokens)))
    return (overlap / float(union_size)) * 100.0


def _substring_match_score(query: str, row_query: str) -> float:
    if not query or not row_query:
        return 0.0
    if query in row_query or row_query in query:
        return 100.0
    return 0.0


def _fuzzy_match_score(query: str, row_query: str) -> float:
    if not query or not row_query:
        return 0.0
    if rapidfuzz_fuzz is not None:
        return float(rapidfuzz_fuzz.token_set_ratio(query, row_query))
    return float(SequenceMatcher(a=query, b=row_query).ratio() * 100.0)


def _increment_runtime_alert_counter(counter_name: str, amount: int = 1) -> None:
    delta = max(0, int(amount))
    if delta <= 0:
        return
    with _RUNTIME_ANALYTICS_LOCK:
        _RUNTIME_ANALYTICS[counter_name] = int(_RUNTIME_ANALYTICS.get(counter_name, 0)) + delta


def _trim_runtime_query_tracking() -> None:
    query_recent_intents = _RUNTIME_ANALYTICS.setdefault("query_recent_intents", {})
    query_last_confidence = _RUNTIME_ANALYTICS.setdefault("query_last_confidence", {})
    while len(query_recent_intents) > MAX_TRACKED_RUNTIME_QUERIES:
        oldest = next(iter(query_recent_intents))
        query_recent_intents.pop(oldest, None)
        query_last_confidence.pop(oldest, None)


def _expected_surface_intent_from_patterns(query: str) -> str:
    lowered = _normalize_query_key(query)
    if not lowered:
        return ""

    for pattern, expected_surface in STRONG_PATTERN_LOCKS:
        if pattern in lowered:
            return expected_surface

    if any(keyword in lowered for keyword in SURFACE_DOCUMENT_KEYWORDS):
        return SURFACE_INTENT_DOCUMENTS_REQUIRED

    if any(keyword in lowered for keyword in SURFACE_BENEFIT_KEYWORDS):
        return SURFACE_INTENT_BENEFITS

    if any(keyword in lowered for keyword in HYBRID_APPLICATION_KEYWORDS):
        return SURFACE_INTENT_APPLICATION_PROCESS

    if any(keyword in lowered for keyword in HYBRID_SCHEME_INFO_KEYWORDS):
        return SURFACE_INTENT_SCHEME_QUERY

    return ""


def _record_runtime_intent_analytics(intent: str, is_fallback: bool, query_fingerprint: str, confidence: float) -> Dict[str, Any]:
    alerts: Dict[str, Any] = {
        "consistency_mismatch": False,
        "confidence_drop": False,
        "fallback_rate_spike": False,
    }
    bounded_confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    with _RUNTIME_ANALYTICS_LOCK:
        _RUNTIME_ANALYTICS["total_queries"] = int(_RUNTIME_ANALYTICS.get("total_queries", 0)) + 1
        intent_frequency = _RUNTIME_ANALYTICS.setdefault("intent_frequency", {})
        intent_frequency[intent] = int(intent_frequency.get(intent, 0)) + 1

        confidence_distribution = _RUNTIME_ANALYTICS.setdefault(
            "confidence_distribution",
            {
                "lt_0_40": 0,
                "0_40_to_0_60": 0,
                "0_60_to_0_80": 0,
                "gte_0_80": 0,
            },
        )
        if bounded_confidence < 0.40:
            bucket = "lt_0_40"
        elif bounded_confidence < 0.60:
            bucket = "0_40_to_0_60"
        elif bounded_confidence < 0.80:
            bucket = "0_60_to_0_80"
        else:
            bucket = "gte_0_80"
        confidence_distribution[bucket] = int(confidence_distribution.get(bucket, 0)) + 1

        if is_fallback:
            _RUNTIME_ANALYTICS["fallback_queries"] = int(_RUNTIME_ANALYTICS.get("fallback_queries", 0)) + 1
            failed_queries = _RUNTIME_ANALYTICS.setdefault("failed_queries", {})
            failed_queries[query_fingerprint] = int(failed_queries.get(query_fingerprint, 0)) + 1

        query_key = (query_fingerprint or "").strip()
        if query_key:
            query_recent_intents = _RUNTIME_ANALYTICS.setdefault("query_recent_intents", {})
            history = list(query_recent_intents.get(query_key, []))
            history.append(intent)
            history = history[-CONSISTENCY_CHECK_RUNS:]
            query_recent_intents[query_key] = history

            if len(history) == CONSISTENCY_CHECK_RUNS and len(set(history)) > 1:
                _RUNTIME_ANALYTICS["consistency_mismatch_events"] = int(_RUNTIME_ANALYTICS.get("consistency_mismatch_events", 0)) + 1
                alerts["consistency_mismatch"] = True
                alerts["recent_intents"] = list(history)

            query_last_confidence = _RUNTIME_ANALYTICS.setdefault("query_last_confidence", {})
            previous_confidence = float(query_last_confidence.get(query_key, bounded_confidence))
            confidence_drop = previous_confidence - bounded_confidence
            if previous_confidence >= 0.55 and confidence_drop >= UNUSUAL_CONFIDENCE_DROP_DELTA:
                _RUNTIME_ANALYTICS["confidence_drop_events"] = int(_RUNTIME_ANALYTICS.get("confidence_drop_events", 0)) + 1
                alerts["confidence_drop"] = True
                alerts["previous_confidence"] = round(previous_confidence, 4)
                alerts["current_confidence"] = round(bounded_confidence, 4)
                alerts["drop_amount"] = round(confidence_drop, 4)
            query_last_confidence[query_key] = bounded_confidence

            _trim_runtime_query_tracking()

        recent_fallback_window = list(_RUNTIME_ANALYTICS.get("recent_fallback_window") or [])
        recent_fallback_window.append(1 if is_fallback else 0)
        if len(recent_fallback_window) > FALLBACK_TREND_WINDOW_SIZE:
            recent_fallback_window = recent_fallback_window[-FALLBACK_TREND_WINDOW_SIZE:]
        _RUNTIME_ANALYTICS["recent_fallback_window"] = recent_fallback_window

        window_rate = (sum(recent_fallback_window) / float(len(recent_fallback_window))) if recent_fallback_window else 0.0
        _RUNTIME_ANALYTICS["recent_fallback_window_rate"] = round(window_rate, 4)

        total_queries = int(_RUNTIME_ANALYTICS.get("total_queries", 0))
        fallback_queries = int(_RUNTIME_ANALYTICS.get("fallback_queries", 0))
        global_rate = fallback_queries / float(max(1, total_queries))
        if (
            len(recent_fallback_window) >= max(10, FALLBACK_TREND_WINDOW_SIZE // 2)
            and window_rate >= FALLBACK_SPIKE_MIN_RATE
            and (window_rate - global_rate) >= FALLBACK_SPIKE_DELTA
        ):
            _RUNTIME_ANALYTICS["fallback_rate_alerts"] = int(_RUNTIME_ANALYTICS.get("fallback_rate_alerts", 0)) + 1
            alerts["fallback_rate_spike"] = True
            alerts["recent_fallback_rate"] = round(window_rate, 4)
            alerts["global_fallback_rate"] = round(global_rate, 4)

        total_queries = int(_RUNTIME_ANALYTICS.get("total_queries", 0))
        if total_queries and total_queries % 100 == 0:
            fallback_queries = int(_RUNTIME_ANALYTICS.get("fallback_queries", 0))
            logger.info(
                "intent_runtime_analytics total_queries=%s fallback_rate=%.4f confidence_distribution=%s",
                total_queries,
                fallback_queries / float(max(1, total_queries)),
                dict(confidence_distribution),
            )
    return alerts


def get_intent_runtime_analytics() -> Dict[str, Any]:
    with _RUNTIME_ANALYTICS_LOCK:
        total_queries = int(_RUNTIME_ANALYTICS.get("total_queries", 0))
        fallback_queries = int(_RUNTIME_ANALYTICS.get("fallback_queries", 0))
        failed_queries = dict(_RUNTIME_ANALYTICS.get("failed_queries", {}))
        top_failed_queries = sorted(
            failed_queries.items(),
            key=lambda item: int(item[1]),
            reverse=True,
        )[:10]
        return {
            "total_queries": total_queries,
            "fallback_queries": fallback_queries,
            "fallback_rate": round(fallback_queries / float(max(1, total_queries)), 4),
            "intent_frequency": dict(_RUNTIME_ANALYTICS.get("intent_frequency", {})),
            "failed_queries": failed_queries,
            "top_failed_queries": top_failed_queries,
            "confidence_distribution": dict(_RUNTIME_ANALYTICS.get("confidence_distribution", {})),
            "consistency_mismatch_events": int(_RUNTIME_ANALYTICS.get("consistency_mismatch_events", 0)),
            "confidence_drop_events": int(_RUNTIME_ANALYTICS.get("confidence_drop_events", 0)),
            "fallback_rate_alerts": int(_RUNTIME_ANALYTICS.get("fallback_rate_alerts", 0)),
            "surface_pattern_drift_events": int(_RUNTIME_ANALYTICS.get("surface_pattern_drift_events", 0)),
            "surface_canonical_drift_events": int(_RUNTIME_ANALYTICS.get("surface_canonical_drift_events", 0)),
            "recent_fallback_window_rate": float(_RUNTIME_ANALYTICS.get("recent_fallback_window_rate", 0.0)),
            "tracked_queries": int(len(_RUNTIME_ANALYTICS.get("query_recent_intents") or {})),
        }


def _dynamic_dataset_threshold(query_token_count: int) -> float:
    # Short queries are noisier, so require much stronger matches.
    if query_token_count < 3:
        return 86.0
    if query_token_count <= 5:
        return 70.0
    return 64.0


def _query_category(tokens: set[str]) -> str:
    for category, keywords in SCHEME_CATEGORY_KEYWORDS.items():
        if tokens.intersection(keywords):
            return category
    return ""


def _scheme_category(scheme_name: str) -> str:
    scheme_tokens = set(_tokenize_for_match(scheme_name))
    return _query_category(scheme_tokens)


def _scheme_boost(query: str, query_tokens: set[str], row_scheme: str) -> float:
    scheme_text = _normalize_query_key(row_scheme)
    if not scheme_text:
        return 0.0
    scheme_tokens = set(_tokenize_for_match(scheme_text))
    if not scheme_tokens:
        return 0.0

    if scheme_text in query:
        return 18.0

    overlap_ratio = len(query_tokens.intersection(scheme_tokens)) / float(max(1, len(scheme_tokens)))
    if overlap_ratio >= 0.6:
        return 12.0
    if overlap_ratio >= 0.3:
        return 6.0
    return 0.0


def _category_mismatch_penalty(query_tokens: set[str], row_scheme: str) -> float:
    query_category = _query_category(query_tokens)
    scheme_category = _scheme_category(row_scheme)
    if query_category and scheme_category and query_category != scheme_category:
        return 15.0
    return 0.0


def _ml_predict_intent_detailed(text: str, session_context: Optional[dict] = None) -> dict:
    from backend.infrastructure.ml.bert_service import predict_intent_detailed

    return predict_intent_detailed(text, session_context=session_context)


def _ml_fallback_intent(text: str) -> Tuple[str, float]:
    from backend.infrastructure.ml.bert_service import fallback_intent

    return fallback_intent(text)


def _contains_keyword(query: str, keywords: set[str]) -> bool:
    lower = query.lower()
    return any(keyword in lower for keyword in keywords)


def detect_intent_and_mode(
    query: str,
    predicted_intent: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Tuple[str, str]:
    text = (query or "").strip().lower()
    if str(predicted_intent or "").strip().lower() == INTENT_FALLBACK:
        return INTENT_GENERAL_QUERY, "clarify"

    canonical_predicted_intent, _ = normalize_intent(predicted_intent, default=INTENT_GENERAL_QUERY)

    if not text:
        return INTENT_GENERAL_QUERY, "info"

    has_action = _contains_keyword(text, ACTION_KEYWORDS)
    has_info = _contains_keyword(text, INFO_KEYWORDS)

    action_intent = ""
    for candidate_intent in (
        INTENT_CHECK_APPLICATION_STATUS,
        INTENT_REGISTER_COMPLAINT,
        INTENT_ACCOUNT_BALANCE,
        INTENT_APPLY_LOAN,
    ):
        if _contains_keyword(text, ACTION_INTENT_HINTS.get(candidate_intent, set())):
            action_intent = candidate_intent
            break

    if has_action and has_info:
        return INTENT_SCHEME_QUERY, "info"

    if action_intent:
        return action_intent, "action"

    if has_action:
        return canonical_predicted_intent if canonical_predicted_intent in ACTION_INTENTS else INTENT_APPLY_LOAN, "action"

    if has_info:
        return INTENT_SCHEME_QUERY, "info"

    if canonical_predicted_intent in ACTION_INTENTS:
        return canonical_predicted_intent, "action"

    if canonical_predicted_intent in INFO_INTENTS:
        return canonical_predicted_intent, "info"

    if confidence is not None and confidence >= 0.65:
        if canonical_predicted_intent in INFO_INTENTS:
            return canonical_predicted_intent, "info"
        if canonical_predicted_intent in ACTION_INTENTS:
            return canonical_predicted_intent, "action"

    return INTENT_GENERAL_QUERY, "info"


def is_followup_info_query(query: str) -> bool:
    text = (query or "").strip().lower()
    if not text:
        return False
    return _contains_keyword(text, FOLLOWUP_INFO_KEYWORDS)


def _normalize_intent_text(text: str) -> str:
    return (text or "").strip().lower()


def detect_multi_intents(text: str) -> List[str]:
    query = _normalize_intent_text(text)
    if not query:
        return []

    scored: List[tuple[str, int]] = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in query:
                score += 2 if " " in keyword else 1
        if score > 0:
            scored.append((intent, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [intent for intent, _ in scored]


def _closest_intent_by_similarity(text: str) -> Optional[str]:
    query = _normalize_intent_text(text)
    if not query:
        return None

    best_intent: Optional[str] = None
    best_score = 0.0
    for intent, keywords in _INTENT_KEYWORDS.items():
        for keyword in keywords:
            score = SequenceMatcher(a=query, b=keyword).ratio()
            if score > best_score:
                best_score = score
                best_intent = intent

    if best_score >= 0.55:
        return best_intent
    return None


def resolve_intent_decision(
    raw_intent: str,
    raw_confidence: float,
    text: str,
    session_context: Optional[dict] = None,
) -> dict:
    session_context = session_context or {}

    migrated_intent, migration_used = migrate_intent(raw_intent)
    normalized_intent, recognized = normalize_intent(migrated_intent)

    multi_intents = detect_multi_intents(text)
    if not multi_intents and normalized_intent in VALID_INTENTS:
        multi_intents = [normalized_intent]

    primary_intent = multi_intents[0] if multi_intents else normalized_intent
    secondary_intents = [intent for intent in multi_intents[1:] if intent != primary_intent]

    calibrated_confidence, keyword_boost_used = calibrate_confidence(raw_confidence, primary_intent, text)
    threshold = get_intent_threshold(primary_intent)
    low_confidence = calibrated_confidence < threshold

    context_used = False
    context_source = ""
    if low_confidence:
        previous_intent, ok = normalize_intent(session_context.get("last_intent"))
        if ok:
            primary_intent = previous_intent
            context_used = True
            context_source = "last_intent"
            calibrated_confidence = max(calibrated_confidence, threshold)
            low_confidence = False
        else:
            last_action = _normalize_intent_text(str(session_context.get("last_action", "")))
            if "apply" in last_action:
                primary_intent = INTENT_APPLY_LOAN
                context_used = True
                context_source = "last_action"
                calibrated_confidence = max(calibrated_confidence, get_intent_threshold(INTENT_APPLY_LOAN))
                low_confidence = False
            elif "status" in last_action:
                primary_intent = INTENT_CHECK_APPLICATION_STATUS
                context_used = True
                context_source = "last_action"
                calibrated_confidence = max(calibrated_confidence, get_intent_threshold(INTENT_CHECK_APPLICATION_STATUS))
                low_confidence = False

    fallback_used = False
    fallback_reason = ""
    if low_confidence:
        closest = _closest_intent_by_similarity(text)
        if closest:
            primary_intent = closest
            fallback_reason = "closest_partial_match"
        else:
            primary_intent = INTENT_GENERAL_QUERY
            fallback_reason = "low_confidence_no_partial_match"
        fallback_used = True

    if primary_intent not in VALID_INTENTS:
        primary_intent = INTENT_GENERAL_QUERY
        fallback_used = True
        fallback_reason = fallback_reason or "unrecognized_intent"

    secondary_intents = [intent for intent in secondary_intents if intent in VALID_INTENTS and intent != primary_intent]

    return {
        "raw_intent": raw_intent,
        "migrated_intent": migrated_intent,
        "migration_used": migration_used,
        "normalized_intent": normalized_intent,
        "recognized": recognized,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
        "confidence": float(calibrated_confidence),
        "raw_confidence": float(raw_confidence),
        "threshold": float(get_intent_threshold(primary_intent)),
        "low_confidence": bool(float(calibrated_confidence) < get_intent_threshold(primary_intent)),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "context_used": context_used,
        "context_source": context_source,
        "keyword_boost_used": keyword_boost_used,
    }


def _normalize_query_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _query_fingerprint(value: str) -> str:
    return fingerprint_text(_normalize_query_key(value))


def _clean_scheme_value(value: Any) -> str:
    cleaned = str(value or "").strip().strip('"').strip()
    if cleaned.lower() in {"nan", "none", "null"}:
        return ""
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _map_dataset_intent(raw_intent: str) -> str:
    normalized = _normalize_query_key(raw_intent)
    direct_map = {
        "application_process": INTENT_APPLY_LOAN,
        "benefits": INTENT_SCHEME_QUERY,
        "documents_required": INTENT_SCHEME_QUERY,
        "eligibility": INTENT_SCHEME_QUERY,
        "scheme_info": INTENT_SCHEME_QUERY,
        "general_query": INTENT_GENERAL_QUERY,
    }
    if normalized in direct_map:
        return direct_map[normalized]

    if any(token in normalized for token in {"apply", "application", "register", "start"}):
        return INTENT_APPLY_LOAN
    if any(token in normalized for token in {"scheme", "info", "information", "eligibility", "benefit", "details", "query"}):
        return INTENT_SCHEME_QUERY
    if any(token in normalized for token in {"correct", "change", "edit", "wrong"}):
        return INTENT_GENERAL_QUERY
    return normalize_intent(normalized, default=INTENT_GENERAL_QUERY)[0]


def warmup_intent_dataset_cache(force: bool = False) -> Dict[str, Any]:
    with _INTENT_DATASET_LOCK:
        if _INTENT_DATASET_CACHE.get("loaded") and not force:
            return dict(_INTENT_DATASET_CACHE)

        rows: List[Dict[str, Any]] = []
        exact_map: Dict[str, Dict[str, str]] = {}
        first_token_index: Dict[str, List[int]] = {}
        token_index: Dict[str, List[int]] = {}
        intent_groups: Dict[str, List[int]] = {}
        csv_path = Path(os.getenv("INTENT_DATASET_PATH") or str(INTENT_DATASET_PATH)).resolve()

        if pd is None:
            logger.warning("pandas_not_available_for_intent_dataset")
        elif csv_path.exists():
            try:
                frame = pd.read_csv(csv_path)
                for record in frame.to_dict(orient="records"):
                    query = _normalize_query_key(str(record.get("query") or ""))
                    if not query:
                        continue
                    intent = _map_dataset_intent(str(record.get("intent") or ""))
                    scheme = _clean_scheme_value(record.get("scheme"))
                    if not scheme:
                        scheme = _clean_scheme_value(record.get("scheme_name"))
                    if not scheme:
                        scheme = _clean_scheme_value(record.get("scheme_title"))
                    ordered_tokens = _tokenize_for_match(query)
                    query_tokens = set(ordered_tokens)
                    row = {
                        "query": query,
                        "intent": intent,
                        "scheme": scheme,
                        "tokens": query_tokens,
                    }
                    row_index = len(rows)
                    rows.append(row)
                    exact_map.setdefault(query, {"intent": intent, "scheme": scheme})
                    intent_groups.setdefault(intent, []).append(row_index)
                    first_token = ordered_tokens[0] if ordered_tokens else ""
                    if first_token:
                        first_token_index.setdefault(first_token, []).append(row_index)
                    for token in query_tokens:
                        token_index.setdefault(token, []).append(row_index)
            except Exception as exc:
                logger.warning("intent_dataset_load_failed error_type=%s", type(exc).__name__)
        else:
            logger.warning("intent_dataset_missing path=%s", str(csv_path))

        _INTENT_DATASET_CACHE.clear()
        _INTENT_DATASET_CACHE.update(
            {
                "rows": rows,
                "exact_map": exact_map,
                "first_token_index": first_token_index,
                "token_index": token_index,
                "intent_groups": intent_groups,
                "query_cache": {},
                "loaded": True,
                "row_count": len(rows),
                "path": str(csv_path),
            }
        )
        return dict(_INTENT_DATASET_CACHE)


def get_intent_dataset_status() -> Dict[str, Any]:
    with _INTENT_DATASET_LOCK:
        if not _INTENT_DATASET_CACHE.get("loaded"):
            snapshot = warmup_intent_dataset_cache(force=False)
        else:
            snapshot = dict(_INTENT_DATASET_CACHE)
    return {
        "loaded": bool(snapshot.get("loaded")),
        "path": str(snapshot.get("path") or str(INTENT_DATASET_PATH)),
        "row_count": int(snapshot.get("row_count") or 0),
        "indexed_first_tokens": int(len(snapshot.get("first_token_index") or {})),
        "indexed_tokens": int(len(snapshot.get("token_index") or {})),
        "intent_groups": int(len(snapshot.get("intent_groups") or {})),
        "runtime_analytics": get_intent_runtime_analytics(),
    }


def _dataset_intent_signal(normalized_text: str) -> Dict[str, Any]:
    query = _normalize_query_key(normalized_text)
    if not query:
        return {"hit": True, "intent": INTENT_FALLBACK, "confidence": 0.0, "scheme": "", "score": 0.0}

    cache = warmup_intent_dataset_cache(force=False)
    query_cache = cache.get("query_cache") or {}
    cached_signal = query_cache.get(query)
    if isinstance(cached_signal, dict):
        return cached_signal

    exact_map = cache.get("exact_map") or {}
    exact = exact_map.get(query)
    if isinstance(exact, dict):
        signal = {
            "hit": True,
            "intent": str(exact.get("intent") or INTENT_GENERAL_QUERY),
            "confidence": 0.99,
            "scheme": _clean_scheme_value(str(exact.get("scheme") or "")),
            "score": 100.0,
        }
        query_cache[query] = signal
        return signal

    rows = cache.get("rows") or []
    if not rows:
        return {"hit": True, "intent": INTENT_FALLBACK, "confidence": 0.0, "scheme": "", "score": 0.0}

    ordered_query_tokens = _tokenize_for_match(query)
    query_tokens = set(ordered_query_tokens)
    dynamic_threshold = _dynamic_dataset_threshold(len(ordered_query_tokens))
    first_token_index = cache.get("first_token_index") or {}
    token_index = cache.get("token_index") or {}
    intent_groups = cache.get("intent_groups") or {}

    candidate_indices: set[int] = set()
    if ordered_query_tokens:
        candidate_indices.update(first_token_index.get(ordered_query_tokens[0], []))
        for token in ordered_query_tokens:
            candidate_indices.update(token_index.get(token, []))

    keyword_hint = _universal_keyword_signal(query)
    hinted_intent = str(keyword_hint.get("intent") or "")
    if bool(keyword_hint.get("hit")) and hinted_intent:
        candidate_indices.update(intent_groups.get(hinted_intent, []))

    if not candidate_indices:
        signal = {
            "hit": True,
            "intent": INTENT_FALLBACK,
            "confidence": 0.0,
            "scheme": "",
            "score": 0.0,
        }
        query_cache[query] = signal
        if len(query_cache) > 5000:
            query_cache.clear()
        return signal

    top_candidates: List[Tuple[float, Dict[str, Any]]] = []
    for row_index in candidate_indices:
        if row_index < 0 or row_index >= len(rows):
            continue
        row = rows[row_index]
        row_query = str(row.get("query") or "")
        row_tokens = set(row.get("tokens") or set())
        if not row_query:
            continue

        fuzzy_score = _fuzzy_match_score(query, row_query)
        token_overlap = _token_overlap_score(query_tokens, row_tokens)
        substring_score = _substring_match_score(query, row_query)
        score = (0.5 * fuzzy_score) + (0.3 * token_overlap) + (0.2 * substring_score)
        top_candidates.append((score, row))

    if not top_candidates:
        signal = {
            "hit": True,
            "intent": INTENT_FALLBACK,
            "confidence": 0.0,
            "scheme": "",
            "score": 0.0,
        }
        query_cache[query] = signal
        if len(query_cache) > 5000:
            query_cache.clear()
        return signal

    top_candidates.sort(key=lambda item: item[0], reverse=True)
    shortlisted = top_candidates[:3]

    best_row: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for base_score, row in shortlisted:
        row_scheme = _clean_scheme_value(str(row.get("scheme") or ""))
        scheme_bonus = _scheme_boost(query, query_tokens, row_scheme)
        mismatch_penalty = _category_mismatch_penalty(query_tokens, row_scheme)
        refined_score = max(0.0, base_score + scheme_bonus - mismatch_penalty)
        if refined_score > best_score:
            best_score = refined_score
            best_row = row

    if best_row is None or best_score < dynamic_threshold:
        confidence = round(max(0.0, best_score / 100.0), 4)
        signal = {
            "hit": True,
            "intent": INTENT_FALLBACK,
            "confidence": confidence,
            "scheme": "",
            "score": round(best_score, 2),
            "dynamic_threshold": round(dynamic_threshold, 2),
            "top_k": 3,
        }
        query_cache[query] = signal
        if len(query_cache) > 5000:
            query_cache.clear()
        return signal

    confidence = min(0.99, max(0.70, best_score / 100.0))
    signal = {
        "hit": True,
        "intent": str(best_row.get("intent") or INTENT_GENERAL_QUERY),
        "confidence": float(confidence),
        "scheme": _clean_scheme_value(str(best_row.get("scheme") or "")),
        "score": round(best_score, 2),
        "dynamic_threshold": round(dynamic_threshold, 2),
        "top_k": 3,
    }
    query_cache[query] = signal
    if len(query_cache) > 5000:
        query_cache.clear()
    return signal


def _detect_language(text: str) -> str:
    raw = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    if not raw:
        return "en"
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", raw))
    has_latin = bool(re.search(r"[a-z]", raw))
    if has_devanagari and has_latin:
        return "mixed"
    if has_devanagari:
        return "hi"
    return "en"


def _is_active_flow(session_context: dict | None) -> bool:
    context = session_context or {}
    last_action = str(context.get("last_action") or "").strip().lower()
    last_intent = str(context.get("last_intent") or "").strip().lower()
    if any(token in last_action for token in {"action", "collect", "confirm", "form", "apply"}):
        return True
    if last_intent in {"apply_loan", INTENT_PROVIDE_INFO}:
        return True
    return False


def normalize_text(text: str) -> Dict[str, str]:
    original = unicodedata.normalize("NFKC", str(text or "")).strip()
    normalized = original.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    token_map = {
        "yojana": "yojana",
        "yojna": "yojana",
        "kisaan": "kisan",
        "kishan": "kisan",
    }
    normalized_tokens: List[str] = []
    index = 0
    while index < len(tokens):
        pair = ""
        if index + 1 < len(tokens):
            pair = f"{tokens[index]} {tokens[index + 1]}"
        if pair and pair in HINGLISH_REPLACEMENTS:
            normalized_tokens.append(HINGLISH_REPLACEMENTS[pair])
            index += 2
            continue
        normalized_tokens.append(token_map.get(tokens[index], tokens[index]))
        index += 1
    normalized = " ".join(normalized_tokens)
    return {
        "original_text": original,
        "normalized_text": normalized,
        "detected_language": _detect_language(original),
    }


def _universal_keyword_signal(normalized_text: str) -> Dict[str, Any]:
    query = (normalized_text or "").strip().lower()
    if not query:
        return {
            "hit": False,
            "intent": INTENT_GENERAL_QUERY,
            "confidence": 0.0,
            "matched_group": "none",
        }

    correction_hits = sum(1 for token in KEYWORD_RULES["correction"] if token in query)
    if correction_hits > 0:
        return {
            "hit": True,
            "intent": INTENT_GENERAL_QUERY,
            "confidence": min(0.95, 0.78 + (0.06 * correction_hits)),
            "matched_group": "correction",
        }

    greeting_hits = sum(1 for token in KEYWORD_RULES["greeting"] if token in query)
    if greeting_hits > 0 and len(query.split()) <= 4:
        return {
            "hit": True,
            "intent": INTENT_GENERAL_QUERY,
            "confidence": min(0.9, 0.74 + (0.05 * greeting_hits)),
            "matched_group": "greeting",
        }

    apply_hits = sum(1 for token in KEYWORD_RULES[INTENT_APPLY_LOAN] if token in query)
    query_hits = sum(1 for token in KEYWORD_RULES[INTENT_SCHEME_QUERY] if token in query)
    if apply_hits <= 0 and query_hits <= 0:
        return {
            "hit": False,
            "intent": INTENT_GENERAL_QUERY,
            "confidence": 0.0,
            "matched_group": "none",
        }

    if apply_hits >= query_hits:
        return {
            "hit": True,
            "intent": INTENT_APPLY_LOAN,
            "confidence": min(0.95, 0.58 + (0.10 * apply_hits)),
            "matched_group": "apply",
        }
    return {
        "hit": True,
        "intent": INTENT_SCHEME_QUERY,
        "confidence": min(0.95, 0.56 + (0.10 * query_hits)),
        "matched_group": "query",
    }


def _classify_conversation_intent(text: str, model_intent: str, model_conf: float) -> Tuple[str, float]:
    query = (text or "").strip().lower()
    if any(token in query for token in CORRECTION_MARKERS):
        return "correction", max(0.78, float(model_conf))

    intent = (model_intent or "").strip().lower()
    if any(tag in intent for tag in {"scheme", "query", "info", "information"}):
        return "info", float(model_conf)
    if any(tag in intent for tag in {"apply", "action", "start", "form"}):
        return "apply", float(model_conf)

    if len(query.split()) <= 1:
        return "unknown", min(0.4, float(model_conf))
    return "unknown", float(model_conf)


def _normalize_surface_intent_label(surface_intent: str) -> str:
    normalized = str(surface_intent or "").strip().lower()
    alias_map = {
        "application_process": SURFACE_INTENT_APPLICATION_PROCESS,
        "apply_process": SURFACE_INTENT_APPLICATION_PROCESS,
        "apply_loan": SURFACE_INTENT_APPLICATION_PROCESS,
        "benefit": SURFACE_INTENT_BENEFITS,
        "benefits": SURFACE_INTENT_BENEFITS,
        "documents": SURFACE_INTENT_DOCUMENTS_REQUIRED,
        "document": SURFACE_INTENT_DOCUMENTS_REQUIRED,
        "documents_required": SURFACE_INTENT_DOCUMENTS_REQUIRED,
        "scheme": SURFACE_INTENT_SCHEME_QUERY,
        "scheme_query": SURFACE_INTENT_SCHEME_QUERY,
        "scheme_info": SURFACE_INTENT_SCHEME_QUERY,
        "general": SURFACE_INTENT_GENERAL_QUERY,
        "general_query": SURFACE_INTENT_GENERAL_QUERY,
        "check_application_status": SURFACE_INTENT_CHECK_APPLICATION_STATUS,
        "fallback": INTENT_FALLBACK,
        "provide_information": INTENT_PROVIDE_INFO,
    }
    return alias_map.get(normalized, normalized)


def _canonical_intent_from_surface(surface_intent: str) -> str:
    normalized_surface = _normalize_surface_intent_label(surface_intent)
    return SURFACE_TO_CANONICAL_INTENT_MAP.get(normalized_surface, INTENT_GENERAL_QUERY)


def _surface_intent_from_canonical(canonical_intent: str) -> str:
    normalized_canonical = str(canonical_intent or INTENT_GENERAL_QUERY).strip().lower()
    return CANONICAL_TO_SURFACE_INTENT_MAP.get(normalized_canonical, SURFACE_INTENT_GENERAL_QUERY)


def _resolve_surface_intent(canonical_intent: str, preferred_surface: str = "") -> str:
    canonical = str(canonical_intent or INTENT_GENERAL_QUERY).strip().lower() or INTENT_GENERAL_QUERY
    preferred = _normalize_surface_intent_label(preferred_surface)
    if preferred and _canonical_intent_from_surface(preferred) == canonical:
        return preferred
    return _surface_intent_from_canonical(canonical)


def _strong_pattern_lock_signal(query: str) -> Dict[str, Any]:
    lowered = (query or "").strip().lower()
    for pattern, locked_surface_intent in STRONG_PATTERN_LOCKS:
        if pattern in lowered:
            canonical_intent = _canonical_intent_from_surface(locked_surface_intent)
            return {
                "hit": True,
                "pattern": pattern,
                "surface_intent": locked_surface_intent,
                "canonical_intent": canonical_intent,
                "confidence": 0.99,
            }
    return {
        "hit": False,
        "pattern": "",
        "surface_intent": "",
        "canonical_intent": "",
        "confidence": 0.0,
    }


def _hybrid_intent_correction(
    query: str,
    model_intent: str,
) -> Dict[str, Any]:
    lowered = (query or "").strip().lower()
    app_hits = sorted([token for token in HYBRID_APPLICATION_KEYWORDS if token in lowered])
    doc_hits = sorted([token for token in HYBRID_DOCUMENT_KEYWORDS if token in lowered])
    benefit_hits = sorted([token for token in HYBRID_BENEFIT_KEYWORDS if token in lowered])
    info_hits = sorted([token for token in HYBRID_SCHEME_INFO_KEYWORDS if token in lowered])
    scheme_hits = sorted([token for token in HYBRID_SCHEME_KEYWORDS if token in lowered])

    corrected_surface_intent = ""
    corrected_canonical_intent = ""
    reason = ""
    # Step 1: strong action intent -> application flow intent.
    if app_hits:
        corrected_surface_intent = SURFACE_INTENT_APPLICATION_PROCESS
        corrected_canonical_intent = INTENT_APPLY_LOAN
        reason = "application_process_keywords"
    # Step 2: explicit document intent keywords.
    elif doc_hits:
        corrected_surface_intent = SURFACE_INTENT_DOCUMENTS_REQUIRED
        corrected_canonical_intent = INTENT_SCHEME_QUERY
        reason = "documents_keywords"
    # Step 2: benefit intent keywords.
    elif benefit_hits:
        corrected_surface_intent = SURFACE_INTENT_BENEFITS
        corrected_canonical_intent = INTENT_SCHEME_QUERY
        reason = "benefits_keywords"
    # Step 3: explicit scheme-info questions.
    elif info_hits:
        corrected_surface_intent = SURFACE_INTENT_SCHEME_QUERY
        corrected_canonical_intent = INTENT_SCHEME_QUERY
        reason = "scheme_info_keywords"
    # Step 4: only if model is still general_query, apply scheme-aware correction.
    elif scheme_hits and model_intent == INTENT_GENERAL_QUERY:
        corrected_surface_intent = SURFACE_INTENT_SCHEME_QUERY
        corrected_canonical_intent = INTENT_SCHEME_QUERY
        reason = "scheme_aware_general_query_boost"

    return {
        "applied": bool(corrected_surface_intent),
        "surface_intent": corrected_surface_intent,
        "canonical_intent": corrected_canonical_intent,
        "reason": reason,
        "application_hits": app_hits,
        "document_hits": doc_hits,
        "benefit_hits": benefit_hits,
        "info_hits": info_hits,
        "scheme_hits": scheme_hits,
    }


def _combine_signals(
    model_intent: str,
    model_conf: float,
    text: str,
    normalized_text: str,
    session_context: dict | None = None,
    dataset_signal: dict | None = None,
) -> Dict[str, Any]:
    if str(model_intent or "").strip().lower() == INTENT_PROVIDE_INFO:
        canonical_model_intent = INTENT_PROVIDE_INFO
        model_recognized = True
    else:
        canonical_model_intent, model_recognized = normalize_intent(model_intent, default=INTENT_GENERAL_QUERY)

    hybrid_correction = _hybrid_intent_correction(text or normalized_text, canonical_model_intent)
    strong_pattern_lock = _strong_pattern_lock_signal(text or normalized_text)
    fallback_intent_name, fallback_conf = _ml_fallback_intent(normalized_text or text)
    active_flow = _is_active_flow(session_context)
    info_detected = fallback_intent_name == INTENT_PROVIDE_INFO

    dataset = dataset_signal or {"hit": False, "intent": INTENT_GENERAL_QUERY, "confidence": 0.0, "scheme": "", "score": 0.0}

    legacy_keyword_intent, legacy_keyword_conf, legacy_keyword_hit = keyword_intent_signal(normalized_text or text)
    universal_keyword = _universal_keyword_signal(normalized_text)

    keyword_hit = bool(legacy_keyword_hit or universal_keyword["hit"])
    keyword_intent = legacy_keyword_intent
    keyword_conf = float(legacy_keyword_conf)
    keyword_group = "legacy"
    if float(universal_keyword["confidence"]) > keyword_conf:
        keyword_intent = str(universal_keyword["intent"])
        keyword_conf = float(universal_keyword["confidence"])
        keyword_group = str(universal_keyword.get("matched_group") or "universal")

    selected_intent = INTENT_GENERAL_QUERY
    selected_surface_intent = SURFACE_INTENT_GENERAL_QUERY
    selected_conf = 0.0
    source = "unresolved"
    hierarchy_stage = "unresolved"
    fallback_used = False
    fallback_reason = ""

    # Priority 1: strong keyword and phrase locks.
    if bool(strong_pattern_lock.get("hit")):
        selected_surface_intent = str(strong_pattern_lock.get("surface_intent") or SURFACE_INTENT_SCHEME_QUERY)
        selected_intent = _canonical_intent_from_surface(selected_surface_intent)
        selected_conf = max(float(strong_pattern_lock.get("confidence") or 0.0), 0.99)
        source = "strong_pattern_lock"
        hierarchy_stage = "strong_keyword"
    elif keyword_hit and float(keyword_conf) >= 0.75:
        keyword_canonical_intent, _ = normalize_intent(keyword_intent, default=INTENT_GENERAL_QUERY)
        selected_intent = keyword_canonical_intent
        selected_surface_intent = _resolve_surface_intent(selected_intent)
        selected_conf = max(float(keyword_conf), float(model_conf), 0.75)
        source = "keyword_strong_override"
        hierarchy_stage = "strong_keyword"
    # Priority 2: hybrid correction.
    elif bool(hybrid_correction.get("applied")):
        selected_surface_intent = str(hybrid_correction.get("surface_intent") or SURFACE_INTENT_SCHEME_QUERY)
        selected_intent = str(hybrid_correction.get("canonical_intent") or INTENT_SCHEME_QUERY)
        selected_conf = max(float(model_conf), MEDIUM_CONFIDENCE_THRESHOLD + 0.30, float(keyword_conf))
        source = f"hybrid_{hybrid_correction.get('reason') or 'correction'}"
        hierarchy_stage = "hybrid_correction"
    # Priority 3: model prediction.
    elif model_recognized and canonical_model_intent:
        selected_intent = canonical_model_intent
        selected_surface_intent = _resolve_surface_intent(selected_intent)
        selected_conf = max(float(model_conf), 0.0)
        source = "model"
        hierarchy_stage = "model_prediction"
    # Priority 4: fallback.
    else:
        fallback_canonical_intent, _ = normalize_intent(fallback_intent_name, default=INTENT_GENERAL_QUERY)
        if fallback_canonical_intent == INTENT_FALLBACK:
            fallback_canonical_intent = INTENT_GENERAL_QUERY
        selected_intent = fallback_canonical_intent
        selected_surface_intent = _resolve_surface_intent(selected_intent)
        selected_conf = max(float(fallback_conf), 0.0)
        source = "fallback_recovery"
        hierarchy_stage = "fallback"
        fallback_used = True
        fallback_reason = "model_unrecognized"

    if selected_intent == INTENT_PROVIDE_INFO:
        if active_flow:
            selected_surface_intent = INTENT_PROVIDE_INFO
            selected_conf = max(selected_conf, 0.8)
            source = "provide_info_active_flow"
            fallback_used = False
            fallback_reason = ""
        else:
            selected_intent = INTENT_APPLY_LOAN if "apply" in (normalized_text or "") else INTENT_SCHEME_QUERY
            selected_surface_intent = _resolve_surface_intent(selected_intent)
            selected_conf = max(selected_conf, 0.6)
            source = "provide_info_deferred"

    # Fallback policy: only low-confidence (<0.4) model results may fallback.
    if hierarchy_stage == "model_prediction" and selected_conf < MEDIUM_CONFIDENCE_THRESHOLD:
        fallback_canonical_intent, _ = normalize_intent(fallback_intent_name, default=INTENT_GENERAL_QUERY)
        if fallback_canonical_intent == INTENT_FALLBACK:
            fallback_canonical_intent = INTENT_GENERAL_QUERY
        selected_intent = fallback_canonical_intent
        selected_surface_intent = _resolve_surface_intent(selected_intent)
        selected_conf = max(0.35, selected_conf, float(fallback_conf))
        source = "fallback_low_confidence_model"
        hierarchy_stage = "fallback"
        fallback_used = True
        fallback_reason = "low_confidence_model_intent"

    if hierarchy_stage == "fallback" and selected_conf < MEDIUM_CONFIDENCE_THRESHOLD:
        selected_intent = INTENT_FALLBACK
        selected_surface_intent = INTENT_FALLBACK
        selected_conf = max(0.0, selected_conf)
        fallback_used = True
        fallback_reason = fallback_reason or "low_confidence_below_0_4"

    threshold = get_flexible_intent_threshold(selected_intent, text)
    low_confidence = bool(selected_conf < HIGH_CONFIDENCE_THRESHOLD)

    return {
        "intent": selected_intent,
        "surface_intent": _normalize_surface_intent_label(selected_surface_intent),
        "confidence": float(selected_conf),
        "threshold": float(threshold),
        "low_confidence": bool(low_confidence),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "source": source,
        "hierarchy_stage": hierarchy_stage,
        "keyword_signal": {
            "hit": keyword_hit,
            "intent": keyword_intent,
            "confidence": float(keyword_conf),
            "matched_group": keyword_group,
        },
        "model_signal": {
            "intent": canonical_model_intent,
            "confidence": float(model_conf),
            "recognized": bool(model_recognized),
        },
        "fallback_signal": {
            "intent": fallback_intent_name,
            "confidence": float(fallback_conf),
        },
        "dataset_signal": {
            "hit": bool(dataset.get("hit")),
            "intent": str(dataset.get("intent") or INTENT_GENERAL_QUERY),
            "confidence": float(dataset.get("confidence") or 0.0),
            "scheme": _clean_scheme_value(str(dataset.get("scheme") or "")),
            "score": float(dataset.get("score") or 0.0),
        },
        "strong_pattern_lock": strong_pattern_lock,
        "hybrid_correction": hybrid_correction,
        "info_detected": bool(info_detected),
    }


class IntentService:
    def detect(self, text: str, debug: bool = False, timings: dict | None = None, session_context: dict | None = None) -> Dict[str, Any]:
        start = time.perf_counter()
        log_event("intent_service_start", endpoint="intent_service", status="success", user_input_length=len(text or ""))
        try:
            normalization_layer = normalize_text(text)
            normalized = normalize_for_intent(
                normalization_layer["normalized_text"],
                language_hint=normalization_layer["detected_language"],
            )
            model_decision = _ml_predict_intent_detailed(normalized.intent_text, session_context=session_context)
            model_intent = str(model_decision.get("primary_intent", INTENT_GENERAL_QUERY))
            model_conf = float(model_decision.get("confidence", 0.0))
            dataset_signal = _dataset_intent_signal(normalized.intent_text)
            combined = _combine_signals(
                model_intent=model_intent,
                model_conf=model_conf,
                text=normalization_layer["original_text"],
                normalized_text=normalized.intent_text,
                session_context=session_context,
                dataset_signal=dataset_signal,
            )

            decision = {
                **model_decision,
                "primary_intent": combined["intent"],
                "confidence": combined["confidence"],
                "low_confidence": combined.get("low_confidence", False),
                "fallback_used": bool(combined["fallback_used"] or combined["intent"] == INTENT_FALLBACK),
                "fallback_reason": combined["fallback_reason"] or model_decision.get("fallback_reason", ""),
                "threshold": combined["threshold"],
                "resolution_source": combined["source"],
                "keyword_signal": combined["keyword_signal"],
                "normalized_input": normalized.intent_text,
                "normalized_language": normalized.language,
                "original_text": normalization_layer["original_text"],
                "detected_language": normalization_layer["detected_language"],
                "model_signal": combined["model_signal"],
                "fallback_signal": combined["fallback_signal"],
                "dataset_signal": combined["dataset_signal"],
                "strong_pattern_lock": combined.get("strong_pattern_lock", {}),
                "hybrid_correction": combined.get("hybrid_correction", {}),
                "hierarchy_stage": combined.get("hierarchy_stage", ""),
                "info_detected": combined["info_detected"],
            }
            matched_scheme = _clean_scheme_value(str(combined.get("dataset_signal", {}).get("scheme") or ""))
            confidence_01 = max(0.0, min(1.0, float(decision.get("confidence") or 0.0)))
            if confidence_01 < 0.60:
                matched_scheme = ""
            if not matched_scheme and confidence_01 >= 0.60:
                matched_scheme = _clean_scheme_value(str((session_context or {}).get("last_scheme") or ""))
            if matched_scheme:
                decision["matched_scheme"] = matched_scheme

            input_text = text or ""
            normalized_text = normalized.intent_text or ""
            log_event(
                "intent_service_trace",
                endpoint="intent_service",
                status="success",
                input_length=len(input_text),
                input_fingerprint=_query_fingerprint(input_text),
                normalized_length=len(normalized_text),
                normalized_fingerprint=_query_fingerprint(normalized_text),
                detected_language=normalization_layer["detected_language"],
                model_intent=model_intent,
                model_confidence=round(model_conf, 4),
                keyword_intent=combined["keyword_signal"].get("intent"),
                keyword_confidence=round(float(combined["keyword_signal"].get("confidence", 0.0)), 4),
                fallback_intent=combined["fallback_signal"].get("intent"),
                fallback_confidence=round(float(combined["fallback_signal"].get("confidence", 0.0)), 4),
                dataset_match=bool(combined["dataset_signal"].get("hit", False)),
                dataset_scheme=combined["dataset_signal"].get("scheme"),
                info_detected=bool(combined.get("info_detected", False)),
                final_intent=decision.get("primary_intent"),
                final_confidence=round(float(decision.get("confidence", 0.0)), 4),
                fallback_used=bool(decision.get("fallback_used")),
                resolution_source=decision.get("resolution_source"),
                hierarchy_stage=decision.get("hierarchy_stage"),
            )
            logger.debug(
                "intent_trace input_len=%s input_fp=%s normalized_len=%s normalized_fp=%s lang=%s model=(%s, %.4f) keyword=(%s, %.4f) fb=(%s, %.4f) final=(%s, %.4f) fallback=%s source=%s",
                len(input_text),
                _query_fingerprint(input_text),
                len(normalized_text),
                _query_fingerprint(normalized_text),
                normalization_layer["detected_language"],
                model_intent,
                model_conf,
                combined["keyword_signal"].get("intent"),
                float(combined["keyword_signal"].get("confidence", 0.0)),
                combined["fallback_signal"].get("intent"),
                float(combined["fallback_signal"].get("confidence", 0.0)),
                decision.get("primary_intent"),
                float(decision.get("confidence", 0.0)),
                bool(decision.get("fallback_used")),
                decision.get("resolution_source"),
            )

            convo_intent, convo_conf = _classify_conversation_intent(
                normalized.intent_text,
                str(decision.get("primary_intent") or ""),
                float(decision.get("confidence") or 0.0),
            )
            decision["conversation_intent"] = convo_intent
            decision["conversation_confidence"] = round(max(0.0, min(1.0, convo_conf)), 4)
            decision["canonical_intent"] = str(decision.get("primary_intent") or INTENT_GENERAL_QUERY).strip().lower() or INTENT_GENERAL_QUERY
            decision["surface_intent"] = _resolve_surface_intent(
                decision["canonical_intent"],
                str(combined.get("surface_intent") or ""),
            )
            decision["expected_surface_intent"] = _expected_surface_intent_from_patterns(normalized.intent_text)
            decision["surface_pattern_mismatch"] = bool(
                decision["expected_surface_intent"]
                and decision["surface_intent"] != decision["expected_surface_intent"]
            )
            decision["canonical_from_surface"] = _canonical_intent_from_surface(decision["surface_intent"])
            decision["surface_canonical_mismatch"] = bool(decision["canonical_from_surface"] != decision["canonical_intent"])
            if decision["surface_canonical_mismatch"]:
                _increment_runtime_alert_counter("surface_canonical_drift_events")
                log_event(
                    "intent_surface_canonical_drift",
                    level="warning",
                    endpoint="intent_service",
                    status="failure",
                    canonical_intent=decision["canonical_intent"],
                    surface_intent=decision["surface_intent"],
                    mapped_canonical_intent=decision["canonical_from_surface"],
                    input_length=len(text or ""),
                    input_fingerprint=_query_fingerprint(text or ""),
                )
            if decision["surface_pattern_mismatch"]:
                _increment_runtime_alert_counter("surface_pattern_drift_events")
                log_event(
                    "intent_surface_pattern_drift",
                    level="warning",
                    endpoint="intent_service",
                    status="failure",
                    expected_surface_intent=decision["expected_surface_intent"],
                    actual_surface_intent=decision["surface_intent"],
                    canonical_intent=decision["canonical_intent"],
                    input_length=len(text or ""),
                    input_fingerprint=_query_fingerprint(text or ""),
                )

            resolved_intent = str(decision.get("primary_intent") or INTENT_GENERAL_QUERY)
            normalized_query = _normalize_query_key(normalized.intent_text)
            query_fingerprint = _query_fingerprint(normalized_query)
            is_fallback_intent = bool(decision.get("fallback_used") or resolved_intent == INTENT_FALLBACK)
            confidence_01 = max(0.0, min(1.0, float(decision.get("confidence") or 0.0)))
            runtime_alerts = _record_runtime_intent_analytics(resolved_intent, is_fallback_intent, query_fingerprint, confidence_01)
            decision["runtime_alerts"] = runtime_alerts
            if runtime_alerts.get("consistency_mismatch"):
                log_event(
                    "intent_consistency_drift",
                    level="error",
                    endpoint="intent_service",
                    status="failure",
                    query_fingerprint=query_fingerprint,
                    query_length=len(normalized_query or ""),
                    recent_intents=runtime_alerts.get("recent_intents", []),
                )
            if runtime_alerts.get("confidence_drop"):
                log_event(
                    "intent_confidence_drop",
                    level="warning",
                    endpoint="intent_service",
                    status="failure",
                    query_fingerprint=query_fingerprint,
                    query_length=len(normalized_query or ""),
                    previous_confidence=runtime_alerts.get("previous_confidence"),
                    current_confidence=runtime_alerts.get("current_confidence"),
                    drop_amount=runtime_alerts.get("drop_amount"),
                )
            if runtime_alerts.get("fallback_rate_spike"):
                log_event(
                    "intent_fallback_rate_spike",
                    level="warning",
                    endpoint="intent_service",
                    status="failure",
                    recent_fallback_rate=runtime_alerts.get("recent_fallback_rate"),
                    global_fallback_rate=runtime_alerts.get("global_fallback_rate"),
                )
            record_intent_event(
                intent=resolved_intent,
                confidence=float(decision.get("confidence") or 0.0),
                fallback_used=bool(decision.get("fallback_used") or is_fallback_intent),
                low_confidence=bool(decision.get("low_confidence", False)),
                raw_intent=str(model_intent or ""),
            )

            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["intent_classification_ms"] = elapsed_ms
            if decision.get("fallback_used"):
                record_fallback()
            if elapsed_ms > 1000.0:
                log_event(
                    "intent_service_slow_response",
                    level="warning",
                    endpoint="intent_service",
                    status="failure",
                    response_time_ms=elapsed_ms,
                    threshold_ms=1000.0,
                    intent=decision.get("primary_intent"),
                )
            log_event(
                "intent_service_success",
                endpoint="intent_service",
                status="success",
                response_time_ms=elapsed_ms,
                intent=decision.get("primary_intent"),
                confidence=round(float(decision.get("confidence", 0.0)) * 100.0, 2),
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None:
                timings["intent_classification_ms"] = elapsed_ms
            log_event(
                "intent_service_failure",
                level="error",
                endpoint="intent_service",
                status="failure",
                error_type=type(exc).__name__,
                response_time_ms=elapsed_ms,
            )
            raise

        if debug:
            return {
                "intent": decision.get("surface_intent") or decision["primary_intent"],
                "canonical_intent": decision.get("canonical_intent") or decision["primary_intent"],
                "secondary_intents": decision.get("secondary_intents", []),
                "confidence": round(float(decision["confidence"]) * 100.0, 2),
                "confidence_01": round(max(0.0, min(1.0, float(decision["confidence"]))), 4),
                "fallback_used": bool(decision.get("fallback_used")),
                "debug": {
                    "raw_model_output": decision.get("raw_model_output"),
                    "normalized_intent": decision.get("normalized_intent"),
                    "fallback_used": decision.get("fallback_used"),
                    "fallback_reason": decision.get("fallback_reason"),
                    "context_used": decision.get("context_used"),
                    "intent_version": decision.get("intent_version"),
                    "resolution_source": decision.get("resolution_source"),
                    "keyword_signal": decision.get("keyword_signal"),
                    "normalized_input": decision.get("normalized_input"),
                    "normalized_language": decision.get("normalized_language"),
                    "original_text": decision.get("original_text"),
                    "normalized_text": decision.get("normalized_input"),
                    "detected_language": decision.get("detected_language"),
                    "model_intent": (decision.get("model_signal") or {}).get("intent"),
                    "model_confidence": (decision.get("model_signal") or {}).get("confidence"),
                    "keyword_intent": (decision.get("keyword_signal") or {}).get("intent"),
                    "keyword_confidence": (decision.get("keyword_signal") or {}).get("confidence"),
                    "dataset_signal": decision.get("dataset_signal"),
                    "strong_pattern_lock": decision.get("strong_pattern_lock"),
                    "hybrid_correction": decision.get("hybrid_correction"),
                    "low_confidence": decision.get("low_confidence"),
                    "matched_scheme": decision.get("matched_scheme"),
                    "info_detected": decision.get("info_detected"),
                    "final_intent": decision.get("primary_intent"),
                    "surface_intent": decision.get("surface_intent"),
                    "canonical_intent": decision.get("canonical_intent"),
                    "canonical_from_surface": decision.get("canonical_from_surface"),
                    "expected_surface_intent": decision.get("expected_surface_intent"),
                    "surface_pattern_mismatch": decision.get("surface_pattern_mismatch"),
                    "surface_canonical_mismatch": decision.get("surface_canonical_mismatch"),
                    "hierarchy_stage": decision.get("hierarchy_stage"),
                    "conversation_intent": decision.get("conversation_intent"),
                    "conversation_confidence": decision.get("conversation_confidence"),
                    "runtime_alerts": decision.get("runtime_alerts"),
                    "confidence_01": round(max(0.0, min(1.0, float(decision.get("confidence") or 0.0))), 4),
                },
            }

        return {
            "intent": decision.get("surface_intent") or decision["primary_intent"],
            "canonical_intent": decision.get("canonical_intent") or decision["primary_intent"],
            "confidence": round(float(decision["confidence"]) * 100.0, 2),
            "confidence_01": round(max(0.0, min(1.0, float(decision.get("confidence") or 0.0))), 4),
            "fallback_used": bool(decision.get("fallback_used")),
            "conversation_intent": decision.get("conversation_intent"),
            "conversation_confidence": round(float(decision.get("conversation_confidence", 0.0)) * 100.0, 2),
            "scheme": decision.get("matched_scheme") or "",
        }

    async def detect_async(self, text: str, debug: bool = False, timings: dict | None = None, session_context: dict | None = None) -> Dict[str, Any]:
        start = time.perf_counter()
        log_event("intent_service_async_start", endpoint="intent_service", status="success", user_input_length=len(text or ""))
        try:
            result = await asyncio.to_thread(self.detect, text, debug, timings, session_context)
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None and "intent_classification_ms" not in timings:
                timings["intent_classification_ms"] = elapsed_ms
            log_event("intent_service_async_success", endpoint="intent_service", status="success", response_time_ms=elapsed_ms, intent=result.get("intent"), confidence=result.get("confidence"))
            return result
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
            if timings is not None and "intent_classification_ms" not in timings:
                timings["intent_classification_ms"] = elapsed_ms
            log_event("intent_service_async_failure", level="error", endpoint="intent_service", status="failure", error_type=type(exc).__name__, response_time_ms=elapsed_ms)
            raise

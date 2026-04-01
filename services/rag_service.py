from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from backend.infrastructure.ml.scheme_registry import (
    get_scheme_registry,
    get_scheme_registry_debug_payload,
    load_dataset,
    warmup_scheme_registry,
)
from backend.shared.security.privacy import fingerprint_text


logger = logging.getLogger(__name__)
RAG_TIMEOUT_SECONDS = 2.0
MAX_RESPONSE_WORDS = 300
RAG_QUERY_CACHE_MAX = 2000
RAG_ENABLE_SEMANTIC_SIMILARITY = (os.getenv("RAG_ENABLE_SEMANTIC_SIMILARITY") or "0").strip() == "1"


_RAG_QUERY_CACHE_LOCK = threading.RLock()
_RAG_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


@lru_cache(maxsize=1)
def _dataset_records() -> List[Dict[str, Any]]:
    rows = load_dataset()
    return [row for row in rows if isinstance(row, dict)]


@lru_cache(maxsize=1)
def _embedding_model() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def _normalize(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _rag_cache_key(query: str) -> str:
    return _normalize(query)


def _get_rag_cache(query: str) -> Optional[Dict[str, Any]]:
    key = _rag_cache_key(query)
    if not key:
        return None
    with _RAG_QUERY_CACHE_LOCK:
        cached = _RAG_QUERY_CACHE.get(key)
        if not isinstance(cached, dict):
            return None
        # Return shallow copies to avoid accidental external mutation.
        return {
            "rows": list(cached.get("rows") or []),
            "filtered": bool(cached.get("filtered")),
            "detected_scheme": str(cached.get("detected_scheme") or ""),
        }


def _set_rag_cache(query: str, rows: List[Dict[str, Any]], filtered: bool, detected_scheme: str) -> None:
    key = _rag_cache_key(query)
    if not key:
        return
    with _RAG_QUERY_CACHE_LOCK:
        _RAG_QUERY_CACHE[key] = {
            "rows": list(rows),
            "filtered": bool(filtered),
            "detected_scheme": str(detected_scheme or ""),
        }
        if len(_RAG_QUERY_CACHE) > RAG_QUERY_CACHE_MAX:
            oldest_key = next(iter(_RAG_QUERY_CACHE.keys()), "")
            if oldest_key:
                _RAG_QUERY_CACHE.pop(oldest_key, None)


def _extract_scheme_name(record: Dict[str, Any]) -> str:
    for key in ("scheme_name", "name", "title", "scheme"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _extract_keywords(record: Dict[str, Any]) -> List[str]:
    items = record.get("keywords")
    if isinstance(items, list):
        return [str(item).strip() for item in items if str(item).strip()]
    if isinstance(items, str) and items.strip():
        return [items.strip()]
    return []


def _extract_record_type(record: Dict[str, Any]) -> str:
    return str(record.get("type") or "scheme").strip().lower() or "scheme"


def _extract_content(record: Dict[str, Any], language: str = "en") -> str:
    if str(record.get("content") or "").strip():
        return str(record.get("content") or "").strip()
    if language == "hi":
        return str(record.get("details_hi") or record.get("summary_hi") or "").strip()
    return str(record.get("details_en") or record.get("summary_en") or "").strip()


def _infer_intent_type(query: str) -> str:
    text = _normalize(query)
    if any(token in text for token in {"eligibility", "eligible", "पात्र", "criteria"}):
        return "eligibility"
    if any(token in text for token in {"benefit", "benefits", "लाभ", "amount", "subsidy"}):
        return "benefits"
    if any(token in text for token in {"document", "documents", "दस्तावेज", "paper"}):
        return "documents"
    if any(token in text for token in {"apply", "application", "register", "process"}):
        return "application"
    return "details"


def _intent_type_alignment_score(query_intent_type: str, record_type: str) -> float:
    normalized_record_type = _normalize(record_type)
    if not normalized_record_type:
        return 0.0
    if query_intent_type == normalized_record_type:
        return 1.0
    aliases = {
        "application": {"apply", "process", "registration", "how_to_apply"},
        "details": {"overview", "detail", "information"},
    }
    for canonical, values in aliases.items():
        if query_intent_type == canonical and any(value in normalized_record_type for value in values):
            return 0.75
    return 0.0


def _resolve_detected_scheme(scheme_context: Optional[Dict[str, object]], query: str) -> str:
    explicit = _normalize(
        str(
            (scheme_context or {}).get("scheme")
            or (scheme_context or {}).get("scheme_name")
            or (scheme_context or {}).get("scheme_id")
            or ""
        )
    )
    return explicit


def _keyword_overlap_score(query: str, values: List[str]) -> float:
    query_tokens = {token for token in _normalize(query).split(" ") if token}
    if not query_tokens:
        return 0.0

    best = 0.0
    for value in values:
        value_tokens = {token for token in _normalize(value).split(" ") if token}
        if not value_tokens:
            continue
        overlap = len(query_tokens.intersection(value_tokens)) / float(max(1, len(value_tokens)))
        if overlap > best:
            best = overlap
    return max(0.0, min(1.0, best))


def _keyword_overlap_query_content(query: str, content: str) -> float:
    query_tokens = {token for token in _normalize(query).split(" ") if token}
    content_tokens = {token for token in _normalize(content).split(" ") if token}
    if not query_tokens or not content_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(content_tokens))
    return max(0.0, min(1.0, overlap / float(max(1, len(query_tokens)))))


def _semantic_similarity_score(query: str, text: str) -> float:
    if not RAG_ENABLE_SEMANTIC_SIMILARITY:
        return 0.0
    model = _embedding_model()
    if model is None:
        return 0.0
    if not query.strip() or not text.strip():
        return 0.0
    try:
        from sentence_transformers import util  # type: ignore

        query_emb = model.encode([query], normalize_embeddings=True)
        text_emb = model.encode([text], normalize_embeddings=True)
        score = float(util.cos_sim(query_emb, text_emb).max().item())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _result_rows(
    transcript: str,
    *,
    detected_scheme: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    rows = _dataset_records()
    filtered_by_scheme = bool(detected_scheme)
    query_intent_type = _infer_intent_type(transcript)

    scored: List[Dict[str, Any]] = []
    for record in rows:
        scheme_name = _extract_scheme_name(record)
        if not scheme_name:
            continue

        normalized_name = _normalize(scheme_name)
        if filtered_by_scheme and normalized_name != detected_scheme:
            continue

        keywords = _extract_keywords(record)
        record_type = _extract_record_type(record)
        content = _extract_content(record)
        name_score = _keyword_overlap_score(transcript, [scheme_name])
        keyword_score = _keyword_overlap_score(transcript, keywords)
        semantic_text = " ".join([scheme_name] + keywords + ([content] if content else []))
        semantic_score = _semantic_similarity_score(transcript, semantic_text)
        intent_type_score = _intent_type_alignment_score(query_intent_type, record_type)
        content_overlap_score = _keyword_overlap_query_content(transcript, content)
        scheme_match_score = 1.0 if detected_scheme and normalized_name == detected_scheme else name_score

        chunk_priority_boost = 0.0
        if record_type != "scheme":
            chunk_priority_boost = 0.22
            if intent_type_score > 0.0:
                chunk_priority_boost += 0.10

        irrelevance_penalty = 0.0
        if keyword_score < 0.05 and content_overlap_score < 0.05 and semantic_score < 0.08:
            irrelevance_penalty += 0.18
        if record_type != "scheme" and intent_type_score <= 0.0 and keyword_score < 0.08:
            irrelevance_penalty += 0.08

        combined = (
            (0.38 * scheme_match_score)
            + (0.22 * intent_type_score)
            + (0.16 * keyword_score)
            + (0.16 * content_overlap_score)
            + (0.08 * semantic_score)
            + chunk_priority_boost
            - irrelevance_penalty
        )
        if combined <= 0:
            continue

        summary_en = str(record.get("summary_en") or "").strip()
        summary_hi = str(record.get("summary_hi") or "").strip()
        details_en = str(record.get("details_en") or "").strip()
        details_hi = str(record.get("details_hi") or "").strip()

        # Chunks may only provide a generic `content` field.
        if content and not details_en:
            details_en = content
        if content and not details_hi:
            details_hi = content
        if content and not summary_en:
            summary_en = content[:280].strip()
        if content and not summary_hi:
            summary_hi = content[:280].strip()

        scored.append(
            {
                "scheme": scheme_name,
                "score": round(combined, 4),
                "type": record_type,
                "summary_en": summary_en,
                "summary_hi": summary_hi,
                "details_en": details_en,
                "details_hi": details_hi,
                "content": content,
            }
        )

    scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return scored[: max(1, limit)], filtered_by_scheme


def _rag_debug(query: str, *, filtered_by_scheme: bool, scheme_used: str, total_results: int) -> Dict[str, Any]:
    return {
        "query": query,
        "filtered_by_scheme": bool(filtered_by_scheme),
        "scheme_used": scheme_used,
        "total_results": int(total_results),
    }


def _safety_flags(*, low_confidence: bool, ambiguous: bool, fallback_triggered: bool) -> Dict[str, bool]:
    return {
        "low_confidence": bool(low_confidence),
        "ambiguous": bool(ambiguous),
        "fallback_triggered": bool(fallback_triggered),
    }


def _is_ambiguous(rows: List[Dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    first = float(rows[0].get("score") or 0.0)
    second = float(rows[1].get("score") or 0.0)
    return abs(first - second) <= 0.08


def _is_broad_discovery_query(query: str) -> bool:
    text = _normalize(query)
    if not text:
        return False
    broad_markers = {
        "scheme",
        "yojana",
        "for family",
        "family",
        "kisan",
        "farmer",
        "batao",
        "suggest",
        "recommend",
    }
    return any(marker in text for marker in broad_markers)


def _forced_scheme_from_query(query: str) -> str:
    return ""


def _intent_guidance(query_intent_type: str, language: str, scheme_name: str) -> Tuple[str, str]:
    label = str(scheme_name or "this scheme").strip() or "this scheme"
    if query_intent_type == "application":
        if language == "hi":
            return (
                f"{label} के लिए आवेदन प्रक्रिया: Step 1 पात्रता जांचें, Step 2 दस्तावेज़ तैयार करें, Step 3 पोर्टल/CSC पर आवेदन करें।",
                "Step 4 आवेदन जमा होने के बाद acknowledgement रखें और status ट्रैक करें।",
            )
        return (
            f"Application process for {label}: Step 1 check eligibility, Step 2 prepare documents, Step 3 apply on portal or CSC.",
            "Step 4 keep acknowledgement and track application status.",
        )
    if query_intent_type == "documents":
        if language == "hi":
            return (
                f"{label} के लिए सामान्य दस्तावेज़: Aadhaar, address proof, bank passbook और income certificate.",
                "अगर state-specific दस्तावेज़ हों तो portal list भी देखें।",
            )
        return (
            f"Common documents for {label}: Aadhaar, address proof, bank passbook, and income certificate.",
            "Also check state-specific document requirements on the official portal.",
        )
    if query_intent_type == "benefits":
        if language == "hi":
            return (
                f"{label} में लाभ eligibility के आधार पर मिलते हैं और सहायता राशि/benefits state rules से तय होते हैं।",
                "सटीक amount के लिए अपनी category और state portal विवरण देखें।",
            )
        return (
            f"Benefits under {label} depend on eligibility, and assistance amount varies by state rules.",
            "Check your category and state portal details for exact benefit amount.",
        )
    if language == "hi":
        return (
            f"{label} एक सरकारी योजना है जो eligibility, benefits और application support देती है।",
            "क्या आप पात्रता, दस्तावेज़ या आवेदन प्रक्रिया जानना चाहते हैं?",
        )
    return (
        f"{label} is a government scheme that provides eligibility, benefit, and application support.",
        "Would you like eligibility, required documents, or application steps?",
    )


def _summarize_to_max_words(text: str, max_words: int = MAX_RESPONSE_WORDS) -> str:
    content = str(text or "").strip()
    if not content:
        return ""
    words = content.split()
    if len(words) <= max_words:
        return content
    trimmed = " ".join(words[:max_words]).rstrip(" ,.;:")
    return f"{trimmed}..."


def _apply_response_length_control(payload: Dict[str, Any], max_words: int = MAX_RESPONSE_WORDS) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    for key in ("confirmation", "explanation", "next_step"):
        if isinstance(payload.get(key), str):
            payload[key] = _summarize_to_max_words(payload[key], max_words=max_words)
    return payload


def _timeout_fallback(language: str, query: str, detected_scheme: str) -> Dict[str, Any]:
    message = "I couldn't find a clear match. Can you clarify your need?" if language != "hi" else "मुझे स्पष्ट मैच नहीं मिला। क्या आप अपनी जरूरत स्पष्ट कर सकते हैं?"
    return {
        "confirmation": message,
        "explanation": message,
        "next_step": message,
        "results": [],
        "schemes": [],
        "confidence": 0.0,
        "rag_debug": _rag_debug(query, filtered_by_scheme=bool(detected_scheme), scheme_used=detected_scheme, total_results=0),
        "safety": _safety_flags(low_confidence=True, ambiguous=False, fallback_triggered=True),
    }


def _error_fallback(language: str, query: str, detected_scheme: str) -> Dict[str, Any]:
    message = "I couldn't find a clear match. Can you clarify your need?" if language != "hi" else "मुझे स्पष्ट मैच नहीं मिला। क्या आप अपनी जरूरत स्पष्ट कर सकते हैं?"
    return {
        "confirmation": message,
        "explanation": message,
        "next_step": message,
        "results": [],
        "schemes": [],
        "confidence": 0.0,
        "rag_debug": _rag_debug(query, filtered_by_scheme=bool(detected_scheme), scheme_used=detected_scheme, total_results=0),
        "safety": _safety_flags(low_confidence=True, ambiguous=False, fallback_triggered=True),
    }


def _retrieve_scheme_impl(
    transcript: str,
    language: str = "en",
    need_category: Optional[str] = None,
    user_profile: Optional[Dict[str, str]] = None,
    scheme_context: Optional[Dict[str, object]] = None,
    session_feedback: Optional[Dict[str, object]] = None,
    context_fusion: Optional[Dict[str, object]] = None,
) -> Optional[dict]:
    query = str(transcript or "").strip()
    detected_scheme = _resolve_detected_scheme(scheme_context, query)
    query_intent_type = _infer_intent_type(query)
    forced_scheme = _forced_scheme_from_query(query)
    if not query:
        return _error_fallback(language, query, detected_scheme)

    cached = _get_rag_cache(query)
    if cached is not None and str(cached.get("detected_scheme") or "") == detected_scheme:
        cached_rows = list(cached.get("rows") or [])
        cached_filtered = bool(cached.get("filtered"))
        top_k = 1 if detected_scheme else 5
        rows = cached_rows[:top_k]
        filtered = cached_filtered
    else:
        top_k = 1 if detected_scheme else 5
        rows, filtered = _result_rows(query, detected_scheme=detected_scheme, limit=top_k)
        _set_rag_cache(query, rows, filtered, detected_scheme)

    ambiguous = _is_ambiguous(rows)
    top_score = float(rows[0].get("score") or 0.0) if rows else 0.0
    low_confidence = (not rows) or (top_score < 0.3)

    if low_confidence:
        if forced_scheme:
            explanation, next_step = _intent_guidance(query_intent_type, language, forced_scheme)
            return {
                "confirmation": forced_scheme,
                "explanation": explanation,
                "next_step": next_step,
                "results": rows,
                "schemes": [forced_scheme],
                "confidence": 0.55,
                "rag_debug": _rag_debug(query, filtered_by_scheme=filtered, scheme_used=detected_scheme, total_results=len(rows)),
                "safety": _safety_flags(low_confidence=False, ambiguous=False, fallback_triggered=False),
            }
        message = "I couldn't find a clear match. Can you clarify your need?" if language != "hi" else "मुझे स्पष्ट मैच नहीं मिला। क्या आप अपनी जरूरत स्पष्ट कर सकते हैं?"
        return {
            "confirmation": message,
            "explanation": message,
            "next_step": message,
            "results": rows,
            "schemes": [str(item.get("scheme") or "").strip() for item in rows],
            "confidence": round(top_score, 4),
            "rag_debug": _rag_debug(query, filtered_by_scheme=filtered, scheme_used=detected_scheme, total_results=0),
            "safety": _safety_flags(low_confidence=True, ambiguous=ambiguous, fallback_triggered=True),
        }

    top = rows[0]
    schemes = [str(item.get("scheme") or "").strip() for item in rows if str(item.get("scheme") or "").strip()]
    if forced_scheme and all(str(item or "").strip().lower() != forced_scheme.lower() for item in schemes):
        schemes = [forced_scheme] + schemes
    confidence = float(top.get("score") or 0.0)

    if ambiguous and not forced_scheme and not (not detected_scheme and _is_broad_discovery_query(query)):
        first = schemes[0] if schemes else ""
        second = schemes[1] if len(schemes) > 1 else ""
        confirm = (
            f"Did you mean: {first} or {second}?"
            if language != "hi"
            else f"क्या आपका मतलब {first} या {second} है?"
        )
        return {
            "type": "clarification",
            "message": "Multiple schemes found. Please clarify.",
            "options": schemes[:5],
            "confirmation": confirm,
            "explanation": confirm,
            "next_step": confirm,
            "results": rows,
            "schemes": schemes,
            "confidence": round(confidence, 4),
            "rag_debug": _rag_debug(query, filtered_by_scheme=filtered, scheme_used=detected_scheme, total_results=len(rows)),
            "safety": _safety_flags(low_confidence=False, ambiguous=True, fallback_triggered=False),
        }

    if detected_scheme:
        selected_scheme = forced_scheme or str(top.get("scheme") or detected_scheme)
        explanation = str(top.get("details_hi") if language == "hi" else top.get("details_en")) or str(
            top.get("summary_hi") if language == "hi" else top.get("summary_en")
        )
        guided_explanation, guided_next_step = _intent_guidance(query_intent_type, language, selected_scheme)
        if query_intent_type in {"application", "documents", "benefits"}:
            explanation = guided_explanation
        elif not explanation:
            explanation = guided_explanation
        next_step = guided_next_step
        confirmation = selected_scheme
    else:
        selected_scheme = forced_scheme or (schemes[0] if schemes else "Relevant Scheme")
        guided_explanation, guided_next_step = _intent_guidance(query_intent_type, language, selected_scheme)
        if query_intent_type in {"application", "documents", "benefits"}:
            explanation = guided_explanation
            next_step = guided_next_step
        else:
            listed = ", ".join(schemes[:5])
            explanation = (
                f"यहाँ सबसे प्रासंगिक योजनाएँ हैं: {listed}."
                if language == "hi"
                else f"Here are the most relevant schemes from the dataset: {listed}."
            )
            next_step = guided_next_step
        confirmation = selected_scheme

    return {
        "confirmation": confirmation,
        "explanation": explanation,
        "next_step": next_step,
        "results": rows,
        "schemes": schemes,
        "confidence": round(confidence, 4),
        "rag_debug": _rag_debug(query, filtered_by_scheme=filtered, scheme_used=detected_scheme, total_results=len(rows)),
        "safety": _safety_flags(low_confidence=False, ambiguous=False, fallback_triggered=False),
    }


def retrieve_scheme(
    transcript: str,
    language: str = "en",
    need_category: Optional[str] = None,
    user_profile: Optional[Dict[str, str]] = None,
    scheme_context: Optional[Dict[str, object]] = None,
    session_feedback: Optional[Dict[str, object]] = None,
    context_fusion: Optional[Dict[str, object]] = None,
) -> Optional[dict]:
    query = str(transcript or "").strip()
    detected_scheme = _resolve_detected_scheme(scheme_context, query)
    if RAG_TIMEOUT_SECONDS <= 0.001:
        fallback = _timeout_fallback(language, query, detected_scheme)
        return _apply_response_length_control(fallback)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _retrieve_scheme_impl,
                transcript,
                language,
                need_category,
                user_profile,
                scheme_context,
                session_feedback,
                context_fusion,
            )
            result = future.result(timeout=RAG_TIMEOUT_SECONDS)

        result = _apply_response_length_control(result or {})
        rag_debug = (result or {}).get("rag_debug") or {}
        safety = (result or {}).get("safety") or {}
        logger.info(
            "rag_critical_events query_fp=%s query_len=%s scheme=%s rag_debug=%s safety=%s",
            fingerprint_text(query),
            len(query or ""),
            detected_scheme,
            rag_debug,
            safety,
        )
        return result
    except FuturesTimeoutError:
        fallback = _timeout_fallback(language, query, detected_scheme)
        logger.warning(
            "rag_timeout query_fp=%s query_len=%s scheme=%s timeout_seconds=%s",
            fingerprint_text(query),
            len(query or ""),
            detected_scheme,
            RAG_TIMEOUT_SECONDS,
        )
        return _apply_response_length_control(fallback)
    except Exception as exc:
        logger.exception(
            "rag_error query_fp=%s query_len=%s scheme=%s error_type=%s",
            fingerprint_text(query),
            len(query or ""),
            detected_scheme,
            type(exc).__name__,
        )
        return _apply_response_length_control(_error_fallback(language, query, detected_scheme))


def recommend_schemes(
    transcript: str,
    language: str = "en",
    limit: int = 3,
    need_category: Optional[str] = None,
    user_profile: Optional[Dict[str, str]] = None,
    scheme_context: Optional[Dict[str, object]] = None,
    session_feedback: Optional[Dict[str, object]] = None,
    context_fusion: Optional[Dict[str, object]] = None,
) -> List[str]:
    query = str(transcript or "").strip()
    detected_scheme = _resolve_detected_scheme(scheme_context, query)
    top_k = 1 if detected_scheme else max(3, min(5, int(limit or 3)))
    cached = _get_rag_cache(query)
    if cached is not None and str(cached.get("detected_scheme") or "") == detected_scheme:
        rows = list(cached.get("rows") or [])[:top_k]
    else:
        rows, filtered = _result_rows(query, detected_scheme=detected_scheme, limit=top_k)
        _set_rag_cache(query, rows, filtered, detected_scheme)
    return [str(item.get("scheme") or "").strip() for item in rows if str(item.get("scheme") or "").strip()]


def recommend_schemes_with_reasons(
    transcript: str,
    language: str = "en",
    limit: int = 3,
    need_category: Optional[str] = None,
    user_profile: Optional[Dict[str, str]] = None,
    scheme_context: Optional[Dict[str, object]] = None,
    session_feedback: Optional[Dict[str, object]] = None,
    context_fusion: Optional[Dict[str, object]] = None,
) -> List[Dict[str, str]]:
    query = str(transcript or "").strip()
    detected_scheme = _resolve_detected_scheme(scheme_context, query)
    top_k = 1 if detected_scheme else max(3, min(5, int(limit or 3)))
    cached = _get_rag_cache(query)
    if cached is not None and str(cached.get("detected_scheme") or "") == detected_scheme:
        rows = list(cached.get("rows") or [])[:top_k]
    else:
        rows, filtered = _result_rows(query, detected_scheme=detected_scheme, limit=top_k)
        _set_rag_cache(query, rows, filtered, detected_scheme)

    reason_rows: List[Dict[str, str]] = []
    for item in rows:
        summary = str(item.get("summary_hi") if language == "hi" else item.get("summary_en"))
        score = float(item.get("score") or 0.0)
        reason = (
            f"Matched by keyword overlap and semantic similarity (score {round(score, 3)})."
            if language != "hi"
            else f"कीवर्ड और semantic similarity के आधार पर मैच (स्कोर {round(score, 3)})."
        )
        reason_rows.append(
            {
                "scheme": str(item.get("scheme") or "").strip(),
                "summary": summary,
                "reason": reason,
            }
        )
    return reason_rows


def retrieve_scheme_with_recommendations(
    transcript: str,
    language: str = "en",
    limit: int = 3,
    need_category: Optional[str] = None,
    user_profile: Optional[Dict[str, str]] = None,
    scheme_context: Optional[Dict[str, object]] = None,
    session_feedback: Optional[Dict[str, object]] = None,
    context_fusion: Optional[Dict[str, object]] = None,
) -> Tuple[Optional[dict], List[str], bool]:
    match = retrieve_scheme(
        transcript,
        language,
        need_category=need_category,
        user_profile=user_profile,
        scheme_context=scheme_context,
        session_feedback=session_feedback,
        context_fusion=context_fusion,
    )
    recommendations = recommend_schemes(
        transcript,
        language,
        limit=limit,
        need_category=need_category,
        user_profile=user_profile,
        scheme_context=scheme_context,
        session_feedback=session_feedback,
        context_fusion=context_fusion,
    )
    exact_match = bool(match and bool(match.get("results")) and bool(match.get("rag_debug", {}).get("filtered_by_scheme", False)))
    return match, recommendations, exact_match


def get_rag_status() -> dict:
    rows = _dataset_records()
    registry = get_scheme_registry() or {}
    return {
        "dataset_path": "backend/data/chunks.json",
        "total_schemes": len(rows),
        "loaded": bool(rows),
        "embedding_model_loaded": bool(_embedding_model() is not None),
        "chunk_rows": int(registry.get("chunk_rows", 0)),
        "scheme_rows": int(registry.get("scheme_rows", 0)),
    }


def warmup_rag_resources(precompute_embeddings: bool = True) -> None:
    _dataset_records()
    if precompute_embeddings:
        _embedding_model()


def get_scheme_registry_snapshot() -> dict:
    return get_scheme_registry_debug_payload()


def warmup_scheme_registry_cache() -> dict:
    return warmup_scheme_registry()

__all__ = [
    "recommend_schemes",
    "recommend_schemes_with_reasons",
    "retrieve_scheme",
    "retrieve_scheme_with_recommendations",
    "get_rag_status",
    "warmup_rag_resources",
    "get_scheme_registry_snapshot",
    "warmup_scheme_registry_cache",
]

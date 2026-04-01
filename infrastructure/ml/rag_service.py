"""Compatibility wrapper for the canonical RAG engine.

Canonical source: backend.services.rag_service
This module intentionally contains no independent retrieval logic.
Chunk-prioritized retrieval behavior is implemented in backend.services.rag_service.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.domain.engines.eligibility import check_eligibility

from backend.services.rag_service import (
    get_rag_status,
    get_scheme_registry_snapshot,
    recommend_schemes,
    recommend_schemes_with_reasons,
    retrieve_scheme,
    retrieve_scheme_with_recommendations,
    warmup_rag_resources,
    warmup_scheme_registry_cache,
)


MIN_SCORING_WEIGHT = 0.1
MAX_SCORING_WEIGHT = 0.6

# Backward-compatible globals used by legacy tests and monkeypatches.
PREPARED_SCHEMES: List[Tuple[Dict[str, Any], List[str]]] = []
SCHEME_BLOBS: List[Tuple[Dict[str, Any], str]] = []


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    bounded = {key: _clamp(float(val), MIN_SCORING_WEIGHT, MAX_SCORING_WEIGHT) for key, val in raw.items()}
    total = sum(bounded.values()) or 1.0
    normalized = {key: val / total for key, val in bounded.items()}

    # Re-clamp after normalization and normalize one final time.
    normalized = {key: _clamp(val, MIN_SCORING_WEIGHT, MAX_SCORING_WEIGHT) for key, val in normalized.items()}
    final_total = sum(normalized.values()) or 1.0
    return {key: val / final_total for key, val in normalized.items()}


def _dynamic_scoring_weights(query: str, confidence: float, profile_completeness: float) -> Dict[str, float]:
    text = str(query or "").strip().lower()
    conf = _clamp(float(confidence), 0.0, 1.0)
    profile = _clamp(float(profile_completeness), 0.0, 1.0)

    keyword_bias = 0.2 if any(token in text for token in ("loan", "scheme", "apply", "benefit")) else 0.0

    raw = {
        "embedding": 0.35 + (0.1 * conf),
        "keyword": 0.25 + keyword_bias + (0.1 * (1.0 - conf)),
        "category": 0.2 + (0.15 * profile),
        "eligibility": 0.2 + (0.15 * profile),
    }
    return _normalize_weights(raw)


def _score_scheme_match(query: str, scheme: Dict[str, Any], blob: str = "") -> float:
    name = str((scheme or {}).get("name") or "").lower()
    text = str(query or "").lower()
    score = 50.0
    if name and name in text:
        score += 20.0
    if blob:
        overlap = len(set(text.split()).intersection(set(str(blob).lower().split())))
        score += float(overlap) * 3.0
    return score


def _select_diverse_top(ranked: List[Tuple[float, Dict[str, Any]]], limit: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    if not ranked or limit <= 0:
        return []

    selected: List[Tuple[float, Dict[str, Any]]] = [ranked[0]]
    seen_categories = {str((ranked[0][1] or {}).get("category") or "")}

    for item in ranked[1:]:
        if len(selected) >= limit:
            break
        category = str((item[1] or {}).get("category") or "")
        if category and category not in seen_categories:
            selected.append(item)
            seen_categories.add(category)

    if len(selected) < limit:
        for item in ranked[1:]:
            if len(selected) >= limit:
                break
            if item not in selected:
                selected.append(item)

    return selected[:limit]


def _rank_schemes(query: str, need_category: str | None = None, user_profile: Dict[str, Any] | None = None) -> List[Tuple[float, Dict[str, Any]]]:
    profile = user_profile or {}
    ranked: List[Tuple[float, Dict[str, Any]]] = []

    source = PREPARED_SCHEMES or [(scheme, []) for scheme, _ in SCHEME_BLOBS]
    for scheme, keywords in source:
        if not isinstance(scheme, dict):
            continue
        blob = " ".join(str(item) for item in (keywords or []))
        base_score = float(_score_scheme_match(query, scheme, blob))
        eligibility = check_eligibility(profile, scheme)
        eligibility_score = float(eligibility.get("score") or 0.0)

        # Strongly ineligible schemes are filtered out.
        if not bool(eligibility.get("eligible", False)) and eligibility_score <= 0.1:
            continue

        category_bonus = 0.0
        if need_category and str(scheme.get("category") or "").lower() == str(need_category).lower():
            category_bonus = 10.0

        final_score = base_score + (eligibility_score * 20.0) + category_bonus
        ranked.append((final_score, scheme))

    ranked.sort(key=lambda item: float(item[0]), reverse=True)
    return ranked

__all__ = [
    "MIN_SCORING_WEIGHT",
    "MAX_SCORING_WEIGHT",
    "PREPARED_SCHEMES",
    "SCHEME_BLOBS",
    "_dynamic_scoring_weights",
    "_score_scheme_match",
    "_select_diverse_top",
    "_rank_schemes",
    "retrieve_scheme",
    "recommend_schemes",
    "recommend_schemes_with_reasons",
    "retrieve_scheme_with_recommendations",
    "get_rag_status",
    "warmup_rag_resources",
    "get_scheme_registry_snapshot",
    "warmup_scheme_registry_cache",
]

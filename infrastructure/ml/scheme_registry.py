from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_CHUNKS_JSON_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks.json"

_GENERIC_KEYWORDS = {
    "scheme",
    "yojana",
    "apply",
    "application",
    "benefit",
    "benefits",
    "government",
    "india",
}

_SCHEME_NAME_KEYS = ("scheme_name", "name", "title", "scheme")

_LOCK = threading.RLock()

SCHEME_REGISTRY: Dict[str, Any] = {
    "schemes": [],
    "total": 0,
    "keywords_map": {},
    "source": "uninitialized",
    "loaded_at": 0.0,
    "scheme_rows": 0,
    "chunk_rows": 0,
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalize_query_for_matching(value: str) -> str:
    normalized = _normalize_text(value)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = normalized.replace("yojana", " ")
    normalized = normalized.replace("scheme", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _canonical_key(value: str) -> str:
    normalized = _normalize_text(value)
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    return normalized


def _display_name(value: str) -> str:
    raw = re.sub(r"\s+", " ", str(value or "").strip())
    if not raw:
        return ""
    titled = " ".join(part.capitalize() for part in raw.split(" "))
    replacements = {
        "Pm": "PM",
        "Pmjay": "PMJAY",
        "Pmay": "PMAY",
        "Dbt": "DBT",
    }
    for source, target in replacements.items():
        titled = re.sub(rf"\b{source}\b", target, titled)
    return titled


def _extract_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("data", "items", "records", "schemes"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]

    return []


def _normalize_unified_record(record: Dict[str, Any], *, default_type: str) -> Dict[str, Any]:
    scheme_name = str(
        record.get("scheme_name")
        or record.get("name")
        or record.get("title")
        or record.get("scheme")
        or ""
    ).strip()
    content = str(
        record.get("content")
        or record.get("details_en")
        or record.get("summary_en")
        or record.get("details_hi")
        or record.get("summary_hi")
        or ""
    ).strip()
    row_type = str(record.get("type") or default_type).strip().lower() or default_type
    return {
        **record,
        "scheme_name": scheme_name,
        "type": row_type,
        "content": content,
    }


def _load_chunks_json() -> List[Dict[str, Any]]:
    path = _CHUNKS_JSON_PATH
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []

    records = _extract_records(payload)
    return [_normalize_unified_record(row, default_type="chunk") for row in records if isinstance(row, dict)]


def load_dataset(source: Optional[str] = None) -> List[Dict[str, Any]]:
    _ = source
    return _load_chunks_json()


def _extract_name(record: Dict[str, Any]) -> str:
    for key in _SCHEME_NAME_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_schemes(dataset: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    unique: Dict[str, str] = {}

    for record in dataset:
        if not isinstance(record, dict):
            continue
        raw_name = _extract_name(record)
        if not raw_name:
            continue
        key = _canonical_key(raw_name)
        if not key:
            continue
        unique.setdefault(key, _display_name(raw_name))

    schemes = sorted(unique.values())
    return {
        "schemes": schemes,
        "total": len(schemes),
    }


def _keyword_candidates(record: Dict[str, Any], scheme_name: str) -> List[str]:
    values: List[str] = [scheme_name]

    for key in ("keywords", "tags", "aliases", "synonyms"):
        item = record.get(key)
        if isinstance(item, list):
            values.extend(str(entry) for entry in item)
        elif isinstance(item, str):
            values.append(item)

    target_user = record.get("target_user")
    if isinstance(target_user, str) and target_user.strip():
        values.append(target_user)

    category = record.get("category")
    if isinstance(category, str) and category.strip():
        values.append(category)

    scheme_parts = re.split(r"\s+", scheme_name)
    for token in scheme_parts:
        token = _normalize_text(token)
        if len(token) > 2 and token not in _GENERIC_KEYWORDS:
            values.append(token)

    normalized = []
    for value in values:
        token = _normalize_text(str(value))
        token = re.sub(r"[^a-z0-9\s]", "", token)
        if token and token not in _GENERIC_KEYWORDS:
            normalized.append(token)

    return sorted(set(normalized), key=len, reverse=True)


def _build_registry(dataset: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
    extracted = extract_schemes(dataset)
    scheme_to_keywords: Dict[str, List[str]] = {}

    display_by_key = {_canonical_key(name): name for name in extracted["schemes"]}

    for record in dataset:
        if not isinstance(record, dict):
            continue
        raw_name = _extract_name(record)
        key = _canonical_key(raw_name)
        if not key:
            continue
        display_name = display_by_key.get(key) or _display_name(raw_name)
        scheme_to_keywords.setdefault(display_name.lower(), [])
        scheme_to_keywords[display_name.lower()] = sorted(
            set(scheme_to_keywords[display_name.lower()] + _keyword_candidates(record, display_name))
        )

    scheme_rows = sum(1 for row in dataset if str(row.get("type") or "").strip().lower() == "scheme")
    chunk_rows = sum(1 for row in dataset if str(row.get("type") or "").strip().lower() != "scheme")

    return {
        "schemes": extracted["schemes"],
        "total": extracted["total"],
        "keywords_map": scheme_to_keywords,
        "source": source,
        "loaded_at": time.time(),
        "scheme_rows": int(scheme_rows),
        "chunk_rows": int(chunk_rows),
    }


def warmup_scheme_registry(force: bool = False, source: Optional[str] = None) -> Dict[str, Any]:
    with _LOCK:
        if SCHEME_REGISTRY.get("total", 0) > 0 and not force:
            return SCHEME_REGISTRY

        dataset = load_dataset(source)
        registry = _build_registry(dataset, "chunks_json")

        SCHEME_REGISTRY.clear()
        SCHEME_REGISTRY.update(registry)
        return SCHEME_REGISTRY


def get_scheme_registry() -> Dict[str, Any]:
    with _LOCK:
        if SCHEME_REGISTRY.get("total", 0) > 0:
            return SCHEME_REGISTRY
    return warmup_scheme_registry()


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase)
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _score_scheme_match(query: str, scheme_name: str, keywords: List[str]) -> float:
    score = 0.0
    normalized_name = _normalize_query_for_matching(scheme_name)

    if normalized_name and _contains_phrase(query, normalized_name):
        score += 1.0

    if normalized_name:
        name_tokens = [token for token in normalized_name.split(" ") if len(token) >= 4]
        if name_tokens and any(token in query for token in name_tokens):
            score += 0.6

    if any(_contains_phrase(query, _normalize_query_for_matching(keyword)) for keyword in keywords if _normalize_query_for_matching(keyword)):
        score += 0.5

    return round(score, 3)


def find_schemes_in_text(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    query = _normalize_query_for_matching(text)
    if not query:
        return []

    registry = get_scheme_registry()
    hits: List[Dict[str, Any]] = []

    for scheme_name in registry.get("schemes", []):
        normalized_name = _normalize_text(scheme_name)
        keywords = registry.get("keywords_map", {}).get(normalized_name, [])
        score = _score_scheme_match(query, scheme_name, list(keywords))
        if score > 0:
            hits.append({"scheme": scheme_name, "score": score})

    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    return hits[:limit]


def get_scheme_registry_debug_payload() -> Dict[str, Any]:
    registry = get_scheme_registry()
    return {
        "total_schemes": int(registry.get("total", 0)),
        "schemes": list(registry.get("schemes", [])),
        "scheme_rows": int(registry.get("scheme_rows", 0)),
        "chunk_rows": int(registry.get("chunk_rows", 0)),
    }


__all__ = [
    "load_dataset",
    "warmup_scheme_registry",
    "get_scheme_registry",
    "get_scheme_registry_debug_payload",
    "find_schemes_in_text",
]

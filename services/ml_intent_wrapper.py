import logging
import re
import csv
import time
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, Optional

from backend.services.helpers.response_builder import build_hackathon_response
from backend.text_normalizer import normalize_text
from backend.data import SCHEME_DATA


logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 1.5
_RETRY_ATTEMPTS = 1
_GENERIC_WORDS = {"yojana", "scheme", "kisan", "yojna"}
_TOKEN_RE = re.compile(r"[a-z0-9\u0900-\u097F]+")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_STOPWORDS = {"s", "of", "for", "under", "the", "and", "than"}
_NON_DISCRIMINATIVE_TOKENS = {
    "kya",
    "hai",
    "batao",
    "mujhe",
    "please",
    "tell",
    "about",
    "show",
    "info",
    "details",
}
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-intent-wrapper")
_SCHEME_DATASET_CACHE: Optional[list[Dict[str, Any]]] = None
_ROOT_PATH = Path(__file__).resolve().parents[2]
_CLEANED_SCHEME_DATASET_PATH = _ROOT_PATH / "cleaned_dataset.csv"
_SCHEME_DATASET_PATH = _ROOT_PATH / "data" / "final_voice_ready_dataset.csv"
_RESOLVER_PHRASE_WEIGHT = 0.9
_RESOLVER_MIN_SCORE = 0.5

# Global in-memory context for multi-turn fallback when request-scoped context is absent.
session_context = {
    "last_scheme": None,
}

CONTROLLED_SCHEME_CLARIFICATION = (
    "I currently provide detailed information for selected schemes. Please ask about a specific scheme like solar, housing, loan, etc."
)


def detect_scheme(text: str) -> Optional[str]:
    normalized_text = str(text or "").strip().lower()
    if not normalized_text:
        return None

    for key in SCHEME_DATA.keys():
        normalized_key = str(key or "").strip().lower()
        if normalized_key and normalized_key in normalized_text:
            return normalized_key

    if "किसान" in normalized_text:
        return "pm kisan"
    if "सोलर" in normalized_text:
        return "solar rooftop subsidy"

    return _resolve_scheme_from_static_data(normalized_text)


def _fallback_payload(intent: str = "general_query") -> Dict[str, Any]:
    return {
        "intent": intent,
        "scheme_name": None,
        "entities": {},
        "confidence": "low",
        "source": "fallback",
    }


def _to_confidence_number(value: Any) -> float:
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            number = float(str(value).strip())
        if number < 0.0:
            return 0.0
        if number > 1.0:
            return 1.0
        return number
    except Exception:
        return 0.0


class _DefaultMLModel:
    def predict(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            from backend.infrastructure.ml.bert_service import predict_intent_detailed

            decision = predict_intent_detailed(text)
            if not isinstance(decision, dict):
                return None
            return {
                "intent": decision.get("primary_intent"),
                "scheme_name": decision.get("matched_scheme"),
                "entities": decision.get("entities") if isinstance(decision.get("entities"), dict) else {},
                "confidence": decision.get("confidence", 0.0),
                "response_template": decision.get("response_template"),
            }
        except Exception:
            return None


ml_model: Any = _DefaultMLModel()


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _prepare_match_text(text: Any) -> tuple[str, list[str], set[str]]:
    tokens = [token for token in _tokenize(str(text or "")) if token and token not in _STOPWORDS]
    clean_text = " ".join(tokens)
    return clean_text, tokens, set(tokens)


def _levenshtein_ratio(left: str, right: str) -> float:
    left_clean = str(left or "").strip()
    right_clean = str(right or "").strip()
    if not left_clean or not right_clean:
        return 0.0
    if left_clean == right_clean:
        return 1.0
    return float(SequenceMatcher(None, left_clean, right_clean).ratio())


def _debug(message: str, **fields: Any) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("%s %s", message, fields)


def _is_generic_only(tokens: list[str]) -> bool:
    if not tokens:
        return True
    return all(token in _GENERIC_WORDS for token in tokens)


def _is_non_discriminative_query(text: str) -> bool:
    tokens = _informative_tokens_from_text(text)
    if not tokens:
        return True
    filtered = [t for t in tokens if t not in _NON_DISCRIMINATIVE_TOKENS]
    return len(filtered) < 2


def _sanitize_scheme_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    scheme_name = value.strip()
    if not scheme_name:
        return None
    return scheme_name


def _is_supported_scheme(scheme_name: Optional[str]) -> bool:
    clean = str(scheme_name or "").strip().lower()
    if not clean:
        return False
    return clean in SCHEME_DATA


def _canonical_scheme_name(scheme_name: Optional[str]) -> Optional[str]:
    clean = str(scheme_name or "").strip().lower()
    if not clean:
        return None
    if clean in SCHEME_DATA:
        return clean
    return None


def _scheme_display_name(scheme_name: Optional[str]) -> str:
    canonical = _canonical_scheme_name(scheme_name)
    if not canonical:
        return str(scheme_name or "").strip()

    tokens = canonical.split()
    formatted: list[str] = []
    for token in tokens:
        if token in {"pm", "sc", "st"}:
            formatted.append(token.upper())
        else:
            formatted.append(token.capitalize())
    return " ".join(formatted)


def _resolve_scheme_from_static_data(text: str) -> Optional[str]:
    normalized_text = str(text or "").strip().lower()
    if not normalized_text:
        return None

    best_match: Optional[str] = None
    best_len = 0
    for scheme_key in SCHEME_DATA.keys():
        key = str(scheme_key or "").strip().lower()
        if not key:
            continue
        if key in normalized_text and len(key) > best_len:
            best_match = key
            best_len = len(key)

    if best_match:
        return best_match

    query_tokens = {token for token in _tokenize(normalized_text) if token not in _GENERIC_WORDS and len(token) >= 3}
    if not query_tokens:
        return None

    best_token_match: Optional[str] = None
    best_overlap = 0
    for scheme_key in SCHEME_DATA.keys():
        key = str(scheme_key or "").strip().lower()
        if not key:
            continue
        key_tokens = {token for token in _tokenize(key) if token not in _GENERIC_WORDS and len(token) >= 3}
        overlap = len(query_tokens.intersection(key_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_token_match = key

    if best_overlap >= 2:
        return best_token_match

    return best_match


def _infer_followup_intent_from_text(text: str) -> Optional[str]:
    clean_text = str(text or "").strip().lower()
    if not clean_text:
        return None

    apply_keywords = ("apply", "kaise", "process", "application", "register")
    eligibility_keywords = ("eligibility", "eligible", "yogya", "patr", "patrata", "criteria", "पात्र")

    if any(keyword in clean_text for keyword in apply_keywords):
        return "apply_help"
    if any(keyword in clean_text for keyword in eligibility_keywords):
        return "eligibility_check"
    return None


def _detect_response_language(text: Any) -> str:
    raw = str(text or "")
    if _DEVANAGARI_RE.search(raw):
        return "hi"
    return "en"


def _parse_keyword_blob(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        parts = [str(item).strip().lower() for item in raw]
        return [item for item in parts if item]
    text = str(raw).strip().lower()
    if not text:
        return []
    text = text.replace("|", ",").replace(";", ",")
    parts = [item.strip() for item in text.split(",")]
    return [item for item in parts if item]


def _derive_fallback_keywords(scheme_name: str) -> list[str]:
    _, tokens, _ = _prepare_match_text(scheme_name)
    filtered = [token for token in tokens if token not in _GENERIC_WORDS and len(token) >= 4]
    # Keep a deterministic coverage set: head + tail unigrams and adjacent bigrams.
    derived: list[str] = []

    head = filtered[:8]
    tail = filtered[-6:] if len(filtered) > 8 else []
    for token in head + tail:
        if token not in derived:
            derived.append(token)

    for idx in range(len(filtered) - 1):
        bigram = f"{filtered[idx]} {filtered[idx + 1]}"
        derived.append(bigram)
        if len(derived) >= 24:
            break
    return derived


def _informative_tokens_from_text(value: Any) -> list[str]:
    _, tokens, _ = _prepare_match_text(value)
    return [token for token in tokens if token not in _GENERIC_WORDS and len(token) >= 3]


def _filter_scheme_keywords(raw_keywords: list[str], scheme_tokens: set[str]) -> list[str]:
    filtered: list[str] = []
    for keyword in raw_keywords:
        key_tokens = _informative_tokens_from_text(keyword)
        if not key_tokens:
            continue
        if any(token in scheme_tokens for token in key_tokens):
            filtered.append(str(keyword).strip().lower())
    return filtered


def _get_cached_scheme_dataset() -> list[Dict[str, Any]]:
    global _SCHEME_DATASET_CACHE
    if _SCHEME_DATASET_CACHE is not None:
        return _SCHEME_DATASET_CACHE

    dataset: list[Dict[str, Any]] = []
    merged: Dict[str, Dict[str, Any]] = {}
    try:
        if not _SCHEME_DATASET_PATH.exists():
            _SCHEME_DATASET_CACHE = []
            return _SCHEME_DATASET_CACHE

        with _SCHEME_DATASET_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scheme_name = str((row or {}).get("scheme_name") or "").strip()
                if not scheme_name:
                    continue
                raw_keywords = _parse_keyword_blob((row or {}).get("keywords"))
                if not raw_keywords:
                    raw_keywords = _parse_keyword_blob((row or {}).get("query_variants"))
                fallback_keywords = _derive_fallback_keywords(scheme_name)
                _, _, scheme_key_tokens = _prepare_match_text(scheme_name)
                if not scheme_key_tokens:
                    continue
                keywords = _filter_scheme_keywords(raw_keywords, scheme_key_tokens)
                dedup_key = " ".join(sorted(scheme_key_tokens))
                entry = merged.get(dedup_key)
                combined_keywords = set(keywords)
                combined_keywords.update(fallback_keywords)
                if entry is None:
                    merged[dedup_key] = {
                        "scheme_name": scheme_name,
                        "keywords": sorted(item for item in combined_keywords if str(item).strip()),
                    }
                else:
                    existing = set(_parse_keyword_blob(entry.get("keywords")))
                    existing.update(combined_keywords)
                    entry["keywords"] = sorted(item for item in existing if str(item).strip())
        dataset = list(merged.values())
    except Exception:
        dataset = []

    _SCHEME_DATASET_CACHE = dataset
    return _SCHEME_DATASET_CACHE


_CLEANED_SCHEME_DATASET_CACHE: Optional[list[Dict[str, Any]]] = None


def _load_scheme_dataset_from_path(dataset_path: Path) -> list[Dict[str, Any]]:
    dataset: list[Dict[str, Any]] = []
    merged: Dict[str, Dict[str, Any]] = {}
    try:
        if not dataset_path.exists():
            return []

        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scheme_name = str((row or {}).get("scheme_name") or "").strip()
                if not scheme_name:
                    continue
                raw_keywords = _parse_keyword_blob((row or {}).get("keywords"))
                if not raw_keywords:
                    raw_keywords = _parse_keyword_blob((row or {}).get("query_variants"))
                fallback_keywords = _derive_fallback_keywords(scheme_name)
                _, _, scheme_key_tokens = _prepare_match_text(scheme_name)
                if not scheme_key_tokens:
                    continue
                keywords = _filter_scheme_keywords(raw_keywords, scheme_key_tokens)
                dedup_key = " ".join(sorted(scheme_key_tokens))
                entry = merged.get(dedup_key)
                combined_keywords = set(keywords)
                combined_keywords.update(fallback_keywords)
                if entry is None:
                    merged[dedup_key] = {
                        "scheme_name": scheme_name,
                        "keywords": sorted(item for item in combined_keywords if str(item).strip()),
                    }
                else:
                    existing = set(_parse_keyword_blob(entry.get("keywords")))
                    existing.update(combined_keywords)
                    entry["keywords"] = sorted(item for item in existing if str(item).strip())
        dataset = list(merged.values())
    except Exception:
        dataset = []
    return dataset


def _get_cached_cleaned_scheme_dataset() -> list[Dict[str, Any]]:
    global _CLEANED_SCHEME_DATASET_CACHE
    if _CLEANED_SCHEME_DATASET_CACHE is None:
        _CLEANED_SCHEME_DATASET_CACHE = _load_scheme_dataset_from_path(_CLEANED_SCHEME_DATASET_PATH)
    return _CLEANED_SCHEME_DATASET_CACHE


def _get_cached_original_scheme_dataset() -> list[Dict[str, Any]]:
    return _get_cached_scheme_dataset()


def _resolve_scheme_two_stage(text: str) -> Optional[str]:
    try:
        cleaned_match = resolve_scheme_from_dataset(text, _get_cached_cleaned_scheme_dataset())
        if cleaned_match is not None:
            canonical = _map_cleaned_match_to_original(cleaned_match, text)
            return canonical or cleaned_match
        return resolve_scheme_from_dataset(text, _get_cached_original_scheme_dataset())
    except Exception:
        return None


def _map_cleaned_match_to_original(cleaned_match: str, query_text: str) -> Optional[str]:
    try:
        cleaned_name = str(cleaned_match or "").strip()
        if not cleaned_name:
            return None

        _, cleaned_tokens, cleaned_set = _prepare_match_text(cleaned_name)
        if not cleaned_tokens:
            return None

        _, query_tokens, query_set = _prepare_match_text(query_text)
        original_dataset = _get_cached_original_scheme_dataset()
        best_name: Optional[str] = None
        best_score = 0.0

        for entry in original_dataset:
            if not isinstance(entry, dict):
                continue
            original_name = _sanitize_scheme_name(entry.get("scheme_name"))
            if not original_name:
                continue

            clean_original, original_tokens, original_set = _prepare_match_text(original_name)
            if not original_tokens:
                continue

            key_overlap = len(cleaned_set & original_set)
            if key_overlap == 0:
                continue

            query_overlap = len(query_set & original_set)
            similarity = _levenshtein_ratio(" ".join(cleaned_tokens), clean_original)
            score = (key_overlap * 1.0) + (query_overlap * 0.6) + (similarity * 0.8)

            if score > best_score:
                best_score = score
                best_name = original_name

        return best_name
    except Exception:
        return None


def resolve_scheme_from_dataset(text: str, dataset: list) -> Optional[str]:
    try:
        if not isinstance(dataset, list) or not dataset:
            return None

        clean_text, input_tokens, input_token_set = _prepare_match_text(text)
        if not input_tokens or _is_generic_only(input_tokens):
            return None
        meaningful_tokens = _informative_tokens_from_text(text)
        if len(meaningful_tokens) < 2:
            return None

        token_freq: Dict[str, int] = {}
        for entry in dataset:
            if not isinstance(entry, dict):
                continue
            scheme_name = _sanitize_scheme_name(entry.get("scheme_name"))
            if not scheme_name:
                continue
            _, scheme_tokens_for_freq, _ = _prepare_match_text(scheme_name)
            for token in set(scheme_tokens_for_freq):
                token_freq[token] = int(token_freq.get(token, 0)) + 1

        token_weight: Dict[str, float] = {
            token: 1.0 / (1.0 + float(freq)) for token, freq in token_freq.items()
        }

        best_scheme: Optional[str] = None
        best_score = 0.0
        best_token_hits = 0
        best_overlap_hits = 0
        best_similarity = 0.0
        best_phrase_len = 0
        scored_candidates: list[tuple[str, float, int, int, int, float, int, set[str], set[str]]] = []

        for entry in dataset:
            if not isinstance(entry, dict):
                continue

            scheme_name = _sanitize_scheme_name(entry.get("scheme_name"))
            if not scheme_name:
                continue

            score = 0.0
            clean_scheme_name, scheme_tokens, _ = _prepare_match_text(scheme_name)
            token_hits = 0

            if clean_scheme_name and clean_scheme_name in clean_text:
                score += _RESOLVER_PHRASE_WEIGHT

            scheme_token_set = set(scheme_tokens)
            overlap_hits = len(scheme_token_set & input_token_set)
            partial_hits = 0

            for token in scheme_tokens:
                if token in input_token_set:
                    score += 0.05 * float(token_weight.get(token, 1.0))
                    token_hits += 1
                else:
                    token_len = len(token)
                    if token_len >= 4 and any(
                        token in query_token or query_token in token
                        for query_token in input_tokens
                        if len(query_token) >= 4
                    ):
                        score += 0.03
                        partial_hits += 1

            if overlap_hits:
                score += min(0.35, overlap_hits * 0.08)

            keyword_score = 0.0
            for keyword in _parse_keyword_blob(entry.get("keywords")):
                keyword_tokens = _informative_tokens_from_text(keyword)
                if not keyword_tokens:
                    continue
                if all(token in input_token_set for token in keyword_tokens):
                    keyword_score += 0.1
            score += min(keyword_score, 0.3)

            similarity = _levenshtein_ratio(clean_text, clean_scheme_name)

            if token_hits == 0 and overlap_hits == 0 and similarity < 0.55:
                continue

            if similarity > 0.75:
                score += 0.3

            score = min(score, 1.0)
            scored_candidates.append(
                (
                    scheme_name,
                    score,
                    overlap_hits,
                    token_hits,
                    partial_hits,
                    similarity,
                    len(clean_scheme_name),
                    set(_informative_tokens_from_text(scheme_name)),
                    scheme_token_set,
                )
            )

            if score > best_score or (
                score == best_score
                and (
                    overlap_hits > best_overlap_hits
                    or token_hits > best_token_hits
                    or similarity > best_similarity
                    or len(clean_scheme_name) > best_phrase_len
                )
            ):
                best_score = score
                best_scheme = scheme_name
                best_token_hits = token_hits
                best_overlap_hits = overlap_hits
                best_similarity = similarity
                best_phrase_len = len(clean_scheme_name)
        if not scored_candidates:
            return None

        ranked = sorted(
            scored_candidates,
            key=lambda item: (item[1], item[2], item[3], item[4], item[5], item[6]),
            reverse=True,
        )

        best_item = ranked[0]
        best_scheme = best_item[0]
        best_score = float(best_item[1])
        best_key_tokens = set(best_item[7])
        best_scheme_tokens = set(best_item[8])

        # Query-scheme alignment: avoid returning a scheme when no key token from scheme appears in query.
        if best_key_tokens:
            aligned_hits = sum(
                1
                for token in best_key_tokens
                if token in input_token_set
                or (len(token) >= 4 and any(token in query_token for query_token in input_tokens if len(query_token) >= 4))
            )
            if aligned_hits == 0:
                return None

        if len(ranked) >= 2:
            best_margin = float(ranked[0][1]) - float(ranked[1][1])
            if best_margin < 0.15:
                return None

            # Ambiguity cluster check: if multiple high-ranking schemes share >70% tokens, abstain.
            cluster_candidates = [item for item in ranked if float(item[1]) >= (best_score - 0.15)]
            for candidate in cluster_candidates[1:]:
                candidate_tokens = set(candidate[8])
                if not best_scheme_tokens or not candidate_tokens:
                    continue
                shared = len(best_scheme_tokens & candidate_tokens)
                base = float(min(len(best_scheme_tokens), len(candidate_tokens)))
                if base > 0 and (shared / base) > 0.7:
                    return None

        if best_score < _RESOLVER_MIN_SCORE:
            return None
        return best_scheme
    except Exception:
        return None


def _validate_prediction(prediction: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(prediction, dict):
        _debug("intent_wrapper_invalid_prediction", reason="prediction_not_dict")
        return None

    intent = prediction.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        _debug("intent_wrapper_invalid_prediction", reason="intent_invalid")
        return None

    confidence_raw = prediction.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        _debug("intent_wrapper_invalid_prediction", reason="confidence_not_float", confidence=confidence_raw)
        return None

    if confidence < 0.0 or confidence > 1.0:
        _debug("intent_wrapper_invalid_prediction", reason="confidence_out_of_range", confidence=confidence)
        return None

    entities = prediction.get("entities")
    if not isinstance(entities, dict):
        entities = {}

    scheme_name = _sanitize_scheme_name(prediction.get("scheme_name"))

    response_template = prediction.get("response_template")
    if response_template is not None and not isinstance(response_template, str):
        response_template = None

    return {
        "intent": intent.strip(),
        "scheme_name": scheme_name,
        "entities": entities,
        "confidence": confidence,
        "response_template": response_template,
    }


def _apply_anti_bias_scheme_filter(text: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    scheme_name = prediction.get("scheme_name")
    if not scheme_name:
        return prediction

    confidence = float(prediction.get("confidence") or 0.0)
    input_tokens = _tokenize(text)

    # A) Low-confidence scheme claims are discarded.
    if confidence < 0.6:
        _debug("intent_wrapper_scheme_removed", reason="low_confidence", confidence=confidence, scheme_name=scheme_name)
        prediction["scheme_name"] = None
        return prediction

    # C) Generic-only input must not produce a concrete scheme.
    if _is_generic_only(input_tokens):
        _debug("intent_wrapper_scheme_removed", reason="generic_only_query", scheme_name=scheme_name)
        prediction["scheme_name"] = None
        return prediction

    # B) Require at least one non-generic scheme token to appear in input.
    scheme_tokens = _tokenize(scheme_name)
    non_generic_scheme_tokens = [token for token in scheme_tokens if token not in _GENERIC_WORDS]
    if not non_generic_scheme_tokens:
        _debug("intent_wrapper_scheme_removed", reason="scheme_tokens_only_generic", scheme_name=scheme_name)
        prediction["scheme_name"] = None
        return prediction

    input_token_set = set(input_tokens)
    if not any(token in input_token_set for token in non_generic_scheme_tokens):
        _debug(
            "intent_wrapper_scheme_removed",
            reason="no_strong_non_generic_match",
            scheme_name=scheme_name,
            non_generic_scheme_tokens=non_generic_scheme_tokens,
        )
        prediction["scheme_name"] = None
        return prediction

    return prediction


def _has_strong_scheme_match(text: str, scheme_name: str) -> bool:
    try:
        clean_text = str(text or "").strip().lower()
        if not clean_text:
            return False
        scheme = _sanitize_scheme_name(scheme_name)
        if not scheme:
            return False
        tokens = [tok for tok in _tokenize(scheme) if tok not in _GENERIC_WORDS and tok not in _STOPWORDS and len(tok) > 2]
        if not tokens:
            return False
        input_tokens = set(_tokenize(clean_text))
        return any(tok in input_tokens for tok in tokens)
    except Exception:
        return False


def _scheme_overlap_count(text: str, scheme_name: str) -> int:
    try:
        scheme = _sanitize_scheme_name(scheme_name)
        if not scheme:
            return 0
        query_tokens = set(_informative_tokens_from_text(text))
        scheme_tokens = set(_informative_tokens_from_text(scheme))
        if not query_tokens or not scheme_tokens:
            return 0
        return len(query_tokens & scheme_tokens)
    except Exception:
        return 0


def _predict_with_timeout(text: str) -> Optional[Dict[str, Any]]:
    for attempt in range(_RETRY_ATTEMPTS + 1):
        try:
            future = _EXECUTOR.submit(ml_model.predict, text)
            prediction = future.result(timeout=_TIMEOUT_SECONDS)
            _debug("intent_wrapper_ml_success", attempt=attempt)
            return prediction
        except TimeoutError:
            _debug("intent_wrapper_ml_failure", reason="timeout", timeout_seconds=_TIMEOUT_SECONDS, attempt=attempt)
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(0.05)
                continue
            return None
        except Exception as exc:
            _debug("intent_wrapper_ml_failure", reason="exception", error_type=type(exc).__name__, attempt=attempt)
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(0.05)
                continue
            return None
    return None


def _resolve_scheme_with_timeout(text: str) -> Optional[str]:
    for attempt in range(_RETRY_ATTEMPTS + 1):
        try:
            future = _EXECUTOR.submit(_resolve_scheme_two_stage, text)
            return future.result(timeout=_TIMEOUT_SECONDS)
        except TimeoutError:
            _debug("resolver_timeout", timeout_seconds=_TIMEOUT_SECONDS, attempt=attempt)
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(0.05)
                continue
            return None
        except Exception as exc:
            _debug("resolver_failure", error_type=type(exc).__name__, attempt=attempt)
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(0.05)
                continue
            return None
    return None


def get_intent(text: str) -> Optional[Dict[str, Any]]:
    try:
        if text is None:
            return None
        clean_text = str(text).strip()
        if not clean_text:
            return None

        prediction = _predict_with_timeout(clean_text)
        if prediction is None:
            return None

        validated = _validate_prediction(prediction)
        if validated is None:
            return None

        filtered = _apply_anti_bias_scheme_filter(clean_text, validated)
        return {
            "intent": filtered["intent"],
            "scheme_name": filtered.get("scheme_name"),
            "entities": filtered.get("entities") if isinstance(filtered.get("entities"), dict) else {},
            "confidence": float(filtered["confidence"]),
            "response_template": filtered.get("response_template"),
            "source": "ml",
        }
    except Exception:
        _debug("intent_wrapper_failure", reason="unexpected_exception")
        return None


def fallback_intent(text: str) -> Dict[str, Any]:
    try:
        if text is None:
            _debug("fallback_intent_used", reason="none_input")
            return _fallback_payload("general_query")

        clean_text = str(text).strip().lower()
        if not clean_text:
            _debug("fallback_intent_used", reason="empty_input")
            return _fallback_payload("general_query")

        apply_keywords = ("apply", "kaise milega", "apply kaise kare", "application", "process")
        eligibility_keywords = ("eligibility", "eligible", "yogya", "patr", "patrata", "criteria", "पात्र")
        scheme_keywords = ("yojana", "scheme", "योजना")

        if any(keyword in clean_text for keyword in apply_keywords):
            intent = "apply_help"
        elif any(keyword in clean_text for keyword in eligibility_keywords):
            intent = "eligibility_check"
        elif any(keyword in clean_text for keyword in scheme_keywords):
            intent = "scheme_search"
        else:
            intent = "general_query"

        _debug("fallback_intent_used", reason="ml_failure_or_unavailable", intent=intent)
        return _fallback_payload(intent)
    except Exception:
        _debug("fallback_intent_used", reason="unexpected_exception")
        return _fallback_payload("general_query")


def _update_session_context(session_context: Optional[dict], intent: str, scheme_name: Optional[str]) -> None:
    if not isinstance(session_context, dict):
        return
    try:
        if scheme_name and _is_supported_scheme(scheme_name):
            session_context["last_scheme"] = scheme_name
        session_context["last_intent"] = intent
    except Exception:
        # Memory update should never break response generation.
        return


def process_user_query(text: str, user_profile: dict = None, session_context: Optional[dict] = None) -> Dict[str, Any]:
    active_session_context = (
        session_context
        if isinstance(session_context, dict)
        else (user_profile if isinstance(user_profile, dict) else globals().get("session_context"))
    )
    response_language = _detect_response_language(text)
    try:
        normalized = normalize_text(text)
        _debug("process_user_query_normalized", normalized=normalized)

        if not normalized:
            return {
                "success": False,
                "message": "Please enter a valid query",
                "type": "error",
                "data": {},
                "confidence": 0.0,
            }

        source = "ml"
        scheme_resolved_by_dataset = False
        static_scheme = detect_scheme(normalized)
        remembered_scheme = None
        if isinstance(active_session_context, dict):
            remembered_scheme = _sanitize_scheme_name(active_session_context.get("last_scheme"))
        inferred_followup = _infer_followup_intent_from_text(normalized)

        if static_scheme:
            intent_data = {
                "intent": "scheme_info",
                "scheme_name": static_scheme,
                "entities": {},
                "confidence": 1.0,
                "response_template": None,
                "source": "static_scheme_data",
            }
            source = "static_scheme_data"
        elif remembered_scheme and inferred_followup:
            intent_data = {
                "intent": inferred_followup,
                "scheme_name": remembered_scheme,
                "entities": {},
                "confidence": 1.0,
                "response_template": None,
                "source": "context_followup",
            }
            source = "context_followup"
        else:
            intent_data = get_intent(normalized)

        if static_scheme and intent_data is not None:
            intent_data = dict(intent_data)
            intent_data["scheme_name"] = static_scheme
            if str(intent_data.get("intent") or "") in {"general", "general_query", "scheme_search", "scheme_query"}:
                intent_data["intent"] = "scheme_info"
            source = "static_scheme_data"

        if intent_data is not None:
            current_scheme = intent_data.get("scheme_name")
            resolved_scheme = _resolve_scheme_with_timeout(normalized)
            ml_scheme_str = str(current_scheme).strip() if isinstance(current_scheme, str) else ""

            if resolved_scheme is not None:
                should_override = False
                if not ml_scheme_str:
                    should_override = True
                else:
                    if not _has_strong_scheme_match(normalized, ml_scheme_str):
                        should_override = True
                    elif resolved_scheme.strip().lower() != ml_scheme_str.strip().lower():
                        resolver_overlap = _scheme_overlap_count(normalized, resolved_scheme)
                        ml_overlap = _scheme_overlap_count(normalized, ml_scheme_str)
                        if resolver_overlap > ml_overlap:
                            should_override = True

                if should_override:
                    intent_data = dict(intent_data)
                    intent_data["scheme_name"] = resolved_scheme
                    scheme_resolved_by_dataset = True
                    _debug(
                        "process_user_query_dataset_scheme_resolved",
                        scheme_name=resolved_scheme,
                        replaced_ml_scheme=ml_scheme_str or None,
                    )

        if intent_data is None:
            fallback_preview = fallback_intent(normalized)
            preview_intent = str(fallback_preview.get("intent") or "general_query")
            should_try_resolver = preview_intent not in {"apply_help", "eligibility_check"}
            if preview_intent == "scheme_search" and _is_non_discriminative_query(normalized):
                should_try_resolver = False

            if should_try_resolver:
                resolved_scheme = _resolve_scheme_with_timeout(normalized)
                if resolved_scheme is not None:
                    intent_data = {
                        "intent": "scheme_info",
                        "scheme_name": resolved_scheme,
                        "entities": {},
                        "confidence": 0.0,
                        "response_template": None,
                        "source": "resolver",
                    }
                    scheme_resolved_by_dataset = True
                    source = "resolver"
                    _debug("process_user_query_dataset_scheme_resolved_after_ml_none", scheme_name=resolved_scheme)

        if intent_data is None:
            intent_data = fallback_intent(normalized)
            source = "fallback"
        _debug("process_user_query_source", source=source)

        intent = str(intent_data.get("intent") or "general_query")
        scheme = intent_data.get("scheme_name")
        protected_intents = {"apply_help", "eligibility_check"}
        followup_intents = {"eligibility", "eligibility_check", "application_help", "apply_help"}

        if isinstance(scheme, str) and scheme.strip() and intent not in protected_intents:
            weak_or_generic_intents = {"general_query", "scheme_search", "scheme_query", "unknown", "other"}
            if scheme_resolved_by_dataset and intent in weak_or_generic_intents:
                intent = "scheme_info"
            elif intent in {"scheme_query", "scheme_info"}:
                intent = "scheme_info"

        if isinstance(scheme, str) and scheme.strip() and intent in {"general", "general_query", "scheme_search"}:
            intent = "scheme_info"

        if intent in followup_intents and not _sanitize_scheme_name(scheme):
            remembered_scheme = None
            if isinstance(active_session_context, dict):
                remembered_scheme = _sanitize_scheme_name(active_session_context.get("last_scheme"))
                if remembered_scheme and not _is_supported_scheme(remembered_scheme):
                    remembered_scheme = None

            if remembered_scheme:
                scheme = remembered_scheme
                intent_data = dict(intent_data)
                intent_data["scheme_name"] = remembered_scheme
                _debug(
                    "process_user_query_followup_scheme_reused",
                    intent=intent,
                    remembered_scheme=remembered_scheme,
                )
            else:
                response_type = "application_help" if intent in {"apply_help", "application_help"} else "eligibility"
                response = build_hackathon_response(
                    success=True,
                    response_type=response_type,
                    message=(
                        "कृपया योजना का नाम बताइए, फिर मैं पात्रता या आवेदन में मदद करूँगा।"
                        if response_language == "hi"
                        else "Please share the scheme name for eligibility or application guidance."
                    ),
                    summary=("फॉलो-अप प्रश्न के लिए योजना संदर्भ चाहिए" if response_language == "hi" else "Need scheme context for follow-up query"),
                    reason=("वर्तमान प्रश्न में योजना नाम नहीं मिला।" if response_language == "hi" else "Follow-up intent detected without a scheme in the current query."),
                    next_step=("पहले योजना का नाम बताएं, फिर पात्रता या आवेदन पूछें।" if response_language == "hi" else "Mention the scheme name, then ask your eligibility or application question."),
                    data={"mode": "clarification"},
                    confidence=_to_confidence_number(intent_data.get("confidence")),
                    language=response_language,
                )
                _update_session_context(active_session_context, "general", None)
                return response

        if not _sanitize_scheme_name(scheme):
            remembered_scheme = None
            if isinstance(active_session_context, dict):
                remembered_scheme = _sanitize_scheme_name(active_session_context.get("last_scheme"))
                if remembered_scheme and not _is_supported_scheme(remembered_scheme):
                    remembered_scheme = None
            inferred_followup = _infer_followup_intent_from_text(normalized)
            if remembered_scheme and inferred_followup:
                scheme = remembered_scheme
                intent = inferred_followup
                intent_data = dict(intent_data)
                intent_data["scheme_name"] = remembered_scheme

        scheme = _sanitize_scheme_name(scheme)
        canonical_scheme = _canonical_scheme_name(scheme)
        display_scheme = _scheme_display_name(canonical_scheme or scheme)
        scheme_details = SCHEME_DATA.get(canonical_scheme, {}) if canonical_scheme else {}
        scheme_summary_text = str((scheme_details or {}).get("summary") or "").strip()
        scheme_eligibility_text = str((scheme_details or {}).get("eligibility") or "").strip()
        scheme_steps_text = str((scheme_details or {}).get("steps") or "").strip()

        if intent in {"scheme_info", "eligibility", "eligibility_check", "application_help", "apply_help"}:
            if not canonical_scheme:
                response = build_hackathon_response(
                    success=True,
                    response_type="general",
                    message=CONTROLLED_SCHEME_CLARIFICATION,
                    summary=("Need a supported scheme name" if response_language != "hi" else "समर्थित योजना का नाम चाहिए"),
                    reason=("Detailed responses are limited to supported scheme data." if response_language != "hi" else "विस्तृत उत्तर समर्थित योजना डेटा तक सीमित हैं।"),
                    next_step=("Ask using a supported scheme name from the available list." if response_language != "hi" else "उपलब्ध सूची में से योजना नाम लेकर पूछें।"),
                    data={"mode": "clarification"},
                    confidence=_to_confidence_number(intent_data.get("confidence")),
                    language=response_language,
                )
                _update_session_context(active_session_context, "general", None)
                return response

        _debug("process_user_query_intent", intent=intent, scheme_present=bool(scheme))

        if intent == "scheme_info" and canonical_scheme:
            scheme_summary = scheme_summary_text or (
                f"{display_scheme} पात्र नागरिकों को सरकारी सहायता और योजना-विशेष लाभ प्रदान करती है।"
                if response_language == "hi"
                else f"{display_scheme} helps eligible citizens through government support and scheme-specific benefits."
            )
            response = build_hackathon_response(
                success=True,
                response_type="scheme_info",
                message=intent_data.get("response_template", "Here is the information"),
                summary=(f"{display_scheme} के लिए जानकारी मिली" if response_language == "hi" else f"Information found for {display_scheme}"),
                reason=("मॉडल ने आपके प्रश्न से योजना का मिलान किया।" if response_language == "hi" else "The model detected a specific scheme match from your query."),
                next_step=("पात्रता, दस्तावेज़ या आवेदन चरण पूछें।" if response_language == "hi" else "Ask about eligibility, documents, or application steps."),
                data={
                    "scheme": display_scheme,
                    "summary": scheme_summary,
                    "eligibility": scheme_eligibility_text,
                    "steps": scheme_steps_text,
                },
                language=response_language,
            )
            response["confidence"] = _to_confidence_number(intent_data.get("confidence"))
            _update_session_context(active_session_context, intent, canonical_scheme)
            return response

        if intent == "eligibility_check":
            resolved_scheme = display_scheme if canonical_scheme else None
            eligibility_message = "मैं आपकी पात्रता जांचता हूँ।" if response_language == "hi" else "Let me check your eligibility."
            if resolved_scheme:
                eligibility_message = (
                    f"मैं {resolved_scheme} के लिए आपकी पात्रता जांचता हूँ।"
                    if response_language == "hi"
                    else f"Let me check your eligibility for {resolved_scheme}."
                )
            eligibility_text = scheme_eligibility_text or (
                (
                    f"{resolved_scheme} की पात्रता योजना नियमों पर निर्भर करती है, जैसे श्रेणी, आय सीमा और आवश्यक दस्तावेज़।"
                    if response_language == "hi"
                    else f"Eligibility for {resolved_scheme} depends on scheme rules such as applicant category, income limits, and required documents."
                )
                if resolved_scheme
                else (
                    "पात्रता योजना नियमों पर निर्भर करती है, जैसे श्रेणी, आय सीमा और आवश्यक दस्तावेज़।"
                    if response_language == "hi"
                    else "Eligibility depends on scheme rules such as applicant category, income limits, and required documents."
                )
            )
            response = build_hackathon_response(
                success=True,
                response_type="eligibility",
                message=eligibility_message,
                summary=("पात्रता जाँच शुरू हुई" if response_language == "hi" else "Eligibility check started"),
                reason=("आपके प्रश्न में पात्रता जाँच की मांग है।" if response_language == "hi" else "Your query asks whether you qualify."),
                next_step=("अपनी मुख्य जानकारी साझा करें ताकि मैं पात्रता जाँच सकूँ।" if response_language == "hi" else "Share your key details so I can assess eligibility."),
                data={
                    "status": "needs_more_info",
                    "scheme": resolved_scheme,
                    "summary": "",
                    "eligibility": eligibility_text,
                    "steps": "",
                },
                language=response_language,
            )
            response["confidence"] = _to_confidence_number(intent_data.get("confidence"))
            _update_session_context(active_session_context, intent, canonical_scheme)
            return response

        if intent == "apply_help":
            resolved_scheme = display_scheme if canonical_scheme else None
            apply_message = "मैं आवेदन प्रक्रिया में आपकी मदद कर सकता हूँ।" if response_language == "hi" else "I can guide you on how to apply."
            if resolved_scheme:
                apply_message = (
                    f"मैं {resolved_scheme} के लिए आवेदन करने में आपकी मदद कर सकता हूँ।"
                    if response_language == "hi"
                    else f"I can guide you on how to apply for {resolved_scheme}."
                )
            apply_steps = scheme_steps_text or (
                "1) पात्रता जांचें। 2) आवश्यक दस्तावेज़ तैयार रखें। 3) आधिकारिक पोर्टल/कार्यालय में आवेदन जमा करें। 4) स्थिति ट्रैक करें और सत्यापन पूरा करें।"
                if response_language == "hi"
                else "1) Check eligibility. 2) Keep required documents ready. 3) Submit application on the official portal or office. 4) Track status and complete verification."
            )
            response = build_hackathon_response(
                success=True,
                response_type="application_help",
                message=apply_message,
                summary=("आवेदन मार्गदर्शन तैयार है" if response_language == "hi" else "Application guidance ready"),
                reason=("आपका प्रश्न योजना के आवेदन से संबंधित है।" if response_language == "hi" else "Your query is about applying for a scheme."),
                next_step=("योजना का नाम बताएं, मैं चरण-दर-चरण मार्गदर्शन दूँगा।" if response_language == "hi" else "Tell me the scheme name to get step-by-step guidance."),
                data={
                    "scheme": resolved_scheme,
                    "summary": "",
                    "eligibility": "",
                    "steps": apply_steps,
                },
                language=response_language,
            )
            response["confidence"] = _to_confidence_number(intent_data.get("confidence"))
            _update_session_context(active_session_context, intent, canonical_scheme)
            return response

        if intent == "scheme_search":
            response = build_hackathon_response(
                success=True,
                response_type="scheme_search",
                message=("मैं योजना खोजने में मदद कर सकता हूँ। कृपया थोड़ी और जानकारी दें।" if response_language == "hi" else "I can help you find schemes. Please share a bit more detail."),
                summary=("सही योजना पहचानने के लिए अधिक विवरण चाहिए" if response_language == "hi" else "Need more details to identify the right scheme"),
                reason=("किसी विशेष योजना का मजबूत मिलान नहीं मिला।" if response_language == "hi" else "No specific scheme was confidently detected."),
                next_step=("अपना राज्य, श्रेणी या सटीक योजना नाम बताएं।" if response_language == "hi" else "Share your state, category, or exact scheme name."),
                data={"mode": "clarification", "summary": "", "eligibility": "", "steps": ""},
                language=response_language,
            )
            response["confidence"] = _to_confidence_number(intent_data.get("confidence"))
            _update_session_context(active_session_context, intent, _sanitize_scheme_name(scheme))
            return response

        if canonical_scheme:
            response = build_hackathon_response(
                success=True,
                response_type="scheme_info",
                message=(f"{display_scheme} की जानकारी नीचे दी गई है।" if response_language == "hi" else f"Here are the details for {display_scheme}."),
                summary=(f"{display_scheme} के लिए जानकारी मिली" if response_language == "hi" else f"Information found for {display_scheme}"),
                reason=("योजना पहचान ली गई, इसलिए योजना-विशिष्ट उत्तर दिया गया।" if response_language == "hi" else "A supported scheme was detected, so a scheme-specific response was returned."),
                next_step=("पात्रता या आवेदन प्रक्रिया के बारे में आगे पूछें।" if response_language == "hi" else "Ask next about eligibility or application process."),
                data={
                    "scheme": display_scheme,
                    "summary": scheme_summary_text,
                    "eligibility": scheme_eligibility_text,
                    "steps": scheme_steps_text,
                },
                language=response_language,
            )
            response["confidence"] = _to_confidence_number(intent_data.get("confidence"))
            _update_session_context(active_session_context, "scheme_info", canonical_scheme)
            return response

        response = build_hackathon_response(
            success=True,
            response_type="general",
            message=("कृपया योजना का नाम बताएं।" if response_language == "hi" else "Please mention one supported scheme name."),
            summary=("मुझे थोड़ा स्पष्ट प्रश्न चाहिए" if response_language == "hi" else "I need a clearer request"),
            reason=("प्रश्न अभी किसी विशिष्ट इंटेंट से मेल नहीं खा रहा।" if response_language == "hi" else "The query does not map to a specific intent yet."),
            next_step=("पात्रता, आवेदन या योजना नाम के बारे में पूछें।" if response_language == "hi" else "Try asking about eligibility, application help, or a scheme name."),
            data={"mode": "clarification", "summary": "", "eligibility": "", "steps": ""},
            language=response_language,
        )
        response["confidence"] = _to_confidence_number(intent_data.get("confidence"))
        _update_session_context(active_session_context, intent, _sanitize_scheme_name(scheme))
        return response
    except Exception:
        return {
            "success": False,
            "type": "fallback",
            "message": "Something went wrong. Please try again.",
            "data": {},
            "confidence": 0.0,
        }

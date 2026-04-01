import logging
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.core.intent_analytics import record_intent_event
from backend.domain.use_cases.intent_resolver import resolve_intent_decision
from backend.intents import (
    INTENT_APPLY_LOAN,
    INTENT_CHECK_APPLICATION_STATUS,
    INTENT_GENERAL_QUERY,
    INTENT_SCHEME_QUERY,
    INTENT_VERSION,
    normalize_intent,
)


INTENT_PROVIDE_INFO = "provide_information"


logger = logging.getLogger(__name__)

# Runtime-configurable model path with stable DistilBERT default.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "./models/intent_model_distilbert")).resolve()

MODEL: Optional[Any] = None
TOKENIZER: Optional[Any] = None
ID2LABEL: Dict[int, str] = {}
MODEL_DEVICE: Optional[Any] = None
MODEL_LOADED = False
LOW_CONFIDENCE_THRESHOLD = 0.40
HIGH_CONFIDENCE_THRESHOLD = 0.70
_MODEL_LOAD_ATTEMPTED = False
_MODEL_LOAD_LOCK = threading.Lock()
_model_load_error: Optional[str] = None

HINGLISH_NORMALIZATION = {
    "apply karna": "apply",
    "apply karo": "apply",
    "yojana": "scheme",
    "kya": "what",
    "chahiye": "need",
    "aply": "apply",
    "documnts": "documents",
    "yojna": "scheme",
    "milgya": "mil gaya",
}

FALLBACK_RULES = {
    "apply": {
        "intent": INTENT_APPLY_LOAN,
        "keywords": {
            "apply",
            "register",
            "application",
            "loan",
            "आवेदन",
            "loan apply",
            "form",
            "registration",
            "apply process",
            "aply process",
            "application process",
            "kaise kare",
        },
        "base_confidence": 0.62,
    },
    "documents": {
        "intent": INTENT_SCHEME_QUERY,
        "keywords": {
            "documents",
            "kya chahiye",
            "required",
        },
        "base_confidence": 0.58,
    },
    "status": {
        "intent": INTENT_CHECK_APPLICATION_STATUS,
        "keywords": {
            "pending",
            "not received",
            "application status",
            "status check",
            "track status",
        },
        "base_confidence": 0.52,
    },
    "benefits": {
        "intent": INTENT_SCHEME_QUERY,
        "keywords": {
            "kitna milega",
            "benefit",
            "amount",
            "paisa",
        },
        "base_confidence": 0.59,
    },
    "query": {
        "intent": INTENT_SCHEME_QUERY,
        "keywords": {
            "what",
            "scheme",
            "yojana",
            "योजना",
            "किसान",
            "kisan",
            "eligibility",
            "kya hai",
            "yojna kya hai",
            "scheme kya hai",
            "batao",
            "mujhe batao",
            "zara batao",
            "help chahiye",
            "kaise hoga",
            "kya karna hai",
            "uske liye kya chahiye",
            "next step",
            "what now",
            "need update",
            "tell me quickly",
            "where can i see status",
            "what is the next step",
            "what is needed for that",
            "how much amount will i get",
            "kab tak milega",
            "kitna paisa milega",
            "ab kya karna hai",
            "next step kya",
            "कब तक मिलेगा",
            "कितना पैसा मिलेगा",
            "अब क्या करना है",
            "अगला स्टेप क्या है",
            "उसके लिए क्या चाहिए",
            "स्थिति कहां दिखेगी",
            "for my case",
            "meri side",
            "mere case",
            "स्थिति बताइए",
            "जानकारी दें",
        },
        "base_confidence": 0.62,
    },
    "correction": {
        "intent": INTENT_GENERAL_QUERY,
        "keywords": {
            "wrong",
            "गलत",
            "बदल",
            "सुधार",
            "change",
            "edit",
        },
        "base_confidence": 0.8,
    },
    "greeting": {
        "intent": INTENT_GENERAL_QUERY,
        "keywords": {
            "hello",
            "hi",
            "hey",
            "namaste",
            "नमस्ते",
        },
        "base_confidence": 0.74,
    },
}

PROVIDE_INFO_KEYWORDS = {
    "number",
    "mobile",
    "aadhaar",
    "aadhar",
    "phone",
    "name",
}

STATUS_CONTEXT_PATTERNS = {
    "mera paisa nahi aya",
    "paisa nahi aya",
    "naam list me nahi hai",
    "naam list me nahi",
    "form bhar diya next kya",
}

AADHAAR_PATTERN = re.compile(r"(?<!\d)\d{12}(?!\d)")
PHONE_PATTERN = re.compile(r"(?<!\d)\d{10}(?!\d)")
NAME_VALUE_PATTERN = re.compile(
    r"(?:my\s+name\s+is|name\s+is|i\s+am|mera\s+naam)\s*[:\-]?\s*([a-z\u0900-\u097f][a-z\u0900-\u097f\s.'-]{1,80})",
    re.IGNORECASE,
)


def _resolve_model_dir() -> Optional[Path]:
    if MODEL_PATH.exists() and MODEL_PATH.is_dir():
        return MODEL_PATH
    return None


def _extract_id2label(model: Any) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    raw_id2label = getattr(getattr(model, "config", None), "id2label", {}) or {}
    if isinstance(raw_id2label, dict):
        for key, value in raw_id2label.items():
            try:
                mapping[int(key)] = str(value)
            except (TypeError, ValueError):
                continue

    if mapping:
        return mapping

    raw_label2id = getattr(getattr(model, "config", None), "label2id", {}) or {}
    if isinstance(raw_label2id, dict):
        for label, idx in raw_label2id.items():
            try:
                mapping[int(idx)] = str(label)
            except (TypeError, ValueError):
                continue
    return mapping


def load_model() -> None:
    global MODEL, TOKENIZER, ID2LABEL, MODEL_DEVICE, MODEL_LOADED, _model_load_error, _MODEL_LOAD_ATTEMPTED

    if MODEL_LOADED:
        return

    with _MODEL_LOAD_LOCK:
        _MODEL_LOAD_ATTEMPTED = True
        if MODEL_LOADED:
            return

        resolved_dir = _resolve_model_dir()
        if resolved_dir is None:
            MODEL = None
            TOKENIZER = None
            ID2LABEL = {}
            MODEL_DEVICE = None
            MODEL_LOADED = False
            _model_load_error = "intent_model_not_found"
            return

        try:
            import torch
            from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

            TOKENIZER = DistilBertTokenizerFast.from_pretrained(str(resolved_dir), local_files_only=True)
            MODEL = DistilBertForSequenceClassification.from_pretrained(str(resolved_dir), local_files_only=True)
            MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            MODEL.to(MODEL_DEVICE)
            MODEL.eval()

            ID2LABEL = _extract_id2label(MODEL)
            if not ID2LABEL:
                raise RuntimeError("model_config_missing_id2label")

            MODEL_LOADED = True
            _model_load_error = None
            logger.info("intent_model_loaded model_path=%s device=%s labels=%s", str(resolved_dir), str(MODEL_DEVICE), _label_count())
        except Exception as exc:
            MODEL = None
            TOKENIZER = None
            ID2LABEL = {}
            MODEL_DEVICE = None
            MODEL_LOADED = False
            _model_load_error = str(exc)
            logger.exception("Intent model load failed, using fallback")


def _label_from_index(index: int) -> Optional[str]:
    if index in ID2LABEL:
        return str(ID2LABEL[index])
    return None


def _label_count() -> int:
    return int(len(ID2LABEL))


def _keyword_hit_count(normalized_text: str, keywords: set[str]) -> int:
    hits = 0
    for keyword in keywords:
        token = str(keyword or "").strip().lower()
        if not token:
            continue

        # Multi-word patterns rely on substring checks; single tokens use boundary checks.
        if " " in token:
            if token in normalized_text:
                hits += 1
            continue

        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", normalized_text):
            hits += 1
    return hits


def _confidence_tier(raw_confidence: float) -> str:
    if raw_confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if raw_confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _log_prediction_outcome(
    *,
    source: str,
    raw_intent: str,
    final_intent: str,
    raw_confidence: float,
    final_confidence: float,
    model_used: bool,
    fallback_used: bool,
    low_confidence: bool,
) -> None:
    logger.info(
        "intent_prediction_outcome source=%s raw_intent=%s final_intent=%s raw_confidence=%.3f final_confidence=%.3f model_used=%s fallback_used=%s low_confidence=%s",
        source,
        raw_intent,
        final_intent,
        float(raw_confidence),
        float(final_confidence),
        bool(model_used),
        bool(fallback_used),
        bool(low_confidence),
    )


def _finalize_prediction(
    decision: Dict[str, Any],
    *,
    intent: str,
    source: str,
    raw_intent: str,
    raw_confidence: float,
    raw_model_output: Dict[str, Any],
    model_used: bool,
    fallback_used: bool,
) -> dict:
    decision.update(
        {
            "intent": intent,
            "intent_version": INTENT_VERSION,
            "source": source,
            "raw_model_output": raw_model_output,
            "model_loaded": bool(MODEL_LOADED),
            "model_used": bool(model_used),
            "fallback_used": bool(fallback_used),
        }
    )
    if fallback_used and not decision.get("fallback_reason"):
        decision["fallback_reason"] = source

    _log_prediction_outcome(
        source=source,
        raw_intent=raw_intent,
        final_intent=str(decision.get("primary_intent") or intent),
        raw_confidence=float(raw_confidence),
        final_confidence=float(decision.get("confidence") or 0.0),
        model_used=bool(model_used),
        fallback_used=bool(fallback_used),
        low_confidence=bool(decision.get("low_confidence", False)),
    )
    record_intent_event(
        intent=decision["primary_intent"],
        confidence=decision["confidence"],
        fallback_used=decision["fallback_used"],
        low_confidence=decision["low_confidence"],
        raw_intent=decision["raw_intent"],
    )
    return decision


def fallback_intent(text: str) -> Tuple[str, float]:
    lowered = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    if not lowered:
        return INTENT_GENERAL_QUERY, 0.0

    normalized = re.sub(r"\s+", " ", lowered)
    for source, target in sorted(HINGLISH_NORMALIZATION.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = normalized.replace(source, target)

    has_aadhaar = bool(AADHAAR_PATTERN.search(normalized))
    has_phone = bool(PHONE_PATTERN.search(normalized))
    provide_keyword_hits = sum(1 for token in PROVIDE_INFO_KEYWORDS if token in normalized)
    status_pattern_hits = sum(1 for pattern in STATUS_CONTEXT_PATTERNS if pattern in normalized)
    provide_info_detected = bool(has_aadhaar or has_phone or provide_keyword_hits > 0)

    scored_groups = []
    for group, config in FALLBACK_RULES.items():
        hits = _keyword_hit_count(normalized, config["keywords"])
        if hits <= 0:
            continue
        confidence = min(0.95, float(config["base_confidence"]) + (0.08 * hits))
        scored_groups.append((group, int(hits), float(confidence), str(config["intent"])))

    apply_detected = any(group == "apply" for group, *_ in scored_groups)
    # Do not override APPLY here when both are present; active-flow handling is done upstream.
    if provide_info_detected and not apply_detected and status_pattern_hits <= 0:
        info_conf = 0.8
        if has_aadhaar or has_phone:
            info_conf = 0.84
        elif provide_keyword_hits >= 2:
            info_conf = 0.82
        return INTENT_PROVIDE_INFO, info_conf

    if not scored_groups:
        return INTENT_GENERAL_QUERY, 0.0

    priority = {"correction": 5, "query": 4, "apply": 3, "status": 2, "greeting": 1}
    scored_groups.sort(key=lambda row: (row[2], priority.get(row[0], 0), row[1]), reverse=True)
    _, _, confidence, intent = scored_groups[0]
    return intent, confidence


def detect_information_input(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    if not lowered:
        return False

    has_aadhaar_digits = bool(AADHAAR_PATTERN.search(lowered))
    has_phone_digits = bool(PHONE_PATTERN.search(lowered))
    has_aadhaar_context = any(token in lowered for token in {"aadhaar", "aadhar"})
    has_phone_context = any(token in lowered for token in {"phone", "mobile", "contact", "number"})

    if has_aadhaar_digits and has_aadhaar_context:
        return True
    if has_phone_digits and has_phone_context:
        return True

    # Name should be treated as user profile info only when explicitly provided as a value,
    # not when used in scheme/status phrases like "name in list".
    name_match = NAME_VALUE_PATTERN.search(lowered)
    if name_match:
        denied_markers = {
            "list",
            "status",
            "scheme",
            "yojana",
            "eligibility",
            "benefit",
            "documents",
            "required",
            "kab milega",
            "nahi aya",
        }
        if not any(marker in lowered for marker in denied_markers):
            return True

    # Standalone Aadhaar with no explicit keyword is still strong enough.
    if has_aadhaar_digits:
        return True

    return False


def get_intent_model_status() -> dict:
    loaded = bool(MODEL_LOADED and MODEL is not None and TOKENIZER is not None and ID2LABEL)
    return {
        "model_path": str(MODEL_PATH),
        "device": str(MODEL_DEVICE) if MODEL_DEVICE is not None else "uninitialized",
        "loaded": loaded,
        "fallback_enabled": True,
        "attempted": _MODEL_LOAD_ATTEMPTED,
        "error": _model_load_error,
        "num_labels": int(MODEL.config.num_labels) if MODEL is not None else 0,
        "label_count": _label_count(),
    }


def predict_intent_detailed(text: str, session_context: Optional[dict] = None) -> dict:
    clean_text = (text or "").strip()
    session_context = session_context or {}
    if not clean_text:
        decision = resolve_intent_decision(
            raw_intent=INTENT_GENERAL_QUERY,
            raw_confidence=0.0,
            text=clean_text,
            session_context=session_context,
        )
        return _finalize_prediction(
            decision,
            intent=INTENT_GENERAL_QUERY,
            source="empty_input",
            raw_intent=INTENT_GENERAL_QUERY,
            raw_confidence=0.0,
            raw_model_output={"intent": INTENT_GENERAL_QUERY, "confidence": 0.0},
            model_used=False,
            fallback_used=True,
        )

    if detect_information_input(clean_text):
        decision = resolve_intent_decision(
            raw_intent=INTENT_PROVIDE_INFO,
            raw_confidence=0.95,
            text=clean_text,
            session_context=session_context,
        )
        decision["primary_intent"] = INTENT_PROVIDE_INFO
        decision["normalized_intent"] = INTENT_PROVIDE_INFO
        decision["low_confidence"] = False
        decision["fallback_used"] = False
        decision["fallback_reason"] = ""
        decision["confidence"] = 0.95
        return _finalize_prediction(
            decision,
            intent=INTENT_PROVIDE_INFO,
            source="rule_based_info_detection",
            raw_intent=INTENT_PROVIDE_INFO,
            raw_confidence=0.95,
            raw_model_output={"intent": INTENT_PROVIDE_INFO, "confidence": 0.95},
            model_used=False,
            fallback_used=False,
        )

    load_model()
    if not MODEL_LOADED:
        raw_intent, raw_confidence = fallback_intent(clean_text)
        decision = resolve_intent_decision(
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            text=clean_text,
            session_context=session_context,
        )
        if raw_intent == INTENT_PROVIDE_INFO:
            decision["primary_intent"] = INTENT_PROVIDE_INFO
            decision["normalized_intent"] = INTENT_PROVIDE_INFO
            decision["fallback_used"] = True
            decision["fallback_reason"] = decision.get("fallback_reason") or "provide_info_detected"
        return _finalize_prediction(
            decision,
            intent=raw_intent,
            source="heuristic_fallback",
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            raw_model_output={"intent": raw_intent, "confidence": raw_confidence},
            model_used=False,
            fallback_used=True,
        )

    if MODEL is None or TOKENIZER is None or not ID2LABEL:
        raw_intent, raw_confidence = fallback_intent(clean_text)
        decision = resolve_intent_decision(
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            text=clean_text,
            session_context=session_context,
        )
        if raw_intent == INTENT_PROVIDE_INFO:
            decision["primary_intent"] = INTENT_PROVIDE_INFO
            decision["normalized_intent"] = INTENT_PROVIDE_INFO
            decision["fallback_used"] = True
            decision["fallback_reason"] = decision.get("fallback_reason") or "provide_info_detected"
        return _finalize_prediction(
            decision,
            intent=raw_intent,
            source="model_unavailable_fallback",
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            raw_model_output={"intent": raw_intent, "confidence": raw_confidence},
            model_used=False,
            fallback_used=True,
        )

    try:
        import torch

        device = MODEL_DEVICE or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = TOKENIZER(
            clean_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = MODEL(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        idx = int(torch.argmax(probs, dim=1).item())
        raw_intent = _label_from_index(idx) or INTENT_GENERAL_QUERY
        raw_confidence = float(probs[0][idx].item())
        confidence_tier = _confidence_tier(raw_confidence)

        top_k = min(3, probs.shape[1])
        top_vals, top_indices = torch.topk(probs[0], k=top_k)
        top_candidates = [
            {
                "intent": _label_from_index(int(i.item())) or INTENT_GENERAL_QUERY,
                "confidence": float(v.item()),
            }
            for v, i in zip(top_vals, top_indices)
        ]

        if confidence_tier == "low":
            rule_intent, rule_confidence = fallback_intent(clean_text)
            rule_matched = rule_intent != INTENT_GENERAL_QUERY

            # Rule-assisted correction path for low confidence predictions.
            if rule_matched and rule_confidence >= LOW_CONFIDENCE_THRESHOLD:
                decision = resolve_intent_decision(
                    raw_intent=rule_intent,
                    raw_confidence=rule_confidence,
                    text=clean_text,
                    session_context=session_context,
                )
                if rule_intent == INTENT_PROVIDE_INFO:
                    decision["primary_intent"] = INTENT_PROVIDE_INFO
                    decision["normalized_intent"] = INTENT_PROVIDE_INFO
                decision["fallback_used"] = False
                decision["fallback_reason"] = ""
                decision["low_confidence"] = True
                decision["confidence"] = float(max(rule_confidence, LOW_CONFIDENCE_THRESHOLD))
                return _finalize_prediction(
                    decision,
                    intent=rule_intent,
                    source="hybrid_rule_correction_low_conf",
                    raw_intent=raw_intent,
                    raw_confidence=raw_confidence,
                    raw_model_output={
                        "intent": raw_intent,
                        "confidence": raw_confidence,
                        "top_candidates": top_candidates,
                        "confidence_tier": confidence_tier,
                        "rule_intent": rule_intent,
                        "rule_confidence": rule_confidence,
                        "thresholds": {
                            "low": LOW_CONFIDENCE_THRESHOLD,
                            "high": HIGH_CONFIDENCE_THRESHOLD,
                        },
                    },
                    model_used=False,
                    fallback_used=False,
                )

            # Hard fallback is allowed only when no rule match exists.
            fallback_confidence = min(float(raw_confidence), LOW_CONFIDENCE_THRESHOLD - 0.01)
            decision = resolve_intent_decision(
                raw_intent=INTENT_GENERAL_QUERY,
                raw_confidence=fallback_confidence,
                text=clean_text,
                session_context=session_context,
            )
            decision["primary_intent"] = INTENT_GENERAL_QUERY
            decision["normalized_intent"] = INTENT_GENERAL_QUERY
            decision["confidence"] = float(fallback_confidence)
            decision["low_confidence"] = True
            decision["fallback_used"] = True
            decision["fallback_reason"] = "low_confidence_no_rule_match"
            return _finalize_prediction(
                decision,
                intent=INTENT_GENERAL_QUERY,
                source="low_confidence_fallback",
                raw_intent=raw_intent,
                raw_confidence=raw_confidence,
                raw_model_output={
                    "intent": raw_intent,
                    "confidence": raw_confidence,
                    "top_candidates": top_candidates,
                    "confidence_tier": confidence_tier,
                    "thresholds": {
                        "low": LOW_CONFIDENCE_THRESHOLD,
                        "high": HIGH_CONFIDENCE_THRESHOLD,
                    },
                },
                model_used=False,
                fallback_used=True,
            )

        decision = resolve_intent_decision(
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            text=clean_text,
            session_context=session_context,
        )
        canonical_model_intent, _ = normalize_intent(raw_intent, default=INTENT_GENERAL_QUERY)

        # Medium-confidence correction is only allowed for uncertain general predictions.
        if (
            LOW_CONFIDENCE_THRESHOLD <= raw_confidence < 0.60
            and raw_confidence < 0.75
            and canonical_model_intent == INTENT_GENERAL_QUERY
        ):
            rule_intent, rule_confidence = fallback_intent(clean_text)
            if rule_intent != INTENT_GENERAL_QUERY and rule_confidence >= LOW_CONFIDENCE_THRESHOLD:
                decision = resolve_intent_decision(
                    raw_intent=rule_intent,
                    raw_confidence=max(raw_confidence, rule_confidence),
                    text=clean_text,
                    session_context=session_context,
                )
                if rule_intent == INTENT_PROVIDE_INFO:
                    decision["primary_intent"] = INTENT_PROVIDE_INFO
                    decision["normalized_intent"] = INTENT_PROVIDE_INFO
                decision["fallback_used"] = False
                decision["fallback_reason"] = ""
                decision["low_confidence"] = True
                decision["confidence"] = float(max(raw_confidence, min(rule_confidence, 0.69)))
                return _finalize_prediction(
                    decision,
                    intent=rule_intent,
                    source="hybrid_rule_correction_medium_conf",
                    raw_intent=raw_intent,
                    raw_confidence=raw_confidence,
                    raw_model_output={
                        "intent": raw_intent,
                        "confidence": raw_confidence,
                        "top_candidates": top_candidates,
                        "confidence_tier": confidence_tier,
                        "rule_intent": rule_intent,
                        "rule_confidence": rule_confidence,
                        "thresholds": {
                            "low": LOW_CONFIDENCE_THRESHOLD,
                            "high": HIGH_CONFIDENCE_THRESHOLD,
                        },
                    },
                    model_used=True,
                    fallback_used=False,
                )

        decision["primary_intent"] = canonical_model_intent
        decision["normalized_intent"] = canonical_model_intent
        decision["confidence"] = float(raw_confidence)
        decision["low_confidence"] = confidence_tier == "medium"
        decision["fallback_used"] = False
        decision["fallback_reason"] = ""
        decision["context_used"] = False

        model_source = "model" if confidence_tier == "high" else "model_medium_confidence"
        return _finalize_prediction(
            decision,
            intent=raw_intent,
            source=model_source,
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            raw_model_output={
                "intent": raw_intent,
                "confidence": raw_confidence,
                "top_candidates": top_candidates,
                "confidence_tier": confidence_tier,
                "thresholds": {
                    "low": LOW_CONFIDENCE_THRESHOLD,
                    "high": HIGH_CONFIDENCE_THRESHOLD,
                },
            },
            model_used=True,
            fallback_used=False,
        )
    except Exception:
        logger.exception("Intent inference failed, using fallback")
        raw_intent, raw_confidence = fallback_intent(clean_text)
        decision = resolve_intent_decision(
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            text=clean_text,
            session_context=session_context,
        )
        if raw_intent == INTENT_PROVIDE_INFO:
            decision["primary_intent"] = INTENT_PROVIDE_INFO
            decision["normalized_intent"] = INTENT_PROVIDE_INFO
            decision["fallback_used"] = True
            decision["fallback_reason"] = decision.get("fallback_reason") or "provide_info_detected"
        return _finalize_prediction(
            decision,
            intent=raw_intent,
            source="exception_fallback",
            raw_intent=raw_intent,
            raw_confidence=raw_confidence,
            raw_model_output={"intent": raw_intent, "confidence": raw_confidence},
            model_used=False,
            fallback_used=True,
        )


def predict_intent(text: str) -> Tuple[str, float]:
    # Backward-compatible API retained for existing callers.
    decision = predict_intent_detailed(text=text, session_context=None)
    return decision["primary_intent"], float(decision["confidence"])

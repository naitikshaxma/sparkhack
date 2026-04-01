import re
import unicodedata
import logging
from dataclasses import dataclass
from typing import List, Optional


logger = logging.getLogger(__name__)


PHRASE_MAP = {
    "पी एम": "pm",
    "पीएम": "pm",
    "p m": "pm",
}

WORD_MAP = {
    "पीएम": "pm",
    "प्रधानमंत्री": "pm",
    "pm": "pm",
    "किसान": "kisan",
    "किसानो": "kisan",
    "किसानों": "kisan",
    "kisaan": "kisan",
    "kissan": "kisan",
    "kisan": "kisan",
    "योजना": "yojana",
    "योजनाएं": "yojana",
    "योजनाओं": "yojana",
    "yojana": "yojana",
    "yojna": "yojana",
    "scheme": "scheme",
    "loan": "loan",
    "rin": "loan",
    "ऋण": "loan",
    "लोन": "loan",
    "madad": "help",
    "मदद": "help",
    "sahayata": "help",
    "सहायता": "help",
    "apply": "apply",
    "aavedan": "apply",
    "आवेदन": "apply",
    "status": "status",
    "स्थिति": "status",
    "eligibility": "eligibility",
    "पात्रता": "eligibility",
    "benefits": "benefits",
    "लाभ": "benefits",
    "documents": "documents",
    "दस्तावेज": "documents",
    "document": "documents",
    "आयुष्मान": "ayushman",
    "ayushman": "ayushman",
    "भारत": "bharat",
    "bharat": "bharat",
    "राशन": "ration",
    "rashan": "ration",
    "कार्ड": "card",
    "card": "card",
}

STOPWORDS = {
    "के",
    "का",
    "की",
    "को",
    "से",
    "में",
    "मे",
    "पर",
    "और",
    "या",
    "कि",
    "kya",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "me",
    "mein",
    "hai",
    "hain",
    "ho",
    "hoga",
    "please",
    "plz",
    "about",
    "batao",
    "bataye",
    "batayen",
    "batayein",
    "बताएं",
    "बताइए",
    "बताओ",
    "जानकारी",
    "जानना",
    "bhi",
    "भी",
}

HINDI_SIGNAL_TOKENS = {
    "kya",
    "kaise",
    "yojana",
    "madad",
    "aavedan",
    "haan",
    "nahi",
    "ji",
}


_NOISE_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u097f]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([.!?।])\1+")
_PUNCT_SPLIT_RE = re.compile(r"([.!?।])")
_PM_VARIANT_RE = re.compile(r"^p+m+$")
_KISAN_VARIANT_RE = re.compile(r"^k(?:i|ee)?s+h?a+a?n$")
_YOJANA_VARIANT_RE = re.compile(r"^yojna+a*$|^yojan+a+$|^yojana+a*$")


TOKEN_MAP = {
    "पीएम": "pm",
    "पी": "pm",
    "एम": "pm",
    "p": "p",
    "m": "m",
    "प्रधानमंत्री": "pradhanmantri",
    "kishan": "kisan",
    "kisaan": "kisan",
    "kissan": "kisan",
    "किसान": "kisan",
    "किसानो": "kisan",
    "किसानों": "kisan",
    "yojna": "yojana",
    "yojnaa": "yojana",
    "yojanaa": "yojana",
    "योजना": "yojana",
    "आवेदन": "apply",
    "पात्र": "eligible",
    "कैसे": "how",
    "क्या": "kya",
    "है": "hai",
    "मुझे": "mujhe",
    "करना": "karna",
}


def _normalize_token(token: str) -> str:
    current = str(token or "").strip()
    if not current:
        return ""
    if current in {".", "?", "!", "।"}:
        return current

    mapped = TOKEN_MAP.get(current, current)
    if _PM_VARIANT_RE.fullmatch(mapped):
        return "pm"
    if _KISAN_VARIANT_RE.fullmatch(mapped):
        return "kisan"
    if _YOJANA_VARIANT_RE.fullmatch(mapped):
        return "yojana"
    return mapped


def _normalize_token_sequence(tokens: List[str]) -> List[str]:
    normalized: List[str] = []
    idx = 0
    while idx < len(tokens):
        token = _normalize_token(tokens[idx])
        if not token:
            idx += 1
            continue

        # Token-level merge for split PM forms: "p m" and "पी एम".
        if idx + 1 < len(tokens):
            nxt = _normalize_token(tokens[idx + 1])
            if (token == "p" and nxt == "m") or (token == "pi" and nxt == "em"):
                normalized.append("pm")
                idx += 2
                continue

        # Keep punctuation deterministic by collapsing repeats.
        if token in {".", "?", "!", "।"} and normalized and normalized[-1] == token:
            idx += 1
            continue

        normalized.append(token)
        idx += 1
    return normalized


def _join_tokens(tokens: List[str]) -> str:
    if not tokens:
        return ""

    parts: List[str] = []
    for token in tokens:
        if token in {".", "?", "!", "।"}:
            if parts:
                parts[-1] = f"{parts[-1]}{token}"
            else:
                parts.append(token)
            continue
        parts.append(token)
    return " ".join(parts).strip()


@dataclass(frozen=True)
class NormalizedInput:
    raw_text: str
    normalized_text: str
    intent_text: str
    language: str
    tokens: List[str]


def _replace_phrases(text: str) -> str:
    output = text
    for phrase in sorted(PHRASE_MAP, key=len, reverse=True):
        output = output.replace(phrase, PHRASE_MAP[phrase])
    return output


def detect_text_language(text: str, language_hint: Optional[str] = None) -> str:
    hint = (language_hint or "").strip().lower()
    if hint in {"hi", "en"}:
        return hint

    content = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    if not content:
        return "en"

    if re.search(r"[\u0900-\u097F]", content):
        return "hi"

    tokens = set(content.split())
    if tokens.intersection(HINDI_SIGNAL_TOKENS):
        return "hi"
    return "en"


def _tokenize_core(text: str) -> List[str]:
    normalized = _replace_phrases(text)
    normalized = re.sub(r"[^\w\s\u0900-\u097f]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    tokens: List[str] = []
    for token in normalized.split():
        mapped = WORD_MAP.get(token, token)
        if mapped in STOPWORDS:
            continue
        if tokens and tokens[-1] == mapped:
            continue
        tokens.append(mapped)
    return tokens


def normalize_text(text: str) -> str:
    try:
        if text is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("normalize_text received None input")
            return ""

        raw_input = str(text)
        if not raw_input or not raw_input.strip():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("normalize_text received empty input")
            return ""

        if len(raw_input) > 500:
            raw_input = raw_input[:500]

        # 1) Lowercase
        lowered = unicodedata.normalize("NFKC", raw_input).lower()
        # 2) Trim
        lowered = lowered.strip()
        # 3) Collapse spaces
        lowered = _MULTI_SPACE_RE.sub(" ", lowered)
        # 4) Remove punctuation noise
        cleaned = _NOISE_PUNCT_RE.sub(" ", lowered)
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            return ""

        # 5) Tokenize and normalize each token independently
        tokens = cleaned.split(" ")
        normalized_tokens: List[str] = []
        for token in tokens:
            normalized = _normalize_token(token)
            if normalized:
                normalized_tokens.append(normalized)

        # 6) Rejoin deterministically
        output = " ".join(normalized_tokens).strip()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "normalize_text completed",
                extra={"input_len": len(raw_input), "token_count": len(tokens), "output_len": len(output)},
            )

        return output
    except Exception:
        try:
            fallback = str(text).lower().strip() if text is not None else ""
            fallback = _NOISE_PUNCT_RE.sub(" ", fallback)
            fallback = _MULTI_SPACE_RE.sub(" ", fallback).strip()
            return fallback[:500]
        except Exception:
            return ""


def normalize_for_intent(text: str, language_hint: Optional[str] = None) -> NormalizedInput:
    raw_text = unicodedata.normalize("NFKC", str(text or "")).strip()
    lowered = raw_text.lower()
    normalized_text = normalize_text(lowered)
    intent_text = normalized_text or re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", lowered)).strip()
    tokens = intent_text.split() if intent_text else []
    language = detect_text_language(raw_text, language_hint=language_hint)
    return NormalizedInput(
        raw_text=raw_text,
        normalized_text=normalized_text,
        intent_text=intent_text,
        language=language,
        tokens=tokens,
    )

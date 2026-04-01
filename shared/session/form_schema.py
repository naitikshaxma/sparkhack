import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from backend.infrastructure.ml.scheme_registry import load_dataset
from backend.shared.validators.validator import validate


DEFAULT_SCHEME_NAME = "Loan Assistance"
GENERIC_FALLBACK_FIELDS = ["full_name", "phone", "aadhaar_number"]

FIELD_QUESTIONS = {
    "full_name": {
        "en": "Please share your full name.",
        "hi": "कृपया अपना पूरा नाम बताएं।",
    },
    "phone": {
        "en": "Please share your 10-digit mobile number.",
        "hi": "कृपया अपना 10 अंकों का मोबाइल नंबर बताएं।",
    },
    "aadhaar_number": {
        "en": "Please share your 12-digit Aadhaar number.",
        "hi": "कृपया अपना 12 अंकों का आधार नंबर बताएं।",
    },
    "annual_income": {
        "en": "Please share your annual income.",
        "hi": "कृपया अपनी वार्षिक आय बताएं।",
    },
    "land_holding_acres": {
        "en": "Please share your total land holding in acres.",
        "hi": "कृपया अपनी कुल भूमि (एकड़ में) बताएं।",
    },
    "farmer_id": {
        "en": "Please share your farmer ID or registration number.",
        "hi": "कृपया अपना किसान आईडी या पंजीकरण नंबर बताएं।",
    },
    "health_card_number": {
        "en": "Please share your health card number if available.",
        "hi": "यदि उपलब्ध हो, तो कृपया अपना हेल्थ कार्ड नंबर बताएं।",
    },
    "family_size": {
        "en": "Please share your family size.",
        "hi": "कृपया परिवार के सदस्यों की संख्या बताएं।",
    },
    "residential_status": {
        "en": "Please share your residential status (urban or rural).",
        "hi": "कृपया अपना आवासीय क्षेत्र बताएं (शहरी या ग्रामीण)।",
    },
    "property_ownership": {
        "en": "Please share whether you own a house or land.",
        "hi": "कृपया बताएं कि आपके पास घर या जमीन है या नहीं।",
    },
}

SCHEME_FORM_CONFIG = {
    "pm kisan": {
        "fields": ["full_name", "aadhaar_number", "phone", "land_holding_acres", "annual_income", "farmer_id"],
        "category": "financial",
        "target_user": "farmer",
        "benefits_type": "financial",
    },
    "ayushman bharat": {
        "fields": ["full_name", "aadhaar_number", "phone", "family_size", "annual_income", "health_card_number"],
        "category": "health",
        "target_user": "family",
        "benefits_type": "health",
    },
    "pmay": {
        "fields": ["full_name", "aadhaar_number", "phone", "annual_income", "residential_status", "property_ownership"],
        "category": "housing",
        "target_user": "low_income",
        "benefits_type": "housing",
    },
    "loan assistance": {
        "fields": ["full_name", "phone", "aadhaar_number", "annual_income"],
        "category": "financial",
        "target_user": "general",
        "benefits_type": "financial",
    },
}

SCHEME_ALIASES = {
    "pm kisan": "pm kisan",
    "pm किसान": "pm kisan",
    "प्रधानमंत्री किसान": "pm kisan",
    "ayushman": "ayushman bharat",
    "ayushman bharat": "ayushman bharat",
    "pmay": "pmay",
    "pm आवास": "pmay",
    "housing": "pmay",
    "loan": "loan assistance",
    "loans": "loan assistance",
}


def _normalize(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


@lru_cache(maxsize=1)
def _load_dataset() -> List[Dict[str, Any]]:
    rows = load_dataset()
    return [row for row in rows if isinstance(row, dict)]


def _dataset_record_for_scheme(scheme_name: str) -> Optional[Dict[str, Any]]:
    target = _normalize(scheme_name)
    if not target:
        return None

    candidates = _load_dataset()
    if not candidates:
        return None

    for row in candidates:
        row_name = _normalize(row.get("name"))
        row_id = _normalize(str(row.get("id") or "").replace("-", " "))
        if target == row_name or target == row_id:
            return row

    for row in candidates:
        row_name = _normalize(row.get("name"))
        if row_name and (target in row_name or row_name in target):
            return row

    return None


def _infer_optional_fields_from_record(record: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(record, dict):
        return []

    text_sources: List[str] = []
    keywords = record.get("keywords")
    if isinstance(keywords, list):
        text_sources.extend(str(item) for item in keywords)

    for key in (
        "required_documents",
        "required_documents_en",
        "required_documents_hi",
        "eligibility",
        "eligibility_en",
        "eligibility_hi",
        "details_en",
        "details_hi",
        "summary_en",
        "summary_hi",
    ):
        value = record.get(key)
        if isinstance(value, list):
            text_sources.extend(str(item) for item in value)
        elif isinstance(value, str):
            text_sources.append(value)

    corpus = _normalize(" ".join(text_sources))
    if not corpus:
        return []

    inferred: List[str] = []

    if any(token in corpus for token in {"income", "annual income", "आय", "salary"}):
        inferred.append("annual_income")

    if any(token in corpus for token in {"farmer", "kisan", "land", "acre", "krishi", "crop"}):
        inferred.extend(["land_holding_acres", "farmer_id"])

    if any(token in corpus for token in {"family", "household", "परिवार"}):
        inferred.append("family_size")

    if any(token in corpus for token in {"health", "medical", "ayushman", "hospital", "card"}):
        inferred.append("health_card_number")

    if any(token in corpus for token in {"housing", "house", "home", "rural", "urban", "property", "rent"}):
        inferred.extend(["residential_status", "property_ownership"])

    unique: List[str] = []
    for field in inferred:
        if field not in unique:
            unique.append(field)
    return unique


def resolve_scheme_name(scheme_name: Optional[str]) -> str:
    raw = _normalize(scheme_name)
    if not raw:
        return DEFAULT_SCHEME_NAME
    if raw in SCHEME_FORM_CONFIG:
        return raw
    for alias, canonical in SCHEME_ALIASES.items():
        if alias in raw:
            return canonical
    return raw


def get_default_scheme_for_category(category: Optional[str]) -> str:
    value = (category or "").strip().lower()
    if value == "health":
        return "ayushman bharat"
    if value == "housing":
        return "pmay"
    if value == "financial":
        return "pm kisan"
    return "loan assistance"


def _build_form_plan(scheme_name: Optional[str]) -> Dict[str, Any]:
    canonical = resolve_scheme_name(scheme_name)
    if canonical in SCHEME_FORM_CONFIG:
        configured_fields = list(SCHEME_FORM_CONFIG[canonical].get("fields", GENERIC_FALLBACK_FIELDS))
        return {
            "form_type": "schema",
            "fields": configured_fields or list(GENERIC_FALLBACK_FIELDS),
            "inferred_fields": [],
        }

    fields: List[str] = list(GENERIC_FALLBACK_FIELDS)

    return {
        "form_type": "generic",
        "fields": fields,
        "inferred_fields": [],
    }


def get_form_type_for_scheme(scheme_name: Optional[str]) -> str:
    return str(_build_form_plan(scheme_name).get("form_type") or "generic")


def get_fields_for_scheme(scheme_name: Optional[str]) -> List[str]:
    plan = _build_form_plan(scheme_name)
    return list(plan.get("fields") or GENERIC_FALLBACK_FIELDS)


def ensure_dynamic_field_completion(session: Dict[str, Any]) -> Dict[str, bool]:
    scheme_name = session.get("selected_scheme") or DEFAULT_SCHEME_NAME
    fields = get_fields_for_scheme(scheme_name)
    completion = dict(session.get("field_completion", {}))
    for field in fields:
        completion.setdefault(field, False)

    # Keep only active dynamic fields.
    filtered = {field: bool(completion.get(field, False)) for field in fields}
    session["field_completion"] = filtered
    return filtered


def get_next_field(session: Dict[str, Any]) -> Optional[str]:
    fields = get_fields_for_scheme(session.get("selected_scheme") or DEFAULT_SCHEME_NAME)
    field_completion = ensure_dynamic_field_completion(session)
    for field in fields:
        if not field_completion.get(field, False):
            return field
    return None


def get_previous_field(current_field: Optional[str], session: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not current_field:
        return None
    fields = get_fields_for_scheme((session or {}).get("selected_scheme") or DEFAULT_SCHEME_NAME)
    try:
        index = fields.index(current_field)
    except ValueError:
        return None
    if index <= 0:
        return None
    return fields[index - 1]


def get_field_question(field: Optional[str], language: str = "en", scheme_name: Optional[str] = None) -> str:
    if not field:
        return (
            "Thank you. Your scheme application form is complete."
            if language == "en"
            else "धन्यवाद। आपका योजना आवेदन फॉर्म पूरा हो गया है।"
        )
    entry = FIELD_QUESTIONS.get(field)
    if not entry:
        return "Please provide the next required detail." if language == "en" else "कृपया अगली आवश्यक जानकारी बताएं।"
    if language == "en":
        return str(entry.get("en") or "Please provide the next required detail.")
    return str(entry.get("hi") or "कृपया अगली आवश्यक जानकारी बताएं।")


# Backward-compatible alias for older call sites.
LOAN_FIELDS = get_fields_for_scheme("loan assistance")


def validate_field(field: str, value: str, language: str = "en") -> Dict[str, Any]:
    """Central validation engine for dynamic form fields.

    Returns API-safe payload with bilingual, user-facing error text.
    """
    is_valid, normalized, error_message = validate(field, value)
    lang = "hi" if (language or "").strip().lower() == "hi" else "en"

    if is_valid:
        return {
            "valid": True,
            "normalized": normalized,
            "error_code": None,
            "error_message": None,
        }

    code = "invalid_input"
    if field == "phone":
        code = "invalid_phone"
    elif field == "aadhaar_number":
        code = "invalid_aadhaar"
    elif field == "annual_income":
        code = "invalid_income"

    localized = {
        "invalid_phone": {
            "en": "Please enter a valid 10-digit mobile number.",
            "hi": "कृपया सही 10 अंकों का मोबाइल नंबर दर्ज करें।",
        },
        "invalid_aadhaar": {
            "en": "Please enter a valid 12-digit Aadhaar number.",
            "hi": "कृपया सही 12 अंकों का आधार नंबर दर्ज करें।",
        },
        "invalid_income": {
            "en": "Please enter annual income as numbers only.",
            "hi": "कृपया वार्षिक आय केवल संख्याओं में दर्ज करें।",
        },
        "invalid_input": {
            "en": "The value looks invalid. Please try again.",
            "hi": "दर्ज की गई जानकारी सही नहीं लग रही। कृपया दोबारा बताएं।",
        },
    }

    message = localized.get(code, localized["invalid_input"]).get(lang) or error_message or localized["invalid_input"]["en"]
    return {
        "valid": False,
        "normalized": None,
        "error_code": code,
        "error_message": message,
    }

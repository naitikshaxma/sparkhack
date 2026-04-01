import re
from typing import Optional, Tuple

from backend.shared.security.privacy import mask_aadhaar


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def validate(field: str, value: str) -> Tuple[bool, Optional[str], Optional[str]]:
    cleaned_value = (value or "").strip()
    if not cleaned_value:
        return False, None, "Value is empty."

    if field == "phone":
        digits = _digits_only(cleaned_value)
        if len(digits) != 10:
            return False, None, "Phone number must be exactly 10 digits."
        return True, digits, None

    if field == "aadhaar_number":
        digits = _digits_only(cleaned_value)
        if len(digits) != 12:
            return False, None, "Aadhaar number must be exactly 12 digits."
        return True, mask_aadhaar(digits), None

    if field == "annual_income":
        candidate = cleaned_value.replace(",", "")
        if not re.fullmatch(r"\d+(\.\d+)?", candidate):
            return False, None, "Annual income must be numeric."
        return True, candidate, None

    return True, cleaned_value, None

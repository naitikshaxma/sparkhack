import json
import time
from typing import Any, Dict
import asyncio
import re

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

from backend.core.logger import log_event

SYSTEM_PROMPT = (
    "You are an OCR data extractor. Extract structured Aadhaar information from raw OCR text. "
    "Return ONLY valid JSON. If unsure, return null."
)


def _empty_extraction() -> Dict[str, Any]:
    return {
        "full_name": None,
        "aadhaar_number": None,
        "date_of_birth": None,
        "address": None,
        "confidence": 0.0,
    }


def extract_text(image_path: str, timings: dict | None = None) -> str:
    start = time.perf_counter()
    log_event("ocr_extract_text_start", endpoint="ocr_service", status="success")
    # Requires system-level Tesseract installation and accessible PATH.
    try:
        if Image is None or pytesseract is None:
            raise RuntimeError("OCR dependencies are not installed")
        image_module: Any = Image
        ocr_module: Any = pytesseract
        with image_module.open(image_path) as img:
            text = ocr_module.image_to_string(img)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if timings is not None:
            timings["ocr_text_extraction_ms"] = elapsed_ms
        log_event("ocr_extract_text_success", endpoint="ocr_service", status="success", response_time_ms=elapsed_ms)
        return text
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if timings is not None:
            timings["ocr_text_extraction_ms"] = elapsed_ms
        log_event(
            "ocr_extract_text_failure",
            level="error",
            endpoint="ocr_service",
            status="failure",
            error_type=type(exc).__name__,
            response_time_ms=elapsed_ms,
        )
        raise


def extract_structured_data(ocr_text: str, timings: dict | None = None) -> Dict[str, Any]:
    start = time.perf_counter()
    log_event("ocr_structured_data_start", endpoint="ocr_service", status="success", user_input_length=len(ocr_text or ""))
    if not (ocr_text or "").strip():
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if timings is not None:
            timings["ocr_structuring_ms"] = elapsed_ms
        return _empty_extraction()

    try:
        text = str(ocr_text or "")
        aadhaar_match = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
        dob_match = re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        name = lines[0] if lines else None
        address = " ".join(lines[1:4]) if len(lines) > 1 else None

        result = {
            "full_name": name,
            "aadhaar_number": aadhaar_match.group(0).replace(" ", "") if aadhaar_match else None,
            "date_of_birth": dob_match.group(0) if dob_match else None,
            "address": address,
            "confidence": 0.35,
            "mode": "local",
        }
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if timings is not None:
            timings["ocr_structuring_ms"] = elapsed_ms
        log_event("ocr_structured_data_local", endpoint="ocr_service", status="success", response_time_ms=elapsed_ms)
        return result
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if timings is not None:
            timings["ocr_structuring_ms"] = elapsed_ms
        log_event("ocr_structured_data_failure", level="error", endpoint="ocr_service", status="failure", error_type="OcrStructuringError", response_time_ms=elapsed_ms)
        return _empty_extraction()


class OcrService:
    def extract_text(self, image_path: str, timings: dict | None = None) -> str:
        return extract_text(image_path, timings=timings)

    def extract_structured_data(self, ocr_text: str, timings: dict | None = None) -> Dict[str, Any]:
        return extract_structured_data(ocr_text, timings=timings)

    async def extract_text_async(self, image_path: str, timings: dict | None = None) -> str:
        return await asyncio.to_thread(extract_text, image_path, timings)

    async def extract_structured_data_async(self, ocr_text: str, timings: dict | None = None) -> Dict[str, Any]:
        return await asyncio.to_thread(extract_structured_data, ocr_text, timings)

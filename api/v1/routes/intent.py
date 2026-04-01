from fastapi import APIRouter, Depends, HTTPException, Query, Request

from backend.container import inject_container
from backend.schemas import IntentRequest
from backend.shared.language.language import detect_input_language
from backend.routes.response_utils import standardized_success
from backend.services.ml_intent_wrapper import process_user_query


router = APIRouter(tags=["intent"])


@router.post("/intent")
async def detect_intent(
    payload: IntentRequest,
    request: Request,
    debug: bool = Query(False),
    container=Depends(inject_container),
):
    client_ip = request.client.host if request.client else "unknown"
    validation = container.input_validator.validate_input(payload.text, client_ip=client_ip, endpoint=request.url.path)
    if not validation.is_valid:
        raise HTTPException(status_code=400, detail=validation.rejected_reason or "Invalid input.")

    result = process_user_query(validation.normalized_text)
    if not isinstance(result, dict):
        result = {
            "success": False,
            "type": "fallback",
            "message": "Something went wrong. Please try again.",
            "data": {},
        }

    if "confidence" not in result:
        result["confidence"] = 0.0

    request.state.intent = result.get("intent")
    request.state.confidence = result.get("confidence")
    request.state.user_input_length = len(validation.normalized_text)
    if debug:
        result["debug"] = {
            **(result.get("debug") or {}),
            "normalized_input": validation.normalized_text,
            "detected_language": detect_input_language(validation.normalized_text),
        }
    return standardized_success(result)

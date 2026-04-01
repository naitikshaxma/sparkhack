from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func

from backend.container import inject_container
from backend.infrastructure.database.connection import db_session_scope
from backend.infrastructure.database import ConversationHistory
from backend.routes.response_utils import standardized_success


router = APIRouter(tags=["system"])


def _extract_category_from_scheme_name(scheme_name: str) -> str:
    text = str(scheme_name or "").strip().lower()
    if not text:
        return "general"

    category_keywords = {
        "agriculture": {"kisan", "krishi", "farmer", "crop", "agri"},
        "housing": {"housing", "house", "home", "awas", "rental"},
        "health": {"health", "bima", "insurance", "medical", "ayush", "care"},
        "education": {"student", "scholarship", "education", "vidya", "school"},
        "employment": {"employment", "job", "skill", "startup", "business", "self"},
        "finance": {"loan", "credit", "finance", "pension", "savings"},
    }

    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "general"


@router.get("/health")
def health(container=Depends(inject_container)):
    return standardized_success(container.system_service.health())


@router.get("/metrics")
def metrics(container=Depends(inject_container)):
    return standardized_success(container.system_service.metrics())


@router.get("/status")
def status(container=Depends(inject_container)):
    return standardized_success(container.system_service.status())


@router.get("/history")
def get_history(request: Request, limit: int = Query(20, ge=1, le=100)):
    raw_user_id = str(getattr(request.state, "user_id", "") or "").strip()
    user_id = int(raw_user_id) if raw_user_id.isdigit() else None
    if user_id is None:
        return standardized_success([])

    with db_session_scope() as db:
        rows = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.user_id == user_id)
            .order_by(ConversationHistory.timestamp.desc())
            .limit(limit)
            .all()
        )

        payload = [
            {
                "query": str(row.query or ""),
                "response": str(row.response or row.message or ""),
                "scheme": str(row.detected_scheme or ""),
                "intent": str(row.intent or ""),
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            }
            for row in rows
        ]
    return standardized_success(payload)


@router.get("/history/summary")
def get_history_summary(request: Request):
    raw_user_id = str(getattr(request.state, "user_id", "") or "").strip()
    user_id = int(raw_user_id) if raw_user_id.isdigit() else None
    if user_id is None:
        return standardized_success(
            {
                "most_used_schemes": [],
                "most_used_categories": [],
                "smart_suggestions": [],
                "last_scheme": "",
                "total_interactions": 0,
            }
        )

    with db_session_scope() as db:
        total_interactions = int(
            db.query(func.count(ConversationHistory.id))
            .filter(ConversationHistory.user_id == user_id)
            .scalar()
            or 0
        )

        last_scheme = (
            db.query(ConversationHistory.detected_scheme)
            .filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.detected_scheme.isnot(None),
                ConversationHistory.detected_scheme != "",
            )
            .order_by(ConversationHistory.timestamp.desc())
            .limit(1)
            .scalar()
        )

        scheme_counts_raw = (
            db.query(ConversationHistory.detected_scheme, func.count(ConversationHistory.id))
            .filter(
                ConversationHistory.user_id == user_id,
                ConversationHistory.detected_scheme.isnot(None),
                ConversationHistory.detected_scheme != "",
            )
            .group_by(ConversationHistory.detected_scheme)
            .order_by(func.count(ConversationHistory.id).desc())
            .limit(10)
            .all()
        )

        most_used_schemes = [
            {
                "scheme": str(name or ""),
                "count": int(count or 0),
            }
            for name, count in scheme_counts_raw
        ]

        category_counts = {}
        for scheme_name, count in scheme_counts_raw:
            category = _extract_category_from_scheme_name(str(scheme_name or ""))
            category_counts[category] = int(category_counts.get(category, 0)) + int(count or 0)

        most_used_categories = [
            {
                "category": category,
                "count": count,
            }
            for category, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)
        ]

        top_category = most_used_categories[0]["category"] if most_used_categories else ""
        smart_suggestions = [
            str(name or "")
            for name, _ in scheme_counts_raw
            if _extract_category_from_scheme_name(str(name or "")) == top_category
        ][:3]

    return standardized_success(
        {
            "most_used_schemes": most_used_schemes,
            "most_used_categories": most_used_categories,
            "smart_suggestions": smart_suggestions,
            "last_scheme": str(last_scheme or ""),
            "total_interactions": total_interactions,
        }
    )

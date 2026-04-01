from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, TypedDict

from backend.text_normalizer import normalize_text


class DecisionResult(TypedDict):
    category: str
    confidence: float
    reasoning: str


SUPPORTED_CATEGORIES = (
    "financial",
    "health",
    "housing",
    "education",
    "employment",
    "agriculture",
    "social_welfare",
)


_NEED_KEYWORDS: Dict[str, set[str]] = {
    "financial": {
        "loan",
        "paisa",
        "money",
        "income",
        "cash",
        "finance",
        "credit",
        "rin",
        "लोन",
        "पैसा",
        "आय",
    },
    "health": {
        "bimari",
        "hospital",
        "health",
        "treatment",
        "medical",
        "doctor",
        "बीमारी",
        "अस्पताल",
        "इलाज",
        "स्वास्थ्य",
    },
    "housing": {
        "ghar",
        "house",
        "home",
        "housing",
        "rent",
        "property",
        "घर",
        "मकान",
        "आवास",
    },
    "education": {
        "education",
        "school",
        "college",
        "student",
        "scholarship",
        "tuition",
        "course",
        "education loan",
        "शिक्षा",
        "छात्र",
        "स्कॉलरशिप",
        "छात्रवृत्ति",
    },
    "employment": {
        "job",
        "employment",
        "rojgar",
        "skill",
        "training",
        "self employment",
        "startup",
        "work",
        "रोजगार",
        "नौकरी",
        "कौशल",
    },
    "agriculture": {
        "farmer",
        "kisan",
        "crop",
        "seed",
        "irrigation",
        "fertilizer",
        "krishi",
        "agriculture",
        "किसान",
        "फसल",
        "खेती",
        "कृषि",
    },
    "social_welfare": {
        "pension",
        "ration",
        "widow",
        "disability",
        "senior citizen",
        "bpl",
        "welfare",
        "old age",
        "social security",
        "पेंशन",
        "राशन",
        "वृद्ध",
        "सामाजिक",
        "कल्याण",
    },
}

_CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    "financial": [
        "I need money support and a government loan scheme",
        "Need income assistance and financial help",
        "Mujhe paisa aur loan madad chahiye",
    ],
    "health": [
        "I need health coverage for hospital treatment",
        "Medical support and disease treatment scheme",
        "Mujhe bimari aur hospital ke liye madad chahiye",
    ],
    "housing": [
        "I need home and house support",
        "Housing subsidy for house construction",
        "Mujhe ghar ke liye awas yojana chahiye",
    ],
    "education": [
        "I need scholarship and education support",
        "Student fee support and study loan scheme",
        "Mujhe padhai aur scholarship mein madad chahiye",
    ],
    "employment": [
        "I need job and employment support",
        "Skill training and rojgar assistance",
        "Mujhe rojgar aur naukri ke liye madad chahiye",
    ],
    "agriculture": [
        "I need agriculture support for crop and irrigation",
        "Farmer assistance for krishi and input costs",
        "Mujhe kheti aur fasal ke liye madad chahiye",
    ],
    "social_welfare": [
        "I need pension and social welfare support",
        "Ration and old age social security scheme",
        "Mujhe pension aur samajik kalyan madad chahiye",
    ],
}


def _empty_scores() -> Dict[str, float]:
    return {category: 0.0 for category in SUPPORTED_CATEGORIES}


def _keyword_scores(query: str) -> Dict[str, float]:
    scores = _empty_scores()
    for category, keywords in _NEED_KEYWORDS.items():
        score = 0.0
        for keyword in keywords:
            if keyword in query:
                score += 2.0 if " " in keyword else 1.0
        scores[category] = score
    return scores


def _context_scores(session_context: Optional[dict]) -> Dict[str, float]:
    scores = _empty_scores()
    context = session_context or {}
    profile = context.get("user_need_profile") or {}
    history = context.get("conversation_history") or []

    preferred = normalize_text(str(profile.get("need_category") or ""))
    if preferred in scores:
        scores[preferred] += 1.25

    user_type = normalize_text(str(profile.get("user_type") or ""))
    if "farmer" in user_type:
        scores["agriculture"] += 0.6
        scores["financial"] += 0.25
    if "student" in user_type:
        scores["education"] += 0.6
    if "business" in user_type:
        scores["employment"] += 0.55
        scores["financial"] += 0.2

    # Use recent user turns as weak context prior.
    recent_user_text = " ".join(
        str(item.get("content", ""))
        for item in history[-4:]
        if isinstance(item, dict) and item.get("role") == "user"
    )
    if recent_user_text:
        contextual_keywords = _keyword_scores(normalize_text(recent_user_text))
        for category in scores:
            scores[category] += 0.15 * contextual_keywords[category]

    return scores


@lru_cache(maxsize=1)
def _get_sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


@lru_cache(maxsize=1)
def _prototype_embeddings() -> Optional[Dict[str, list]]:
    model = _get_sentence_transformer_model()
    if model is None:
        return None

    embeddings: Dict[str, list] = {}
    for category, texts in _CATEGORY_PROTOTYPES.items():
        embeddings[category] = model.encode(texts, normalize_embeddings=True)
    return embeddings


def _embedding_scores(query: str) -> Dict[str, float]:
    scores = _empty_scores()
    model = _get_sentence_transformer_model()
    prototypes = _prototype_embeddings()
    if model is None or prototypes is None:
        return scores

    from sentence_transformers import util  # type: ignore

    query_embedding = model.encode([query], normalize_embeddings=True)
    for category, emb in prototypes.items():
        similarity = util.cos_sim(query_embedding, emb).max().item()
        scores[category] = max(0.0, float(similarity))
    return scores


def detect_user_need(user_input: str, session_context: Optional[dict] = None) -> DecisionResult:
    normalized = normalize_text(user_input)
    query = (normalized or user_input or "").strip().lower()
    if not query:
        return {
            "category": "social_welfare",
            "confidence": 0.0,
            "reasoning": "No input text available; defaulting to social_welfare for safe assistance.",
        }

    keyword_scores = _keyword_scores(query)
    embed_scores = _embedding_scores(query)
    context_scores = _context_scores(session_context)

    merged = _empty_scores()
    for category in merged:
        merged[category] = (
            0.50 * keyword_scores[category]
            + 0.35 * embed_scores[category]
            + 0.15 * context_scores[category]
        )

    best_category = max(merged, key=merged.get)
    best_score = float(merged[best_category])
    total = sum(merged.values())

    if best_score <= 0:
        return {
            "category": "social_welfare",
            "confidence": 0.35,
            "reasoning": "No strong category signal detected; using default social welfare support path.",
        }

    confidence = min(1.0, max(0.35, best_score / max(1e-6, total)))
    reasoning = (
        f"Hybrid decision selected {best_category}: "
        f"keyword={keyword_scores[best_category]:.2f}, "
        f"embedding={embed_scores[best_category]:.2f}, "
        f"context={context_scores[best_category]:.2f}."
    )
    return {
        "category": best_category,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
    }

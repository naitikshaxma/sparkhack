from typing import Any, Dict, List


def recommendation_suffix(language: str, recommendations: List[str]) -> str:
    if not recommendations:
        return ""
    if language == "hi":
        return f"\n\nआप इन योजनाओं के बारे में पूछ सकते हैं: {', '.join(recommendations)}"
    return f"\n\nYou can ask about these schemes: {', '.join(recommendations)}"


def guided_followup_question(user_input: str, language: str) -> str:
    query = (user_input or "").strip().lower()
    if any(token in query for token in {"loan", "credit", "finance", "financial", "लोन", "ऋण"}):
        if language == "hi":
            return "आपको किस तरह का लोन चाहिए: किसान, छात्र, या छोटे बिज़नेस के लिए?"
        return "Which type of loan do you need: farmer, student, or small business?"
    if any(token in query for token in {"health", "hospital", "medical", "स्वास्थ्य", "इलाज"}):
        if language == "hi":
            return "क्या मदद अस्पताल खर्च, बीमा, या परिवार कवरेज के लिए चाहिए?"
        return "Do you need help with hospital costs, insurance, or family coverage?"
    if any(token in query for token in {"house", "home", "housing", "घर", "आवास"}):
        if language == "hi":
            return "आपका फोकस घर खरीदना है, घर बनाना है, या किराए से राहत चाहिए?"
        return "Is your focus buying a house, building one, or rental support?"
    return (
        "मैं सही योजना चुनने के लिए एक बात जानना चाहूँगा: आपको तुरंत किस चीज़ में मदद चाहिए?"
        if language == "hi"
        else "To pick the best scheme, what is your top priority right now?"
    )


def smart_clarification_message(language: str, recommendations: List[str], user_input: str = "") -> str:
    base = guided_followup_question(user_input, language)
    return f"{base}{recommendation_suffix(language, recommendations)}"


def adaptive_recommendation_limit(confidence: float, low_threshold: float, high_threshold: float) -> int:
    if confidence > high_threshold:
        return 1
    if confidence < low_threshold:
        return 3
    return 2


def apply_recommendation_continuity(session: Dict[str, Any], recommendations: List[str]) -> List[str]:
    previous = [str(item) for item in session.get("last_recommendations", []) if str(item).strip()]
    filtered = [item for item in recommendations if item not in previous]
    final_list = filtered or recommendations[:1]
    session["last_recommendations"] = final_list[:3]
    return final_list

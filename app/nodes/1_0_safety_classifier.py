"""Safety classification node."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.safety_classifier import SAFETY_CLASSIFIER_SYSTEM_PROMPT
from app.utils.messages import format_conversation


SafetyCategory = Literal[
    "emergency",
    "self_harm",
    "diagnosis_request",
    "prescription_request",
    "dosage_request",
    "drug_interaction",
    "medicine_info",
    "symptom_info",
    "general_health_info",
    "latest_health_info",
    "unknown",
]


class SafetyClassification(BaseModel):
    """Structured result returned by the classifier model."""

    safety_category: SafetyCategory
    is_safe_to_answer: bool
    safety_reason: str = Field(
        min_length=1,
        description="A brief reason for the classification.",
    )


UNSAFE_CATEGORIES = {
    "emergency",
    "self_harm",
    "diagnosis_request",
    "prescription_request",
    "dosage_request",
}


def safety_classifier_node(state: GraphState) -> dict[str, object]:
    """Classify the current query before retrieval or web search."""

    user_query = state.get("user_query", "").strip()
    messages = state.get("messages", [])

    if not user_query:
        return {
            "safety_category": "unknown",
            "is_safe_to_answer": False,
            "safety_reason": "No user query was provided.",
        }

    try:
        classifier = load_llm().with_structured_output(SafetyClassification)
        result = classifier.invoke(
            [
                SystemMessage(content=SAFETY_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Conversation context:\n"
                        f"{format_conversation(messages)}\n\n"
                        f"Current user query:\n{user_query}"
                    )
                ),
            ]
        )
        classification = SafetyClassification.model_validate(result)
    except Exception:
        return {
            "safety_category": "unknown",
            "is_safe_to_answer": False,
            "safety_reason": "Safety classification was unavailable.",
        }

    category = classification.safety_category

    return {
        "safety_category": category,
        "is_safe_to_answer": category not in UNSAFE_CATEGORIES,
        "safety_reason": classification.safety_reason,
    }

"""Intent planner node for selecting the required knowledge source."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.intent_planner import INTENT_PLANNER_SYSTEM_PROMPT
from app.utils.messages import format_conversation


Route = Literal["rag_only", "web_only", "rag_and_web", "direct_answer"]


class IntentPlan(BaseModel):
    """Structured routing decision returned by the planner model."""

    route: Route
    planner_reason: str = Field(
        min_length=1,
        description="A brief explanation of why this route is appropriate.",
    )


def _fallback_route(safety_category: str) -> Route:
    """Choose a conservative route if structured planning is unavailable."""

    if safety_category in {"medicine_info", "drug_interaction"}:
        return "web_only"

    if safety_category in {
        "symptom_info",
        "general_health_info",
        "latest_health_info",
    }:
        return "web_only"

    return "direct_answer"


def intent_planner_node(state: GraphState) -> dict[str, str]:
    """Decide whether the query needs RAG, web search, both, or neither."""

    user_query = state.get("user_query", "").strip()
    messages = state.get("messages", [])

    if not user_query:
        return {
            "route": "direct_answer",
            "planner_reason": "No user query was provided.",
        }

    try:
        planner = load_llm().with_structured_output(IntentPlan)
        result = planner.invoke(
            [
                SystemMessage(content=INTENT_PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Safety category: "
                        f"{state.get('safety_category', 'unknown')}\n\n"
                        f"Conversation context:\n"
                        f"{format_conversation(messages)}\n\n"
                        f"Current user query:\n{user_query}"
                    )
                ),
            ]
        )
        plan = IntentPlan.model_validate(result)
    except Exception:
        fallback = _fallback_route(state.get("safety_category", "unknown"))
        return {
            "route": fallback,
            "planner_reason": (
                "Intent planning was unavailable; a conservative fallback "
                "route was selected."
            ),
        }

    return {
        "route": plan.route,
        "planner_reason": plan.planner_reason,
    }

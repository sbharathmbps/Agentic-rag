"""FastAPI routes."""

from typing import Any

from fastapi import APIRouter

from app.api.schemas import ChatRequest, ChatResponse
from app.graph.runner import invoke_graph


router = APIRouter()


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert graph state into JSON-friendly values."""

    serialized = dict(state)
    messages = serialized.get("messages")

    if messages is not None:
        serialized["messages"] = [
            {
                "type": getattr(message, "type", "message"),
                "content": getattr(message, "content", str(message)),
            }
            for message in messages
        ]

    return serialized


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Invoke the graph for one user query."""

    final_state = invoke_graph(request.query)

    return ChatResponse(
        safety_category=final_state.get("safety_category"),
        is_safe_to_answer=final_state.get("is_safe_to_answer"),
        safety_reason=final_state.get("safety_reason"),
        route=final_state.get("route"),
        planner_reason=final_state.get("planner_reason"),
        final_answer=final_state.get("final_answer"),
        state=_serialize_state(final_state),
    )

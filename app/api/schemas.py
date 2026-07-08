"""API request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user message."""

    query: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Current graph response shape."""

    safety_category: str | None = None
    is_safe_to_answer: bool | None = None
    safety_reason: str | None = None
    route: str | None = None
    planner_reason: str | None = None
    final_answer: str | None = None
    state: dict[str, Any]

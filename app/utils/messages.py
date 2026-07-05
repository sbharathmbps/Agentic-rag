"""Utilities for working with conversation messages."""

from collections.abc import Sequence
from langchain_core.messages import BaseMessage


def format_conversation(messages: Sequence[BaseMessage], max_messages: int = 10,) -> str:
    """Render the most recent messages as compact role/content lines."""

    if max_messages < 1:
        raise ValueError("max_messages must be at least 1.")

    recent_messages = messages[-max_messages:]
    lines = []

    for message in recent_messages:
        role = getattr(message, "type", "message")
        lines.append(f"{role}: {message.content}")

    return "\n".join(lines) or "No previous conversation."

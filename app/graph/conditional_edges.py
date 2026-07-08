"""Conditional edge functions for the LangGraph workflow."""

from app.graph.state import GraphState


def route_after_safety_classifier(state: GraphState) -> str:
    """Route unsafe queries to escalation; safe queries to intent planning."""

    if state.get("is_safe_to_answer") is False:
        return "escalation"

    return "intent_planner"


def route_after_query_rewriter(state: GraphState) -> str:
    """Route rewritten queries to the required retriever/search node."""

    route = state.get("route", "web_only")

    if route in {"rag_only", "rag_and_web"}:
        return "rag_retriever"

    if route == "web_only":
        return "web_search"

    return "web_search"


def route_after_rag_retriever(state: GraphState) -> str:
    """Route after local RAG retrieval."""

    if state.get("route") == "rag_and_web":
        return "web_search"

    return "evidence_evaluator"

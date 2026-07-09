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


def route_after_web_search(state: GraphState) -> str:
    if (
        state.get("route") == "rag_only"
        and state.get("has_enough_evidence") is False
    ):
        return "answer_generator"

    return "evidence_evaluator"


def route_after_evidence_evaluator(state: GraphState) -> str:
    """Route weak rag_only evidence to web search once; otherwise answer."""

    if (
        state.get("route") == "rag_only"
        and state.get("has_enough_evidence") is False
        and not state.get("web_results")
    ):
        return "web_search"

    return "answer_generator"


def route_after_answer_verifier(state: GraphState) -> str:
    """Route passed/finalized answers to END; unsafe answers to revision."""

    if state.get("verification_status") == "revise" and not state.get("final_answer"):
        return "answer_revision"

    return "end"

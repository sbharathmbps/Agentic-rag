"""Conditional edge functions for the LangGraph workflow."""

from app.graph.state import GraphState


def route_after_safety_classifier(state: GraphState) -> str:
    """Route unsafe queries to escalation; safe queries to intent planning."""

    if state.get("is_safe_to_answer") is False:
        return "unsafe_escalate"

    return "safe_continue"


def route_after_query_rewriter(state: GraphState) -> str:
    """Route rewritten queries to the required retriever/search node."""

    route = state.get("route", "web_only")

    if route == "rag_only":
        return "rag_only_to_rag"

    if route == "rag_and_web":
        return "rag_and_web_to_rag"

    if route == "web_only":
        return "web_only_to_web"

    return "fallback_to_web"


def route_after_rag_retriever(state: GraphState) -> str:
    """Route after local RAG retrieval."""

    if state.get("route") == "rag_and_web":
        return "rag_and_web_continue_web"

    return "rag_only_evaluate"


def route_after_web_search(state: GraphState) -> str:
    if (
        state.get("route") == "rag_only"
        and state.get("has_enough_evidence") is False
    ):
        return "rag_fallback_answer"

    return "evaluate_web_evidence"


def route_after_evidence_evaluator(state: GraphState) -> str:
    """Route weak rag_only evidence to web search once; otherwise answer."""

    if (
        state.get("route") == "rag_only"
        and state.get("has_enough_evidence") is False
        and not state.get("web_results")
    ):
        return "weak_rag_try_web"

    return "answer_with_available_evidence"


def route_after_answer_verifier(state: GraphState) -> str:
    """Route passed/finalized answers to END; unsafe answers to revision."""

    if state.get("verification_status") == "revise" and not state.get("final_answer"):
        return "revise_answer"

    return "pass_or_max_revisions"

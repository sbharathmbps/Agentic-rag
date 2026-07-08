"""LangGraph builder for the agentic healthcare RAG workflow."""

from langgraph.graph import END, START, StateGraph

from app.graph.conditional_edges import (
    route_after_query_rewriter,
    route_after_rag_retriever,
    route_after_safety_classifier,
)
from app.graph.state import GraphState
from app.nodes.node_1_0_safety_classifier import safety_classifier_node
from app.nodes.node_2_0_intent_planner import intent_planner_node
from app.nodes.node_2_1_escalation import escalation_node
from app.nodes.node_3_0_query_rewriter import query_rewriter_node
from app.nodes.node_4_0_rag_retriever import rag_retriever_node
from app.nodes.node_4_1_web_search import web_search_node



def evidence_evaluator_node(state: GraphState) -> dict:
    """Temporary endpoint until the evidence evaluator node is implemented."""

    return {}


def build_graph():
    """Build the healthcare assistant graph."""

    graph = StateGraph(GraphState)

    graph.add_node("safety_classifier", safety_classifier_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("intent_planner", intent_planner_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("rag_retriever", rag_retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("evidence_evaluator", evidence_evaluator_node)

    graph.add_edge(START, "safety_classifier")
    graph.add_conditional_edges(
        "safety_classifier", route_after_safety_classifier,
        {
            "escalation": "escalation",
            "intent_planner": "intent_planner",
        },
    )
    graph.add_edge("escalation", END)
    graph.add_edge("intent_planner", "query_rewriter")
    graph.add_conditional_edges(
        "query_rewriter",
        route_after_query_rewriter,
        {
            "rag_retriever": "rag_retriever",
            "web_search": "web_search"
        },
    )
    graph.add_conditional_edges(
        "rag_retriever",
        route_after_rag_retriever,
        {
            "evidence_evaluator": "evidence_evaluator",
            "web_search": "web_search",
        },
    )
    graph.add_edge("web_search", "evidence_evaluator")
    graph.add_edge("evidence_evaluator", END)

    return graph.compile()

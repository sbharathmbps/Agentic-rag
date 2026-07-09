"""LangGraph builder for the agentic healthcare RAG workflow."""

from langgraph.graph import END, START, StateGraph

from app.graph.conditional_edges import (
    route_after_evidence_evaluator,
    route_after_answer_verifier,
    route_after_query_rewriter,
    route_after_rag_retriever,
    route_after_web_search,
    route_after_safety_classifier,
)
from app.graph.state import GraphState
from app.nodes.node_1_0_safety_classifier import safety_classifier_node
from app.nodes.node_2_0_intent_planner import intent_planner_node
from app.nodes.node_2_1_escalation import escalation_node
from app.nodes.node_3_0_query_rewriter import query_rewriter_node
from app.nodes.node_4_0_rag_retriever import rag_retriever_node
from app.nodes.node_4_1_web_search import web_search_node
from app.nodes.node_5_0_evidence_evaluator import evidence_evaluator_node
from app.nodes.node_6_0_answer_generator import answer_generator_node
from app.nodes.node_7_0_answer_verifier import answer_verifier_node
from app.nodes.node_7_1_answer_revision import answer_revision_node



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
    graph.add_node("answer_generator", answer_generator_node)
    graph.add_node("answer_verifier", answer_verifier_node)
    graph.add_node("answer_revision", answer_revision_node)

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
    graph.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "evidence_evaluator": "evidence_evaluator",
            "answer_generator": "answer_generator",
        },
    )    
    graph.add_conditional_edges(
        "evidence_evaluator",
        route_after_evidence_evaluator,
        {
            "web_search": "web_search",
            "answer_generator": "answer_generator",
        },
    )
    graph.add_edge("answer_generator", "answer_verifier")
    graph.add_conditional_edges(
        "answer_verifier",
        route_after_answer_verifier,
        {
            "answer_revision": "answer_revision",
            "end": END,
        },
    )
    graph.add_edge("answer_revision", "answer_verifier")

    return graph.compile()

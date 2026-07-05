from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]

    # Original user query
    user_query: str

    # Safety classification
    safety_category: str
    is_safe_to_answer: bool
    safety_reason: str

    # Planning / routing
    route: str
    planner_reason: str

    # Rewritten queries
    rag_query: str
    web_query: str

    # Retrieved evidence
    rag_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]

    # Evidence evaluation
    has_enough_evidence: bool
    evidence_score: float
    missing_information: str

    # Answer generation
    draft_answer: str
    final_answer: str

    # Verification / revision loop
    verification_status: str
    verification_feedback: str
    revision_count: int

    # Final source references
    sources: List[Dict[str, Any]]

    # Retry / fallback control
    web_search_attempted: bool
    search_retry_count: int
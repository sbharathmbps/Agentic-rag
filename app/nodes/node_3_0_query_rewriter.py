"""Query rewriter node for RAG and web search."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.query_rewriter import QUERY_REWRITER_SYSTEM_PROMPT
from app.utils.messages import format_conversation


class RewrittenQueries(BaseModel):
    """Structured query rewrite result."""

    rag_query: str = Field(description="Optimized local RAG query, or empty.")
    web_query: str = Field(description="Optimized web search query, or empty.")


def _fallback_rewrite(user_query: str, route: str) -> dict[str, str]:
    """Route-aware fallback if the LLM rewrite is unavailable."""

    if route == "rag_only":
        return {"rag_query": user_query, "web_query": ""}

    if route == "web_only":
        return {"rag_query": "", "web_query": user_query}

    if route == "rag_and_web":
        return {"rag_query": user_query, "web_query": user_query}

    return {"rag_query": "", "web_query": user_query}


def query_rewriter_node(state: GraphState) -> dict[str, str]:
    """Rewrite the user query based on the planner route."""

    user_query = state.get("user_query", "").strip()
    route = state.get("route", "web_only")
    messages = state.get("messages", [])

    if not user_query:
        return {"rag_query": "", "web_query": ""}

    try:
        rewriter = load_llm().with_structured_output(RewrittenQueries)
        result = rewriter.invoke(
            [
                SystemMessage(content=QUERY_REWRITER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Route: {route}\n\n"
                        f"Conversation:\n{format_conversation(messages)}\n\n"
                        f"User query:\n{user_query}"
                    )
                ),
            ]
        )
        rewritten = RewrittenQueries.model_validate(result)
    except Exception:
        return _fallback_rewrite(user_query, route)

    return {
        "rag_query": rewritten.rag_query.strip(),
        "web_query": rewritten.web_query.strip(),
    }

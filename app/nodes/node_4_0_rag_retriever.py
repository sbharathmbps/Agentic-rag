"""RAG retriever node for local medicine information."""

from app.graph.state import GraphState
from app.tools.rag import search_medicine_knowledge


def _format_rag_result(document) -> dict:
    """Convert a LangChain Document into graph-state result format."""

    metadata = dict(getattr(document, "metadata", {}) or {})

    return {
        "content": getattr(document, "page_content", ""),
        "metadata": {
            "medicine_name": metadata.get("medicine_name", ""),
            "source": metadata.get("source", ""),
            "last_updated": metadata.get("last_updated", ""),
            **metadata,
        },
    }


def rag_retriever_node(state: GraphState) -> dict[str, list[dict]]:
    """Search the local FAISS vector DB using state['rag_query']."""

    rag_query = state.get("rag_query", "").strip()

    if not rag_query:
        return {"rag_results": []}

    try:
        documents = search_medicine_knowledge(rag_query)
    except Exception:
        return {"rag_results": []}

    return {
        "rag_results": [_format_rag_result(document) for document in documents],
    }

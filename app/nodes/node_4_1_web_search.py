"""Web search node for trusted healthcare sources."""

from urllib.parse import urlparse

from app.graph.state import GraphState
from app.tools.tavily import TRUSTED_HEALTH_DOMAINS, search_trusted_health_web


def _is_trusted_health_source(url: str) -> bool:
    """Return whether a URL belongs to a preferred trusted health domain."""

    hostname = urlparse(url).netloc.lower()
    hostname = hostname.removeprefix("www.")

    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in TRUSTED_HEALTH_DOMAINS
    )


def _format_web_result(result: dict) -> dict:
    """Convert a Tavily result into graph-state result format."""

    url = result.get("url", "")

    return {
        "title": result.get("title", ""),
        "url": url,
        "content": result.get("content", ""),
        "source_type": (
            "trusted_health_source"
            if _is_trusted_health_source(url)
            else "web_source"
        ),
    }


def web_search_node(state: GraphState) -> dict[str, list[dict]]:
    """Search trusted healthcare websites using state['web_query']."""

    web_query = (
        state.get("web_query", "").strip()
        or state.get("rag_query", "").strip()
        or state.get("user_query", "").strip()
    )

    if not web_query:
        return {"web_results": []}

    try:
        results = search_trusted_health_web(web_query)
    except Exception:
        return {"web_results": []}

    return {
        "web_results": [_format_web_result(result) for result in results],
    }

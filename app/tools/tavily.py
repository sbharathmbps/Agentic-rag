"""Tavily web-search helpers."""

from langchain_tavily import TavilySearch

from app.config.settings import load_environment


TRUSTED_HEALTH_DOMAINS = [
    "who.int",
    "cdc.gov",
    "fda.gov",
    "nhs.uk",
    "medlineplus.gov",
    "mayoclinic.org",
    "dailymed.nlm.nih.gov",
]


def search_trusted_health_web(query: str,max_results: int = 5,) -> list[dict]:
    """Search trusted healthcare sources with Tavily."""

    if not query.strip():
        return []

    load_environment()

    search = TavilySearch(
        max_results=max_results,
        search_depth="basic",
        include_domains=TRUSTED_HEALTH_DOMAINS,
        include_answer=False,
        include_raw_content=False,
    )

    response = search.invoke({"query": query})

    if isinstance(response, dict):
        return response.get("results", [])

    if isinstance(response, list):
        return response

    return []

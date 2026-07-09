"""Utilities for formatting retrieved evidence."""


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to a compact prompt-friendly snippet."""

    text = text.strip()
    if len(text) <= max_chars:
        return text

    return f"{text[:max_chars].rstrip()}..."


def format_rag_results(
    results: list[dict],
    max_results: int = 4,
    max_chars: int = 800,
) -> str:
    """Format RAG evidence for LLM prompts."""

    if not results:
        return "No RAG evidence."

    lines = []
    for index, result in enumerate(results[:max_results], start=1):
        metadata = result.get("metadata", {}) or {}
        medicine = metadata.get("medicine_name", "")
        source = metadata.get("source", "local medicine store")
        content = truncate_text(str(result.get("content", "")), max_chars)
        lines.append(
            f"[RAG {index}] medicine={medicine}; source={source}\n{content}"
        )

    return "\n\n".join(lines)


def format_web_results(
    results: list[dict],
    max_results: int = 4,
    max_chars: int = 800,
) -> str:
    """Format web evidence for LLM prompts."""

    if not results:
        return "No web evidence."

    lines = []
    for index, result in enumerate(results[:max_results], start=1):
        title = result.get("title", "")
        url = result.get("url", "")
        content = truncate_text(str(result.get("content", "")), max_chars)
        lines.append(f"[WEB {index}] {title}\nURL: {url}\n{content}")

    return "\n\n".join(lines)

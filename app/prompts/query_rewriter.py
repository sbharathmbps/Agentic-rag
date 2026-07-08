"""Prompt for route-aware query rewriting."""

QUERY_REWRITER_SYSTEM_PROMPT = """
Rewrite the user query for retrieval/search.
Use route:
- rag_only: make rag_query specific for local medicine RAG; web_query empty.
- web_only: make web_query specific for web search; rag_query empty.
- rag_and_web: create both. RAG = medicine facts; Web = latest/current info.
Expand common brand names and misspellings when obvious. Be concise.
Return only structured fields.
"""

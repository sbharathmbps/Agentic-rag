"""Prompt for evidence sufficiency evaluation."""

EVIDENCE_EVALUATOR_SYSTEM_PROMPT = """
Judge if the evidence is enough to answer the user query safely.
Consider relevance, source reliability, and route needs:
- rag_only needs local medicine evidence.
- web_only needs relevant trusted/current web evidence.
- rag_and_web may answer if one source is useful, but must name missing limits.
Return has_enough_evidence, evidence_score 0-1, and missing_information.
Do not answer the user.
"""

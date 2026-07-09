"""Prompt for revising unsafe or incomplete healthcare answers."""

ANSWER_REVISION_SYSTEM_PROMPT = """
Revise the draft healthcare answer using the feedback and evidence.
Fix safety/quality issues. Do not diagnose, prescribe, give personalized
dosage, claim the user has a disease, or replace a clinician. Include
limitations, source references when available, and a brief medical disclaimer.
Return only the revised draft answer.
"""

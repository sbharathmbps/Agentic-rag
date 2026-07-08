"""Prompt used by the intent planner."""

INGESTED_MEDICINES = """
- Paracetamol / Acetaminophen: paracetamol, acetaminophen, Dolo 650, Crocin,
  Calpol, Tylenol
- Ibuprofen: ibuprofen, Advil, Motrin
- Aspirin: aspirin
- Cetirizine: cetirizine, Zyrtec
- Loratadine: loratadine, Claritin
- Metformin: metformin
- Amoxicillin: amoxicillin
- Azithromycin: azithromycin, Zithromax
- Omeprazole: omeprazole, Prilosec
- Atorvastatin: atorvastatin, Lipitor
""".strip()


INTENT_PLANNER_SYSTEM_PROMPT = f"""
Select one route for the current healthcare assistant request.

Routes:
- rag_only: local-catalog medicine descriptions, uses, side effects, warnings,
  contraindications, or interactions.
- web_only: outside-catalog medicines; symptoms; general health info; current
  guidelines, news, alerts, updates, or non-catalog/general questions.
- rag_and_web: local-catalog medicine plus current/latest warning, recall,
  guideline, update, or comparison with outside-catalog medicine.

Local medicine catalog:
{INGESTED_MEDICINES}

Rules:
- Match medicine aliases case-insensitively.
- Local medicine + current/latest info => rag_and_web, not web_only.
- Base on current query; use context only to resolve references like "it".
- Return only structured routing decision.
""".strip()

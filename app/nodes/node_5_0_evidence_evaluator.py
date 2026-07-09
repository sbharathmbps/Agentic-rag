"""Evidence evaluator node."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.evidence_evaluator import EVIDENCE_EVALUATOR_SYSTEM_PROMPT
from app.utils.evidence import format_rag_results, format_web_results


MAX_EVIDENCE_CHARS = 800


class EvidenceEvaluation(BaseModel):
    """Structured evidence sufficiency decision."""

    has_enough_evidence: bool
    evidence_score: float = Field(ge=0.0, le=1.0)
    missing_information: str


def _fallback_evaluation(route: str, rag_results: list[dict], web_results: list[dict]):
    """Conservative fallback if LLM evaluation is unavailable."""

    has_rag = bool(rag_results)
    has_web = bool(web_results)

    if route == "rag_only":
        return {
            "has_enough_evidence": has_rag,
            "evidence_score": 0.5 if has_rag else 0.0,
            "missing_information": (
                "" if has_rag else "Local medicine evidence is missing."
            ),
        }

    if route == "web_only":
        return {
            "has_enough_evidence": has_web,
            "evidence_score": 0.5 if has_web else 0.0,
            "missing_information": (
                "" if has_web else "Trusted web evidence is missing."
            ),
        }

    has_any = has_rag or has_web
    missing = []
    if not has_rag:
        missing.append("Local medicine evidence is missing.")
    if not has_web:
        missing.append("Trusted current web evidence is missing.")

    return {
        "has_enough_evidence": has_any,
        "evidence_score": 0.5 if has_any else 0.0,
        "missing_information": " ".join(missing),
    }


def evidence_evaluator_node(state: GraphState) -> dict[str, object]:
    """Use an LLM to judge whether evidence is enough for the user query."""

    user_query = state.get("user_query", "").strip()
    route = state.get("route", "web_only")
    rag_results = state.get("rag_results", [])
    web_results = state.get("web_results", [])

    try:
        evaluator = load_llm().with_structured_output(EvidenceEvaluation)
        result = evaluator.invoke(
            [
                SystemMessage(content=EVIDENCE_EVALUATOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User query:\n{user_query}\n\n"
                        f"Route: {route}\n\n"
                        f"RAG evidence:\n"
                        f"{format_rag_results(rag_results, max_chars=MAX_EVIDENCE_CHARS)}\n\n"
                        f"Web evidence:\n"
                        f"{format_web_results(web_results, max_chars=MAX_EVIDENCE_CHARS)}"
                    )
                ),
            ]
        )
        evaluation = EvidenceEvaluation.model_validate(result)
    except Exception:
        return _fallback_evaluation(route, rag_results, web_results)

    return {
        "has_enough_evidence": evaluation.has_enough_evidence,
        "evidence_score": evaluation.evidence_score,
        "missing_information": evaluation.missing_information,
    }

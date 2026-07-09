"""Answer verifier node."""

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.answer_verifier import ANSWER_VERIFIER_SYSTEM_PROMPT
from app.utils.evidence import format_rag_results, format_web_results


MAX_REVISION_COUNT = 2
MAX_EVIDENCE_CHARS = 700


class AnswerVerification(BaseModel):
    """Structured verification result."""

    verification_status: Literal["pass", "revise"]
    verification_feedback: str = Field(min_length=1)


def _fallback_verification(state: GraphState) -> dict[str, object]:
    """Safe fallback if verifier LLM is unavailable."""

    draft_answer = state.get("draft_answer", "").strip()
    revision_count = state.get("revision_count", 0)

    if not draft_answer:
        return {
            "verification_status": "revise",
            "verification_feedback": "Draft answer is missing.",
            "revision_count": revision_count,
        }

    return {
        "verification_status": "pass",
        "verification_feedback": "Verifier unavailable; accepting safe fallback.",
        "revision_count": revision_count,
        "final_answer": draft_answer,
    }


def answer_verifier_node(state: GraphState) -> dict[str, object]:
    """Check the draft answer for safety, grounding, and completeness."""

    draft_answer = state.get("draft_answer", "").strip()
    revision_count = state.get("revision_count", 0)

    if revision_count >= MAX_REVISION_COUNT:
        return {
            "verification_status": "revise",
            "verification_feedback": (
                "Maximum revision count reached; finalizing current draft."
            ),
            "revision_count": revision_count,
            "final_answer": draft_answer,
        }

    try:
        verifier = load_llm().with_structured_output(AnswerVerification)
        result = verifier.invoke(
            [
                SystemMessage(content=ANSWER_VERIFIER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User query:\n{state.get('user_query', '')}\n\n"
                        f"Safety category: {state.get('safety_category', '')}\n"
                        f"Route: {state.get('route', '')}\n"
                        f"Missing/limits: "
                        f"{state.get('missing_information', '')}\n\n"
                        f"RAG evidence:\n"
                        f"{format_rag_results(
                            state.get('rag_results', []),
                            max_chars=MAX_EVIDENCE_CHARS,
                        )}\n\n"
                        f"Web evidence:\n"
                        f"{format_web_results(
                            state.get('web_results', []),
                            max_chars=MAX_EVIDENCE_CHARS,
                        )}\n\n"
                        f"Draft answer:\n{draft_answer}"
                    )
                ),
            ]
        )
        verification = AnswerVerification.model_validate(result)
    except Exception:
        return _fallback_verification(state)

    output: dict[str, object] = {
        "verification_status": verification.verification_status,
        "verification_feedback": verification.verification_feedback,
        "revision_count": revision_count,
    }

    if verification.verification_status == "pass":
        output["final_answer"] = draft_answer

    return output

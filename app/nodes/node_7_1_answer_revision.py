"""Answer revision node."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.answer_revision import ANSWER_REVISION_SYSTEM_PROMPT
from app.utils.evidence import format_rag_results, format_web_results


MAX_EVIDENCE_CHARS = 700


class RevisedAnswer(BaseModel):
    """Structured revised answer."""

    draft_answer: str = Field(min_length=1)


def _fallback_revision(state: GraphState) -> str:
    """Basic safe revision if the LLM revision is unavailable."""

    draft_answer = state.get("draft_answer", "").strip()
    feedback = state.get("verification_feedback", "").strip()

    return (
        f"{draft_answer}\n\n"
        "Safety note: This is general health information, not a diagnosis, "
        "prescription, or personalized dosing advice. Please consult a doctor "
        "or pharmacist for personal medical decisions."
        + (f"\n\nLimitation addressed: {feedback}" if feedback else "")
    ).strip()


def answer_revision_node(state: GraphState) -> dict[str, object]:
    """Revise the draft answer using verifier feedback."""

    revision_count = state.get("revision_count", 0) + 1

    try:
        reviser = load_llm().with_structured_output(RevisedAnswer)
        result = reviser.invoke(
            [
                SystemMessage(content=ANSWER_REVISION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User query:\n{state.get('user_query', '')}\n\n"
                        f"Verifier feedback:\n"
                        f"{state.get('verification_feedback', '')}\n\n"
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
                        f"Draft answer:\n{state.get('draft_answer', '')}"
                    )
                ),
            ]
        )
        revised_answer = RevisedAnswer.model_validate(result).draft_answer.strip()
    except Exception:
        revised_answer = _fallback_revision(state)

    return {
        "draft_answer": revised_answer,
        "revision_count": revision_count,
    }

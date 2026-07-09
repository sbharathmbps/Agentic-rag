"""Answer generator node."""

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.state import GraphState
from app.llm.gemini import load_llm
from app.prompts.answer_generator import ANSWER_GENERATOR_SYSTEM_PROMPT
from app.utils.evidence import format_rag_results, format_web_results


MAX_EVIDENCE_CHARS = 900


class DraftAnswer(BaseModel):
    """Structured draft answer returned by the model."""

    draft_answer: str = Field(min_length=1)


def _fallback_answer(state: GraphState) -> str:
    """Safe fallback when model answer generation is unavailable."""

    missing = state.get("missing_information", "")
    if missing:
        return (
            "I do not have enough reliable evidence to give a complete answer. "
            f"Limitation: {missing} Please consult a qualified healthcare "
            "professional for personal medical advice. I cannot diagnose, "
            "prescribe, or provide personalized dosing."
        )

    return (
        "I found some general health information, but I cannot generate a full "
        "answer right now. Please consult a qualified healthcare professional "
        "for personal medical advice. I cannot diagnose, prescribe, or provide "
        "personalized dosing."
    )


def answer_generator_node(state: GraphState) -> dict[str, str]:
    """Generate a safe draft answer using retrieved evidence."""

    user_query = state.get("user_query", "").strip()

    try:
        generator = load_llm().with_structured_output(DraftAnswer)
        result = generator.invoke(
            [
                SystemMessage(content=ANSWER_GENERATOR_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User query:\n{user_query}\n\n"
                        f"Route: {state.get('route', 'web_only')}\n"
                        f"Evidence score: {state.get('evidence_score', 0)}\n"
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
                        )}"
                    )
                ),
            ]
        )
        draft = DraftAnswer.model_validate(result).draft_answer.strip()
    except Exception:
        draft = _fallback_answer(state)

    return {"draft_answer": draft}

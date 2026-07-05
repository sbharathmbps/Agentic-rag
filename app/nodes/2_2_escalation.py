"""Deterministic safe responses for risky healthcare requests."""

from langchain_core.messages import AIMessage

from app.graph.state import GraphState


ESCALATION_RESPONSES = {
    "emergency": (
        "This may require urgent medical attention. Please contact your local "
        "emergency services or go to the nearest emergency department now. "
        "I can provide general information, but I cannot diagnose or prescribe "
        "medication."
    ),
    "self_harm": (
        "I'm sorry you're dealing with this. If you may act on thoughts of "
        "self-harm or are in immediate danger, contact local emergency services "
        "now. Please also reach out to a local crisis hotline or a trusted "
        "person who can stay with you. I cannot provide crisis care, but you "
        "deserve immediate support from someone who can."
    ),
    "diagnosis_request": (
        "I cannot diagnose a medical condition. A qualified healthcare "
        "professional can assess your symptoms and medical history. I can still "
        "provide general educational information about symptoms or conditions."
    ),
    "prescription_request": (
        "I cannot prescribe medication or decide which medicine you personally "
        "should take. Please speak with a doctor or pharmacist who can review "
        "your symptoms, health history, and current medicines."
    ),
    "dosage_request": (
        "I cannot provide a personalized dosage or recommend changing a dose. "
        "Please follow the product label or your prescriber's instructions and "
        "check with a doctor or pharmacist before making changes."
    ),
}

DEFAULT_ESCALATION_RESPONSE = (
    "I cannot safely answer this as a personal medical decision. Please consult "
    "a qualified healthcare professional. I can provide general educational "
    "health information."
)


def escalation_node(state: GraphState) -> dict[str, object]:
    """Return a category-specific safe response for an unsafe query."""

    category = state.get("safety_category", "unknown")
    response = ESCALATION_RESPONSES.get(category, DEFAULT_ESCALATION_RESPONSE)

    return {
        "final_answer": response,
        "messages": [AIMessage(content=response)],
    }

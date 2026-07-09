"""Runtime helpers for invoking the compiled LangGraph app."""

from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config.settings import load_environment
from app.graph.builder import build_graph
from app.graph.state import GraphState

load_environment()


@lru_cache(maxsize=1)
def get_graph():
    """Compile the graph once and reuse it across requests."""

    return build_graph()


def invoke_graph(
    user_query: str,
    messages: list[BaseMessage] | None = None,
) -> GraphState:
    """Invoke the graph with the user's query and return the final state."""

    messages = messages or []
    current_messages = [*messages, HumanMessage(content=user_query)]

    initial_state: GraphState = {
        "user_query": user_query,
        "messages": current_messages,
    }

    return get_graph().invoke(initial_state)


def _get_answer(final_state: GraphState) -> str:
    """Return the best answer field to display to the user."""

    return (
        final_state.get("final_answer")
        or final_state.get("draft_answer")
        or "No final answer generated."
    )


def run_chat_loop() -> None:
    """Run a terminal chat loop with conversation context."""

    messages: list[BaseMessage] = []

    print("Agentic RAG chat started. Type 'exit', 'quit', or 'q' to stop.\n")

    while True:
        user_query = input("You: ").strip()

        if user_query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        if not user_query:
            continue

        final_state = invoke_graph(user_query, messages)
        answer = _get_answer(final_state)

        print(f"\nAssistant: {answer}\n")

        messages.append(HumanMessage(content=user_query))
        messages.append(AIMessage(content=answer))


if __name__ == "__main__":
    run_chat_loop()

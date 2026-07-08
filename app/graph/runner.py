"""Runtime helpers for invoking the compiled LangGraph app."""

from functools import lru_cache

from langchain_core.messages import HumanMessage

from app.graph.builder import build_graph
from app.graph.state import GraphState
from app.config.settings import load_environment
    
load_environment()


@lru_cache(maxsize=1)
def get_graph():
    """Compile the graph once and reuse it across requests."""

    return build_graph()


def invoke_graph(user_query: str) -> GraphState:
    """Invoke the graph with the user's query and return the final state."""

    initial_state: GraphState = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
    }

    return get_graph().invoke(initial_state)

if __name__ == "__main__":
    
    user_query = "What are the uses of paracetamol and its recent guidelines according to age?"
    # user_query = "what shall I do for sucidal thoughts?"
    final_state = invoke_graph(user_query)
    print(final_state)



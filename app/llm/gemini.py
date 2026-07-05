"""Gemini chat-model construction."""

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import load_environment

MODEL_NAME = "gemini-2.5-flash"


@lru_cache(maxsize=8)
def load_llm(
    model: str = MODEL_NAME,
    temperature: float = 0.1,
    max_retries: int = 2,
) -> ChatGoogleGenerativeAI:

    load_environment()

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

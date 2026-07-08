
"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(title="Agentic Healthcare RAG")

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""

    return {"status": "ok"}

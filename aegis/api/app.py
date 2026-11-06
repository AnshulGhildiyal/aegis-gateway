"""FastAPI application and Phase 0 endpoints."""

from fastapi import FastAPI, HTTPException, status

from aegis.api.models import ChatCompletionRequest

app = FastAPI(title="Aegis Gateway", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the API process is responsive."""
    return {"status": "ok"}


@app.post(
    "/v1/chat/completions",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    response_model=None,
)
async def create_chat_completion(request: ChatCompletionRequest) -> None:
    """Reserve the OpenAI-compatible route for a later implementation phase."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Chat completions are not implemented yet.",
    )

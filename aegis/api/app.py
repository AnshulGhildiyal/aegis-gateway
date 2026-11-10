"""FastAPI application and API endpoints."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status

from aegis.api.models import ChatCompletionRequest, ChatCompletionResponse
from aegis.providers.groq import (
    GroqConfigurationError,
    GroqConnectionError,
    GroqHTTPError,
    GroqProvider,
    GroqResponseError,
    GroqTimeoutError,
    GroqUnsupportedRequestError,
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create and close the shared provider for this application instance."""
    provider = GroqProvider()
    application.state.groq_provider = provider
    try:
        yield
    finally:
        await provider.aclose()


app = FastAPI(title="Aegis Gateway", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Report that the API process is responsive."""
    return {"status": "ok"}


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
)
async def create_chat_completion(
    http_request: Request,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Forward one non-streaming chat completion request to Groq."""
    try:
        provider = cast(GroqProvider, http_request.app.state.groq_provider)
        return await provider.complete(request)
    except GroqUnsupportedRequestError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The chat completion request is not supported.",
        ) from None
    except GroqConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The chat completion provider is not configured.",
        ) from None
    except GroqTimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The chat completion provider timed out.",
        ) from None
    except (GroqConnectionError, GroqHTTPError, GroqResponseError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The chat completion provider failed.",
        ) from None

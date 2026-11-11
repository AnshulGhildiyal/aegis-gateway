"""FastAPI application and API endpoints."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import ValidationError

from aegis.api.models import ChatCompletionRequest, ChatCompletionResponse
from aegis.engine.cache_policy import build_cache_key, is_cache_eligible
from aegis.engine.embeddings import FastEmbedder
from aegis.engine.lookup import SemanticLookup
from aegis.providers.groq import (
    GroqConfigurationError,
    GroqConnectionError,
    GroqHTTPError,
    GroqProvider,
    GroqResponseError,
    GroqTimeoutError,
    GroqUnsupportedRequestError,
)
from aegis.storage.cache import CacheOperationalError, SemanticCache

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Create and close the shared providers and semantic-cache resources."""
    async with AsyncExitStack() as stack:
        provider = GroqProvider()
        stack.push_async_callback(provider.aclose)

        embedder = FastEmbedder()
        stack.push_async_callback(embedder.close)

        cache = SemanticCache(
            embedder,
            path=":memory:",
            vector_size=384,
            ttl_seconds=3600,
        )
        stack.callback(cache.close)

        lookup = SemanticLookup(
            cache,
            cosine_threshold=0.95,
            stage0_enabled=True,
            nli_enabled=False,
        )

        application.state.groq_provider = provider
        application.state.embedder = embedder
        application.state.semantic_cache = cache
        application.state.semantic_lookup = lookup
        yield


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
    """Return a validated cached completion or forward the request to Groq."""
    try:
        provider = cast(GroqProvider, http_request.app.state.groq_provider)
        if not is_cache_eligible(request):
            return await provider.complete(request)

        key = build_cache_key(request)
        user_content = request.messages[0].content
        if key is None or not isinstance(user_content, str):
            return await provider.complete(request)

        lookup = cast(SemanticLookup, http_request.app.state.semantic_lookup)
        try:
            result = await lookup.lookup(user_content, key)
        except CacheOperationalError:
            logger.warning(
                "Semantic cache lookup failed; bypassing cache.",
                exc_info=True,
            )
            return await provider.complete(request)

        if result.entry is not None:
            try:
                return ChatCompletionResponse.model_validate(result.entry.response)
            except ValidationError:
                logger.warning(
                    "Semantic cache returned a malformed completion; refreshing it.",
                    exc_info=True,
                )

        response = await provider.complete(request)
        cache = cast(SemanticCache, http_request.app.state.semantic_cache)
        try:
            await cache.put(
                user_content=user_content,
                response=response.model_dump(mode="json"),
                key=key,
            )
        except CacheOperationalError:
            logger.warning(
                "Semantic cache storage failed; returning the provider response.",
                exc_info=True,
            )
        return response
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

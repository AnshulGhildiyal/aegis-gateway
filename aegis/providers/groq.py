"""Asynchronous Groq chat-completion provider using HTTPX."""

import os

import httpx
from pydantic import ValidationError

from aegis.api.models import ChatCompletionRequest, ChatCompletionResponse

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProviderError(Exception):
    """Base class for failures raised by the Groq provider adapter."""


class GroqConfigurationError(GroqProviderError):
    """Raised when the Groq API key is missing or empty."""


class GroqHTTPError(GroqProviderError):
    """Raised when Groq responds with a non-success HTTP status."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"Groq returned HTTP status {status_code}.")


class GroqTimeoutError(GroqProviderError):
    """Raised when a Groq request times out."""


class GroqConnectionError(GroqProviderError):
    """Raised when the HTTP transport cannot complete a Groq request."""


class GroqResponseError(GroqProviderError):
    """Raised when Groq returns a response that is not valid for Aegis."""


class GroqUnsupportedRequestError(GroqProviderError):
    """Raised when the request uses a feature this adapter does not implement."""


class GroqProvider:
    """Send non-streaming OpenAI-compatible chat completions to Groq."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key is None or not api_key.strip():
            raise GroqConfigurationError("GROQ_API_KEY is required.")
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
            transport=transport,
        )

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Forward a non-streaming request and validate Groq's response."""
        if request.stream:
            raise GroqUnsupportedRequestError(
                "Streaming requests are not supported by this adapter."
            )

        if request.presence_penalty is not None:
            raise GroqUnsupportedRequestError(
                "Groq does not support presence_penalty."
            )
        if request.frequency_penalty is not None:
            raise GroqUnsupportedRequestError(
                "Groq does not support frequency_penalty."
            )
        if request.n is not None and request.n != 1:
            raise GroqUnsupportedRequestError("Groq only supports n=1.")
        if request.model_extra:
            extra_fields = ", ".join(sorted(request.model_extra))
            raise GroqUnsupportedRequestError(
                f"Unsupported request fields: {extra_fields}."
            )

        payload = request.model_dump(mode="json", exclude_unset=True)
        max_tokens_was_set = "max_tokens" in request.model_fields_set
        max_completion_tokens_was_set = (
            "max_completion_tokens" in request.model_fields_set
        )
        max_tokens = payload.pop("max_tokens", None)
        if (
            max_tokens_was_set
            and max_completion_tokens_was_set
            and max_tokens != payload.get("max_completion_tokens")
        ):
            raise GroqUnsupportedRequestError(
                "max_tokens and max_completion_tokens must match when both are set."
            )
        if max_tokens is not None:
            payload["max_completion_tokens"] = max_tokens

        try:
            response = await self._client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                json=payload,
            )
        except httpx.TimeoutException:
            raise GroqTimeoutError("The Groq request timed out.") from None
        except httpx.RequestError:
            raise GroqConnectionError(
                "The Groq request could not be completed."
            ) from None

        if not 200 <= response.status_code < 300:
            raise GroqHTTPError(response.status_code)

        try:
            return ChatCompletionResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            raise GroqResponseError(
                "Groq returned a malformed chat completion response."
            ) from None

    async def aclose(self) -> None:
        """Close the owned HTTPX client and release its transport resources."""
        await self._client.aclose()

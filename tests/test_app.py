"""Tests for the FastAPI application and request/response models."""

from collections.abc import AsyncIterator
from typing import cast

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

import aegis.api.app as api_module
from aegis.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from aegis.providers.groq import (
    GroqConfigurationError,
    GroqConnectionError,
    GroqHTTPError,
    GroqResponseError,
    GroqTimeoutError,
    GroqUnsupportedRequestError,
)

REQUEST_BODY = {
    "model": "llama-test",
    "messages": [{"role": "user", "content": "Hello"}],
}
RESPONSE_BODY = {
    "id": "chatcmpl-test",
    "created": 1,
    "model": "llama-test",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }
    ],
}


class MockProvider:
    def __init__(
        self,
        *,
        response: ChatCompletionResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or ChatCompletionResponse.model_validate(RESPONSE_BODY)
        self.error = error
        self.received_request: ChatCompletionRequest | None = None
        self.closed = False

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        self.received_request = request
        if self.error is not None:
            raise self.error
        return self.response

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=api_module.app),
        base_url="http://test",
    ) as test_client:
        yield test_client


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture
async def provider_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    provider = MockProvider()
    monkeypatch.setattr(api_module, "GroqProvider", lambda: provider)
    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            yield test_client


async def test_health_returns_ok_with_provider_lifespan(
    provider_client: httpx.AsyncClient,
) -> None:
    response = await provider_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_chat_completion_returns_provider_response(
    provider_client: httpx.AsyncClient,
) -> None:
    response = await provider_client.post(
        "/v1/chat/completions",
        json=REQUEST_BODY,
    )

    assert response.status_code == 200
    expected = ChatCompletionResponse.model_validate(RESPONSE_BODY)
    assert response.json() == expected.model_dump(mode="json")


async def test_chat_completion_forwards_validated_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockProvider()
    monkeypatch.setattr(api_module, "GroqProvider", lambda: provider)
    expected_request = ChatCompletionRequest.model_validate(REQUEST_BODY)

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=REQUEST_BODY,
            )

    assert response.status_code == 200
    assert provider.received_request == expected_request


@pytest.mark.parametrize(
    ("provider_error", "status_code"),
    [
        (GroqUnsupportedRequestError("secret detail"), 400),
        (GroqConfigurationError("secret detail"), 503),
        (GroqTimeoutError("secret detail"), 504),
        (GroqConnectionError("secret detail"), 502),
        (GroqHTTPError(429), 502),
        (GroqResponseError("secret detail"), 502),
    ],
)
async def test_provider_errors_map_to_generic_http_responses(
    monkeypatch: pytest.MonkeyPatch,
    provider_error: Exception,
    status_code: int,
) -> None:
    provider = MockProvider(error=provider_error)
    monkeypatch.setattr(api_module, "GroqProvider", lambda: provider)

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=REQUEST_BODY,
            )

    assert response.status_code == status_code
    assert "secret detail" not in response.text


async def test_provider_is_closed_during_application_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MockProvider()
    monkeypatch.setattr(api_module, "GroqProvider", lambda: provider)

    async with api_module.app.router.lifespan_context(api_module.app):
        assert cast(MockProvider, api_module.app.state.groq_provider) is provider
        assert not provider.closed

    assert provider.closed


def test_chat_completion_models_validate_openai_shape() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "example-model",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    response = ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-example",
            "created": 1,
            "model": "example-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    assert request.messages[0].content == "Hello"
    assert response.choices[0].message.content == "Hi"
    assert response.object == "chat.completion"

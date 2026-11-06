"""Tests for the Phase 0 API and request/response models."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from aegis.api.app import app
from aegis.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as test_client:
        yield test_client


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_chat_completions_is_not_implemented(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": "example-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 501
    assert response.json() == {
        "detail": "Chat completions are not implemented yet."
    }


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

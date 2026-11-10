"""Isolated tests for the Groq HTTP provider adapter."""

import httpx
import pytest

from aegis.api.models import ChatCompletionRequest, ChatCompletionResponse
from aegis.providers.groq import (
    GROQ_CHAT_COMPLETIONS_URL,
    GroqConfigurationError,
    GroqConnectionError,
    GroqHTTPError,
    GroqProvider,
    GroqResponseError,
    GroqTimeoutError,
    GroqUnsupportedRequestError,
)

_FAKE_API_KEY = "test-groq-key"
_REQUEST_FIELDS: dict[str, object] = {
    "model": "llama-test",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.25,
    "top_p": 0.8,
    "n": 1,
    "stream": False,
    "stop": ["END"],
    "tools": [{"type": "function", "function": {"name": "lookup"}}],
    "tool_choice": "auto",
    "response_format": {"type": "json_object"},
}


def valid_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "created": 1,
        "model": "llama-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        },
    }


def request_model(**overrides: object) -> ChatCompletionRequest:
    fields = {**_REQUEST_FIELDS, **overrides}
    return ChatCompletionRequest.model_validate(
        fields
    )


async def assert_request_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    request: ChatCompletionRequest,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)

    def unexpected_request(http_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unsupported request should not be sent")

    provider = GroqProvider(transport=httpx.MockTransport(unexpected_request))
    try:
        with pytest.raises(GroqUnsupportedRequestError):
            await provider.complete(request)
    finally:
        await provider.aclose()


async def sent_payload(
    monkeypatch: pytest.MonkeyPatch,
    request: ChatCompletionRequest,
) -> dict[str, object]:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)
    observed: dict[str, object] = {}

    async def handler(http_request: httpx.Request) -> httpx.Response:
        observed["body"] = await http_request.aread()
        return httpx.Response(200, json=valid_response())

    provider = GroqProvider(transport=httpx.MockTransport(handler))
    try:
        await provider.complete(request)
    finally:
        await provider.aclose()

    body = observed["body"]
    assert isinstance(body, bytes)
    payload = httpx.Response(200, content=body).json()
    assert isinstance(payload, dict)
    return {str(key): value for key, value in payload.items()}


async def test_streaming_request_is_rejected_without_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        raise AssertionError("streaming request should not be sent")

    provider = GroqProvider(transport=httpx.MockTransport(unexpected_request))
    try:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "llama-test",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            }
        )
        with pytest.raises(GroqUnsupportedRequestError):
            await provider.complete(request)
    finally:
        await provider.aclose()


async def test_complete_sends_endpoint_auth_and_all_supplied_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", f"  {_FAKE_API_KEY} \t")
    observed: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers["Authorization"]
        observed["body"] = await request.aread()
        return httpx.Response(200, json=valid_response())

    provider = GroqProvider(transport=httpx.MockTransport(handler))
    try:
        result = await provider.complete(request_model(max_completion_tokens=64))
    finally:
        await provider.aclose()

    assert observed["url"] == GROQ_CHAT_COMPLETIONS_URL
    assert observed["authorization"] == f"Bearer {_FAKE_API_KEY}"
    assert observed["body"] == (
        b'{"model":"llama-test","messages":[{"role":"user","content":"Hi"}],'
        b'"temperature":0.25,"top_p":0.8,"n":1,"stream":false,"stop":["END"],'
        b'"max_completion_tokens":64,"tools":[{"type":"function","function":'
        b'{"name":"lookup"}}],"tool_choice":"auto","response_format":'
        b'{"type":"json_object"}}'
    )
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].message.content == "Hello"
    assert result.usage is not None
    assert result.usage.total_tokens == 3


async def test_presence_penalty_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await assert_request_is_unsupported(
        monkeypatch,
        request_model(presence_penalty=0.1),
    )


async def test_frequency_penalty_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await assert_request_is_unsupported(
        monkeypatch,
        request_model(frequency_penalty=0.1),
    )


@pytest.mark.parametrize(
    ("n", "is_supported"),
    [(1, True), (0, False), (2, False)],
)
async def test_n_must_be_one(
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    is_supported: bool,
) -> None:
    request = request_model(n=n)
    if not is_supported:
        await assert_request_is_unsupported(monkeypatch, request)
        return

    payload = await sent_payload(monkeypatch, request)
    assert payload["n"] == 1


async def test_max_tokens_is_normalized_to_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = await sent_payload(
        monkeypatch,
        request_model(max_tokens=32),
    )

    assert payload["max_completion_tokens"] == 32
    assert "max_tokens" not in payload


async def test_equal_token_limits_send_only_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = await sent_payload(
        monkeypatch,
        request_model(max_tokens=32, max_completion_tokens=32),
    )

    assert payload["max_completion_tokens"] == 32
    assert "max_tokens" not in payload


async def test_conflicting_token_limits_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await assert_request_is_unsupported(
        monkeypatch,
        request_model(max_tokens=32, max_completion_tokens=64),
    )


async def test_unsupported_extra_fields_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await assert_request_is_unsupported(
        monkeypatch,
        request_model(unknown_option="must not be forwarded"),
    )


async def test_http_error_is_distinguishable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="provider detail")

    provider = GroqProvider(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GroqHTTPError) as error:
            await provider.complete(request_model())
    finally:
        await provider.aclose()

    assert error.value.status_code == 429
    assert "provider detail" not in str(error.value)


async def test_timeout_is_distinguishable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = GroqProvider(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GroqTimeoutError):
            await provider.complete(request_model())
    finally:
        await provider.aclose()


async def test_connection_failure_is_distinguishable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    provider = GroqProvider(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GroqConnectionError):
            await provider.complete(request_model())
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not-json"),
        httpx.Response(200, json={"id": "incomplete"}),
    ],
)
async def test_malformed_response_is_distinguishable(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)

    async def handler(request: httpx.Request) -> httpx.Response:
        return response

    provider = GroqProvider(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GroqResponseError):
            await provider.complete(request_model())
    finally:
        await provider.aclose()


@pytest.mark.parametrize("api_key", [None, ""])
def test_missing_or_empty_api_key_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
) -> None:
    if api_key is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", api_key)

    with pytest.raises(GroqConfigurationError, match="GROQ_API_KEY is required"):
        GroqProvider(transport=httpx.MockTransport(lambda request: httpx.Response(200)))


class TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=valid_response(), request=request)

    async def aclose(self) -> None:
        self.closed = True


async def test_aclose_closes_owned_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", _FAKE_API_KEY)
    transport = TrackingTransport()
    provider = GroqProvider(transport=transport)

    await provider.aclose()

    assert transport.closed

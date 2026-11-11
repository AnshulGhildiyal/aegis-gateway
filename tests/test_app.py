"""Tests for the FastAPI application and request/response models."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import aegis.api.app as api_module
from aegis.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from aegis.engine.cache_policy import build_cache_key
from aegis.engine.lookup import LookupResult
from aegis.providers.groq import (
    GroqConfigurationError,
    GroqConnectionError,
    GroqHTTPError,
    GroqResponseError,
    GroqTimeoutError,
    GroqUnsupportedRequestError,
)
from aegis.storage.cache import CacheEntry, CacheKey, CacheOperationalError

REQUEST_BODY = {
    "model": "llama-test",
    "messages": [{"role": "user", "content": "Hello"}],
}
CACHEABLE_REQUEST_BODY = {
    **REQUEST_BODY,
    "temperature": 0.1,
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
        cleanup_order: list[str] | None = None,
    ) -> None:
        self.response = response or ChatCompletionResponse.model_validate(RESPONSE_BODY)
        self.error = error
        self.received_request: ChatCompletionRequest | None = None
        self.closed = False
        self.close_calls = 0
        self.cleanup_order = cleanup_order

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
        self.close_calls += 1
        if self.cleanup_order is not None:
            self.cleanup_order.append("provider")


class MockEmbedder:
    def __init__(self, cleanup_order: list[str]) -> None:
        self.cleanup_order = cleanup_order
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        self.cleanup_order.append("embedder")


class MockCache:
    def __init__(
        self,
        embedder: MockEmbedder,
        cleanup_order: list[str],
        *,
        path: str,
        vector_size: int,
        ttl_seconds: float,
    ) -> None:
        self.embedder = embedder
        self.cleanup_order = cleanup_order
        self.path = path
        self.vector_size = vector_size
        self.ttl_seconds = ttl_seconds
        self.closed = False
        self.put_calls: list[tuple[str, dict[str, object], CacheKey]] = []
        self.put_error: CacheOperationalError | None = None

    async def put(
        self,
        *,
        user_content: str,
        response: dict[str, object],
        key: CacheKey,
    ) -> None:
        if self.put_error is not None:
            raise self.put_error
        self.put_calls.append((user_content, response, key))

    def close(self) -> None:
        self.closed = True
        self.cleanup_order.append("cache")


class MockLookup:
    def __init__(
        self,
        cache: MockCache,
        *,
        cosine_threshold: float,
        stage0_enabled: bool,
        nli_enabled: bool,
    ) -> None:
        self.cache = cache
        self.cosine_threshold = cosine_threshold
        self.stage0_enabled = stage0_enabled
        self.nli_enabled = nli_enabled
        self.lookup_calls: list[tuple[str, CacheKey]] = []
        self.result = LookupResult(entry=None)
        self.lookup_error: CacheOperationalError | None = None

    async def lookup(self, user_content: str, key: CacheKey) -> LookupResult:
        if self.lookup_error is not None:
            raise self.lookup_error
        self.lookup_calls.append((user_content, key))
        return self.result


@dataclass
class MockResources:
    provider: MockProvider
    embedder: MockEmbedder
    cache: MockCache
    lookup: MockLookup
    cleanup_order: list[str]


def install_mock_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> MockResources:
    cleanup_order: list[str] = []
    provider = MockProvider(cleanup_order=cleanup_order)
    embedder = MockEmbedder(cleanup_order)
    cache = MockCache(
        embedder,
        cleanup_order,
        path=":memory:",
        vector_size=384,
        ttl_seconds=3600,
    )
    lookup = MockLookup(
        cache,
        cosine_threshold=0.95,
        stage0_enabled=True,
        nli_enabled=False,
    )
    monkeypatch.setattr(api_module, "GroqProvider", lambda: provider)
    monkeypatch.setattr(api_module, "FastEmbedder", lambda: embedder)
    def create_cache(
        cache_embedder: MockEmbedder,
        *,
        path: str,
        vector_size: int,
        ttl_seconds: float,
    ) -> MockCache:
        assert cache_embedder is embedder
        cache.path = path
        cache.vector_size = vector_size
        cache.ttl_seconds = ttl_seconds
        return cache

    def create_lookup(
        lookup_cache: MockCache,
        *,
        cosine_threshold: float,
        stage0_enabled: bool,
        nli_enabled: bool,
    ) -> MockLookup:
        assert lookup_cache is cache
        lookup.cosine_threshold = cosine_threshold
        lookup.stage0_enabled = stage0_enabled
        lookup.nli_enabled = nli_enabled
        return lookup

    monkeypatch.setattr(api_module, "SemanticCache", create_cache)
    monkeypatch.setattr(api_module, "SemanticLookup", create_lookup)
    return MockResources(provider, embedder, cache, lookup, cleanup_order)


def assert_resources_not_published(application: FastAPI) -> None:
    for name in (
        "groq_provider",
        "embedder",
        "semantic_cache",
        "semantic_lookup",
    ):
        assert not hasattr(application.state, name)


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
    install_mock_resources(monkeypatch)
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
    resources = install_mock_resources(monkeypatch)
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
    assert resources.provider.received_request == expected_request


async def test_cache_hit_returns_cached_response_without_calling_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    expected = ChatCompletionResponse.model_validate(RESPONSE_BODY)
    resources.lookup.result = LookupResult(
        entry=CacheEntry(
            user_content="Hello",
            response=expected.model_dump(mode="json"),
            created_at=1.0,
            expires_at=2.0,
            last_accessed_at=1.0,
            similarity=1.0,
        )
    )

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=CACHEABLE_REQUEST_BODY,
            )

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")
    assert resources.lookup.lookup_calls[0][0] == "Hello"
    assert resources.provider.received_request is None
    assert resources.cache.put_calls == []


async def test_cache_miss_calls_provider_and_stores_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=CACHEABLE_REQUEST_BODY,
            )

    assert response.status_code == 200
    assert resources.provider.received_request is not None
    assert resources.provider.received_request.model_fields_set == {
        "model",
        "messages",
        "temperature",
    }
    assert len(resources.provider.received_request.messages) == 1
    assert resources.provider.received_request.messages[0].content == "Hello"
    assert resources.lookup.lookup_calls[0][0] == "Hello"
    assert len(resources.cache.put_calls) == 1
    user_content, cached_response, key = resources.cache.put_calls[0]
    assert user_content == "Hello"
    assert cached_response == resources.provider.response.model_dump(mode="json")
    assert key == build_cache_key(resources.provider.received_request)


async def test_ineligible_request_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)

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
    assert resources.provider.received_request is not None
    assert resources.lookup.lookup_calls == []
    assert resources.cache.put_calls == []


async def test_lookup_operational_failure_falls_back_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    resources.lookup.lookup_error = CacheOperationalError("backend unavailable")

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=CACHEABLE_REQUEST_BODY,
            )

    assert response.status_code == 200
    assert resources.provider.received_request is not None
    assert resources.cache.put_calls == []


async def test_storage_operational_failure_returns_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    resources.cache.put_error = CacheOperationalError("backend unavailable")

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=CACHEABLE_REQUEST_BODY,
            )

    assert response.status_code == 200
    assert response.json() == resources.provider.response.model_dump(mode="json")
    assert resources.provider.received_request is not None
    assert len(resources.cache.put_calls) == 0


async def test_invalid_cached_response_falls_back_to_provider_and_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    resources.lookup.result = LookupResult(
        entry=CacheEntry(
            user_content="Hello",
            response={"id": "invalid"},
            created_at=1.0,
            expires_at=2.0,
            last_accessed_at=1.0,
            similarity=1.0,
        )
    )

    async with api_module.app.router.lifespan_context(api_module.app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_module.app),
            base_url="http://test",
        ) as test_client:
            response = await test_client.post(
                "/v1/chat/completions",
                json=CACHEABLE_REQUEST_BODY,
            )

    assert response.status_code == 200
    assert response.json() == resources.provider.response.model_dump(mode="json")
    assert resources.provider.received_request is not None
    assert len(resources.cache.put_calls) == 1


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
    resources = install_mock_resources(monkeypatch)
    resources.provider.error = provider_error

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
    resources = install_mock_resources(monkeypatch)
    test_app = FastAPI(lifespan=api_module.lifespan)

    async with test_app.router.lifespan_context(test_app):
        assert cast(MockProvider, test_app.state.groq_provider) is resources.provider
        assert cast(MockEmbedder, test_app.state.embedder) is resources.embedder
        assert cast(MockCache, test_app.state.semantic_cache) is resources.cache
        assert cast(MockLookup, test_app.state.semantic_lookup) is resources.lookup
        assert resources.cache.path == ":memory:"
        assert resources.cache.vector_size == 384
        assert resources.cache.ttl_seconds == 3600
        assert resources.lookup.cosine_threshold == 0.95
        assert resources.lookup.stage0_enabled
        assert not resources.lookup.nli_enabled
        assert not resources.provider.closed
        assert not resources.embedder.closed
        assert not resources.cache.closed

    assert resources.cache.closed
    assert resources.embedder.closed
    assert resources.provider.closed
    assert resources.cleanup_order == ["cache", "embedder", "provider"]
    assert resources.provider.close_calls == 1


async def test_fast_embedder_startup_failure_closes_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    error = RuntimeError("embedder initialization failed")

    def fail_embedder() -> MockEmbedder:
        raise error

    monkeypatch.setattr(api_module, "FastEmbedder", fail_embedder)
    test_app = FastAPI(lifespan=api_module.lifespan)

    with pytest.raises(RuntimeError) as raised:
        async with test_app.router.lifespan_context(test_app):
            pytest.fail("lifespan should not start")

    assert raised.value is error
    assert resources.provider.closed
    assert resources.cleanup_order == ["provider"]
    assert_resources_not_published(test_app)


async def test_semantic_cache_startup_failure_closes_embedder_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    error = RuntimeError("cache initialization failed")

    def fail_cache(
        embedder: MockEmbedder,
        **kwargs: object,
    ) -> MockCache:
        raise error

    monkeypatch.setattr(api_module, "SemanticCache", fail_cache)
    test_app = FastAPI(lifespan=api_module.lifespan)

    with pytest.raises(RuntimeError) as raised:
        async with test_app.router.lifespan_context(test_app):
            pytest.fail("lifespan should not start")

    assert raised.value is error
    assert resources.embedder.closed
    assert resources.provider.closed
    assert resources.cleanup_order == ["embedder", "provider"]
    assert_resources_not_published(test_app)


async def test_semantic_lookup_startup_failure_closes_prior_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = install_mock_resources(monkeypatch)
    error = RuntimeError("lookup initialization failed")

    def fail_lookup(
        cache: MockCache,
        **kwargs: object,
    ) -> MockLookup:
        raise error

    monkeypatch.setattr(api_module, "SemanticLookup", fail_lookup)
    test_app = FastAPI(lifespan=api_module.lifespan)

    with pytest.raises(RuntimeError) as raised:
        async with test_app.router.lifespan_context(test_app):
            pytest.fail("lifespan should not start")

    assert raised.value is error
    assert resources.cache.closed
    assert resources.embedder.closed
    assert resources.provider.closed
    assert resources.cleanup_order == ["cache", "embedder", "provider"]
    assert_resources_not_published(test_app)


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

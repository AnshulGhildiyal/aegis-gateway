"""Tests for semantic-cache eligibility and key construction."""

import hashlib
import math

import pytest
from pydantic import ValidationError

from aegis.api.models import ChatCompletionRequest
from aegis.engine.cache_policy import build_cache_key, is_cache_eligible

_BASE_REQUEST: dict[str, object] = {
    "model": "llama-test",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.1,
}


def request_model(**overrides: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({**_BASE_REQUEST, **overrides})


def test_request_with_neutral_explicit_options_is_eligible() -> None:
    request = request_model(
        top_p=1.0,
        n=1,
        stop=None,
        presence_penalty=0.0,
        frequency_penalty=None,
        tools=None,
        tool_choice="auto",
        response_format=None,
    )

    assert is_cache_eligible(request)


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Again"},
        ],
        [{"role": "system", "content": "Hello"}],
        [{"role": "assistant", "content": "Hello"}],
        [{"role": "user"}],
        [{"role": "user", "content": None}],
        [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": "Hello", "name": "named-user"}],
        [{"role": "user", "content": "Hello", "tool_call_id": "call-1"}],
        [{"role": "user", "content": "Hello", "tool_calls": []}],
        [{"role": "user", "content": "Hello", "custom": "value"}],
    ],
)
def test_ineligible_message_shapes_are_rejected(
    messages: list[dict[str, object]],
) -> None:
    assert not is_cache_eligible(request_model(messages=messages))


def test_empty_message_list_is_rejected_by_request_model() -> None:
    with pytest.raises(ValidationError):
        request_model(messages=[])


def test_omitted_temperature_is_ineligible() -> None:
    request = ChatCompletionRequest.model_validate(
        {"model": "llama-test", "messages": [{"role": "user", "content": "Hello"}]}
    )

    assert "temperature" not in request.model_fields_set
    assert request.temperature is None
    assert not is_cache_eligible(request)


def test_explicit_null_temperature_is_ineligible() -> None:
    request = request_model(temperature=None)

    assert "temperature" in request.model_fields_set
    assert not is_cache_eligible(request)


@pytest.mark.parametrize("temperature", [-0.01, 0.21, math.nan, math.inf])
def test_temperature_outside_allowed_range_is_ineligible(
    temperature: float,
) -> None:
    assert not is_cache_eligible(request_model(temperature=temperature))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_p", 0.9),
        ("top_p", None),
        ("n", 2),
        ("n", None),
        ("stream", True),
        ("stop", "END"),
        ("stop", ["END"]),
        ("presence_penalty", 0.1),
        ("frequency_penalty", -0.1),
        ("tools", []),
        ("tools", [{"type": "function"}]),
        ("tool_choice", "none"),
        ("tool_choice", {"type": "function"}),
        ("response_format", {"type": "json_object"}),
        ("seed", 7),
    ],
)
def test_output_affecting_options_outside_cache_key_are_ineligible(
    field: str,
    value: object,
) -> None:
    assert not is_cache_eligible(request_model(**{field: value}))


@pytest.mark.parametrize(
    ("max_tokens", "max_completion_tokens"),
    [(32, 64)],
)
def test_conflicting_explicit_token_limits_are_ineligible(
    max_tokens: int,
    max_completion_tokens: int,
) -> None:
    assert not is_cache_eligible(
        request_model(
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
        )
    )


@pytest.mark.parametrize(
    ("max_tokens", "max_completion_tokens", "expected_effective_value"),
    [
        (None, 64, 64),
        (64, None, 64),
        (64, 64, 64),
    ],
)
def test_non_conflicting_token_limits_are_eligible_and_normalized(
    max_tokens: int | None,
    max_completion_tokens: int | None,
    expected_effective_value: int,
) -> None:
    request = request_model(
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
    )

    assert is_cache_eligible(request)
    key = build_cache_key(request)
    assert key is not None
    assert key.max_tokens == expected_effective_value


@pytest.mark.parametrize(
    ("token_fields", "expected_max_tokens"),
    [
        ({}, None),
        ({"max_tokens": 64}, 64),
        ({"max_completion_tokens": 32}, 32),
        ({"max_tokens": 32, "max_completion_tokens": 32}, 32),
    ],
)
def test_cache_key_fields_and_effective_token_limit(
    token_fields: dict[str, object],
    expected_max_tokens: int | None,
) -> None:
    request = request_model(**token_fields)

    key = build_cache_key(request)

    assert key is not None
    assert key.system_prompt_hash == hashlib.sha256(b"").hexdigest()
    assert key.tenant_id == "local-demo"
    assert key.embedding_model_version == "BAAI/bge-small-en-v1.5"
    assert key.target_llm == "llama-test"
    assert key.temperature == 0.1
    assert key.max_tokens == expected_max_tokens
    assert key.tools is None
    assert key.response_format is None


def test_build_cache_key_returns_none_for_ineligible_request() -> None:
    assert build_cache_key(request_model(temperature=0.3)) is None

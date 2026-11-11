"""Eligibility checks and cache-key construction for chat requests."""

import hashlib

from aegis.api.models import ChatCompletionRequest
from aegis.storage.cache import CacheKey

_TENANT_ID = "local-demo"
_EMBEDDING_MODEL_VERSION = "BAAI/bge-small-en-v1.5"
_SYSTEM_PROMPT_HASH = hashlib.sha256(b"").hexdigest()


def is_cache_eligible(request: ChatCompletionRequest) -> bool:
    """Return whether a request's output-affecting options fit the cache key."""
    fields_set = request.model_fields_set
    if "temperature" not in fields_set or request.temperature is None:
        return False
    if not 0.0 <= request.temperature <= 0.2:
        return False

    if len(request.messages) != 1:
        return False
    message = request.messages[0]
    if message.role != "user":
        return False
    if not isinstance(message.content, str) or not message.content.strip():
        return False
    if (
        message.name is not None
        or message.tool_call_id is not None
        or message.tool_calls is not None
        or message.model_extra
    ):
        return False

    if "top_p" in fields_set and request.top_p != 1.0:
        return False
    if "n" in fields_set and request.n != 1:
        return False
    if request.stream:
        return False
    if request.stop is not None:
        return False
    if request.presence_penalty not in (None, 0.0):
        return False
    if request.frequency_penalty not in (None, 0.0):
        return False
    if request.tools is not None or request.response_format is not None:
        return False
    if request.tool_choice not in (None, "auto"):
        return False
    if request.model_extra:
        return False

    if (
        request.max_tokens is not None
        and request.max_completion_tokens is not None
        and request.max_tokens != request.max_completion_tokens
    ):
        return False

    return True


def build_cache_key(request: ChatCompletionRequest) -> CacheKey | None:
    """Build the strict key for an eligible request, otherwise return ``None``."""
    if not is_cache_eligible(request):
        return None

    temperature = request.temperature
    if temperature is None:
        return None
    max_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )

    return CacheKey(
        system_prompt_hash=_SYSTEM_PROMPT_HASH,
        tenant_id=_TENANT_ID,
        embedding_model_version=_EMBEDDING_MODEL_VERSION,
        target_llm=request.model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=None,
        response_format=None,
    )

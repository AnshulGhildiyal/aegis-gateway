"""Pydantic models for the OpenAI-compatible chat completions API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["system", "developer", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class ChatMessage(BaseModel):
    """A message in a chat completion request or response."""

    model_config = ConfigDict(extra="allow")

    role: ChatRole
    content: str | list[dict[str, object]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, object]] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request fields."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: str | dict[str, object] | None = None
    response_format: dict[str, object] | None = None


class ChatCompletionChoice(BaseModel):
    """One generated answer in a chat completion response."""

    index: int
    message: ChatMessage
    finish_reason: FinishReason | None


class ChatCompletionUsage(BaseModel):
    """Token counts reported by a chat completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible non-streaming chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None

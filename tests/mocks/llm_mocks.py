"""Mock LLM server implementations for testing.

This module provides mock implementations for OpenAI-compatible LLM APIs,
including support for:
- OpenAI API
- Ollama API (OpenAI-compatible mode)
- vLLM (OpenAI-compatible mode)
- Any other OpenAI-compatible backend

The mocks support both synchronous and streaming responses, tool calls,
and error simulation.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, AsyncGenerator, Callable
from unittest.mock import AsyncMock, MagicMock, patch


def create_chat_completion_response(
    content: str,
    role: str = "assistant",
    model: str = "mock-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """Create a mock OpenAI-compatible chat completion response.

    Args:
        content: The response content text
        role: Message role (default: "assistant")
        model: Model name (default: "mock-model")
        prompt_tokens: Number of prompt tokens (default: 10)
        completion_tokens: Number of completion tokens (default: 20)
        finish_reason: Reason for completion (default: "stop")

    Returns:
        Dict matching OpenAI chat completion response format
    """
    return {
        "id": "chatcmpl-mock-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": role,
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_tool_call_response(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_call_id: str = "call_mock_123",
    content: str | None = None,
    model: str = "mock-model",
    prompt_tokens: int = 15,
    completion_tokens: int = 25,
) -> dict[str, Any]:
    """Create a mock response with a tool call.

    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool call
        tool_call_id: Unique ID for the tool call
        content: Optional text content alongside tool call
        model: Model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        Dict matching OpenAI chat completion response with tool calls
    """
    return {
        "id": "chatcmpl-mock-tool-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_multiple_tool_calls_response(
    tool_calls: list[tuple[str, dict[str, Any]]],
    content: str | None = None,
    model: str = "mock-model",
) -> dict[str, Any]:
    """Create a mock response with multiple tool calls.

    Args:
        tool_calls: List of (tool_name, tool_args) tuples
        content: Optional text content
        model: Model name

    Returns:
        Dict matching OpenAI chat completion response with multiple tool calls
    """
    return {
        "id": "chatcmpl-mock-multi-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"call_mock_{i}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                        for i, (name, args) in enumerate(tool_calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": 50,
        },
    }


def create_streaming_chunks(
    content: str,
    chunk_size: int = 5,
    include_role: bool = True,
) -> list[str]:
    """Create SSE streaming chunks for a text response.

    Args:
        content: The full content to stream
        chunk_size: Approximate size of each chunk
        include_role: Whether to include role in first chunk

    Returns:
        List of SSE-formatted strings
    """
    chunks = []

    # Role chunk
    if include_role:
        chunks.append('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')

    # Content chunks
    for i in range(0, len(content), chunk_size):
        chunk_content = content[i : i + chunk_size]
        # Escape special characters for JSON
        chunk_content = chunk_content.replace("\\", "\\\\").replace('"', '\\"')
        chunks.append(f'data: {{"choices":[{{"delta":{{"content":"{chunk_content}"}}}}]}}\n\n')

    # Done marker
    chunks.append("data: [DONE]\n\n")

    return chunks


def create_streaming_tool_call_chunks(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_call_id: str = "call_mock_stream",
    prefix_content: str | None = None,
) -> list[str]:
    """Create SSE streaming chunks for a tool call response.

    Args:
        tool_name: Name of the tool
        tool_args: Tool arguments
        tool_call_id: Tool call ID
        prefix_content: Optional text content before tool call

    Returns:
        List of SSE-formatted strings
    """
    chunks = []

    # Role chunk
    chunks.append('data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')

    # Optional prefix content
    if prefix_content:
        escaped = prefix_content.replace("\\", "\\\\").replace('"', '\\"')
        chunks.append(f'data: {{"choices":[{{"delta":{{"content":"{escaped}"}}}}]}}\n\n')

    # Tool call initialization
    chunks.append(
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"{tool_call_id}",'
        f'"type":"function","function":{{"name":"{tool_name}","arguments":""}}}}]}}}}]}}\n\n'
    )

    # Tool arguments (could be chunked, but for simplicity we send all at once)
    args_json = json.dumps(tool_args).replace('"', '\\"')
    chunks.append(
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,'
        f'"function":{{"arguments":"{args_json}"}}}}]}}}}]}}\n\n'
    )

    # Finish reason
    chunks.append('data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n')

    # Done marker
    chunks.append("data: [DONE]\n\n")

    return chunks


def create_streaming_response(
    content: str | None = None,
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    chunk_size: int = 5,
) -> list[str]:
    """Create a complete streaming response with optional tool calls.

    Args:
        content: Text content to stream
        tool_calls: Optional list of (tool_name, tool_args) for tool calls
        chunk_size: Size of content chunks

    Returns:
        List of SSE-formatted strings
    """
    if tool_calls:
        # For simplicity, only support single tool call in streaming
        tool_name, tool_args = tool_calls[0]
        return create_streaming_tool_call_chunks(
            tool_name, tool_args, prefix_content=content
        )
    elif content:
        return create_streaming_chunks(content, chunk_size)
    else:
        return ['data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n', "data: [DONE]\n\n"]


class MockLLMServer:
    """Mock LLM server that can handle multiple request/response scenarios.

    This class provides a flexible way to configure mock responses for
    different prompts or conversation scenarios.

    Example:
        server = MockLLMServer()
        server.add_response("hello", "Hi there!")
        server.add_tool_call_response(
            "turn on light",
            "ha_control",
            {"action": "turn_on", "entity_id": "light.living_room"}
        )

        with server.patch_aiohttp():
            # Run your test
            pass
    """

    def __init__(self, default_response: str = "I'm a mock assistant."):
        """Initialize the mock server.

        Args:
            default_response: Default response when no match is found
        """
        self.default_response = default_response
        self.responses: list[tuple[Callable[[str], bool], dict[str, Any]]] = []
        self.streaming_responses: list[tuple[Callable[[str], bool], list[str]]] = []
        self.call_history: list[dict[str, Any]] = []
        self._call_count = 0

    def add_response(
        self,
        match: str | Callable[[str], bool],
        response: str | dict[str, Any],
    ) -> "MockLLMServer":
        """Add a response for matching prompts.

        Args:
            match: String to match (substring) or callable predicate
            response: Response string or full response dict

        Returns:
            Self for chaining
        """
        if isinstance(match, str):
            matcher = lambda text, m=match: m.lower() in text.lower()
        else:
            matcher = match

        if isinstance(response, str):
            response_dict = create_chat_completion_response(response)
        else:
            response_dict = response

        self.responses.append((matcher, response_dict))
        return self

    def add_tool_call_response(
        self,
        match: str | Callable[[str], bool],
        tool_name: str,
        tool_args: dict[str, Any],
        content: str | None = None,
    ) -> "MockLLMServer":
        """Add a tool call response for matching prompts.

        Args:
            match: String to match or callable predicate
            tool_name: Name of tool to call
            tool_args: Arguments for the tool
            content: Optional text content

        Returns:
            Self for chaining
        """
        if isinstance(match, str):
            matcher = lambda text, m=match: m.lower() in text.lower()
        else:
            matcher = match

        response_dict = create_tool_call_response(tool_name, tool_args, content=content)
        self.responses.append((matcher, response_dict))
        return self

    def add_streaming_response(
        self,
        match: str | Callable[[str], bool],
        content: str,
        chunk_size: int = 5,
    ) -> "MockLLMServer":
        """Add a streaming response for matching prompts.

        Args:
            match: String to match or callable predicate
            content: Content to stream
            chunk_size: Size of each chunk

        Returns:
            Self for chaining
        """
        if isinstance(match, str):
            matcher = lambda text, m=match: m.lower() in text.lower()
        else:
            matcher = match

        chunks = create_streaming_chunks(content, chunk_size)
        self.streaming_responses.append((matcher, chunks))
        return self

    def add_sequence(self, responses: list[str | dict[str, Any]]) -> "MockLLMServer":
        """Add a sequence of responses to return in order.

        Args:
            responses: List of responses to return in sequence

        Returns:
            Self for chaining
        """
        for i, resp in enumerate(responses):
            # Create a matcher that matches based on call count
            call_index = i

            def make_matcher(idx: int) -> Callable[[str], bool]:
                return lambda text, idx=idx: self._call_count == idx

            if isinstance(resp, str):
                response_dict = create_chat_completion_response(resp)
            else:
                response_dict = resp

            self.responses.append((make_matcher(call_index), response_dict))

        return self

    def get_response(self, messages: list[dict[str, Any]], stream: bool = False) -> Any:
        """Get a response for the given messages.

        Args:
            messages: List of message dicts
            stream: Whether streaming is requested

        Returns:
            Response dict or streaming chunks
        """
        # Extract last user message for matching
        user_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        # Record the call
        self.call_history.append({
            "messages": messages,
            "stream": stream,
            "user_text": user_text,
        })

        # Check streaming responses first if streaming
        if stream:
            for matcher, chunks in self.streaming_responses:
                if matcher(user_text):
                    self._call_count += 1
                    return chunks

        # Check regular responses
        for matcher, response in self.responses:
            if matcher(user_text):
                self._call_count += 1
                if stream:
                    # Convert to streaming format
                    content = response["choices"][0]["message"].get("content", "")
                    if content:
                        return create_streaming_chunks(content)
                    return ['data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n', "data: [DONE]\n\n"]
                return response

        # Return default
        self._call_count += 1
        if stream:
            return create_streaming_chunks(self.default_response)
        return create_chat_completion_response(self.default_response)

    def reset(self) -> None:
        """Reset call history and count."""
        self.call_history.clear()
        self._call_count = 0

    @contextmanager
    def patch_aiohttp(self):
        """Context manager to patch aiohttp with this mock server.

        Yields:
            The mock session for additional assertions
        """
        with mock_aiohttp_session(self.get_response) as mock_session:
            yield mock_session


def mock_aiohttp_session(
    response_handler: Callable[[list[dict[str, Any]], bool], Any],
) -> Any:
    """Create a mock aiohttp session with the given response handler.

    Args:
        response_handler: Function that takes (messages, stream) and returns response

    Returns:
        Context manager that patches aiohttp.ClientSession
    """

    @contextmanager
    def _patch():
        async def create_streaming_iterator(chunks: list[str]) -> AsyncGenerator[bytes, None]:
            for chunk in chunks:
                yield chunk.encode("utf-8")

        def mock_post_side_effect(url: str, **kwargs) -> MagicMock:
            payload = kwargs.get("json", {})
            messages = payload.get("messages", [])
            stream = payload.get("stream", False)

            response_data = response_handler(messages, stream)

            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.raise_for_status = MagicMock()

            if stream and isinstance(response_data, list):
                # Streaming response
                mock_response.content = MagicMock()
                mock_response.content.__aiter__ = lambda self: create_streaming_iterator(
                    response_data
                )
            else:
                # Regular response
                mock_response.json = AsyncMock(return_value=response_data)
                mock_response.text = AsyncMock(return_value=json.dumps(response_data))

            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            return mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post_side_effect)
        mock_session.closed = False
        mock_session.close = AsyncMock()  # Make close() async-compatible
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            yield mock_session

    return _patch()


# Pre-built response templates for common scenarios
RESPONSES = {
    "greeting": create_chat_completion_response(
        "Hello! I'm your home assistant. How can I help you today?"
    ),
    "acknowledgment": create_chat_completion_response(
        "Done! I've completed that action for you."
    ),
    "query_light_on": create_chat_completion_response(
        "The living room light is currently on."
    ),
    "query_light_off": create_chat_completion_response(
        "The living room light is currently off."
    ),
    "query_temperature": create_chat_completion_response(
        "The current temperature is 72°F."
    ),
    "error_response": create_chat_completion_response(
        "I'm sorry, I encountered an error processing your request."
    ),
    "turn_on_light": create_tool_call_response(
        "ha_control",
        {"action": "turn_on", "entity_id": "light.living_room"},
    ),
    "turn_off_light": create_tool_call_response(
        "ha_control",
        {"action": "turn_off", "entity_id": "light.living_room"},
    ),
    "query_entity": create_tool_call_response(
        "ha_query",
        {"entity_id": "light.living_room"},
    ),
}

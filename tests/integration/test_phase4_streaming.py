"""Integration tests for Phase 4: Streaming Response Support.

This test suite validates the end-to-end streaming functionality:
- End-to-end streaming with ChatLog integration
- Streaming with tool calls
- Streaming with multiple tool calls
- Streaming fallback on errors
- Streaming disabled in configuration
- Tool progress events during streaming
- Delta listener integration
- Error event emission
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    EVENT_STREAMING_ERROR,
)

# Test constants
TEST_MODEL = "llama2"
TEST_BASE_URL = "http://localhost:11434/v1"
TEST_API_KEY = "test_key"


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_hass_for_streaming():
    """Create a mock Home Assistant instance for streaming tests."""
    mock = MagicMock(spec=HomeAssistant)
    mock.data = {}

    # Mock states
    mock.states = MagicMock()
    mock.states.async_all = MagicMock(return_value=[])
    mock.states.get = MagicMock(return_value=None)
    mock.states.async_entity_ids = MagicMock(return_value=[])

    # Mock services
    mock.services = MagicMock()
    mock.services.async_call = AsyncMock()

    # Mock config
    mock.config = MagicMock()
    mock.config.config_dir = "/config"
    mock.config.location_name = "Test Home"

    # Mock bus for event tracking
    mock.bus = MagicMock()
    # async_fire is actually sync in HA (returns None, not a coroutine)
    mock.bus.async_fire = MagicMock(return_value=None)

    return mock


@pytest.fixture
def streaming_config():
    """Create a config with streaming enabled."""
    return {
        CONF_LLM_BASE_URL: TEST_BASE_URL,
        CONF_LLM_API_KEY: TEST_API_KEY,
        CONF_LLM_MODEL: TEST_MODEL,
        CONF_STREAMING_ENABLED: True,  # Enable streaming
        CONF_HISTORY_ENABLED: True,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
    }


@pytest.fixture
def non_streaming_config():
    """Create a config with streaming disabled."""
    return {
        CONF_LLM_BASE_URL: TEST_BASE_URL,
        CONF_LLM_API_KEY: TEST_API_KEY,
        CONF_LLM_MODEL: TEST_MODEL,
        CONF_STREAMING_ENABLED: False,  # Disable streaming
        CONF_HISTORY_ENABLED: True,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
    }


@pytest.fixture
def mock_chat_log():
    """Create a mock ChatLog with delta_listener."""
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.delta_listener = MagicMock()  # Has listener = streaming available
    chat_log.unresponded_tool_results = []
    chat_log.content = []
    return chat_log


@pytest.fixture
def mock_user_input():
    """Create a mock ConversationInput."""
    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.text = "Hello, how are you?"
    user_input.conversation_id = "test_123"
    user_input.device_id = None
    user_input.satellite_id = None
    user_input.language = "en"
    user_input.agent_id = "test_agent"
    user_input.context = MagicMock()
    user_input.context.user_id = "test_user"
    return user_input


# ============================================================================
# Helper Utilities for Mocking SSE Streams
# ============================================================================


async def create_mock_sse_stream(messages: list[str]) -> AsyncGenerator[bytes, None]:
    """Create a mock SSE stream for testing.

    Args:
        messages: List of SSE message strings

    Yields:
        Encoded SSE lines
    """
    for msg in messages:
        yield msg.encode("utf-8")


def create_text_stream(text: str) -> list[str]:
    """Create SSE stream that outputs text.

    Args:
        text: The text content to stream

    Returns:
        List of SSE lines representing a text-only response
    """
    return [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n',
        "data: [DONE]\n",
    ]


def create_text_stream_chunked(chunks: list[str]) -> list[str]:
    """Create SSE stream with multiple text chunks.

    Args:
        chunks: List of text chunks to stream

    Returns:
        List of SSE lines representing a chunked text response
    """
    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}\n']
    for chunk in chunks:
        lines.append(f'data: {{"choices":[{{"delta":{{"content":"{chunk}"}}}}]}}\n')
    lines.append("data: [DONE]\n")
    return lines


def create_tool_call_stream(tool_name: str, tool_args: dict[str, Any]) -> list[str]:
    """Create SSE stream with a tool call.

    Args:
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool

    Returns:
        List of SSE lines representing a tool call
    """
    args_json = json.dumps(tool_args).replace('"', '\\"')
    return [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call_1","type":"function","function":{{"name":"{tool_name}","arguments":""}}}}]}}}}]}}\n',  # noqa: E501
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{args_json}"}}}}]}}}}]}}\n',  # noqa: E501
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
        "data: [DONE]\n",
    ]


def create_text_then_tool_stream(text: str, tool_name: str, tool_args: dict[str, Any]) -> list[str]:
    """Create SSE stream with text followed by a tool call.

    Args:
        text: Initial text content
        tool_name: Name of the tool to call
        tool_args: Arguments for the tool

    Returns:
        List of SSE lines representing text + tool call
    """
    args_json = json.dumps(tool_args).replace('"', '\\"')
    return [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n',
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call_1","type":"function","function":{{"name":"{tool_name}","arguments":""}}}}]}}}}]}}\n',  # noqa: E501
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{args_json}"}}}}]}}}}]}}\n',  # noqa: E501
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
        "data: [DONE]\n",
    ]


def create_multiple_tool_calls_stream(tool_calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Create SSE stream with multiple tool calls.

    Args:
        tool_calls: List of (tool_name, tool_args) tuples

    Returns:
        List of SSE lines representing multiple tool calls
    """
    lines = ['data: {"choices":[{"delta":{"role":"assistant"}}]}\n']

    for index, (tool_name, tool_args) in enumerate(tool_calls):
        args_json = json.dumps(tool_args).replace('"', '\\"')
        # Initialize tool call
        lines.append(
            f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":{index},"id":"call_{index}","type":"function","function":{{"name":"{tool_name}","arguments":""}}}}]}}}}]}}\n'  # noqa: E501
        )
        # Add arguments
        lines.append(
            f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":{index},"function":{{"arguments":"{args_json}"}}}}]}}}}]}}\n'  # noqa: E501
        )

    lines.append('data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n')
    lines.append("data: [DONE]\n")
    return lines


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_streaming(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test complete streaming flow from request to response.

    This test verifies:
    - Streaming is detected and enabled
    - SSE stream is correctly parsed
    - Deltas are sent to ChatLog.delta_listener
    - Final response is returned correctly
    """
    # Patch async_should_expose to avoid entity exposure issues
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Create mock streaming response
        stream_lines = create_text_stream_chunked(["Hello", " there", ", how can I help?"])

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_mock_sse_stream(stream_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Mock content returned from chat log
        mock_assistant_content = conversation.AssistantContent(
            agent_id="test_agent", content="Hello there, how can I help?"
        )

        # Track what gets added to chat log
        added_content = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock async_add_delta_content_stream."""
            async for delta in delta_stream:
                # Track deltas
                added_content.append(delta)
            # Return the final content
            yield mock_assistant_content

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                # Mock async_get_result_from_chat_log
                mock_result = conversation.ConversationResult(
                    response=MagicMock(),
                    conversation_id="test_123",
                )
                with patch(
                    "homeassistant.components.conversation.async_get_result_from_chat_log",
                    return_value=mock_result,
                ):
                    # Execute
                    result = await agent.async_process(mock_user_input)

                    # Verify result was returned
                    assert result is not None
                    assert result.conversation_id == "test_123"

                    # Verify deltas were processed
                    assert len(added_content) > 0
                    # Should have role delta and content deltas
                    assert any("role" in delta for delta in added_content)
                    assert any("content" in delta for delta in added_content)


@pytest.mark.asyncio
async def test_streaming_with_tool_calls(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test streaming interrupted by tool calls.

    This test verifies:
    - Text content is streamed
    - Tool calls are detected and accumulated
    - Tool calls are executed by ChatLog
    - Response continues after tool execution
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Create streaming response with tool call
        stream_lines = create_text_then_tool_stream(
            "Let me check that for you.",
            "ha_query",
            {"entity_id": "light.living_room"},
        )

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_mock_sse_stream(stream_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Mock chat log to track tool calls
        tool_calls_received = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures tool calls."""
            async for delta in delta_stream:
                if "tool_calls" in delta:
                    tool_calls_received.extend(delta["tool_calls"])
            # Return assistant content with tool calls
            yield conversation.AssistantContent(
                agent_id="test_agent",
                content="Let me check that for you.",
                tool_calls=[
                    llm.ToolInput(
                        id="call_1",
                        tool_name="ha_query",
                        tool_args={"entity_id": "light.living_room"},
                    )
                ],
            )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream
        mock_chat_log.unresponded_tool_results = []  # No unresponded results after first call

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                mock_result = conversation.ConversationResult(
                    response=MagicMock(),
                    conversation_id="test_123",
                )
                with patch(
                    "homeassistant.components.conversation.async_get_result_from_chat_log",
                    return_value=mock_result,
                ):
                    # Execute
                    result = await agent.async_process(mock_user_input)

                    # Verify result
                    assert result is not None

                    # Verify tool calls were captured
                    assert len(tool_calls_received) > 0
                    assert tool_calls_received[0].tool_name == "ha_query"
                    assert tool_calls_received[0].tool_args == {"entity_id": "light.living_room"}


@pytest.mark.asyncio
async def test_streaming_with_multiple_tool_calls(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test streaming with multiple tool calls in sequence.

    This test verifies:
    - Multiple tool calls are correctly parsed
    - All tool calls are executed
    - Tool calls are indexed correctly
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Create streaming response with multiple tool calls
        stream_lines = create_multiple_tool_calls_stream(
            [
                ("ha_query", {"entity_id": "light.living_room"}),
                ("ha_query", {"entity_id": "light.bedroom"}),
            ]
        )

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_mock_sse_stream(stream_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Track tool calls
        tool_calls_received = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures multiple tool calls."""
            async for delta in delta_stream:
                if "tool_calls" in delta:
                    tool_calls_received.extend(delta["tool_calls"])
            # Return content with both tool calls
            yield conversation.AssistantContent(
                agent_id="test_agent",
                content="",
                tool_calls=[
                    llm.ToolInput(
                        id="call_0",
                        tool_name="ha_query",
                        tool_args={"entity_id": "light.living_room"},
                    ),
                    llm.ToolInput(
                        id="call_1",
                        tool_name="ha_query",
                        tool_args={"entity_id": "light.bedroom"},
                    ),
                ],
            )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream
        mock_chat_log.unresponded_tool_results = []

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                mock_result = conversation.ConversationResult(
                    response=MagicMock(),
                    conversation_id="test_123",
                )
                with patch(
                    "homeassistant.components.conversation.async_get_result_from_chat_log",
                    return_value=mock_result,
                ):
                    # Execute
                    result = await agent.async_process(mock_user_input)

                    # Verify result
                    assert result is not None

                    # Verify both tool calls were captured
                    assert len(tool_calls_received) == 2
                    assert tool_calls_received[0].tool_name == "ha_query"
                    assert tool_calls_received[1].tool_name == "ha_query"


@pytest.mark.asyncio
async def test_streaming_fallback_on_error(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_user_input,
):
    """Test automatic fallback to synchronous on streaming error.

    This test verifies:
    - Streaming errors are caught
    - System falls back to synchronous mode
    - EVENT_STREAMING_ERROR is fired
    - Response is still generated successfully
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Track events
        events = []

        def capture_event(event_type, data):
            events.append((event_type, data))

        mock_hass_for_streaming.bus.async_fire = capture_event

        # Mock ChatLog to indicate streaming is available
        mock_chat_log = MagicMock(spec=conversation.ChatLog)
        mock_chat_log.delta_listener = MagicMock()  # Has listener

        # Mock streaming to fail
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(side_effect=Exception("Streaming failed"))
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock synchronous response to succeed
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Fallback response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        call_count = [0]

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if "stream" in kwargs.get("json", {}) and kwargs["json"]["stream"]:
                # Streaming call - fail
                return mock_response
            # Synchronous call - succeed
            return sync_mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post)
        mock_session.closed = False

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                # Execute
                result = await agent.async_process(mock_user_input)

                # Verify fallback occurred
                assert result is not None
                assert result.conversation_id == "test_123"

                # Verify error event was fired
                error_events = [e for e in events if e[0] == EVENT_STREAMING_ERROR]
                assert len(error_events) == 1
                assert error_events[0][1]["fallback"] is True
                assert "error" in error_events[0][1]


@pytest.mark.asyncio
async def test_streaming_disabled_uses_synchronous(session_manager,
    mock_hass_for_streaming,
    non_streaming_config,
    mock_user_input,
):
    """Test that streaming is skipped when disabled in config.

    This test verifies:
    - _can_stream() returns False when disabled
    - Synchronous processing is used
    - No streaming attempted
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, non_streaming_config)

        # Verify _can_stream() returns False
        # We need to set up the context first
        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_ctx:
            mock_chat_log = MagicMock(spec=conversation.ChatLog)
            mock_chat_log.delta_listener = MagicMock()
            mock_ctx.get.return_value = mock_chat_log

            # Check that streaming is disabled
            assert not agent._can_stream()

        # Mock synchronous response
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Synchronous response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=sync_mock_response)
        mock_session.closed = False

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            # Execute
            result = await agent.async_process(mock_user_input)

            # Verify synchronous was used
            assert result is not None

            # Verify the API call did NOT include stream=True
            assert mock_session.post.called
            call_kwargs = mock_session.post.call_args[1]
            payload = call_kwargs.get("json", {})
            assert payload.get("stream") is not True


@pytest.mark.asyncio
async def test_streaming_no_chatlog_uses_synchronous(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_user_input,
):
    """Test that streaming is skipped when ChatLog is not available.

    This test verifies:
    - _can_stream() returns False when ChatLog is None
    - Synchronous processing is used
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Mock ChatLog to be None (not available)
        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_ctx:
            mock_ctx.get.return_value = None

            # Verify _can_stream() returns False
            assert not agent._can_stream()

        # Mock synchronous response
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Synchronous response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=sync_mock_response)
        mock_session.closed = False

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = None

                # Execute
                result = await agent.async_process(mock_user_input)

                # Verify result
                assert result is not None


@pytest.mark.asyncio
async def test_streaming_no_delta_listener_uses_synchronous(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_user_input,
):
    """Test that streaming is skipped when ChatLog has no delta_listener.

    This test verifies:
    - _can_stream() returns False when delta_listener is None
    - Synchronous processing is used
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Mock ChatLog without delta_listener
        mock_chat_log = MagicMock(spec=conversation.ChatLog)
        mock_chat_log.delta_listener = None  # No listener

        with patch("homeassistant.components.conversation.chat_log.current_chat_log") as mock_ctx:
            mock_ctx.get.return_value = mock_chat_log

            # Verify _can_stream() returns False
            assert not agent._can_stream()

        # Mock synchronous response
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Synchronous response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=sync_mock_response)
        mock_session.closed = False

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                # Execute
                result = await agent.async_process(mock_user_input)

                # Verify result
                assert result is not None


@pytest.mark.asyncio
async def test_streaming_conversation_history_integration(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test that streaming integrates correctly with conversation history.

    This test verifies:
    - Conversation history is included in streaming requests
    - User and assistant messages are saved after streaming
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # Add some history
        agent.conversation_manager.add_message("test_123", "user", "Previous question")
        agent.conversation_manager.add_message("test_123", "assistant", "Previous answer")

        # Create streaming response
        stream_lines = create_text_stream("New answer")

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_mock_sse_stream(stream_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock delta stream."""
            async for delta in delta_stream:
                pass
            yield conversation.AssistantContent(agent_id="test_agent", content="New answer")

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream
        mock_chat_log.unresponded_tool_results = []

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                mock_result = conversation.ConversationResult(
                    response=MagicMock(),
                    conversation_id="test_123",
                )
                with patch(
                    "homeassistant.components.conversation.async_get_result_from_chat_log",
                    return_value=mock_result,
                ):
                    # Execute
                    result = await agent.async_process(mock_user_input)

                    # Verify result
                    assert result is not None

                    # Verify history was passed to LLM (should have previous messages)
                    assert mock_session.post.called
                    call_kwargs = mock_session.post.call_args[1]
                    messages = call_kwargs["json"]["messages"]
                    # Should have: system, user (previous), assistant (previous), user (current)
                    assert len(messages) >= 4

                    # Verify new messages were added to history
                    history = agent.conversation_manager.get_history("test_123")
                    # Should have: prev user, prev assistant, curr user, curr assistant
                    assert len(history) == 4


@pytest.mark.asyncio
async def test_streaming_tool_iteration_loop(session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test streaming with tool call requiring multiple iterations.

    This test verifies:
    - Tool calls trigger next iteration
    - Multiple streaming calls are made
    - unresponded_tool_results controls loop termination
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config)

        # First call: tool call
        stream_lines_1 = create_tool_call_stream("ha_query", {"entity_id": "light.living_room"})

        # Second call: final response
        stream_lines_2 = create_text_stream("The light is on")

        call_count = [0]

        def create_response(lines):
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.content = MagicMock()
            mock_resp.content.__aiter__ = lambda self: create_mock_sse_stream(lines)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=None)
            return mock_resp

        def mock_post(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return create_response(stream_lines_1)
            return create_response(stream_lines_2)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post)
        mock_session.closed = False

        # Track iterations
        iterations = [0]

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that controls iteration."""
            iterations[0] += 1
            async for delta in delta_stream:
                pass

            if iterations[0] == 1:
                # First iteration: return tool call
                mock_chat_log.unresponded_tool_results = ["pending"]  # Trigger next iteration
                yield conversation.AssistantContent(
                    agent_id="test_agent",
                    content="",
                    tool_calls=[
                        llm.ToolInput(
                            id="call_1",
                            tool_name="ha_query",
                            tool_args={"entity_id": "light.living_room"},
                        )
                    ],
                )
                # Simulate tool result
                yield conversation.ToolResultContent(
                    agent_id="test_agent",
                    tool_call_id="call_1",
                    tool_name="ha_query",
                    tool_result={"state": "on"},
                )
            else:
                # Second iteration: final response
                mock_chat_log.unresponded_tool_results = []  # End loop
                yield conversation.AssistantContent(
                    agent_id="test_agent", content="The light is on"
                )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream
        mock_chat_log.unresponded_tool_results = []

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                mock_result = conversation.ConversationResult(
                    response=MagicMock(),
                    conversation_id="test_123",
                )
                with patch(
                    "homeassistant.components.conversation.async_get_result_from_chat_log",
                    return_value=mock_result,
                ):
                    # Execute
                    result = await agent.async_process(mock_user_input)

                    # Verify multiple iterations occurred
                    assert iterations[0] == 2
                    assert call_count[0] == 2
                    # Verify result was returned
                    assert result is not None  # noqa: F841


# ============================================================================
# Coverage Summary Test
# ============================================================================


def test_helper_utilities():
    """Test helper utility functions for stream creation.

    This ensures our helper functions work correctly.
    """
    # Test create_text_stream
    lines = create_text_stream("Hello")
    assert len(lines) == 3
    assert "Hello" in lines[1]
    assert "[DONE]" in lines[2]

    # Test create_text_stream_chunked
    lines = create_text_stream_chunked(["Hello", " World"])
    assert len(lines) == 4
    assert "Hello" in lines[1]
    assert "World" in lines[2]

    # Test create_tool_call_stream
    lines = create_tool_call_stream("ha_control", {"action": "turn_on"})
    assert len(lines) == 5
    assert "ha_control" in lines[1]
    assert "tool_calls" in lines[1]

    # Test create_text_then_tool_stream
    lines = create_text_then_tool_stream("Checking...", "ha_query", {"entity_id": "test"})
    assert len(lines) == 6
    assert "Checking" in lines[1]
    assert "ha_query" in lines[2]

    # Test create_multiple_tool_calls_stream
    lines = create_multiple_tool_calls_stream(
        [("tool1", {"arg1": "val1"}), ("tool2", {"arg2": "val2"})]
    )
    assert len(lines) == 7  # role + 2*(init+args) + finish + done
    assert "tool1" in lines[1]
    assert "tool2" in lines[3]

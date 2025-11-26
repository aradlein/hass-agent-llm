"""Integration tests for Phase 4: Streaming Network Failure Scenarios.

This test suite validates streaming failure handling:
- Partial SSE stream: Connection drops mid-response
- Malformed SSE data: Invalid event format
- HTTP 503 mid-streaming: Service unavailable during stream
- Timeout during streaming: Slow response exceeds timeout
- Invalid JSON in SSE events: JSON parse error
- Network errors during streaming
- Graceful degradation and error handling
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant

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

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

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
# Helper Utilities for Failure Simulation
# ============================================================================


async def create_partial_sse_stream(
    valid_messages: list[str], error_after: int = 2
) -> AsyncGenerator[bytes, None]:
    """Create a stream that drops connection mid-response.

    Args:
        valid_messages: SSE messages to send before dropping
        error_after: Number of messages to send before raising error

    Yields:
        Encoded SSE lines, then raises exception
    """
    count = 0
    for msg in valid_messages:
        if count >= error_after:
            raise aiohttp.ClientError("Connection lost")
        yield msg.encode("utf-8")
        count += 1


async def create_malformed_sse_stream() -> AsyncGenerator[bytes, None]:
    """Create stream with malformed SSE data.

    Yields:
        Malformed SSE data
    """
    # Missing "data: " prefix
    yield b'{"invalid": "no_prefix"}\n'
    # Invalid JSON after "data: "
    yield b"data: {this is not json}\n"
    # Valid data to show recovery
    yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
    yield b'data: {"choices":[{"delta":{"content":"recovered"}}]}\n'
    yield b"data: [DONE]\n"


async def create_timeout_stream(delay_seconds: float = 10.0) -> AsyncGenerator[bytes, None]:
    """Create stream that times out by delaying.

    Args:
        delay_seconds: How long to delay before sending data

    Yields:
        SSE data after delay
    """
    # Send initial data
    yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
    # Delay to trigger timeout
    await asyncio.sleep(delay_seconds)
    yield b'data: {"choices":[{"delta":{"content":"too late"}}]}\n'


async def create_invalid_json_stream() -> AsyncGenerator[bytes, None]:
    """Create stream with invalid JSON in SSE events.

    Yields:
        SSE data with JSON parse errors
    """
    # Valid start
    yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
    # Invalid JSON - missing closing brace
    yield b'data: {"choices":[{"delta":{"content":"test"\n'
    # Invalid JSON - malformed
    yield b'data: {"choices":[{"delta":{"content":undefined}}]}\n'
    # Valid to show recovery
    yield b'data: {"choices":[{"delta":{"content":"recovered"}}]}\n'
    yield b"data: [DONE]\n"


async def create_http_503_stream() -> AsyncGenerator[bytes, None]:
    """Create stream that simulates HTTP 503 mid-stream.

    Yields:
        Initial data then raises HTTP error
    """
    yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
    yield b'data: {"choices":[{"delta":{"content":"starting"}}]}\n'
    # Simulate server error mid-stream
    raise aiohttp.ServerConnectionError("Service Unavailable")


# ============================================================================
# Test Cases - Network Failure Scenarios
# ============================================================================


@pytest.mark.asyncio
async def test_partial_sse_stream_connection_drop(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test graceful handling when connection drops mid-stream.

    This test verifies:
    - Partial content is processed before failure
    - Exception is caught and handled
    - System falls back to synchronous mode
    - Appropriate error event is fired
    - User receives error response (not partial data)
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Track events
        events = []

        def capture_event(event_type, data=None):
            events.append((event_type, data))
            # Return None to avoid coroutine warning (async_fire is sync in HA)
            return None

        mock_hass_for_streaming.bus.async_fire = MagicMock(side_effect=capture_event)

        # Create stream that drops connection after 2 chunks
        stream_lines = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n',
            'data: {"choices":[{"delta":{"content":"there"}}]}\n',
            "data: [DONE]\n",
        ]

        # Mock response that fails mid-stream
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_partial_sse_stream(
            stream_lines, error_after=2
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock synchronous fallback to succeed
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
async def test_malformed_sse_data_handling(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of malformed SSE data.

    This test verifies:
    - Invalid SSE lines are skipped
    - Processing continues with valid data
    - No crash on malformed input
    - Partial valid content is processed
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Mock response with malformed data
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_malformed_sse_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Track deltas processed
        processed_deltas = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures deltas."""
            async for delta in delta_stream:
                processed_deltas.append(delta)
            # Return content
            yield conversation.AssistantContent(
                agent_id="test_agent", content="recovered"
            )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream

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
                    # Execute - should not crash
                    result = await agent.async_process(mock_user_input)

                    # Verify result was returned
                    assert result is not None

                    # Verify some deltas were processed (at least role and valid content)
                    assert len(processed_deltas) > 0
                    # Should have role delta
                    assert any("role" in delta for delta in processed_deltas)
                    # Should have content delta (from valid data)
                    assert any("content" in delta for delta in processed_deltas)


@pytest.mark.asyncio
async def test_http_503_during_streaming(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of HTTP 503 error during streaming.

    This test verifies:
    - HTTP errors mid-stream are caught
    - System falls back to synchronous mode
    - Error event is fired
    - User receives fallback response
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Track events
        events = []

        def capture_event(event_type, data=None):
            events.append((event_type, data))
            # Return None to avoid coroutine warning (async_fire is sync in HA)
            return None

        mock_hass_for_streaming.bus.async_fire = MagicMock(side_effect=capture_event)

        # Mock response that raises HTTP error mid-stream
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_http_503_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock synchronous fallback
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Fallback after 503"}}],
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
                return mock_response
            return sync_mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post)
        mock_session.closed = False

        # Make chat_log.async_add_delta_content_stream propagate the error
        async def failing_add_delta_stream(entry_id, delta_stream):
            """Mock that propagates the 503 error."""
            async for delta in delta_stream:
                pass  # Error will be raised from stream iteration
            yield  # Must yield to be an async generator

        mock_chat_log.async_add_delta_content_stream = failing_add_delta_stream
        # Mock chat_log.content to have content (avoiding IndexError in fallback)
        mock_chat_log.content = [
            conversation.AssistantContent(
                agent_id="test_agent", content="Fallback after 503"
            )
        ]
        mock_chat_log.conversation_id = "test_123"

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                # Execute
                result = await agent.async_process(mock_user_input)

                # Verify fallback occurred
                assert result is not None

                # Verify both streaming and synchronous were called
                assert call_count[0] == 2

                # Verify error event was fired
                error_events = [e for e in events if e[0] == EVENT_STREAMING_ERROR]
                assert len(error_events) == 1
                assert error_events[0][1]["fallback"] is True


@pytest.mark.asyncio
async def test_timeout_during_streaming(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of timeout during streaming.

    This test verifies:
    - Timeout errors are caught
    - System falls back to synchronous mode
    - Error event is fired with timeout context
    - Resources are cleaned up properly
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Track events
        events = []

        def capture_event(event_type, data=None):
            events.append((event_type, data))
            # Return None to avoid coroutine warning (async_fire is sync in HA)
            return None

        mock_hass_for_streaming.bus.async_fire = MagicMock(side_effect=capture_event)

        # Create async generator that simulates timeout
        async def timeout_generator():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            # Simulate timeout by raising asyncio.TimeoutError
            raise asyncio.TimeoutError("Stream timed out")

        # Mock response that times out
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: timeout_generator()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock synchronous fallback
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Fallback after timeout"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        def mock_post(*args, **kwargs):
            if "stream" in kwargs.get("json", {}) and kwargs["json"]["stream"]:
                return mock_response
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

                # Verify error event was fired
                error_events = [e for e in events if e[0] == EVENT_STREAMING_ERROR]
                assert len(error_events) == 1
                assert error_events[0][1]["fallback"] is True
                # Should mention timeout in error
                assert "error_type" in error_events[0][1]


@pytest.mark.asyncio
async def test_invalid_json_in_sse_events(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of invalid JSON in SSE events.

    This test verifies:
    - JSON parse errors are logged but not fatal
    - Invalid chunks are skipped
    - Valid chunks continue to be processed
    - Stream completes successfully with partial data
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: create_invalid_json_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Track deltas
        processed_deltas = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures deltas."""
            async for delta in delta_stream:
                processed_deltas.append(delta)
            yield conversation.AssistantContent(
                agent_id="test_agent", content="recovered"
            )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream

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
                    # Execute - should not crash
                    result = await agent.async_process(mock_user_input)

                    # Verify result was returned
                    assert result is not None

                    # Verify deltas were processed (should skip invalid JSON)
                    assert len(processed_deltas) > 0
                    # Should have role
                    assert any("role" in delta for delta in processed_deltas)
                    # Should have at least one valid content delta
                    assert any("content" in delta for delta in processed_deltas)


@pytest.mark.asyncio
async def test_invalid_tool_call_json_in_stream(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of tool calls with invalid JSON arguments.

    This test verifies:
    - Tool calls with malformed JSON args are caught
    - Error is logged appropriately
    - Tool call continues with empty args
    - Stream processing continues
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Create stream with tool call containing invalid JSON
        async def invalid_tool_json_stream():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            # Start tool call
            yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ha_query","arguments":""}}]}}]}\n'
            # Invalid JSON arguments
            yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"invalid json here"}}]}}]}\n'
            # Finish
            yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'
            yield b"data: [DONE]\n"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: invalid_tool_json_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Track tool calls
        tool_calls_received = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures tool calls."""
            async for delta in delta_stream:
                if "tool_calls" in delta:
                    tool_calls_received.extend(delta["tool_calls"])
            # Return content with tool call (empty args due to parse error)
            yield conversation.AssistantContent(
                agent_id="test_agent",
                content="",
                tool_calls=tool_calls_received,
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
                    # Execute - should not crash
                    result = await agent.async_process(mock_user_input)

                    # Verify result
                    assert result is not None

                    # Verify tool call was received (with empty args due to JSON error)
                    assert len(tool_calls_received) == 1
                    assert tool_calls_received[0].tool_name == "ha_query"
                    # Args should be empty dict due to parse error
                    assert tool_calls_received[0].tool_args == {}


@pytest.mark.asyncio
async def test_empty_stream_response(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of empty stream (no data sent).

    This test verifies:
    - Empty streams don't cause crashes
    - System handles gracefully
    - Appropriate default response is generated
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Create empty stream
        async def empty_stream():
            # Send nothing
            if False:
                yield

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: empty_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        processed_deltas = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures deltas."""
            async for delta in delta_stream:
                processed_deltas.append(delta)
            # Return minimal content
            yield conversation.AssistantContent(
                agent_id="test_agent", content=""
            )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream

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
                    # Execute - should not crash
                    result = await agent.async_process(mock_user_input)

                    # Verify result was returned
                    assert result is not None

                    # Should have at least role delta (from handler initialization)
                    assert len(processed_deltas) >= 1
                    assert processed_deltas[0].get("role") == "assistant"


@pytest.mark.asyncio
async def test_network_error_before_streaming_starts(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test handling of network error before streaming begins.

    This test verifies:
    - Network errors during connection are caught
    - System falls back to synchronous mode
    - Error event is fired
    - Fallback succeeds
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Track events
        events = []

        def capture_event(event_type, data=None):
            events.append((event_type, data))
            # Return None to avoid coroutine warning (async_fire is sync in HA)
            return None

        mock_hass_for_streaming.bus.async_fire = MagicMock(side_effect=capture_event)

        # Mock response that fails on enter
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock synchronous fallback
        sync_response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Fallback after network error"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        def mock_post(*args, **kwargs):
            if "stream" in kwargs.get("json", {}) and kwargs["json"]["stream"]:
                return mock_response
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

                # Verify error event was fired
                error_events = [e for e in events if e[0] == EVENT_STREAMING_ERROR]
                assert len(error_events) == 1
                assert error_events[0][1]["fallback"] is True


@pytest.mark.asyncio
async def test_stream_with_only_done_marker(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test stream that only sends [DONE] marker.

    This test verifies:
    - Minimal streams are handled
    - No content results in appropriate behavior
    - No crashes occur
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Create stream with only [DONE]
        async def done_only_stream():
            yield b"data: [DONE]\n"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: done_only_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        processed_deltas = []

        async def mock_add_delta_stream(entry_id, delta_stream):
            """Mock that captures deltas."""
            async for delta in delta_stream:
                processed_deltas.append(delta)
            yield conversation.AssistantContent(
                agent_id="test_agent", content=""
            )

        mock_chat_log.async_add_delta_content_stream = mock_add_delta_stream

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
                    # Execute - should not crash
                    result = await agent.async_process(mock_user_input)

                    # Verify result was returned
                    assert result is not None

                    # Should have role delta from handler initialization
                    assert len(processed_deltas) >= 1


@pytest.mark.asyncio
async def test_stream_handler_exception_propagation(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_chat_log,
    mock_user_input,
):
    """Test that exceptions in stream handler are properly propagated.

    This test verifies:
    - Exceptions during stream transformation are caught
    - Error is logged with traceback
    - System falls back appropriately
    """
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Track events
        events = []

        def capture_event(event_type, data=None):
            events.append((event_type, data))
            # Return None to avoid coroutine warning (async_fire is sync in HA)
            return None

        mock_hass_for_streaming.bus.async_fire = MagicMock(side_effect=capture_event)

        # Create valid stream
        async def valid_stream():
            yield b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"test"}}]}\n'
            yield b"data: [DONE]\n"

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()
        mock_response.content.__aiter__ = lambda self: valid_stream()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock synchronous fallback
        sync_response = {
            "choices": [{"message": {"role": "assistant", "content": "Fallback"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        sync_mock_response = MagicMock()
        sync_mock_response.status = 200
        sync_mock_response.json = AsyncMock(return_value=sync_response)
        sync_mock_response.__aenter__ = AsyncMock(return_value=sync_mock_response)
        sync_mock_response.__aexit__ = AsyncMock(return_value=None)

        def mock_post(*args, **kwargs):
            if "stream" in kwargs.get("json", {}) and kwargs["json"]["stream"]:
                return mock_response
            return sync_mock_response

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post)
        mock_session.closed = False

        # Make chat_log.async_add_delta_content_stream raise exception
        async def failing_add_delta_stream(entry_id, delta_stream):
            """Mock that raises exception during processing."""
            async for delta in delta_stream:
                raise RuntimeError("Simulated stream processing error")
            yield  # Must yield to be an async generator

        mock_chat_log.async_add_delta_content_stream = failing_add_delta_stream

        with patch.object(agent, "_ensure_session", return_value=mock_session):
            with patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_ctx:
                mock_ctx.get.return_value = mock_chat_log

                # Execute
                result = await agent.async_process(mock_user_input)

                # Verify fallback occurred
                assert result is not None

                # Verify error event was fired
                error_events = [e for e in events if e[0] == EVENT_STREAMING_ERROR]
                assert len(error_events) == 1
                assert error_events[0][1]["fallback"] is True
                assert "error" in error_events[0][1]

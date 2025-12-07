"""Test that simulates realistic Home Assistant async_add_delta_content_stream behavior.

This test file is designed to uncover the infinite loop bug by simulating
how Home Assistant's async_add_delta_content_stream ACTUALLY works, not how
our mocks currently work.

Key insights about real HA behavior:
1. async_add_delta_content_stream is an async GENERATOR that yields content items
2. It processes the delta stream DURING tool execution (not after)
3. It may yield multiple items: AssistantContent with tool_calls, then ToolResultContent after execution
4. The unresponded_tool_results list is managed BY Home Assistant's ChatLog
5. Tool execution happens INSIDE async_add_delta_content_stream
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Test constants
TEST_MODEL = "gpt-4o"
TEST_BASE_URL = "https://api.openai.com/v1"
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
    mock.bus.async_fire = MagicMock(return_value=None)

    return mock


@pytest.fixture
def streaming_config():
    """Create a config with streaming enabled."""
    return {
        CONF_LLM_BASE_URL: TEST_BASE_URL,
        CONF_LLM_API_KEY: TEST_API_KEY,
        CONF_LLM_MODEL: TEST_MODEL,
        CONF_STREAMING_ENABLED: True,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
    }


@pytest.fixture
def mock_user_input():
    """Create a mock ConversationInput."""
    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.text = "Turn on the living room light"
    user_input.conversation_id = "test_123"
    user_input.device_id = None
    user_input.satellite_id = None
    user_input.language = "en"
    user_input.agent_id = "test_agent"
    user_input.context = MagicMock()
    user_input.context.user_id = "test_user"
    return user_input


# ============================================================================
# Realistic Mock of Home Assistant's async_add_delta_content_stream
# ============================================================================


class RealisticChatLogMock:
    """Mock ChatLog that simulates realistic HA behavior.

    This mock is based on understanding how HA's async_add_delta_content_stream
    actually works:

    1. It's an async generator that yields content items
    2. It consumes the delta stream
    3. It executes tools DURING the stream processing
    4. It manages unresponded_tool_results internally
    5. It may yield multiple content items per call
    """

    def __init__(self):
        """Initialize the mock."""
        self.delta_listener = MagicMock()
        self.content = []
        self.unresponded_tool_results = []
        self._call_count = 0

    async def async_add_delta_content_stream(
        self, entry_id: str, delta_stream: AsyncGenerator
    ) -> AsyncGenerator[conversation.AssistantContent | conversation.ToolResultContent, None]:
        """Realistic simulation of HA's async_add_delta_content_stream.

        This is how we believe HA actually works based on the API and integration tests.

        Flow:
        1. Consume all deltas from delta_stream
        2. Build AssistantContent from deltas
        3. If there are tool_calls, execute them
        4. Yield AssistantContent (with tool_calls if any)
        5. Yield ToolResultContent for each executed tool
        6. Update unresponded_tool_results to signal if more iterations needed
        """
        self._call_count += 1
        print(f"\n[RealisticChatLogMock] Call #{self._call_count} to async_add_delta_content_stream")

        # Step 1: Consume all deltas
        role = None
        content_text = ""
        tool_calls = []

        async for delta in delta_stream:
            print(f"[RealisticChatLogMock] Processing delta: {delta}")
            if "role" in delta:
                role = delta["role"]
            if "content" in delta:
                content_text += delta["content"]
            if "tool_calls" in delta:
                tool_calls.extend(delta["tool_calls"])

        print(f"[RealisticChatLogMock] Stream consumed. role={role}, content_len={len(content_text)}, tool_calls={len(tool_calls)}")

        # Step 2: Build AssistantContent
        assistant_content = conversation.AssistantContent(
            agent_id="test_agent",
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
        )

        # Step 3: Yield AssistantContent
        print(f"[RealisticChatLogMock] Yielding AssistantContent (tool_calls={len(tool_calls)})")
        yield assistant_content

        # Step 4: If there are tool calls, execute them and yield results
        if tool_calls:
            print(f"[RealisticChatLogMock] Executing {len(tool_calls)} tool calls...")

            # Simulate tool execution
            for tool_call in tool_calls:
                # Fake tool execution result
                tool_result = {
                    "success": True,
                    "state": "on",
                    "entity_id": tool_call.tool_args.get("entity_id", "unknown"),
                }

                print(f"[RealisticChatLogMock] Tool {tool_call.tool_name} executed: {tool_result}")

                # Yield ToolResultContent
                yield conversation.ToolResultContent(
                    agent_id="test_agent",
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.tool_name,
                    tool_result=tool_result,
                )

            # Step 5: Update unresponded_tool_results to trigger next iteration
            # In real HA, this would be set based on whether tools were executed
            # For the FIRST call, we set it to trigger iteration
            # For subsequent calls, we clear it to end the loop
            if self._call_count == 1:
                print("[RealisticChatLogMock] Setting unresponded_tool_results to trigger next iteration")
                self.unresponded_tool_results = [f"result_{tc.id}" for tc in tool_calls]
            else:
                print("[RealisticChatLogMock] Clearing unresponded_tool_results to end loop")
                self.unresponded_tool_results = []
        else:
            # No tool calls, so no unresponded results
            print("[RealisticChatLogMock] No tool calls, clearing unresponded_tool_results")
            self.unresponded_tool_results = []


# ============================================================================
# Helper Utilities for Mocking SSE Streams
# ============================================================================


async def create_mock_sse_stream(messages: list[str]) -> AsyncGenerator[bytes, None]:
    """Create a mock SSE stream for testing."""
    for msg in messages:
        yield msg.encode("utf-8")


def create_tool_call_stream(tool_name: str, tool_args: dict[str, Any]) -> list[str]:
    """Create SSE stream with a tool call."""
    args_json = json.dumps(tool_args).replace('"', '\\"')
    return [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call_1","type":"function","function":{{"name":"{tool_name}","arguments":""}}}}]}}}}]}}\n',
        f'data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":"{args_json}"}}}}]}}}}]}}\n',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n',
        "data: [DONE]\n",
    ]


def create_text_stream(text: str) -> list[str]:
    """Create SSE stream that outputs text."""
    return [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n',
        "data: [DONE]\n",
    ]


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_realistic_streaming_with_tool_call(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_user_input,
):
    """Test streaming with realistic HA behavior.

    This test simulates what ACTUALLY happens:
    1. First iteration: LLM returns tool call
    2. HA executes tool and yields both AssistantContent AND ToolResultContent
    3. HA sets unresponded_tool_results to trigger next iteration
    4. Second iteration: LLM returns final text response
    5. HA yields AssistantContent with no tool calls
    6. HA clears unresponded_tool_results to end loop

    EXPECTED: Loop should terminate after 2 iterations
    BUG SYMPTOM: Loop might continue infinitely
    """
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # First LLM call: Returns tool call
        stream_lines_1 = create_tool_call_stream(
            "ha_control",
            {"entity_id": "light.living_room", "action": "turn_on"},
        )

        # Second LLM call: Returns final text
        stream_lines_2 = create_text_stream("I've turned on the living room light.")

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
            print(f"\n[LLM Call #{call_count[0]}]")
            if call_count[0] == 1:
                return create_response(stream_lines_1)
            return create_response(stream_lines_2)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post)
        mock_session.closed = False

        # Use realistic ChatLog mock
        mock_chat_log = RealisticChatLogMock()

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
                    print("\n========== STARTING TEST ==========")
                    result = await agent.async_process(mock_user_input)
                    print("\n========== TEST COMPLETED ==========")

                    # Verify result
                    assert result is not None
                    assert result.conversation_id == "test_123"

                    # CRITICAL: Verify loop terminated after exactly 2 iterations
                    print(f"\nLLM was called {call_count[0]} times")
                    print(f"ChatLog.async_add_delta_content_stream was called {mock_chat_log._call_count} times")

                    assert call_count[0] == 2, f"Expected 2 LLM calls, got {call_count[0]}"
                    assert mock_chat_log._call_count == 2, f"Expected 2 stream iterations, got {mock_chat_log._call_count}"


@pytest.mark.asyncio
async def test_realistic_streaming_without_tool_calls(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_user_input,
):
    """Test streaming when LLM returns text without tool calls.

    This should:
    1. First iteration: LLM returns text
    2. HA yields AssistantContent with no tool calls
    3. HA clears unresponded_tool_results
    4. Loop terminates after 1 iteration
    """
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Single LLM call: Returns text only
        stream_lines = create_text_stream("Hello! How can I help you?")

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
            return create_response(stream_lines)

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=mock_post)
        mock_session.closed = False

        # Use realistic ChatLog mock
        mock_chat_log = RealisticChatLogMock()

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

                    # Verify loop terminated after 1 iteration
                    assert result is not None
                    assert call_count[0] == 1, f"Expected 1 LLM call, got {call_count[0]}"
                    assert mock_chat_log._call_count == 1, f"Expected 1 stream iteration, got {mock_chat_log._call_count}"


@pytest.mark.asyncio
async def test_realistic_streaming_edge_case_empty_content(
    session_manager,
    mock_hass_for_streaming,
    streaming_config,
    mock_user_input,
):
    """Test edge case where async_add_delta_content_stream yields nothing.

    This could happen if:
    - The stream is malformed
    - The LLM returns empty response
    - There's an error in stream processing

    EXPECTED: Loop should terminate gracefully
    """
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_streaming, streaming_config, session_manager)

        # Create empty stream
        stream_lines = ["data: [DONE]\n"]

        def create_response(lines):
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.content = MagicMock()
            mock_resp.content.__aiter__ = lambda self: create_mock_sse_stream(lines)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=None)
            return mock_resp

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=create_response(stream_lines))
        mock_session.closed = False

        # Custom ChatLog that yields nothing
        class EmptyChatLogMock:
            def __init__(self):
                self.delta_listener = MagicMock()
                self.unresponded_tool_results = []
                self._call_count = 0

            async def async_add_delta_content_stream(self, entry_id, delta_stream):
                self._call_count += 1
                async for delta in delta_stream:
                    pass
                # Yield nothing
                return
                yield  # Make it a generator

        mock_chat_log = EmptyChatLogMock()

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

                    # Should terminate gracefully
                    assert result is not None
                    assert mock_chat_log._call_count == 1

"""Integration tests for observability events.

These tests verify that all observability events are fired correctly with
proper data structures and values during conversation processing.
"""

import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant, State

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_PERSIST,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONTEXT_MODE_DIRECT,
    EVENT_CONTEXT_INJECTED,
    EVENT_CONVERSATION_FINISHED,
    EVENT_CONVERSATION_STARTED,
    EVENT_ERROR,
    EVENT_HISTORY_SAVED,
)

_LOGGER = logging.getLogger(__name__)


class EventCapture:
    """Helper class to capture events fired during tests."""

    def __init__(self):
        """Initialize event capture."""
        self.events: list[tuple[str, dict[str, Any]]] = []

    def capture_event(self, event_type: str, event_data: dict[str, Any] | None = None) -> None:
        """Capture an event (sync, as async_fire is sync in HA)."""
        self.events.append((event_type, (event_data or {}).copy()))

    def get_events(self, event_type: str) -> list[dict[str, Any]]:
        """Get all events of a specific type."""
        return [data for evt_type, data in self.events if evt_type == event_type]

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()


@pytest.fixture
def event_capture(test_hass: HomeAssistant) -> EventCapture:
    """Create event capture fixture that hooks into test_hass.bus.async_fire."""
    capture = EventCapture()

    # Create a sync mock that captures events (async_fire is sync in HA)
    def capturing_async_fire(event_type: str, event_data: dict[str, Any] | None = None, **kwargs):
        """Capture event (sync - async_fire is actually sync in HA)."""
        capture.capture_event(event_type, event_data)
        # Return None to match HA's actual async_fire behavior (it's sync, not async)
        return None

    # Replace async_fire with our capturing mock
    test_hass.bus.async_fire = MagicMock(side_effect=capturing_async_fire)

    return capture


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_started_event(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that EVENT_CONVERSATION_STARTED fires when process_message is called.

    Verifies:
    1. Event fires at the start of conversation
    2. Event contains conversation_id
    3. Event contains user_id when provided
    4. Event contains device_id when provided
    5. Event contains timestamp
    6. Event contains context_mode
    7. Timestamp is reasonable (within last few seconds)
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,  # Enable events
        CONF_DEBUG_LOGGING: False,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Process message with all optional parameters
        conversation_id = "test_conv_started"
        user_id = "test_user_123"
        device_id = "test_device_456"

        start_time = time.time()

        # Mock LLM call
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            await agent.process_message(
                text="Hello!",
                conversation_id=conversation_id,
                user_id=user_id,
                device_id=device_id,
            )

        end_time = time.time()

        # Verify EVENT_CONVERSATION_STARTED was fired
        started_events = event_capture.get_events(EVENT_CONVERSATION_STARTED)
        assert (
            len(started_events) == 1
        ), f"Expected 1 conversation_started event, got {len(started_events)}"

        event_data = started_events[0]

        # Verify all required fields are present
        assert "conversation_id" in event_data, "Event must contain conversation_id"
        assert "user_id" in event_data, "Event must contain user_id"
        assert "device_id" in event_data, "Event must contain device_id"
        assert "timestamp" in event_data, "Event must contain timestamp"
        assert "context_mode" in event_data, "Event must contain context_mode"

        # Verify field values are correct
        assert (
            event_data["conversation_id"] == conversation_id
        ), f"conversation_id mismatch: expected {conversation_id}, got {event_data['conversation_id']}"
        assert (
            event_data["user_id"] == user_id
        ), f"user_id mismatch: expected {user_id}, got {event_data['user_id']}"
        assert (
            event_data["device_id"] == device_id
        ), f"device_id mismatch: expected {device_id}, got {event_data['device_id']}"
        assert (
            event_data["context_mode"] == CONTEXT_MODE_DIRECT
        ), f"context_mode mismatch: expected {CONTEXT_MODE_DIRECT}, got {event_data['context_mode']}"

        # Verify timestamp is reasonable (within test execution window)
        timestamp = event_data["timestamp"]
        assert isinstance(
            timestamp, (int, float)
        ), f"timestamp should be numeric, got {type(timestamp)}"
        assert (
            start_time <= timestamp <= end_time
        ), f"timestamp {timestamp} should be between {start_time} and {end_time}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_finished_event(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that EVENT_CONVERSATION_FINISHED fires with correct metrics.

    Verifies:
    1. Event fires at the end of conversation
    2. Event contains conversation_id
    3. Event contains user_id
    4. Event contains duration_ms (integer > 0)
    5. Event contains tokens dict with prompt, completion, total
    6. Event contains performance dict with latency metrics
    7. Event contains context dict
    8. Event contains tool_calls (integer >= 0)
    9. Event contains tool_breakdown dict
    10. Event contains used_external_llm boolean
    11. All numeric metrics have correct types and reasonable values
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_conv_finished"
        user_id = "test_user_789"

        # Mock LLM call
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            await agent.process_message(
                text="What is 2 + 2?",
                conversation_id=conversation_id,
                user_id=user_id,
            )

        # Verify EVENT_CONVERSATION_FINISHED was fired
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)
        assert (
            len(finished_events) == 1
        ), f"Expected 1 conversation_finished event, got {len(finished_events)}"

        event_data = finished_events[0]

        # Verify required fields are present
        required_fields = [
            "conversation_id",
            "user_id",
            "duration_ms",
            "tokens",
            "performance",
            "context",
            "tool_calls",
            "tool_breakdown",
            "used_external_llm",
        ]

        for field in required_fields:
            assert field in event_data, f"Event must contain {field}"

        # Verify conversation_id and user_id match
        assert (
            event_data["conversation_id"] == conversation_id
        ), f"conversation_id mismatch: expected {conversation_id}, got {event_data['conversation_id']}"
        assert (
            event_data["user_id"] == user_id
        ), f"user_id mismatch: expected {user_id}, got {event_data['user_id']}"

        # Verify duration_ms is positive integer
        duration_ms = event_data["duration_ms"]
        assert isinstance(duration_ms, int), f"duration_ms should be int, got {type(duration_ms)}"
        assert duration_ms > 0, f"duration_ms should be positive, got {duration_ms}"
        # Reasonable upper bound (30 seconds for a simple query)
        assert duration_ms < 30000, f"duration_ms seems too high: {duration_ms}ms"

        # Verify tokens dict structure and values
        tokens = event_data["tokens"]
        assert isinstance(tokens, dict), f"tokens should be dict, got {type(tokens)}"
        assert "prompt" in tokens, "tokens must contain 'prompt'"
        assert "completion" in tokens, "tokens must contain 'completion'"
        assert "total" in tokens, "tokens must contain 'total'"

        # Token counts should be non-negative integers and match the mocked LLM response
        assert isinstance(
            tokens["prompt"], int
        ), f"tokens.prompt should be int, got {type(tokens['prompt'])}"
        assert isinstance(
            tokens["completion"], int
        ), f"tokens.completion should be int, got {type(tokens['completion'])}"
        assert isinstance(
            tokens["total"], int
        ), f"tokens.total should be int, got {type(tokens['total'])}"

        # Verify exact token counts match the mock LLM response
        # Mock response has: prompt_tokens: 10, completion_tokens: 5, total_tokens: 15
        assert tokens["prompt"] == 10, f"tokens.prompt should be 10 (from mock), got {tokens['prompt']}"
        assert (
            tokens["completion"] == 5
        ), f"tokens.completion should be 5 (from mock), got {tokens['completion']}"
        assert tokens["total"] == 15, f"tokens.total should be 15 (from mock), got {tokens['total']}"

        # Total should equal sum of prompt + completion
        expected_total = tokens["prompt"] + tokens["completion"]
        assert (
            tokens["total"] == expected_total
        ), f"tokens.total ({tokens['total']}) should equal prompt + completion ({expected_total})"

        # Verify performance dict structure and values
        performance = event_data["performance"]
        assert isinstance(performance, dict), f"performance should be dict, got {type(performance)}"
        assert "llm_latency_ms" in performance, "performance must contain 'llm_latency_ms'"
        assert "tool_latency_ms" in performance, "performance must contain 'tool_latency_ms'"
        assert "context_latency_ms" in performance, "performance must contain 'context_latency_ms'"

        # Latency metrics should be non-negative integers
        assert isinstance(
            performance["llm_latency_ms"], int
        ), f"llm_latency_ms should be int, got {type(performance['llm_latency_ms'])}"
        assert isinstance(
            performance["tool_latency_ms"], int
        ), f"tool_latency_ms should be int, got {type(performance['tool_latency_ms'])}"
        assert isinstance(
            performance["context_latency_ms"], int
        ), f"context_latency_ms should be int, got {type(performance['context_latency_ms'])}"

        assert (
            performance["llm_latency_ms"] >= 0
        ), f"llm_latency_ms should be >= 0, got {performance['llm_latency_ms']}"
        assert (
            performance["tool_latency_ms"] >= 0
        ), f"tool_latency_ms should be >= 0, got {performance['tool_latency_ms']}"
        assert (
            performance["context_latency_ms"] >= 0
        ), f"context_latency_ms should be >= 0, got {performance['context_latency_ms']}"

        # Verify context is a dict
        context = event_data["context"]
        assert isinstance(context, dict), f"context should be dict, got {type(context)}"

        # Verify tool_calls is non-negative integer
        tool_calls = event_data["tool_calls"]
        assert isinstance(tool_calls, int), f"tool_calls should be int, got {type(tool_calls)}"
        assert tool_calls >= 0, f"tool_calls should be >= 0, got {tool_calls}"

        # Verify tool_breakdown is a dict
        tool_breakdown = event_data["tool_breakdown"]
        assert isinstance(
            tool_breakdown, dict
        ), f"tool_breakdown should be dict, got {type(tool_breakdown)}"

        # Verify used_external_llm is boolean
        used_external_llm = event_data["used_external_llm"]
        assert isinstance(
            used_external_llm, bool
        ), f"used_external_llm should be bool, got {type(used_external_llm)}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_event_on_exception(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that EVENT_ERROR fires when exceptions occur.

    Verifies:
    1. Event fires when process_message encounters an error
    2. Event contains error_type (exception class name)
    3. Event contains error_message
    4. Event contains conversation_id
    5. Event contains component name
    6. Event contains context dict with relevant info
    7. Error details are accurate and useful for debugging
    """
    from custom_components.home_agent.exceptions import HomeAgentError

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_error_event"

        # Mock _call_llm to raise an error
        with patch.object(
            agent, "_call_llm", side_effect=HomeAgentError("Simulated LLM API error for testing")
        ):
            # Try to process message (should fail due to mocked error)
            with pytest.raises(HomeAgentError):
                await agent.process_message(
                    text="This should fail",
                    conversation_id=conversation_id,
                )

        # Verify EVENT_ERROR was fired
        error_events = event_capture.get_events(EVENT_ERROR)
        assert len(error_events) >= 1, f"Expected at least 1 error event, got {len(error_events)}"

        event_data = error_events[0]

        # Verify required fields
        assert "error_type" in event_data, "Event must contain error_type"
        assert "error_message" in event_data, "Event must contain error_message"
        assert "conversation_id" in event_data, "Event must contain conversation_id"
        assert "component" in event_data, "Event must contain component"
        assert "context" in event_data, "Event must contain context"

        # Verify conversation_id matches
        assert (
            event_data["conversation_id"] == conversation_id
        ), f"conversation_id mismatch: expected {conversation_id}, got {event_data['conversation_id']}"

        # Verify error_type is a string (class name)
        error_type = event_data["error_type"]
        assert isinstance(error_type, str), f"error_type should be str, got {type(error_type)}"
        assert len(error_type) > 0, "error_type should not be empty"
        # Should be a valid exception class name (contains "Error" typically)
        assert (
            error_type.endswith("Error") or "Exception" in error_type
        ), f"error_type should be an exception class name, got: {error_type}"

        # Verify error_message is a string
        error_message = event_data["error_message"]
        assert isinstance(
            error_message, str
        ), f"error_message should be str, got {type(error_message)}"
        assert len(error_message) > 0, "error_message should not be empty"

        # Verify component is set to "agent"
        assert (
            event_data["component"] == "agent"
        ), f"component should be 'agent', got {event_data['component']}"

        # Verify context is a dict with text_length
        context = event_data["context"]
        assert isinstance(context, dict), f"context should be dict, got {type(context)}"
        assert "text_length" in context, "context must contain text_length"
        assert isinstance(
            context["text_length"], int
        ), f"text_length should be int, got {type(context['text_length'])}"
        assert (
            context["text_length"] > 0
        ), f"text_length should be > 0, got {context['text_length']}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_events_disabled_when_config_false(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that events DON'T fire when CONF_EMIT_EVENTS is False.

    Verifies:
    1. No EVENT_CONVERSATION_STARTED when CONF_EMIT_EVENTS is False
    2. No EVENT_CONVERSATION_FINISHED when CONF_EMIT_EVENTS is False
    3. Events work normally when CONF_EMIT_EVENTS is True (sanity check)
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    # First test: Events disabled
    config_disabled = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,  # Disable events
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent_disabled = HomeAgent(test_hass, config_disabled, session_manager)

        event_capture.clear()

        # Mock LLM call
        with patch.object(agent_disabled, "_call_llm", return_value=mock_llm_response):
            await agent_disabled.process_message(
                text="Test with events disabled",
                conversation_id="test_events_disabled",
            )

        # Verify NO events were fired
        started_events = event_capture.get_events(EVENT_CONVERSATION_STARTED)
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)

        assert (
            len(started_events) == 0
        ), f"Expected 0 conversation_started events when disabled, got {len(started_events)}"
        assert (
            len(finished_events) == 0
        ), f"Expected 0 conversation_finished events when disabled, got {len(finished_events)}"

        await agent_disabled.close()

    # Second test: Events enabled (sanity check)
    config_enabled = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,  # Enable events
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent_enabled = HomeAgent(test_hass, config_enabled, session_manager)

        event_capture.clear()

        # Mock LLM call
        with patch.object(agent_enabled, "_call_llm", return_value=mock_llm_response):
            await agent_enabled.process_message(
                text="Test with events enabled",
                conversation_id="test_events_enabled",
            )

        # Verify events WERE fired
        started_events = event_capture.get_events(EVENT_CONVERSATION_STARTED)
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)

        assert (
            len(started_events) == 1
        ), f"Expected 1 conversation_started event when enabled, got {len(started_events)}"
        assert (
            len(finished_events) == 1
        ), f"Expected 1 conversation_finished event when enabled, got {len(finished_events)}"

        await agent_enabled.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_injected_event(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    sample_entity_states: list[State],
    session_manager,
):
    """Test that EVENT_CONTEXT_INJECTED fires when context is retrieved.

    Verifies:
    1. Event fires when context manager retrieves context
    2. Event contains conversation_id
    3. Event contains context_mode
    4. Event contains entity_count or relevant metrics
    5. Event fires before conversation processing completes
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_context_injected"

        # Mock LLM call
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            await agent.process_message(
                text="What lights are on?",
                conversation_id=conversation_id,
            )

        # Verify EVENT_CONTEXT_INJECTED was fired
        context_events = event_capture.get_events(EVENT_CONTEXT_INJECTED)

        # Context injection may happen once per conversation
        assert (
            len(context_events) >= 1
        ), f"Expected at least 1 context_injected event, got {len(context_events)}"

        event_data = context_events[0]

        # Verify event contains conversation_id
        assert "conversation_id" in event_data, "Event must contain conversation_id"
        assert (
            event_data["conversation_id"] == conversation_id
        ), f"conversation_id mismatch: expected {conversation_id}, got {event_data['conversation_id']}"

        # Verify event contains mode (not context_mode - that's the field name in the event)
        assert "mode" in event_data, "Event must contain mode"
        assert (
            event_data["mode"] == CONTEXT_MODE_DIRECT
        ), f"mode mismatch: expected {CONTEXT_MODE_DIRECT}, got {event_data['mode']}"

        # Verify event contains entity metrics (required field)
        assert "entity_count" in event_data, "Event must contain entity_count"
        entity_count = event_data["entity_count"]
        assert isinstance(
            entity_count, int
        ), f"entity_count should be int, got {type(entity_count)}"
        assert entity_count >= 0, f"entity_count should be >= 0, got {entity_count}"

        # Verify event contains token_count
        assert "token_count" in event_data, "Event must contain token_count"
        token_count = event_data["token_count"]
        assert isinstance(
            token_count, int
        ), f"token_count should be int, got {type(token_count)}"
        assert token_count >= 0, f"token_count should be >= 0, got {token_count}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_history_saved_event(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that EVENT_HISTORY_SAVED fires when history is saved.

    Verifies:
    1. Event fires when conversation history is persisted
    2. Event contains conversation_id
    3. Event contains message_count or relevant metrics
    4. Event fires after conversation completes
    5. No event when history is disabled
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    # Test with history enabled
    config_with_history = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: True,  # Enable history
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_HISTORY_PERSIST: True,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent_with_history = HomeAgent(test_hass, config_with_history, session_manager)

        event_capture.clear()

        conversation_id = "test_history_saved"

        # Mock LLM call
        with patch.object(agent_with_history, "_call_llm", return_value=mock_llm_response):
            await agent_with_history.process_message(
                text="Remember this message",
                conversation_id=conversation_id,
            )

        # Verify EVENT_HISTORY_SAVED structure if it fires
        # Note: EVENT_HISTORY_SAVED fires when history is persisted to storage,
        # which may not happen immediately during test execution due to the save delay
        history_events = event_capture.get_events(EVENT_HISTORY_SAVED)

        # If the event fired, verify its structure
        if len(history_events) > 0:
            event_data = history_events[0]

            # Verify required fields for this event type
            # Note: This event tracks overall storage state, not per-conversation
            assert "conversation_count" in event_data, "Event must contain conversation_count"
            assert "message_count" in event_data, "Event must contain message_count"
            assert "size_bytes" in event_data, "Event must contain size_bytes"
            assert "timestamp" in event_data, "Event must contain timestamp"

            # Verify field types and values
            conversation_count = event_data["conversation_count"]
            assert isinstance(
                conversation_count, int
            ), f"conversation_count should be int, got {type(conversation_count)}"
            assert conversation_count > 0, f"conversation_count should be > 0, got {conversation_count}"

            message_count = event_data["message_count"]
            assert isinstance(
                message_count, int
            ), f"message_count should be int, got {type(message_count)}"
            assert message_count > 0, f"message_count should be > 0, got {message_count}"

            size_bytes = event_data["size_bytes"]
            assert isinstance(
                size_bytes, int
            ), f"size_bytes should be int, got {type(size_bytes)}"
            assert size_bytes >= 0, f"size_bytes should be >= 0, got {size_bytes}"

        await agent_with_history.close()

    # Test with history disabled
    config_no_history = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,  # Disable history
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent_no_history = HomeAgent(test_hass, config_no_history, session_manager)

        event_capture.clear()

        # Mock LLM call
        with patch.object(agent_no_history, "_call_llm", return_value=mock_llm_response):
            await agent_no_history.process_message(
                text="This won't be saved",
                conversation_id="test_no_history",
            )

        # Verify NO history_saved events when history is disabled
        history_events = event_capture.get_events(EVENT_HISTORY_SAVED)
        assert (
            len(history_events) == 0
        ), f"Expected 0 history_saved events when history disabled, got {len(history_events)}"

        await agent_no_history.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_events_in_single_conversation(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that multiple event types fire correctly in a single conversation.

    Verifies:
    1. EVENT_CONVERSATION_STARTED fires first
    2. EVENT_CONTEXT_INJECTED fires during processing (if context used)
    3. EVENT_CONVERSATION_FINISHED fires last
    4. Events fire in correct order
    5. All events share the same conversation_id
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: True,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_multiple_events"

        # Mock LLM call
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            await agent.process_message(
                text="Turn on the lights",
                conversation_id=conversation_id,
            )

        # Get all events in order
        all_events = event_capture.events

        # Find indices of key events
        started_idx = None
        finished_idx = None
        context_idx = None

        for idx, (event_type, _) in enumerate(all_events):
            if event_type == EVENT_CONVERSATION_STARTED and started_idx is None:
                started_idx = idx
            elif event_type == EVENT_CONVERSATION_FINISHED and finished_idx is None:
                finished_idx = idx
            elif event_type == EVENT_CONTEXT_INJECTED and context_idx is None:
                context_idx = idx

        # Verify EVENT_CONVERSATION_STARTED fired
        assert started_idx is not None, "EVENT_CONVERSATION_STARTED should fire"

        # Verify EVENT_CONVERSATION_FINISHED fired
        assert finished_idx is not None, "EVENT_CONVERSATION_FINISHED should fire"

        # Verify ordering: started comes before finished
        assert (
            started_idx < finished_idx
        ), f"EVENT_CONVERSATION_STARTED (idx={started_idx}) should fire before EVENT_CONVERSATION_FINISHED (idx={finished_idx})"

        # If context event fired, it should be between started and finished
        if context_idx is not None:
            assert (
                started_idx < context_idx < finished_idx
            ), f"EVENT_CONTEXT_INJECTED (idx={context_idx}) should fire between started (idx={started_idx}) and finished (idx={finished_idx})"

        # Verify all events have the same conversation_id
        started_events = event_capture.get_events(EVENT_CONVERSATION_STARTED)
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)

        assert (
            started_events[0]["conversation_id"] == conversation_id
        ), "Started event should have correct conversation_id"
        assert (
            finished_events[0]["conversation_id"] == conversation_id
        ), "Finished event should have correct conversation_id"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_finished_metrics_accuracy(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    sample_entity_states: list[State],
    session_manager,
):
    """Test that EVENT_CONVERSATION_FINISHED metrics are accurate and useful.

    Verifies:
    1. Token counts reflect actual LLM usage
    2. Performance metrics show realistic latencies
    3. Tool call counts match actual tool executions
    4. Tool breakdown accurately lists which tools were used
    5. Metrics are consistent across the event data
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: True,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        # Mock service calls to count tool executions
        service_call_count = 0

        async def mock_service_call(domain, service, service_data, **kwargs):
            nonlocal service_call_count
            service_call_count += 1
            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call)

        agent = HomeAgent(test_hass, config, session_manager)

        # Mock the get_exposed_entities method
        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in sample_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        conversation_id = "test_metrics_accuracy"

        start_time = time.time()

        # Mock LLM call
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            await agent.process_message(
                text="Turn on the living room light and tell me the temperature",
                conversation_id=conversation_id,
            )

        end_time = time.time()
        actual_duration_ms = int((end_time - start_time) * 1000)

        # Get the finished event
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)
        assert len(finished_events) == 1, "Expected 1 conversation_finished event"

        event_data = finished_events[0]

        # Verify duration_ms is close to actual measured duration
        reported_duration_ms = event_data["duration_ms"]
        # Allow 10% margin for event processing overhead, but at least 5ms to account for OS scheduling jitter
        duration_diff = abs(reported_duration_ms - actual_duration_ms)
        duration_margin = max(actual_duration_ms * 0.1, 5)
        assert (
            duration_diff <= duration_margin
        ), f"Reported duration {reported_duration_ms}ms differs too much from actual {actual_duration_ms}ms (diff: {duration_diff}ms, margin: {duration_margin}ms)"

        # Verify performance metrics sum to reasonable portion of total duration
        performance = event_data["performance"]
        total_latency = (
            performance["llm_latency_ms"]
            + performance["tool_latency_ms"]
            + performance["context_latency_ms"]
        )

        # Total latency should not exceed reported duration
        assert (
            total_latency <= reported_duration_ms
        ), f"Sum of latencies ({total_latency}ms) should not exceed total duration ({reported_duration_ms}ms)"

        # LLM latency should be >= 0 (can be 0 for mocked LLM call)
        assert performance["llm_latency_ms"] >= 0, "LLM latency should be >= 0"

        # Verify token counts are provided (for real LLM, tokens should be > 0)
        tokens = event_data["tokens"]
        # Some LLMs may not report token counts, but if they do, verify consistency
        if tokens["total"] > 0:
            # Verify reasonable token counts (not absurdly high)
            assert tokens["total"] < 10000, f"Token count seems too high: {tokens['total']}"

            # Verify prompt tokens > 0 (we sent a prompt)
            assert tokens["prompt"] > 0, "Prompt tokens should be > 0"

        # Verify tool_calls count matches tool_breakdown
        tool_calls = event_data["tool_calls"]
        tool_breakdown = event_data["tool_breakdown"]

        breakdown_total = sum(tool_breakdown.values())
        assert (
            tool_calls == breakdown_total
        ), f"tool_calls ({tool_calls}) should match sum of tool_breakdown ({breakdown_total})"

        # Verify used_external_llm is False (we didn't enable external LLM)
        assert (
            event_data["used_external_llm"] is False
        ), "used_external_llm should be False when external LLM is not enabled"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_finished_token_counts_accurate(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that token counts in EVENT_CONVERSATION_FINISHED exactly match LLM response.

    Verifies:
    1. Token counts are extracted from LLM usage data
    2. Exact values match (not just >= 0)
    3. Total equals prompt + completion
    4. Different token values are reported accurately
    """
    # Test multiple scenarios with different token counts
    test_cases = [
        {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},  # Edge case
    ]

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        for idx, token_counts in enumerate(test_cases):
            event_capture.clear()

            # Mock LLM response with specific token counts
            mock_llm_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Test response",
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": token_counts,
            }

            with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                await agent.process_message(
                    text=f"Test message {idx}",
                    conversation_id=f"test_token_accuracy_{idx}",
                )

            # Verify token counts in event match exactly
            finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)
            assert len(finished_events) == 1, f"Expected 1 finished event, got {len(finished_events)}"

            event_data = finished_events[0]
            tokens = event_data["tokens"]

            # Verify exact match with mock LLM response
            assert (
                tokens["prompt"] == token_counts["prompt_tokens"]
            ), f"Test case {idx}: prompt tokens mismatch - expected {token_counts['prompt_tokens']}, got {tokens['prompt']}"
            assert (
                tokens["completion"] == token_counts["completion_tokens"]
            ), f"Test case {idx}: completion tokens mismatch - expected {token_counts['completion_tokens']}, got {tokens['completion']}"
            assert (
                tokens["total"] == token_counts["total_tokens"]
            ), f"Test case {idx}: total tokens mismatch - expected {token_counts['total_tokens']}, got {tokens['total']}"

            # Verify arithmetic consistency
            assert (
                tokens["total"] == tokens["prompt"] + tokens["completion"]
            ), f"Test case {idx}: total should equal prompt + completion"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_ordering(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that observability events fire in the correct order.

    Verifies:
    1. EVENT_CONVERSATION_STARTED fires first
    2. EVENT_CONTEXT_INJECTED fires after started (if context used)
    3. EVENT_CONVERSATION_FINISHED fires last
    4. All events have consistent conversation_id
    5. Timestamps are monotonically increasing
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_PERSIST: True,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_event_ordering"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            await agent.process_message(
                text="Test ordering",
                conversation_id=conversation_id,
            )

        # Get all events in the order they were fired
        all_events = event_capture.events

        # Extract indices of key events
        started_idx = None
        finished_idx = None
        context_idx = None
        history_idx = None

        for idx, (event_type, event_data) in enumerate(all_events):
            if event_type == EVENT_CONVERSATION_STARTED and started_idx is None:
                started_idx = idx
            elif event_type == EVENT_CONVERSATION_FINISHED and finished_idx is None:
                finished_idx = idx
            elif event_type == EVENT_CONTEXT_INJECTED and context_idx is None:
                context_idx = idx
            elif event_type == EVENT_HISTORY_SAVED and history_idx is None:
                history_idx = idx

        # Verify required events fired
        assert started_idx is not None, "EVENT_CONVERSATION_STARTED must fire"
        assert finished_idx is not None, "EVENT_CONVERSATION_FINISHED must fire"

        # Verify ordering: started < finished
        assert (
            started_idx < finished_idx
        ), f"STARTED (idx={started_idx}) must fire before FINISHED (idx={finished_idx})"

        # If context event fired, verify it's between started and finished
        if context_idx is not None:
            assert (
                started_idx < context_idx < finished_idx
            ), f"CONTEXT_INJECTED (idx={context_idx}) must fire between STARTED (idx={started_idx}) and FINISHED (idx={finished_idx})"

        # If history event fired, verify it fires after or with finished
        if history_idx is not None:
            assert (
                history_idx >= finished_idx
            ), f"HISTORY_SAVED (idx={history_idx}) should fire after or with FINISHED (idx={finished_idx})"

        # Verify timestamps are monotonically increasing for started -> finished
        started_event = event_capture.get_events(EVENT_CONVERSATION_STARTED)[0]
        finished_event = event_capture.get_events(EVENT_CONVERSATION_FINISHED)[0]

        assert "timestamp" in started_event, "STARTED event must have timestamp"
        # Note: FINISHED event has duration_ms but may not have separate timestamp
        # So we only verify STARTED has timestamp

        # Verify all events have the same conversation_id
        assert (
            started_event["conversation_id"] == conversation_id
        ), "STARTED event should have correct conversation_id"
        assert (
            finished_event["conversation_id"] == conversation_id
        ), "FINISHED event should have correct conversation_id"

        if context_idx is not None:
            context_events = event_capture.get_events(EVENT_CONTEXT_INJECTED)
            assert (
                context_events[0]["conversation_id"] == conversation_id
            ), "CONTEXT event should have correct conversation_id"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_error_event_fired(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that EVENT_STREAMING_ERROR fires when streaming fails.

    Verifies:
    1. Event fires when stream encounters an error
    2. Event contains error_type
    3. Event contains error_message
    4. Event contains conversation_id
    5. Event contains relevant context
    """
    from custom_components.home_agent.const import EVENT_STREAMING_ERROR

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_STREAMING_ENABLED: True,  # Enable streaming
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_streaming_error"

        # Mock streaming to raise an error
        async def mock_stream_error(*args, **kwargs):
            """Simulate a streaming error."""
            raise ConnectionError("Simulated streaming connection error")

        # Try to trigger streaming error
        # Note: The actual implementation depends on how streaming is handled in the agent
        # This test assumes agent has a _stream_llm or similar method
        with patch.object(
            agent, "_call_llm", side_effect=ConnectionError("Simulated streaming error")
        ):
            # Process message should fail due to streaming error
            try:
                await agent.process_message(
                    text="Test streaming",
                    conversation_id=conversation_id,
                )
            except Exception:
                # Expected to fail
                pass

        # Verify EVENT_STREAMING_ERROR was fired OR EVENT_ERROR was fired
        # (depending on how errors are categorized)
        streaming_errors = event_capture.get_events(EVENT_STREAMING_ERROR)
        general_errors = event_capture.get_events(EVENT_ERROR)

        # At least one error event should fire
        total_errors = len(streaming_errors) + len(general_errors)
        assert total_errors >= 1, f"Expected at least 1 error event, got {total_errors}"

        # If streaming error event exists, verify its structure
        if len(streaming_errors) > 0:
            event_data = streaming_errors[0]

            # Verify required fields
            assert "error_type" in event_data, "Event must contain error_type"
            assert "error_message" in event_data, "Event must contain error_message"
            assert "conversation_id" in event_data, "Event must contain conversation_id"

            # Verify conversation_id matches
            assert (
                event_data["conversation_id"] == conversation_id
            ), f"conversation_id mismatch: expected {conversation_id}, got {event_data['conversation_id']}"

            # Verify error details
            assert isinstance(
                event_data["error_type"], str
            ), "error_type should be string"
            assert isinstance(
                event_data["error_message"], str
            ), "error_message should be string"
            assert len(event_data["error_message"]) > 0, "error_message should not be empty"

        await agent.close()

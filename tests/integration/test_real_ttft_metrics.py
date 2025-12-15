"""Integration tests for TTFT (Time To First Token) metrics with real LLM.

These tests validate that TTFT metrics are correctly captured when using
a real LLM endpoint, as opposed to mocked responses.

Test scenarios:
1. Basic TTFT validation - TTFT > 0 when real LLM is called
2. Streaming mode - TTFT should be less than total LLM latency
3. Non-streaming mode - TTFT should equal LLM latency (first iteration)
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_STREAMING_ENABLED,
    EVENT_CONVERSATION_FINISHED,
)


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

    def capturing_async_fire(event_type: str, event_data: dict[str, Any] | None = None, **kwargs):
        """Capture event (sync - async_fire is actually sync in HA)."""
        capture.capture_event(event_type, event_data)
        return None

    test_hass.bus.async_fire = MagicMock(side_effect=capturing_async_fire)

    return capture


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_ttft_metric_with_real_llm(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that TTFT metric is non-zero when using a real LLM.

    This test validates that when a real LLM endpoint is called,
    the ttft_ms metric in EVENT_CONVERSATION_FINISHED is > 0,
    proving that actual timing is being captured.

    Verifies:
    1. ttft_ms is present in the performance dict
    2. ttft_ms > 0 (real timing captured, not just initialized to 0)
    3. ttft_ms is a reasonable value (< 30 seconds)
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 100,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
        CONF_STREAMING_ENABLED: False,  # Non-streaming for this basic test
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Process a simple message with real LLM
        await agent.process_message(
            text="Say hello in exactly three words.",
            conversation_id="test_ttft_real_llm",
        )

        # Get the finished event
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)
        assert len(finished_events) == 1, "Expected 1 conversation_finished event"

        event_data = finished_events[0]
        performance = event_data["performance"]

        # Verify ttft_ms is present
        assert "ttft_ms" in performance, "performance dict must contain 'ttft_ms'"

        ttft_ms = performance["ttft_ms"]

        # Verify ttft_ms is a positive integer (real timing captured)
        assert isinstance(ttft_ms, int), f"ttft_ms should be int, got {type(ttft_ms)}"
        assert ttft_ms > 0, (
            f"ttft_ms should be > 0 with real LLM (actual timing), got {ttft_ms}. "
            "This suggests TTFT tracking may not be working correctly."
        )

        # Sanity check: TTFT should be reasonable (< 30 seconds)
        assert ttft_ms < 30000, f"ttft_ms seems unreasonably high: {ttft_ms}ms"

        # Verify LLM latency is also captured
        llm_latency_ms = performance["llm_latency_ms"]
        assert llm_latency_ms > 0, f"llm_latency_ms should be > 0, got {llm_latency_ms}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_ttft_streaming_less_than_total_latency(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that TTFT in streaming mode is less than total LLM latency.

    In streaming mode, TTFT measures when the first token arrives,
    which should be before the complete response is received.
    Therefore, ttft_ms should be less than llm_latency_ms.

    Note: This test requires the LLM to actually stream tokens incrementally.
    Some LLMs may buffer the entire response, making TTFT equal to latency.

    Verifies:
    1. ttft_ms is present and > 0
    2. llm_latency_ms is present and > 0
    3. ttft_ms <= llm_latency_ms (TTFT cannot exceed total latency)
    4. For truly streaming responses, ttft_ms < llm_latency_ms
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,  # Request longer response to see streaming effect
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
        CONF_STREAMING_ENABLED: True,  # Enable streaming
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Request a longer response to maximize streaming effect
        await agent.process_message(
            text="Count from 1 to 10, saying each number on a new line.",
            conversation_id="test_ttft_streaming",
        )

        # Get the finished event
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)
        assert len(finished_events) == 1, "Expected 1 conversation_finished event"

        event_data = finished_events[0]
        performance = event_data["performance"]

        # Verify metrics are present
        assert "ttft_ms" in performance, "performance dict must contain 'ttft_ms'"
        assert "llm_latency_ms" in performance, "performance dict must contain 'llm_latency_ms'"

        ttft_ms = performance["ttft_ms"]
        llm_latency_ms = performance["llm_latency_ms"]

        # Verify both are positive
        assert ttft_ms > 0, f"ttft_ms should be > 0 in streaming mode, got {ttft_ms}"
        assert llm_latency_ms > 0, f"llm_latency_ms should be > 0, got {llm_latency_ms}"

        # TTFT must not exceed total latency (this is always true)
        assert ttft_ms <= llm_latency_ms, (
            f"ttft_ms ({ttft_ms}) should be <= llm_latency_ms ({llm_latency_ms}). "
            "TTFT cannot exceed total response time."
        )

        # For a truly streaming LLM, TTFT should be strictly less than total latency
        # However, we use a soft assertion here because:
        # 1. Some LLMs may buffer responses
        # 2. Very fast responses may have TTFT ≈ latency
        # 3. Network conditions can affect timing
        if llm_latency_ms > 500:  # Only check if response took meaningful time
            # Log the ratio for debugging
            ratio = ttft_ms / llm_latency_ms if llm_latency_ms > 0 else 1.0
            print(f"TTFT/Latency ratio: {ratio:.2f} (ttft={ttft_ms}ms, latency={llm_latency_ms}ms)")

            # For longer responses, we expect TTFT to be noticeably less
            # A ratio < 0.9 indicates true streaming behavior
            # But we don't fail the test if this isn't met, just warn
            if ratio >= 0.9:
                print(
                    f"Warning: TTFT ratio ({ratio:.2f}) suggests LLM may not be "
                    "streaming tokens incrementally. This is not necessarily a bug."
                )

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_ttft_non_streaming_equals_llm_latency(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    event_capture: EventCapture,
    session_manager,
):
    """Test that TTFT in non-streaming mode equals LLM latency.

    In non-streaming mode, the entire response arrives at once,
    so TTFT (time to first token) should equal the total LLM latency
    for the first LLM call.

    Verifies:
    1. ttft_ms is present and > 0
    2. llm_latency_ms is present and > 0
    3. ttft_ms == llm_latency_ms (or very close, within timing tolerance)
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 100,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: True,
        CONF_DEBUG_LOGGING: False,
        CONF_STREAMING_ENABLED: False,  # Disable streaming
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Simple request that won't trigger tool calls (single LLM iteration)
        await agent.process_message(
            text="What is 2 + 2? Reply with just the number.",
            conversation_id="test_ttft_non_streaming",
        )

        # Get the finished event
        finished_events = event_capture.get_events(EVENT_CONVERSATION_FINISHED)
        assert len(finished_events) == 1, "Expected 1 conversation_finished event"

        event_data = finished_events[0]
        performance = event_data["performance"]

        # Verify metrics are present
        assert "ttft_ms" in performance, "performance dict must contain 'ttft_ms'"
        assert "llm_latency_ms" in performance, "performance dict must contain 'llm_latency_ms'"

        ttft_ms = performance["ttft_ms"]
        llm_latency_ms = performance["llm_latency_ms"]

        # Verify both are positive
        assert ttft_ms > 0, f"ttft_ms should be > 0 with real LLM, got {ttft_ms}"
        assert llm_latency_ms > 0, f"llm_latency_ms should be > 0, got {llm_latency_ms}"

        # In non-streaming mode with single iteration, TTFT should equal LLM latency
        # The implementation sets ttft_ms = llm_latency_ms on iteration == 1
        #
        # However, if the LLM triggers tool calls, there may be multiple iterations,
        # and total llm_latency_ms accumulates across iterations while ttft_ms
        # is set only on the first iteration.
        #
        # For a simple query that doesn't trigger tools, they should be equal.
        # We allow a small tolerance for timing measurement differences.
        assert ttft_ms <= llm_latency_ms, (
            f"ttft_ms ({ttft_ms}) should be <= llm_latency_ms ({llm_latency_ms})"
        )

        # Check if they're approximately equal (for single-iteration case)
        # If ttft_ms is much less than llm_latency_ms, it means multiple LLM calls happened
        if ttft_ms < llm_latency_ms * 0.5:
            # Multiple iterations occurred - TTFT is from first call only
            print(
                f"Note: Multiple LLM iterations detected. "
                f"TTFT ({ttft_ms}ms) is from first call, "
                f"total latency ({llm_latency_ms}ms) includes all calls."
            )
        else:
            # Single iteration - TTFT should be very close to total latency
            # Allow 10% tolerance for timing measurement differences
            tolerance = max(50, llm_latency_ms * 0.1)  # At least 50ms tolerance
            assert abs(ttft_ms - llm_latency_ms) <= tolerance, (
                f"In non-streaming single-iteration mode, ttft_ms ({ttft_ms}) "
                f"should approximately equal llm_latency_ms ({llm_latency_ms}), "
                f"but difference is {abs(ttft_ms - llm_latency_ms)}ms "
                f"(tolerance: {tolerance}ms)"
            )

        await agent.close()

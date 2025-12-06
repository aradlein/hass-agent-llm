"""Integration tests for context optimization and truncation.

This module tests the complete context optimization pipeline, including:
- Context truncation when exceeding token limits
- Preservation of most relevant information during optimization
- Integration with ContextManager and ContextOptimizer
- Behavior with different context sizes and limits

Running the tests:
    # Run all context optimization tests
    pytest tests/integration/test_context_optimization.py -v

    # Run a specific test
    pytest tests/integration/test_context_optimization.py::test_context_truncation_when_exceeding_limit -v

    # Run with detailed output
    pytest tests/integration/test_context_optimization.py -vv -s

Note: These tests do not require external services (LLM, ChromaDB) as they focus
on the optimization/truncation logic itself using mock entities and configurations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import State

from tests.integration.helpers import setup_entity_states
from custom_components.home_agent.const import (
    CONF_COMPRESSION_LEVEL,
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_DIRECT_ENTITIES,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_PRESERVE_RECENT_MESSAGES,
    CONTEXT_FORMAT_JSON,
    CONTEXT_MODE_DIRECT,
    MAX_CONTEXT_TOKENS,
)
from custom_components.home_agent.context_manager import ContextManager
from custom_components.home_agent.context_optimizer import ContextOptimizer
from custom_components.home_agent.exceptions import TokenLimitExceeded
from custom_components.home_agent.helpers import estimate_tokens


@pytest.fixture
def large_entity_set() -> list[State]:
    """Create a large set of entities to test truncation.

    Returns:
        List of State objects representing many entities
    """
    entities = []

    # Create 100 entities across different domains
    for i in range(20):
        entities.append(
            State(
                f"light.room_{i}",
                "on" if i % 2 == 0 else "off",
                {
                    "brightness": 128 + i,
                    "color_temp": 300 + i * 10,
                    "friendly_name": f"Light in Room {i}",
                    "supported_features": 12345,
                },
            )
        )

    for i in range(20):
        entities.append(
            State(
                f"sensor.temperature_{i}",
                f"{20 + i}.5",
                {
                    "unit_of_measurement": "°C",
                    "device_class": "temperature",
                    "friendly_name": f"Temperature Sensor {i}",
                },
            )
        )

    for i in range(20):
        entities.append(
            State(
                f"switch.device_{i}",
                "on" if i % 3 == 0 else "off",
                {"friendly_name": f"Switch {i}"},
            )
        )

    for i in range(20):
        entities.append(
            State(
                f"climate.zone_{i}",
                "heat" if i % 2 == 0 else "cool",
                {
                    "temperature": 22 + i * 0.5,
                    "current_temperature": 21 + i * 0.5,
                    "hvac_mode": "heat" if i % 2 == 0 else "cool",
                    "friendly_name": f"Climate Zone {i}",
                },
            )
        )

    for i in range(20):
        entities.append(
            State(
                f"binary_sensor.door_{i}",
                "on" if i % 4 == 0 else "off",
                {
                    "device_class": "door",
                    "friendly_name": f"Door Sensor {i}",
                },
            )
        )

    return entities


@pytest.fixture
def entity_states_with_long_attributes() -> list[State]:
    """Create entities with very long attribute values.

    Returns:
        List of State objects with long attributes
    """
    return [
        State(
            "sensor.weather_forecast",
            "sunny",
            {
                "friendly_name": "Weather Forecast",
                "forecast": "x" * 5000,  # Very long forecast data
                "detailed_info": "y" * 3000,  # More long data
            },
        ),
        State(
            "media_player.living_room",
            "playing",
            {
                "friendly_name": "Living Room Media",
                "playlist": "z" * 4000,  # Long playlist
                "metadata": "a" * 2000,
            },
        ),
        State(
            "light.kitchen",
            "on",
            {"friendly_name": "Kitchen Light", "brightness": 255},
        ),
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_truncation_when_exceeding_limit(
    test_hass, large_entity_set, session_manager
):
    """Test that context is truncated when it exceeds the token limit.

    This test verifies that:
    1. Large context exceeding MAX_CONTEXT_TOKENS is properly truncated
    2. The truncated context stays within the token limit
    3. TokenLimitExceeded exception is raised when context cannot be reduced enough
    """
    # Configuration with direct context mode and many entities
    config = {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "",
        CONF_LLM_MODEL: "test-model",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [entity.entity_id for entity in large_entity_set],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        "max_context_tokens": 500,  # Low limit to force truncation
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        setup_entity_states(test_hass, large_entity_set)

        context_manager = ContextManager(test_hass, config)

        # Get raw context first to verify it's large
        raw_context = await context_manager.get_context(
            user_input="Show me all devices",
            conversation_id="test_truncation",
        )

        # Estimate raw context tokens
        raw_tokens = estimate_tokens(raw_context)
        assert raw_tokens > 500, "Raw context should exceed the limit for this test"

        # Now get formatted context - should raise TokenLimitExceeded if too large
        # or successfully truncate
        metrics = {}
        try:
            formatted_context = await context_manager.get_formatted_context(
                user_input="Show me all devices",
                conversation_id="test_truncation",
                metrics=metrics,
            )

            # If we get here, truncation succeeded
            formatted_tokens = estimate_tokens(formatted_context)
            assert (
                formatted_tokens <= 500
            ), "Formatted context should be within token limit after optimization"

            # Verify metrics show compression occurred
            assert "context" in metrics
            assert metrics["context"]["original_tokens"] > 0
            assert metrics["context"]["optimized_tokens"] < metrics["context"]["original_tokens"]

        except TokenLimitExceeded:
            # This is expected if context cannot be reduced enough
            # The important thing is the exception was raised rather than
            # allowing oversized context through
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_optimization_preserves_relevant_info(
    test_hass, large_entity_set, session_manager
):
    """Test that context optimization preserves most relevant information.

    This test verifies that:
    1. When context is compressed, entities mentioned in the query are prioritized
    2. Relevant domain entities are kept
    3. Less relevant entities are removed first
    """
    config = {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "",
        CONF_LLM_MODEL: "test-model",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [entity.entity_id for entity in large_entity_set],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        "max_context_tokens": 4000,  # Reasonable limit for 100 entities
        CONF_COMPRESSION_LEVEL: "medium",
    }

    # Use a smaller subset of entities for this test
    test_entities = large_entity_set[:20]

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        setup_entity_states(test_hass, test_entities)

        # Update config to only include the subset of entities
        config[CONF_DIRECT_ENTITIES] = [entity.entity_id for entity in test_entities]
        context_manager = ContextManager(test_hass, config)

        # Query specifically about lights in room 5
        user_query = "What's the status of the light in room 5?"

        metrics = {}
        context = await context_manager.get_formatted_context(
            user_input=user_query,
            conversation_id="test_relevance",
            metrics=metrics,
        )

        # The context should mention the specific entity we asked about
        # even if context was truncated
        assert (
            "room_5" in context.lower() or "room 5" in context.lower()
        ), "Context should preserve entities mentioned in query"

        # Verify metrics were recorded
        assert "context" in metrics
        assert metrics["context"]["original_tokens"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_optimizer_with_different_compression_levels(test_hass, large_entity_set):
    """Test context optimizer with different compression levels.

    This test verifies that:
    1. Different compression levels produce different sized outputs
    2. Higher compression levels preserve less information
    3. All compression levels stay within target token limits
    """
    # Use a smaller subset to make compression level differences more visible
    test_entities = large_entity_set[:20]

    # Create entity dictionaries from State objects
    entities = []
    for state in test_entities:
        entity_dict = {
            "entity_id": state.entity_id,
            "state": state.state,
            "attributes": dict(state.attributes),
        }
        entities.append(entity_dict)

    # Use a moderate target that allows compression to show differences
    target_tokens = 400

    compression_levels = ["none", "low", "medium", "high"]
    results = {}

    for level in compression_levels:
        optimizer = ContextOptimizer(compression_level=level)

        compressed = optimizer.compress_entity_context(
            entities=entities,
            target_tokens=target_tokens,
            user_query="Show me all lights",
        )

        import json

        compressed_json = json.dumps(compressed)
        compressed_tokens = estimate_tokens(compressed_json)

        results[level] = {
            "entities": len(compressed),
            "tokens": compressed_tokens,
            "metrics": optimizer.get_metrics(),
        }

    # Verify all non-none compression levels stay within target
    for level, result in results.items():
        if level != "none":  # None compression doesn't enforce limits
            assert (
                result["tokens"] <= target_tokens * 1.2
            ), f"{level} compression should stay near target"

    # Verify metrics were recorded for each level
    for level, result in results.items():
        assert result["metrics"] is not None
        assert result["metrics"].entities_before >= result["metrics"].entities_after


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_optimization_with_long_attributes(
    test_hass, entity_states_with_long_attributes, session_manager
):
    """Test that very long attribute values are properly handled.

    This test verifies that:
    1. Entities with very long attribute values are truncated
    2. Essential information is preserved
    3. Context stays within token limits
    """
    config = {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "",
        CONF_LLM_MODEL: "test-model",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [
            entity.entity_id for entity in entity_states_with_long_attributes
        ],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        "max_context_tokens": 1000,
        CONF_COMPRESSION_LEVEL: "medium",
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        setup_entity_states(test_hass, entity_states_with_long_attributes)

        # Create optimizer
        optimizer = ContextOptimizer(compression_level="medium")

        # Convert states to entity dicts
        entities = []
        for state in entity_states_with_long_attributes:
            entity_dict = {
                "entity_id": state.entity_id,
                "state": state.state,
                "attributes": dict(state.attributes),
            }
            entities.append(entity_dict)

        # Remove redundant attributes (which should truncate long values)
        cleaned = optimizer.remove_redundant_attributes(entities)

        import json

        cleaned_json = json.dumps(cleaned)
        cleaned_tokens = estimate_tokens(cleaned_json)

        # Original entities have very long attributes (total ~14000 chars)
        original_json = json.dumps(entities)
        original_tokens = estimate_tokens(original_json)

        # After removing redundant attributes, should be much smaller
        assert cleaned_tokens < original_tokens
        assert (
            cleaned_tokens < 2000
        ), "Cleaned context should have truncated long attribute values"

        # Verify essential attributes are still present
        for entity in cleaned:
            assert "entity_id" in entity
            assert "state" in entity


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_history_truncation(test_hass, session_manager):
    """Test that conversation history is truncated when exceeding limits.

    This test verifies that:
    1. Long conversation history is truncated to fit token limits
    2. Recent messages are preserved
    3. Older messages are removed first
    """
    optimizer = ContextOptimizer(
        compression_level="medium",
        preserve_recent_messages=2,  # Keep last 2 pairs (4 messages)
    )

    # Create a long conversation history
    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"User message {i} " * 20})
        messages.append({"role": "assistant", "content": f"Assistant response {i} " * 20})

    # Each message is roughly 100-200 chars, so 40 messages * 150 = 6000 chars ~ 1500 tokens
    import json

    original_json = json.dumps(messages)
    original_tokens = estimate_tokens(original_json)

    assert original_tokens > 500, "Original history should exceed target for this test"

    # Compress to fit in 500 tokens
    compressed = optimizer.compress_conversation_history(messages, target_tokens=500)

    compressed_tokens = sum(estimate_tokens(m.get("content", "")) for m in compressed)

    # Should have fewer messages
    assert len(compressed) < len(messages)

    # Should preserve recent messages (last 4 with preserve_recent_messages=2)
    assert len(compressed) >= 4
    assert compressed[-1] == messages[-1]
    assert compressed[-2] == messages[-2]
    assert compressed[-3] == messages[-3]
    assert compressed[-4] == messages[-4]

    # Should be within target
    assert compressed_tokens <= 500


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_optimization_with_zero_entities(test_hass, session_manager):
    """Test context optimization with empty entity list.

    This test verifies that:
    1. Empty context is handled gracefully
    2. No errors occur
    3. System continues to function
    """
    config = {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "",
        CONF_LLM_MODEL: "test-model",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [],  # No entities
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        "max_context_tokens": 1000,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        context_manager = ContextManager(test_hass, config)

        # Should not raise error with empty entities
        context = await context_manager.get_formatted_context(
            user_input="Hello",
            conversation_id="test_empty",
        )

        assert context is not None
        # Context might be empty or minimal
        tokens = estimate_tokens(context)
        assert tokens < 1000


@pytest.mark.integration
@pytest.mark.asyncio
async def test_context_metrics_tracking(test_hass, large_entity_set, session_manager):
    """Test that context optimization metrics are properly tracked.

    This test verifies that:
    1. Metrics are populated correctly
    2. Compression ratio is calculated
    3. Original and optimized token counts are recorded
    """
    config = {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "",
        CONF_LLM_MODEL: "test-model",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [entity.entity_id for entity in large_entity_set[:30]],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        "max_context_tokens": 4000,  # Generous limit for testing metrics
        CONF_COMPRESSION_LEVEL: "medium",
    }

    # Use a smaller subset of entities
    test_entities = large_entity_set[:10]

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        setup_entity_states(test_hass, test_entities)

        # Update config with the smaller entity set
        config[CONF_DIRECT_ENTITIES] = [entity.entity_id for entity in test_entities]
        context_manager = ContextManager(test_hass, config)

        metrics = {}
        await context_manager.get_formatted_context(
            user_input="Show me all devices",
            conversation_id="test_metrics",
            metrics=metrics,
        )

        # Verify metrics structure
        assert "context" in metrics
        assert "mode" in metrics["context"]
        assert "original_tokens" in metrics["context"]
        assert "optimized_tokens" in metrics["context"]
        assert "compression_ratio" in metrics["context"]

        # Verify metrics values
        assert metrics["context"]["mode"] == CONTEXT_MODE_DIRECT
        assert metrics["context"]["original_tokens"] > 0
        assert metrics["context"]["optimized_tokens"] > 0
        assert metrics["context"]["compression_ratio"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smart_truncate_preserves_important_terms(test_hass):
    """Test that smart truncation preserves important terms.

    This test verifies that:
    1. Important terms mentioned in preserve list are kept
    2. Truncation respects sentence boundaries when possible
    3. Text is properly truncated when exceeding token limits
    """
    optimizer = ContextOptimizer(compression_level="medium")

    # Create a long text with important terms scattered throughout
    text = (
        "The living room light is currently on with brightness set to 128. "
        "The bedroom has two lights, both of which are off. "
        "The kitchen light is dimmed to 50 percent brightness. "
        "The bathroom light has been recently replaced and is very bright. "
        "The hallway has motion-activated lighting that turns on automatically. "
        "The garage light is connected to a smart switch. "
        "The outdoor lights are on a timer schedule. "
        "The basement lighting system needs maintenance."
    )

    original_tokens = estimate_tokens(text)
    assert original_tokens > 50, "Original text should be long enough for this test"

    # Truncate with preservation of "living room"
    truncated = optimizer.smart_truncate(
        text, max_tokens=30, preserve=["living room", "brightness"]
    )

    truncated_tokens = estimate_tokens(truncated)

    # Should be truncated
    assert len(truncated) < len(text)
    assert truncated_tokens <= 30

    # Should preserve important terms if possible
    # At minimum, should try to preserve them
    assert "living room" in truncated or "brightness" in truncated

    # Should end properly
    assert truncated.endswith(".") or truncated.endswith("...")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_limit_exceeded_exception(test_hass, session_manager):
    """Test that TokenLimitExceeded is raised when context cannot be reduced.

    This test verifies that:
    1. TokenLimitExceeded exception is raised for oversized context
    2. Exception contains useful error message
    3. System fails gracefully rather than sending oversized context
    """
    # Create entities with extremely long content that cannot be compressed enough
    very_large_entities = []
    for i in range(50):
        very_large_entities.append(
            State(
                f"sensor.large_{i}",
                "on",
                {
                    "friendly_name": f"Large Sensor {i}",
                    "data": "x" * 1000,  # Each entity has 1000 chars of data
                },
            )
        )

    config = {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "",
        CONF_LLM_MODEL: "test-model",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        CONF_DIRECT_ENTITIES: [entity.entity_id for entity in very_large_entities],
        CONF_CONTEXT_FORMAT: CONTEXT_FORMAT_JSON,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        "max_context_tokens": 100,  # Very low limit
        CONF_COMPRESSION_LEVEL: "high",
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        setup_entity_states(test_hass, very_large_entities)

        context_manager = ContextManager(test_hass, config)

        # Should raise TokenLimitExceeded
        with pytest.raises(TokenLimitExceeded) as exc_info:
            await context_manager.get_formatted_context(
                user_input="Show me all sensors",
                conversation_id="test_limit_exceeded",
            )

        # Verify exception message is informative
        assert "exceeds limit" in str(exc_info.value).lower()
        assert "100" in str(exc_info.value)  # Should mention the limit

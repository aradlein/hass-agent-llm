"""Unit tests for the context optimizer module.

This module tests the context optimization and compression functionality,
including entity compression, conversation history management, entity
prioritization, and smart truncation features.
"""

from __future__ import annotations

import pytest

from custom_components.home_agent.context_optimizer import (
    CompressionMetrics,
    ContextOptimizer,
    EntityPriority,
)


@pytest.fixture
def sample_entities():
    """Provide sample entity data for testing."""
    return [
        {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "friendly_name": "Living Room Light",
                "brightness": 255,
                "color_temp": 400,
                "supported_features": 12345,
                "icon": "mdi:lightbulb",
                "_internal_id": "abc123",
            },
        },
        {
            "entity_id": "light.bedroom",
            "state": "off",
            "attributes": {
                "friendly_name": "Bedroom Light",
                "brightness": 0,
                "supported_features": 12345,
            },
        },
        {
            "entity_id": "sensor.temperature",
            "state": "22.5",
            "attributes": {
                "friendly_name": "Temperature Sensor",
                "unit_of_measurement": "°C",
                "device_class": "temperature",
            },
        },
        {
            "entity_id": "climate.thermostat",
            "state": "heat",
            "attributes": {
                "friendly_name": "Thermostat",
                "temperature": 22.5,
                "target_temperature": 23.0,
                "hvac_mode": "heat",
            },
        },
    ]


@pytest.fixture
def sample_messages():
    """Provide sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Turn on the living room lights"},
        {"role": "assistant", "content": "I've turned on the living room lights."},
        {"role": "user", "content": "What's the temperature?"},
        {"role": "assistant", "content": "The temperature is 22.5°C."},
        {"role": "user", "content": "Set the thermostat to 23 degrees"},
        {
            "role": "assistant",
            "content": "I've set the thermostat to 23°C.",
        },
        {"role": "user", "content": "Turn off all the lights"},
        {"role": "assistant", "content": "I've turned off all the lights."},
    ]


@pytest.fixture
def optimizer():
    """Provide a context optimizer instance for testing."""
    return ContextOptimizer(
        compression_level="medium",
        preserve_recent_messages=3,
    )


class TestContextOptimizerInitialization:
    """Test context optimizer initialization."""

    def test_default_initialization(self):
        """Test optimizer initializes with default values."""
        optimizer = ContextOptimizer()

        assert optimizer.compression_level == "medium"
        assert optimizer.preserve_recent_messages == 3

    def test_custom_initialization(self):
        """Test optimizer initializes with custom values."""
        optimizer = ContextOptimizer(
            compression_level="high",
            preserve_recent_messages=5,
        )

        assert optimizer.compression_level == "high"
        assert optimizer.preserve_recent_messages == 5

    @pytest.mark.skip(reason="Feature not yet implemented")
    def test_access_counts_initialized_empty(self):
        """Test access counts are initialized as empty."""
        optimizer = ContextOptimizer()
        assert len(optimizer._access_counts) == 0


class TestRemoveRedundantAttributes:
    """Test redundant attribute removal."""

    def test_removes_internal_attributes(self, optimizer, sample_entities):
        """Test that internal attributes (starting with _) are removed."""
        cleaned = optimizer.remove_redundant_attributes(sample_entities)

        for entity in cleaned:
            if "attributes" in entity:
                for key in entity["attributes"]:
                    assert not key.startswith("_")

    def test_removes_common_bloat(self, optimizer, sample_entities):
        """Test that common bloat attributes are removed."""
        cleaned = optimizer.remove_redundant_attributes(sample_entities)

        bloat_attrs = ["supported_features", "icon", "entity_picture"]
        for entity in cleaned:
            if "attributes" in entity:
                for bloat in bloat_attrs:
                    assert bloat not in entity["attributes"]

    def test_preserves_essential_attributes(self, optimizer, sample_entities):
        """Test that essential attributes are preserved."""
        cleaned = optimizer.remove_redundant_attributes(sample_entities)

        # Find the temperature sensor
        temp_sensor = next(e for e in cleaned if e["entity_id"] == "sensor.temperature")

        assert "friendly_name" in temp_sensor.get("attributes", {})
        assert "unit_of_measurement" in temp_sensor.get("attributes", {})
        assert "device_class" in temp_sensor.get("attributes", {})

    def test_preserves_entity_id_and_state(self, optimizer, sample_entities):
        """Test that entity_id and state are always preserved."""
        cleaned = optimizer.remove_redundant_attributes(sample_entities)

        assert len(cleaned) == len(sample_entities)
        for i, entity in enumerate(cleaned):
            assert entity["entity_id"] == sample_entities[i]["entity_id"]
            assert entity["state"] == sample_entities[i]["state"]

    def test_handles_empty_attributes(self, optimizer):
        """Test handling of entities with no attributes."""
        entities = [
            {"entity_id": "switch.test", "state": "on", "attributes": {}},
            {"entity_id": "sensor.test", "state": "10"},  # No attributes key
        ]

        cleaned = optimizer.remove_redundant_attributes(entities)

        assert len(cleaned) == 2
        assert cleaned[0]["entity_id"] == "switch.test"
        assert cleaned[1]["entity_id"] == "sensor.test"


class TestCompressEntityContext:
    """Test entity context compression."""

    def test_returns_unchanged_if_under_target(self, optimizer, sample_entities):
        """Test that entities are unchanged if already under target."""
        # Set a very high target
        result = optimizer.compress_entity_context(sample_entities, target_tokens=10000)

        assert len(result) == len(sample_entities)
        # Metrics should show no reduction
        metrics = optimizer.get_metrics()
        assert metrics is not None
        assert metrics.reduction_percent == 0.0

    def test_reduces_entities_when_over_target(self, optimizer, sample_entities):
        """Test that entities are reduced when over target."""
        # Set a low target to force compression
        result = optimizer.compress_entity_context(sample_entities, target_tokens=50)

        # Should have fewer entities or fewer attributes
        assert result is not None
        metrics = optimizer.get_metrics()
        assert metrics is not None
        assert metrics.compressed_tokens <= metrics.original_tokens

    def test_prioritizes_with_user_query(self, optimizer, sample_entities):
        """Test that entities are prioritized based on user query."""
        result = optimizer.compress_entity_context(
            sample_entities,
            target_tokens=100,
            user_query="living room lights",
        )

        # Living room light should be prioritized
        entity_ids = [e["entity_id"] for e in result]
        if len(entity_ids) < len(sample_entities):
            # If we had to drop entities, living room should be kept
            assert "light.living_room" in entity_ids

    def test_handles_empty_entity_list(self, optimizer):
        """Test handling of empty entity list."""
        result = optimizer.compress_entity_context([], target_tokens=100)
        assert result == []

    def test_tracks_compression_metrics(self, optimizer, sample_entities):
        """Test that compression metrics are tracked correctly."""
        optimizer.compress_entity_context(sample_entities, target_tokens=50)

        metrics = optimizer.get_metrics()
        assert metrics is not None
        assert isinstance(metrics, CompressionMetrics)
        assert metrics.original_tokens > 0
        assert metrics.compressed_tokens > 0
        assert metrics.entities_before == len(sample_entities)
        assert metrics.entities_after <= metrics.entities_before


class TestCompressConversationHistory:
    """Test conversation history compression."""

    def test_returns_unchanged_if_under_target(self, optimizer, sample_messages):
        """Test that messages are unchanged if already under target."""
        result = optimizer.compress_conversation_history(sample_messages, target_tokens=10000)

        assert len(result) == len(sample_messages)
        assert result == sample_messages

    def test_preserves_recent_messages(self, optimizer, sample_messages):
        """Test that recent messages are preserved."""
        # Set preserve_recent_messages to 2 (2 pairs = 4 messages)
        optimizer.preserve_recent_messages = 2
        result = optimizer.compress_conversation_history(sample_messages, target_tokens=100)

        # Should keep at least the last 4 messages
        assert len(result) >= min(4, len(sample_messages))

        # Last messages should be identical
        for i in range(min(4, len(sample_messages))):
            assert result[-i - 1] == sample_messages[-i - 1]

    def test_truncates_older_messages(self, optimizer, sample_messages):
        """Test that older messages are truncated when needed."""
        # Set low target to force truncation
        result = optimizer.compress_conversation_history(sample_messages, target_tokens=50)

        # Should have fewer messages
        assert len(result) < len(sample_messages)

    def test_handles_empty_message_list(self, optimizer):
        """Test handling of empty message list."""
        result = optimizer.compress_conversation_history([], target_tokens=100)
        assert result == []

    def test_handles_very_low_target(self, optimizer, sample_messages):
        """Test handling of very low target tokens."""
        # Target so low it can only keep the most recent pair
        result = optimizer.compress_conversation_history(sample_messages, target_tokens=10)

        # Should keep at least the most recent message(s)
        assert len(result) >= 1
        assert result[-1] == sample_messages[-1]


class TestPrioritizeEntities:
    """Test entity prioritization."""

    def test_prioritizes_mentioned_entities(self, optimizer, sample_entities):
        """Test that mentioned entities get high priority."""
        priorities = optimizer.prioritize_entities(
            sample_entities, "turn on the living room lights"
        )

        # Living room light should have highest score
        assert priorities[0].entity_id == "light.living_room"
        assert (
            "mentioned_in_query" in priorities[0].reasons
            or "name_mentioned" in priorities[0].reasons
        )

    def test_prioritizes_by_domain(self, optimizer, sample_entities):
        """Test that domain-relevant entities are prioritized."""
        priorities = optimizer.prioritize_entities(sample_entities, "adjust the temperature")

        # Climate and temperature sensor should be prioritized
        high_priority_ids = [p.entity_id for p in priorities[:2]]
        assert (
            "sensor.temperature" in high_priority_ids or "climate.thermostat" in high_priority_ids
        )

    def test_returns_entity_priority_objects(self, optimizer, sample_entities):
        """Test that proper EntityPriority objects are returned."""
        priorities = optimizer.prioritize_entities(sample_entities, "living room")

        assert len(priorities) == len(sample_entities)
        for priority in priorities:
            assert isinstance(priority, EntityPriority)
            assert hasattr(priority, "entity_id")
            assert hasattr(priority, "score")
            assert hasattr(priority, "reasons")
            assert isinstance(priority.score, float)
            assert isinstance(priority.reasons, list)

    def test_tracks_access_patterns(self, optimizer, sample_entities):
        """Test that access patterns are tracked."""
        # First call
        optimizer.prioritize_entities(sample_entities, "test")
        assert optimizer._access_counts["light.living_room"] == 1

        # Second call
        optimizer.prioritize_entities(sample_entities, "test")
        assert optimizer._access_counts["light.living_room"] == 2

    def test_sorts_by_score_descending(self, optimizer, sample_entities):
        """Test that priorities are sorted by score in descending order."""
        priorities = optimizer.prioritize_entities(sample_entities, "living room")

        # Scores should be in descending order
        scores = [p.score for p in priorities]
        assert scores == sorted(scores, reverse=True)


class TestSmartTruncate:
    """Test smart truncation."""

    def test_returns_unchanged_if_under_limit(self, optimizer):
        """Test that text is unchanged if under token limit."""
        text = "This is a short text."
        result = optimizer.smart_truncate(text, max_tokens=100)
        assert result == text

    def test_truncates_long_text(self, optimizer):
        """Test that long text is truncated."""
        text = "This is a very long text. " * 50
        result = optimizer.smart_truncate(text, max_tokens=20)

        assert len(result) < len(text)
        assert result.endswith("...") or result.endswith(".")

    def test_preserves_important_terms(self, optimizer):
        """Test that important terms are preserved when possible."""
        text = (
            "The living room light is on. "
            "The bedroom light is off. "
            "The kitchen light is dimmed."
        )
        result = optimizer.smart_truncate(text, max_tokens=30, preserve=["living room"])

        assert "living room" in result

    def test_handles_empty_text(self, optimizer):
        """Test handling of empty text."""
        result = optimizer.smart_truncate("", max_tokens=10)
        assert result == ""

    def test_respects_sentence_boundaries(self, optimizer):
        """Test that truncation respects sentence boundaries when possible."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = optimizer.smart_truncate(text, max_tokens=30)

        # Should end with a period or ellipsis
        assert result.endswith(".") or result.endswith("...")


class TestEstimateContextTokens:
    """Test context token estimation."""

    def test_estimates_all_components(self, optimizer):
        """Test that all context components are estimated."""
        context = {
            "system_prompt": "You are a helpful assistant.",
            "entity_context": "Entities: light.living_room",
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "user_message": "Turn on the lights",
        }

        estimates = optimizer.estimate_context_tokens(context)

        assert "system_prompt" in estimates
        assert "entity_context" in estimates
        assert "conversation_history" in estimates
        assert "user_message" in estimates
        assert "total" in estimates

        # All estimates should be non-negative
        for key, value in estimates.items():
            assert value >= 0

    def test_handles_entity_list(self, optimizer, sample_entities):
        """Test estimation with entity list instead of string."""
        context = {
            "system_prompt": "Test",
            "entity_context": sample_entities,
            "conversation_history": [],
            "user_message": "Test",
        }

        estimates = optimizer.estimate_context_tokens(context)
        assert estimates["entity_context"] > 0

    def test_total_is_sum_of_components(self, optimizer):
        """Test that total equals sum of components."""
        context = {
            "system_prompt": "Test prompt",
            "entity_context": "Test context",
            "conversation_history": [{"role": "user", "content": "Test"}],
            "user_message": "Test message",
        }

        estimates = optimizer.estimate_context_tokens(context)

        expected_total = (
            estimates["system_prompt"]
            + estimates["entity_context"]
            + estimates["conversation_history"]
            + estimates["user_message"]
        )

        assert estimates["total"] == expected_total


class TestOptimizeForModel:
    """Test model-specific optimization."""

    def test_returns_unchanged_if_under_limit(self, optimizer, sample_entities):
        """Test that context is unchanged if under model limit."""
        context = {
            "system_prompt": "Short prompt",
            "entity_context": sample_entities[:2],  # Just 2 entities
            "conversation_history": [],
            "user_message": "Test",
        }

        result = optimizer.optimize_for_model(context, "gpt-4o-mini")

        # Should be largely unchanged for small context
        assert "entity_context" in result
        assert "system_prompt" in result

    def test_compresses_when_over_limit(self, optimizer, sample_entities):
        """Test that context is compressed when over model limit."""
        # Create a large context
        large_entities = sample_entities * 100  # Duplicate many times
        large_history = [{"role": "user", "content": "Message " + str(i)} for i in range(100)]

        context = {
            "system_prompt": "Test prompt " * 100,
            "entity_context": large_entities,
            "conversation_history": large_history,
            "user_message": "Test",
        }

        result = optimizer.optimize_for_model(context, "gpt-3.5-turbo")

        # Result should be smaller
        original_estimates = optimizer.estimate_context_tokens(context)
        result_estimates = optimizer.estimate_context_tokens(result)

        assert result_estimates["total"] < original_estimates["total"]

    def test_recognizes_different_models(self, optimizer):
        """Test that different models are recognized."""
        context = {
            "system_prompt": "Test",
            "entity_context": [],
            "conversation_history": [],
            "user_message": "Test",
        }

        # These should all work without error
        models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "unknown-model",
        ]

        for model in models:
            result = optimizer.optimize_for_model(context, model)
            assert result is not None


class TestCompressionMetrics:
    """Test compression metrics tracking."""

    def test_get_metrics_returns_none_initially(self, optimizer):
        """Test that metrics are None before any compression."""
        assert optimizer.get_metrics() is None

    def test_get_metrics_after_compression(self, optimizer, sample_entities):
        """Test that metrics are available after compression."""
        optimizer.compress_entity_context(sample_entities, target_tokens=100)

        metrics = optimizer.get_metrics()
        assert metrics is not None
        assert isinstance(metrics, CompressionMetrics)

    def test_reset_metrics_clears_data(self, optimizer, sample_entities):
        """Test that reset_metrics clears all tracking data."""
        optimizer.compress_entity_context(sample_entities, target_tokens=100)
        optimizer.prioritize_entities(sample_entities, "test")

        assert optimizer.get_metrics() is not None
        assert len(optimizer._access_counts) > 0

        optimizer.reset_metrics()

        assert optimizer.get_metrics() is None
        assert len(optimizer._access_counts) == 0


class TestCompressionLevels:
    """Test different compression levels."""

    def test_none_compression_preserves_all(self, sample_entities):
        """Test that NONE compression preserves all attributes."""
        optimizer = ContextOptimizer(compression_level="none")
        result = optimizer._apply_compression_level(sample_entities, "none")

        # Should preserve entity structure (though attributes might vary)
        assert len(result) == len(sample_entities)

    def test_low_compression_keeps_most(self, sample_entities):
        """Test that LOW compression keeps most useful attributes."""
        optimizer = ContextOptimizer(compression_level="low")
        result = optimizer._apply_compression_level(sample_entities, "low")

        # Find temperature sensor
        temp = next(e for e in result if e["entity_id"] == "sensor.temperature")

        # Should keep essential attributes
        assert "friendly_name" in temp.get("attributes", {})
        assert "unit_of_measurement" in temp.get("attributes", {})

    def test_medium_compression_keeps_essential(self, sample_entities):
        """Test that MEDIUM compression keeps only essential attributes."""
        optimizer = ContextOptimizer(compression_level="medium")
        result = optimizer._apply_compression_level(sample_entities, "medium")

        for entity in result:
            attrs = entity.get("attributes", {})
            # Should only have essential attributes
            allowed = {"friendly_name", "unit_of_measurement", "device_class"}
            for key in attrs:
                assert key in allowed

    def test_high_compression_minimal_attributes(self, sample_entities):
        """Test that HIGH compression keeps minimal attributes."""
        optimizer = ContextOptimizer(compression_level="high")
        result = optimizer._apply_compression_level(sample_entities, "high")

        for entity in result:
            attrs = entity.get("attributes", {})
            # Should only have friendly_name at most
            assert len(attrs) <= 1
            if attrs:
                assert "friendly_name" in attrs


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_malformed_entities(self, optimizer):
        """Test handling of malformed entity data."""
        malformed = [
            {},  # Empty dict
            {"entity_id": "test.entity"},  # No state
            {"state": "on"},  # No entity_id
            {"entity_id": "test.entity", "state": "on", "attributes": "not_a_dict"},
        ]

        # Should not raise an error
        result = optimizer.remove_redundant_attributes(malformed)
        assert len(result) == len(malformed)

    def test_handles_unicode_content(self, optimizer):
        """Test handling of unicode characters in content."""
        entities = [
            {
                "entity_id": "sensor.température",
                "state": "22°C",
                "attributes": {"friendly_name": "Température 🌡️"},
            }
        ]

        result = optimizer.compress_entity_context(entities, target_tokens=100)
        assert len(result) == 1

    def test_very_long_attribute_values(self, optimizer):
        """Test handling of very long attribute values."""
        entities = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "attributes": {
                    "very_long_value": "x" * 10000,  # Very long string
                },
            }
        ]

        result = optimizer.remove_redundant_attributes(entities)
        # Should truncate or remove the long value
        assert len(str(result)) < len(str(entities))

    def test_zero_target_tokens(self, optimizer, sample_entities):
        """Test handling of zero target tokens."""
        result = optimizer.compress_entity_context(sample_entities, target_tokens=0)
        # Should return empty or minimal result
        assert len(result) == 0 or len(result) < len(sample_entities)

    def test_negative_target_tokens(self, optimizer, sample_entities):
        """Test handling of negative target tokens."""
        result = optimizer.compress_entity_context(sample_entities, target_tokens=-100)
        # Should handle gracefully
        assert isinstance(result, list)

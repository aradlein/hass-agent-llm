"""Unit tests for memory quality validation.

Tests the enhanced validation logic that rejects low-quality memories including:
- Meta-information about conversations
- Negative existence statements
- Content that is too short
- Content with low importance scores
- Transient states and low-value patterns

These tests validate both positive cases (good memories that should be stored)
and negative cases (low-quality memories that should be rejected).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_MEMORY_EXTRACTION_LLM,
)
from custom_components.home_agent.memory_manager import MemoryManager


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    mock = MagicMock()
    mock.data = {}
    mock.bus = MagicMock()
    mock.bus.async_fire = MagicMock()
    mock.async_create_task = MagicMock()
    mock.async_add_executor_job = AsyncMock()
    return mock


@pytest.fixture
def agent_config():
    """Create agent configuration."""
    return {
        CONF_LLM_BASE_URL: "http://test.com",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_EXTRACTION_ENABLED: True,
        CONF_MEMORY_EXTRACTION_LLM: "local",
    }


@pytest.fixture
def memory_manager(mock_hass):
    """Create a MemoryManager instance for testing."""
    config = {
        "memory_dedup_threshold": 0.9,
    }
    with patch("custom_components.home_agent.vector_db_manager.VectorDBManager"):
        mock_vector_db = MagicMock()
        manager = MemoryManager(mock_hass, mock_vector_db, config)
        return manager


@pytest.fixture
def home_agent(mock_hass, agent_config, memory_manager):
    """Create a HomeAgent instance for testing."""
    with patch("custom_components.home_agent.agent.ContextManager"):
        with patch("custom_components.home_agent.agent.ConversationHistoryManager"):
            with patch("custom_components.home_agent.agent.ToolHandler"):
                from custom_components.home_agent.conversation_session import (
                    ConversationSessionManager,
                )

                session_manager = ConversationSessionManager(mock_hass)
                agent = HomeAgent(mock_hass, agent_config, session_manager)
                agent._memory_manager = memory_manager
                return agent


class TestTransientStateLowValuePatternDetection:
    """Test the enhanced _is_transient_state method that detects low-value patterns."""

    def test_detects_device_state_patterns(self, memory_manager):
        """Test that device state patterns are detected as transient."""
        transient_contents = [
            "The light is on in the bedroom",
            "Temperature is currently 72 degrees",
            "The door is closed",
            "Garage door is open",
            "Thermostat status is heating",
            "All lights are off",
        ]

        for content in transient_contents:
            assert memory_manager._is_transient_state(content), f"Should detect: {content}"

    def test_detects_conversational_meta_patterns(self, memory_manager):
        """Test that conversational meta-information is detected as low-quality."""
        low_value_contents = [
            "The conversation occurred at 3pm",
            "We discussed the temperature settings",
            "User asked about the bedroom sensor",
            "I mentioned the thermostat earlier",
            "During the conversation, the user seemed confused",
            "We talked about automation routines",
        ]

        for content in low_value_contents:
            assert memory_manager._is_transient_state(content), f"Should detect: {content}"

    def test_detects_negative_existence_patterns(self, memory_manager):
        """Test that negative existence statements are detected as low-quality."""
        negative_contents = [
            "There is no specific bed occupancy sensor",
            "There are no motion sensors in the garage",
            "The bedroom does not have a temperature sensor",
            "No specific automation for morning routines",
            "User doesn't have a smart doorbell",
        ]

        for content in negative_contents:
            assert memory_manager._is_transient_state(content), f"Should detect: {content}"

    def test_allows_high_quality_content(self, memory_manager):
        """Test that high-quality content is NOT flagged as transient/low-value."""
        good_contents = [
            "User prefers bedroom temperature at 68°F for sleeping",
            "The bedroom has blackout curtains installed last month",
            "User works night shift and sleeps during the day",
            "Morning routine starts at 6:30am with gradual lighting",
            "User is sensitive to bright lights in the evening",
        ]

        for content in good_contents:
            assert not memory_manager._is_transient_state(content), f"Should NOT detect: {content}"


class TestMemoryQualityValidation:
    """Test memory quality validation in _parse_and_store_memories."""

    @pytest.mark.asyncio
    async def test_rejects_short_memories(self, home_agent):
        """Test that memories with less than 10 meaningful words are rejected."""
        # Mock the memory manager's add_memory to track calls
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        short_memories = [
            {
                "type": "fact",
                "content": "User likes blue",  # Only 3 words
                "importance": 0.8,
            },
            {
                "type": "preference",
                "content": "Temperature at 70",  # Only 3 words
                "importance": 0.7,
            },
        ]

        extraction_result = json.dumps(short_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        # No memories should be stored
        assert stored_count == 0
        assert home_agent.memory_manager.add_memory.call_count == 0

    @pytest.mark.asyncio
    async def test_rejects_low_importance_memories(self, home_agent):
        """Test that memories with importance < 0.4 are rejected."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        low_importance_memories = [
            {
                "type": "fact",
                "content": "User mentioned they like the color blue for accent lighting in rooms",
                "importance": 0.2,  # Too low
            },
            {
                "type": "preference",
                "content": "User prefers the bedroom temperature to be set at exactly 70 degrees",
                "importance": 0.35,  # Too low
            },
        ]

        extraction_result = json.dumps(low_importance_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert stored_count == 0
        assert home_agent.memory_manager.add_memory.call_count == 0

    @pytest.mark.asyncio
    async def test_rejects_low_value_starting_patterns(self, home_agent):
        """Test that memories starting with low-value phrases are rejected."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        low_value_memories = [
            {
                "type": "fact",
                "content": "There is no specific bed occupancy sensor in the bedroom at this time",
                "importance": 0.6,
            },
            {
                "type": "context",
                "content": "We discussed the temperature settings for the bedroom during evening hours",
                "importance": 0.7,
            },
            {
                "type": "event",
                "content": "User asked about the thermostat settings during our conversation earlier today",
                "importance": 0.5,
            },
        ]

        extraction_result = json.dumps(low_value_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert stored_count == 0
        assert home_agent.memory_manager.add_memory.call_count == 0

    @pytest.mark.asyncio
    async def test_rejects_transient_state_content(self, home_agent):
        """Test that transient state and low-quality content is rejected for all types."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        transient_memories = [
            {
                "type": "fact",
                "content": "The bedroom light is currently on and has been since earlier this evening",
                "importance": 0.6,
            },
            {
                "type": "event",
                "content": "The conversation occurred at three pm when the user asked about controls",
                "importance": 0.5,
            },
        ]

        extraction_result = json.dumps(transient_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert stored_count == 0
        assert home_agent.memory_manager.add_memory.call_count == 0

    @pytest.mark.asyncio
    async def test_accepts_high_quality_memories(self, home_agent):
        """Test that high-quality memories pass all validation checks and are stored."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        good_memories = [
            {
                "type": "preference",
                "content": "User prefers bedroom temperature at 68°F for sleeping and 72°F when awake",
                "importance": 0.8,
                "entities": ["climate.bedroom"],
                "topics": ["temperature", "bedroom", "sleep"],
            },
            {
                "type": "fact",
                "content": "The bedroom has blackout curtains that were installed last month for better sleep",
                "importance": 0.7,
                "entities": [],
                "topics": ["bedroom", "curtains", "sleep"],
            },
            {
                "type": "context",
                "content": "User works night shift schedule and sleeps during daytime hours between 8am and 4pm",
                "importance": 0.9,
                "entities": [],
                "topics": ["schedule", "work", "sleep"],
            },
            {
                "type": "event",
                "content": "User installed new smart thermostat last week and configured heating schedules for winter",
                "importance": 0.6,
                "entities": ["climate.bedroom"],
                "topics": ["thermostat", "installation", "heating"],
            },
        ]

        extraction_result = json.dumps(good_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        # All 4 memories should be stored
        assert stored_count == 4
        assert home_agent.memory_manager.add_memory.call_count == 4

        # Verify the content of stored memories
        for i, memory in enumerate(good_memories):
            call_kwargs = home_agent.memory_manager.add_memory.call_args_list[i][1]
            assert call_kwargs["content"] == memory["content"]
            assert call_kwargs["memory_type"] == memory["type"]
            assert call_kwargs["importance"] == memory["importance"]

    @pytest.mark.asyncio
    async def test_mixed_quality_memories(self, home_agent):
        """Test that only high-quality memories are stored when mixed with low-quality."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        mixed_memories = [
            # Good memory - should be stored
            {
                "type": "preference",
                "content": "User prefers bedroom temperature at 68°F for optimal sleeping conditions every night",
                "importance": 0.8,
            },
            # Too short - should be rejected
            {
                "type": "fact",
                "content": "User likes blue lights",
                "importance": 0.7,
            },
            # Good memory - should be stored
            {
                "type": "context",
                "content": "User is very sensitive to bright lights in the evening after 8pm due to migraines",
                "importance": 0.9,
            },
            # Low importance - should be rejected
            {
                "type": "fact",
                "content": "The user mentioned they sometimes check the weather forecast before leaving home in morning",
                "importance": 0.3,
            },
            # Starts with low-value pattern - should be rejected
            {
                "type": "fact",
                "content": "There is no specific automation configured for the morning routine at this moment",
                "importance": 0.6,
            },
            # Good memory - should be stored
            {
                "type": "fact",
                "content": "The living room has three smart bulbs configured for different lighting scenes and moods",
                "importance": 0.7,
            },
            # Transient state - should be rejected
            {
                "type": "event",
                "content": "The bedroom light is currently on and the temperature is at seventy two degrees",
                "importance": 0.5,
            },
        ]

        extraction_result = json.dumps(mixed_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        # Only 3 good memories should be stored
        assert stored_count == 3
        assert home_agent.memory_manager.add_memory.call_count == 3

        # Verify that the right memories were stored (the good ones)
        stored_contents = [
            call[1]["content"] for call in home_agent.memory_manager.add_memory.call_args_list
        ]
        assert "User prefers bedroom temperature at 68°F" in stored_contents[0]
        assert "sensitive to bright lights" in stored_contents[1]
        assert "three smart bulbs" in stored_contents[2]


class TestMemoryValidationEdgeCases:
    """Test edge cases in memory quality validation."""

    @pytest.mark.asyncio
    async def test_handles_empty_extraction_result(self, home_agent):
        """Test handling of empty memory extraction results."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        extraction_result = json.dumps([])
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert stored_count == 0
        assert home_agent.memory_manager.add_memory.call_count == 0

    @pytest.mark.asyncio
    async def test_word_count_excludes_short_words(self, home_agent):
        """Test that word count validation excludes very short words (<=2 chars)."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        # Content has 15 total words but only 6 meaningful words (>2 chars)
        # "a", "to", "at", "in", "is", "on", "of", "it", "or" are all <=2 chars
        short_word_memory = [
            {
                "type": "fact",
                "content": "A user is in a room at a time or on it",  # Only short words
                "importance": 0.8,
            }
        ]

        extraction_result = json.dumps(short_word_memory)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        # Should be rejected due to insufficient meaningful words
        assert stored_count == 0

    @pytest.mark.asyncio
    async def test_importance_threshold_at_boundary(self, home_agent):
        """Test importance threshold at exactly 0.4 (boundary case)."""
        home_agent.memory_manager.add_memory = AsyncMock(return_value="mem_123")

        # Test at threshold boundary
        boundary_memories = [
            {
                "type": "fact",
                "content": "User mentioned they prefer using voice commands for controlling bedroom lights primarily",
                "importance": 0.4,  # Exactly at threshold - should be stored
            },
            {
                "type": "fact",
                "content": "User sometimes checks the temperature sensor readings in the morning before work",
                "importance": 0.39,  # Just below threshold - should be rejected
            },
        ]

        extraction_result = json.dumps(boundary_memories)
        stored_count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        # Only the first one (0.4) should be stored
        assert stored_count == 1
        assert home_agent.memory_manager.add_memory.call_count == 1

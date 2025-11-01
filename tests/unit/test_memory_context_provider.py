"""Unit tests for MemoryContextProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.const import (
    CONF_MEMORY_CONTEXT_TOP_K,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MIN_IMPORTANCE,
    DEFAULT_MEMORY_CONTEXT_TOP_K,
    DEFAULT_MEMORY_ENABLED,
    DEFAULT_MEMORY_MIN_IMPORTANCE,
)
from custom_components.home_agent.context_providers.memory import (
    MemoryContextProvider,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager instance."""
    mock_manager = MagicMock()
    mock_manager.search_memories = AsyncMock()
    return mock_manager


@pytest.fixture
def memory_context_provider(mock_hass, mock_memory_manager):
    """Create a MemoryContextProvider instance."""
    config = {
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_CONTEXT_TOP_K: 5,
        CONF_MEMORY_MIN_IMPORTANCE: 0.3,
    }
    return MemoryContextProvider(mock_hass, config, mock_memory_manager)


@pytest.mark.asyncio
async def test_get_context_memory_disabled(mock_hass, mock_memory_manager):
    """Test that get_context returns empty string when memory is disabled."""
    config = {CONF_MEMORY_ENABLED: False}
    provider = MemoryContextProvider(mock_hass, config, mock_memory_manager)

    context = await provider.get_context("turn on the lights")

    assert context == ""
    mock_memory_manager.search_memories.assert_not_called()


@pytest.mark.asyncio
async def test_get_context_no_memories_found(memory_context_provider, mock_memory_manager):
    """Test get_context when no relevant memories are found."""
    mock_memory_manager.search_memories.return_value = []

    context = await memory_context_provider.get_context("what's the weather?")

    assert context == ""
    mock_memory_manager.search_memories.assert_called_once_with(
        query="what's the weather?",
        top_k=5,
        min_importance=0.3,
    )


@pytest.mark.asyncio
async def test_get_context_with_memories(memory_context_provider, mock_memory_manager):
    """Test get_context successfully retrieves and formats memories."""
    mock_memories = [
        {
            "id": "mem1",
            "type": "preference",
            "content": "User prefers bedroom temperature at 68°F",
            "importance": 0.8,
        },
        {
            "id": "mem2",
            "type": "fact",
            "content": "Living room has smart lights",
            "importance": 0.6,
        },
        {
            "id": "mem3",
            "type": "context",
            "content": "User usually wakes up at 7 AM",
            "importance": 0.5,
        },
    ]
    mock_memory_manager.search_memories.return_value = mock_memories

    context = await memory_context_provider.get_context("set bedroom temperature")

    # Verify search was called with correct parameters
    mock_memory_manager.search_memories.assert_called_once_with(
        query="set bedroom temperature",
        top_k=5,
        min_importance=0.3,
    )

    # Verify context formatting
    assert "## Relevant Information from Past Conversations" in context
    assert "[Preference] User prefers bedroom temperature at 68°F" in context
    assert "[Fact] Living room has smart lights" in context
    assert "[Context] User usually wakes up at 7 AM" in context


@pytest.mark.asyncio
async def test_get_context_uses_config_parameters(mock_hass, mock_memory_manager):
    """Test that get_context uses configured parameters."""
    config = {
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_CONTEXT_TOP_K: 10,
        CONF_MEMORY_MIN_IMPORTANCE: 0.7,
    }
    provider = MemoryContextProvider(mock_hass, config, mock_memory_manager)
    mock_memory_manager.search_memories.return_value = []

    await provider.get_context("test query")

    mock_memory_manager.search_memories.assert_called_once_with(
        query="test query",
        top_k=10,
        min_importance=0.7,
    )


@pytest.mark.asyncio
async def test_get_context_uses_defaults(mock_hass, mock_memory_manager):
    """Test that get_context uses default values when not configured."""
    config = {}  # Empty config
    provider = MemoryContextProvider(mock_hass, config, mock_memory_manager)
    mock_memory_manager.search_memories.return_value = []

    await provider.get_context("test query")

    mock_memory_manager.search_memories.assert_called_once_with(
        query="test query",
        top_k=DEFAULT_MEMORY_CONTEXT_TOP_K,
        min_importance=DEFAULT_MEMORY_MIN_IMPORTANCE,
    )


@pytest.mark.asyncio
async def test_get_context_handles_search_error(memory_context_provider, mock_memory_manager):
    """Test that get_context handles errors gracefully."""
    mock_memory_manager.search_memories.side_effect = Exception("Database error")

    context = await memory_context_provider.get_context("test query")

    # Should return empty string on error
    assert context == ""


@pytest.mark.asyncio
async def test_format_memories_empty_list(memory_context_provider):
    """Test _format_memories with empty list."""
    result = memory_context_provider._format_memories([])

    assert result == ""


@pytest.mark.asyncio
async def test_format_memories_missing_fields(memory_context_provider):
    """Test _format_memories handles missing fields gracefully."""
    memories = [
        {"id": "mem1"},  # Missing type and content
        {"id": "mem2", "content": "Test content"},  # Missing type
    ]

    result = memory_context_provider._format_memories(memories)

    assert "## Relevant Information from Past Conversations" in result
    assert "[Fact]" in result  # Default type
    assert "Test content" in result


@pytest.mark.asyncio
async def test_format_memories_type_capitalization(memory_context_provider):
    """Test that memory types are properly capitalized."""
    memories = [
        {"type": "preference", "content": "Test 1"},
        {"type": "fact", "content": "Test 2"},
        {"type": "context", "content": "Test 3"},
    ]

    result = memory_context_provider._format_memories(memories)

    assert "[Preference]" in result
    assert "[Fact]" in result
    assert "[Context]" in result


@pytest.mark.asyncio
async def test_get_context_with_conversation_id(memory_context_provider, mock_memory_manager):
    """Test get_context accepts conversation_id parameter."""
    mock_memory_manager.search_memories.return_value = []

    # conversation_id is accepted but not currently used in search
    context = await memory_context_provider.get_context(
        "test query",
        conversation_id="conv123",
    )

    assert context == ""
    # Verify search was called (conversation_id might be used in future)
    mock_memory_manager.search_memories.assert_called_once()


@pytest.mark.asyncio
async def test_get_context_integration(memory_context_provider, mock_memory_manager):
    """Test complete integration flow from query to formatted output."""
    mock_memories = [
        {
            "type": "preference",
            "content": "Prefers lights dimmed at night",
            "importance": 0.9,
        }
    ]
    mock_memory_manager.search_memories.return_value = mock_memories

    context = await memory_context_provider.get_context(
        "turn on bedroom lights",
        conversation_id="test_conv",
    )

    # Verify full output structure
    assert context.startswith("## Relevant Information from Past Conversations\n\n")
    assert "- [Preference] Prefers lights dimmed at night\n" in context
    assert context.endswith("\n")

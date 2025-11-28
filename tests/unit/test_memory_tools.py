"""Unit tests for memory tools."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_agent.exceptions import ToolExecutionError
from custom_components.home_agent.tools.memory_tools import (
    TOOL_RECALL_MEMORY,
    TOOL_STORE_MEMORY,
    RecallMemoryTool,
    StoreMemoryTool,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    return MagicMock()


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager instance."""
    mock_manager = MagicMock()
    mock_manager.add_memory = AsyncMock()
    mock_manager.search_memories = AsyncMock()
    return mock_manager


@pytest.fixture
def store_memory_tool(mock_hass, mock_memory_manager):
    """Create a StoreMemoryTool instance."""
    return StoreMemoryTool(mock_hass, mock_memory_manager, conversation_id="test_conv")


@pytest.fixture
def recall_memory_tool(mock_hass, mock_memory_manager):
    """Create a RecallMemoryTool instance."""
    return RecallMemoryTool(mock_hass, mock_memory_manager)


class TestStoreMemoryTool:
    """Tests for StoreMemoryTool."""

    def test_tool_name(self, store_memory_tool):
        """Test that tool name is correct."""
        assert store_memory_tool.name == TOOL_STORE_MEMORY

    def test_tool_description(self, store_memory_tool):
        """Test that tool has a description."""
        description = store_memory_tool.description
        assert isinstance(description, str)
        assert len(description) > 0
        assert "store" in description.lower()

    def test_tool_parameters_schema(self, store_memory_tool):
        """Test that parameter schema is valid."""
        params = store_memory_tool.parameters

        assert params["type"] == "object"
        assert "content" in params["properties"]
        assert "memory_type" in params["properties"]
        assert "importance" in params["properties"]
        assert params["required"] == ["content"]

    def test_tool_openai_format(self, store_memory_tool):
        """Test tool definition in OpenAI format."""
        definition = store_memory_tool.to_openai_format()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == TOOL_STORE_MEMORY
        assert "description" in definition["function"]
        assert "parameters" in definition["function"]

    @pytest.mark.asyncio
    async def test_execute_success(self, store_memory_tool, mock_memory_manager):
        """Test successful memory storage."""
        mock_memory_manager.add_memory.return_value = "mem_123"

        result = await store_memory_tool.execute(
            content="User prefers lights at 50% brightness",
            memory_type="preference",
            importance=0.8,
        )

        assert result["success"] is True
        assert "mem_123" in result["message"]

        # Verify memory was stored correctly
        mock_memory_manager.add_memory.assert_called_once_with(
            content="User prefers lights at 50% brightness",
            memory_type="preference",
            conversation_id="test_conv",
            importance=0.8,
            metadata={
                "extraction_method": "manual",
                "tool": TOOL_STORE_MEMORY,
            },
        )

    @pytest.mark.asyncio
    async def test_execute_with_defaults(self, store_memory_tool, mock_memory_manager):
        """Test execute with default parameters."""
        mock_memory_manager.add_memory.return_value = "mem_456"

        result = await store_memory_tool.execute(
            content="Test fact",
        )

        assert result["success"] is True

        # Verify defaults were used
        mock_memory_manager.add_memory.assert_called_once()
        call_kwargs = mock_memory_manager.add_memory.call_args[1]
        assert call_kwargs["memory_type"] == "fact"
        assert call_kwargs["importance"] == 0.5

    @pytest.mark.asyncio
    async def test_execute_missing_content(self, store_memory_tool, mock_memory_manager):
        """Test execute fails when content is missing."""
        with pytest.raises(ToolExecutionError, match="Missing required parameter: content"):
            await store_memory_tool.execute()

    @pytest.mark.asyncio
    async def test_execute_empty_content(self, store_memory_tool, mock_memory_manager):
        """Test execute fails when content is empty."""
        with pytest.raises(ToolExecutionError, match="Missing required parameter: content"):
            await store_memory_tool.execute(content="")

    @pytest.mark.asyncio
    async def test_execute_memory_manager_error(self, store_memory_tool, mock_memory_manager):
        """Test execute handles memory manager errors."""
        mock_memory_manager.add_memory.side_effect = Exception("Database error")

        with pytest.raises(ToolExecutionError, match="Failed to store memory"):
            await store_memory_tool.execute(content="Test")


class TestRecallMemoryTool:
    """Tests for RecallMemoryTool."""

    def test_tool_name(self, recall_memory_tool):
        """Test that tool name is correct."""
        assert recall_memory_tool.name == TOOL_RECALL_MEMORY

    def test_tool_description(self, recall_memory_tool):
        """Test that tool has a description."""
        description = recall_memory_tool.description
        assert isinstance(description, str)
        assert len(description) > 0
        assert "search" in description.lower() or "recall" in description.lower()

    def test_tool_parameters_schema(self, recall_memory_tool):
        """Test that parameter schema is valid."""
        params = recall_memory_tool.parameters

        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert params["required"] == ["query"]

    def test_tool_openai_format(self, recall_memory_tool):
        """Test tool definition in OpenAI format."""
        definition = recall_memory_tool.to_openai_format()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == TOOL_RECALL_MEMORY
        assert "description" in definition["function"]
        assert "parameters" in definition["function"]

    @pytest.mark.asyncio
    async def test_execute_success(self, recall_memory_tool, mock_memory_manager):
        """Test successful memory recall."""
        mock_memories = [
            {
                "type": "preference",
                "content": "Prefers bedroom at 68°F",
                "importance": 0.8,
            },
            {
                "type": "fact",
                "content": "Has 3 bedrooms",
                "importance": 0.6,
            },
        ]
        mock_memory_manager.search_memories.return_value = mock_memories

        result = await recall_memory_tool.execute(
            query="bedroom temperature",
            limit=5,
        )

        assert result["success"] is True
        assert "Found 2 relevant memories" in result["message"]
        assert "[Preference] Prefers bedroom at 68°F" in result["message"]
        assert "[Fact] Has 3 bedrooms" in result["message"]

        # Verify search parameters
        mock_memory_manager.search_memories.assert_called_once_with(
            query="bedroom temperature",
            top_k=5,
            min_importance=0.0,
        )

    @pytest.mark.asyncio
    async def test_execute_with_defaults(self, recall_memory_tool, mock_memory_manager):
        """Test execute with default limit."""
        mock_memory_manager.search_memories.return_value = []

        result = await recall_memory_tool.execute(query="test")

        # Verify default limit was used
        mock_memory_manager.search_memories.assert_called_once()
        call_kwargs = mock_memory_manager.search_memories.call_args[1]
        assert call_kwargs["top_k"] == 5
        # Verify result structure
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_no_memories_found(self, recall_memory_tool, mock_memory_manager):
        """Test execute when no memories are found."""
        mock_memory_manager.search_memories.return_value = []

        result = await recall_memory_tool.execute(query="nonexistent")

        assert result["success"] is True
        assert "No relevant memories found" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_missing_query(self, recall_memory_tool, mock_memory_manager):
        """Test execute fails when query is missing."""
        with pytest.raises(ToolExecutionError, match="Missing required parameter: query"):
            await recall_memory_tool.execute()

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, recall_memory_tool, mock_memory_manager):
        """Test execute fails when query is empty."""
        with pytest.raises(ToolExecutionError, match="Missing required parameter: query"):
            await recall_memory_tool.execute(query="")

    @pytest.mark.asyncio
    async def test_execute_memory_manager_error(self, recall_memory_tool, mock_memory_manager):
        """Test execute handles memory manager errors."""
        mock_memory_manager.search_memories.side_effect = Exception("Database error")

        with pytest.raises(ToolExecutionError, match="Failed to recall memories"):
            await recall_memory_tool.execute(query="test")

    @pytest.mark.asyncio
    async def test_execute_formats_importance(self, recall_memory_tool, mock_memory_manager):
        """Test that importance scores are formatted correctly."""
        mock_memories = [
            {
                "type": "fact",
                "content": "Test memory",
                "importance": 0.755,
            }
        ]
        mock_memory_manager.search_memories.return_value = mock_memories

        result = await recall_memory_tool.execute(query="test")

        # Check that importance is formatted with 2 decimal places
        assert (
            "(importance: 0.76)" in result["message"] or "(importance: 0.75)" in result["message"]
        )

    @pytest.mark.asyncio
    async def test_execute_handles_missing_fields(self, recall_memory_tool, mock_memory_manager):
        """Test execute handles memories with missing fields."""
        mock_memories = [
            {
                "content": "Memory without type",
                # Missing type and importance
            },
            {
                "type": "preference",
                # Missing content and importance
            },
        ]
        mock_memory_manager.search_memories.return_value = mock_memories

        result = await recall_memory_tool.execute(query="test")

        # Should not crash, should use defaults
        assert result["success"] is True
        assert "Found 2 relevant memories" in result["message"]


class TestToolIntegration:
    """Integration tests for memory tools."""

    @pytest.mark.asyncio
    async def test_store_and_recall_workflow(self, mock_hass, mock_memory_manager):
        """Test complete workflow of storing and recalling a memory."""
        # Store a memory
        store_tool = StoreMemoryTool(mock_hass, mock_memory_manager, "conv_123")
        mock_memory_manager.add_memory.return_value = "mem_xyz"

        store_result = await store_tool.execute(
            content="User's favorite color is blue",
            memory_type="preference",
            importance=0.7,
        )

        assert store_result["success"] is True

        # Recall the memory
        recall_tool = RecallMemoryTool(mock_hass, mock_memory_manager)
        mock_memory_manager.search_memories.return_value = [
            {
                "type": "preference",
                "content": "User's favorite color is blue",
                "importance": 0.7,
            }
        ]

        recall_result = await recall_tool.execute(query="favorite color")

        assert recall_result["success"] is True
        assert "favorite color is blue" in recall_result["message"]

    def test_both_tools_registered_correctly(self, mock_hass, mock_memory_manager):
        """Test that both tools can be instantiated and have correct names."""
        store_tool = StoreMemoryTool(mock_hass, mock_memory_manager)
        recall_tool = RecallMemoryTool(mock_hass, mock_memory_manager)

        assert store_tool.name == TOOL_STORE_MEMORY
        assert recall_tool.name == TOOL_RECALL_MEMORY
        assert store_tool.name != recall_tool.name

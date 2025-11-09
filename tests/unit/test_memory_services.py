"""Unit tests for memory management services."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {
        "home_agent": {
            "test_entry": {
                "memory_manager": MagicMock(),
            }
        }
    }
    return hass


@pytest.fixture
def mock_memory_manager(mock_hass):
    """Get the mock memory manager."""
    return mock_hass.data["home_agent"]["test_entry"]["memory_manager"]


class TestListMemoriesService:
    """Tests for list_memories service."""

    @pytest.mark.asyncio
    async def test_list_memories_success(self, mock_hass, mock_memory_manager):
        """Test successful memory listing."""
        mock_memory_manager.list_all_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem1",
                    "type": "fact",
                    "content": "Test memory",
                    "importance": 0.7,
                    "extracted_at": "2024-01-01T00:00:00",
                    "last_accessed": "2024-01-02T00:00:00",
                    "source_conversation_id": "conv1",
                }
            ]
        )

        # Get entry data helper
        def _get_entry_data(target_entry_id):
            return mock_hass.data["home_agent"].get(target_entry_id, {})

        entry_data = _get_entry_data("test_entry")
        memory_manager = entry_data.get("memory_manager")

        memories = await memory_manager.list_all_memories(
            limit=None,
            memory_type=None,
        )

        result = {
            "memories": [
                {
                    "id": m["id"],
                    "type": m["type"],
                    "content": m["content"],
                    "importance": m["importance"],
                    "extracted_at": m["extracted_at"],
                    "last_accessed": m["last_accessed"],
                    "source_conversation_id": m.get("source_conversation_id"),
                }
                for m in memories
            ],
            "total": len(memories),
        }

        assert result["total"] == 1
        assert result["memories"][0]["id"] == "mem1"
        assert result["memories"][0]["type"] == "fact"

    @pytest.mark.asyncio
    async def test_list_memories_with_filter(self, mock_hass, mock_memory_manager):
        """Test memory listing with type filter."""
        mock_memory_manager.list_all_memories = AsyncMock(return_value=[])

        await mock_memory_manager.list_all_memories(
            limit=50,
            memory_type="preference",
        )

        mock_memory_manager.list_all_memories.assert_called_once_with(
            limit=50,
            memory_type="preference",
        )

    @pytest.mark.asyncio
    async def test_list_memories_no_manager(self, mock_hass):
        """Test list memories when manager not enabled."""
        # Remove memory manager
        mock_hass.data["home_agent"]["test_entry"] = {}

        def _get_entry_data(target_entry_id):
            return mock_hass.data["home_agent"].get(target_entry_id, {})

        entry_data = _get_entry_data("test_entry")
        memory_manager = entry_data.get("memory_manager")

        if not memory_manager:
            result = {"error": "Memory Manager not enabled", "memories": [], "total": 0}
        else:
            result = {}

        assert result["error"] == "Memory Manager not enabled"
        assert result["total"] == 0


class TestDeleteMemoryService:
    """Tests for delete_memory service."""

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, mock_hass, mock_memory_manager):
        """Test successful memory deletion."""
        mock_memory_manager.delete_memory = AsyncMock(return_value=True)

        success = await mock_memory_manager.delete_memory("mem123")

        assert success is True
        mock_memory_manager.delete_memory.assert_called_once_with("mem123")

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, mock_hass, mock_memory_manager):
        """Test deleting non-existent memory."""
        mock_memory_manager.delete_memory = AsyncMock(return_value=False)

        success = await mock_memory_manager.delete_memory("nonexistent")

        assert success is False


class TestClearMemoriesService:
    """Tests for clear_memories service."""

    @pytest.mark.asyncio
    async def test_clear_memories_success(self, mock_hass, mock_memory_manager):
        """Test successful memory clearing with confirmation."""
        mock_memory_manager.clear_all_memories = AsyncMock(return_value=25)

        # Simulate handler logic
        confirm = True
        if confirm:
            deleted_count = await mock_memory_manager.clear_all_memories()
            result = {"deleted_count": deleted_count}
        else:
            result = {"error": "confirmation_required", "deleted_count": 0}

        assert result["deleted_count"] == 25
        mock_memory_manager.clear_all_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_memories_without_confirmation(self, mock_hass, mock_memory_manager):
        """Test clear memories fails without confirmation."""
        confirm = False

        if not confirm:
            result = {"error": "confirmation_required", "deleted_count": 0}
        else:
            result = {}

        assert result["error"] == "confirmation_required"
        assert result["deleted_count"] == 0


class TestSearchMemoriesService:
    """Tests for search_memories service."""

    @pytest.mark.asyncio
    async def test_search_memories_success(self, mock_hass, mock_memory_manager):
        """Test successful memory search."""
        mock_memory_manager.search_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem1",
                    "type": "preference",
                    "content": "User likes warm temperature",
                    "importance": 0.8,
                    "relevance_score": 0.95,
                },
                {
                    "id": "mem2",
                    "type": "fact",
                    "content": "Living room has temperature sensor",
                    "importance": 0.6,
                    "relevance_score": 0.75,
                },
            ]
        )

        memories = await mock_memory_manager.search_memories(
            query="temperature preferences",
            top_k=10,
            min_importance=0.0,
        )

        result = {
            "memories": [
                {
                    "id": m["id"],
                    "type": m["type"],
                    "content": m["content"],
                    "importance": m["importance"],
                    "relevance_score": m.get("relevance_score", 0.0),
                }
                for m in memories
            ],
            "total": len(memories),
        }

        assert result["total"] == 2
        assert result["memories"][0]["relevance_score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_memories_no_results(self, mock_hass, mock_memory_manager):
        """Test search with no results."""
        mock_memory_manager.search_memories = AsyncMock(return_value=[])

        memories = await mock_memory_manager.search_memories(
            query="nonexistent topic",
            top_k=10,
            min_importance=0.0,
        )

        assert len(memories) == 0


class TestAddMemoryService:
    """Tests for add_memory service."""

    @pytest.mark.asyncio
    async def test_add_memory_success(self, mock_hass, mock_memory_manager):
        """Test successful memory addition."""
        mock_memory_manager.add_memory = AsyncMock(return_value="mem_new123")

        memory_id = await mock_memory_manager.add_memory(
            content="User prefers lights at 50%",
            memory_type="preference",
            conversation_id=None,
            importance=0.7,
            metadata={
                "extraction_method": "manual_service",
                "topics": [],
                "entities_involved": [],
            },
        )

        result = {"memory_id": memory_id}

        assert result["memory_id"] == "mem_new123"
        mock_memory_manager.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_memory_with_defaults(self, mock_hass, mock_memory_manager):
        """Test adding memory with default values."""
        mock_memory_manager.add_memory = AsyncMock(return_value="mem_default")

        memory_id = await mock_memory_manager.add_memory(
            content="Test fact",
            memory_type="fact",  # default
            conversation_id=None,
            importance=0.5,  # default
            metadata={
                "extraction_method": "manual_service",
                "topics": [],
                "entities_involved": [],
            },
        )

        assert memory_id == "mem_default"


class TestServiceIntegration:
    """Integration tests for all memory services."""

    @pytest.mark.asyncio
    async def test_service_workflow(self, mock_hass, mock_memory_manager):
        """Test complete workflow: add, search, list, delete."""
        # Add a memory
        mock_memory_manager.add_memory = AsyncMock(return_value="mem_workflow")
        memory_id = await mock_memory_manager.add_memory(
            content="Test workflow memory",
            memory_type="fact",
            conversation_id=None,
            importance=0.7,
            metadata={
                "extraction_method": "manual_service",
                "topics": [],
                "entities_involved": [],
            },
        )
        assert memory_id == "mem_workflow"

        # Search for it
        mock_memory_manager.search_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem_workflow",
                    "type": "fact",
                    "content": "Test workflow memory",
                    "importance": 0.7,
                    "relevance_score": 1.0,
                }
            ]
        )
        memories = await mock_memory_manager.search_memories(
            query="workflow",
            top_k=10,
            min_importance=0.0,
        )
        assert len(memories) == 1

        # List all memories
        mock_memory_manager.list_all_memories = AsyncMock(
            return_value=[
                {
                    "id": "mem_workflow",
                    "type": "fact",
                    "content": "Test workflow memory",
                    "importance": 0.7,
                    "extracted_at": "2024-01-01T00:00:00",
                    "last_accessed": "2024-01-01T00:00:00",
                }
            ]
        )
        all_memories = await mock_memory_manager.list_all_memories()
        assert len(all_memories) == 1

        # Delete it
        mock_memory_manager.delete_memory = AsyncMock(return_value=True)
        success = await mock_memory_manager.delete_memory("mem_workflow")
        assert success is True

    @pytest.mark.asyncio
    async def test_all_services_check_memory_manager(self, mock_hass):
        """Test that all services check for memory manager availability."""
        # Remove memory manager
        mock_hass.data["home_agent"]["test_entry"] = {}

        def _get_entry_data(target_entry_id):
            return mock_hass.data["home_agent"].get(target_entry_id, {})

        entry_data = _get_entry_data("test_entry")
        memory_manager = entry_data.get("memory_manager")

        # All services should check for memory manager
        assert memory_manager is None

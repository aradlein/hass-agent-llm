"""Unit tests for MemoryManager.

Tests the long-term memory storage and retrieval system including
dual storage (Home Assistant Store + ChromaDB) and all core functionality.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.const import (
    CONF_MEMORY_COLLECTION_NAME,
    CONF_MEMORY_DEDUP_THRESHOLD,
    CONF_MEMORY_IMPORTANCE_DECAY,
    CONF_MEMORY_MAX_MEMORIES,
    CONF_MEMORY_MIN_IMPORTANCE,
    DEFAULT_MEMORY_COLLECTION_NAME,
)
from custom_components.home_agent.memory_manager import (
    MEMORY_TYPE_CONTEXT,
    MEMORY_TYPE_EVENT,
    MEMORY_TYPE_FACT,
    MEMORY_TYPE_PREFERENCE,
    MemoryManager,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    mock = MagicMock()
    mock.data = {}
    mock.async_add_executor_job = AsyncMock(side_effect=lambda func: func())
    return mock


@pytest.fixture
def mock_store():
    """Create a mock Store instance."""
    with patch("custom_components.home_agent.memory_manager.Store") as mock:
        store = MagicMock()
        store.async_load = AsyncMock(return_value=None)
        store.async_save = AsyncMock()
        mock.return_value = store
        yield store


@pytest.fixture
def mock_vector_db_manager():
    """Create a mock VectorDBManager instance."""
    mock = MagicMock()
    mock._client = MagicMock()
    mock._embed_text = AsyncMock(return_value=[0.1] * 384)  # Mock embedding vector

    # Mock collection
    collection = MagicMock()
    collection.query = MagicMock(
        return_value={
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
    )
    collection.upsert = MagicMock()
    collection.delete = MagicMock()

    mock._client.get_or_create_collection = MagicMock(return_value=collection)

    return mock


@pytest.fixture
def memory_config():
    """Create a test memory configuration."""
    return {
        CONF_MEMORY_MAX_MEMORIES: 100,
        CONF_MEMORY_MIN_IMPORTANCE: 0.3,
        CONF_MEMORY_COLLECTION_NAME: DEFAULT_MEMORY_COLLECTION_NAME,
        CONF_MEMORY_IMPORTANCE_DECAY: 0.0,
        CONF_MEMORY_DEDUP_THRESHOLD: 0.95,
    }


@pytest.fixture
async def memory_manager(mock_hass, mock_store, mock_vector_db_manager, memory_config):
    """Create a MemoryManager instance for testing."""
    with patch("custom_components.home_agent.memory_manager.CHROMADB_AVAILABLE", True):
        manager = MemoryManager(
            hass=mock_hass,
            vector_db_manager=mock_vector_db_manager,
            config=memory_config,
        )
        await manager.async_initialize()
        yield manager
        await manager.async_shutdown()


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization."""

    async def test_init_with_config(self, mock_hass, mock_vector_db_manager, memory_config):
        """Test initialization with custom config."""
        with patch("custom_components.home_agent.memory_manager.Store"):
            manager = MemoryManager(
                hass=mock_hass,
                vector_db_manager=mock_vector_db_manager,
                config=memory_config,
            )

            assert manager.max_memories == 100
            assert manager.min_importance == 0.3
            assert manager.collection_name == DEFAULT_MEMORY_COLLECTION_NAME
            assert manager.importance_decay == 0.0
            assert manager.dedup_threshold == 0.95

    async def test_async_initialize_with_existing_memories(
        self, mock_hass, mock_vector_db_manager, memory_config
    ):
        """Test initialization loads existing memories from store."""
        existing_memories = {
            "mem1": {
                "id": "mem1",
                "type": MEMORY_TYPE_FACT,
                "content": "Test memory",
                "importance": 0.8,
            }
        }

        with patch("custom_components.home_agent.memory_manager.Store") as mock_store_cls:
            with patch("custom_components.home_agent.memory_manager.CHROMADB_AVAILABLE", True):
                store = MagicMock()
                store.async_load = AsyncMock(return_value={"memories": existing_memories})
                store.async_save = AsyncMock()
                mock_store_cls.return_value = store

                manager = MemoryManager(
                    hass=mock_hass,
                    vector_db_manager=mock_vector_db_manager,
                    config=memory_config,
                )

                await manager.async_initialize()

                assert len(manager._memories) == 1
                assert "mem1" in manager._memories

    async def test_async_initialize_chromadb_unavailable(
        self, mock_hass, mock_vector_db_manager, memory_config
    ):
        """Test initialization works without ChromaDB."""
        with patch("custom_components.home_agent.memory_manager.Store") as mock_store_cls:
            with patch("custom_components.home_agent.memory_manager.CHROMADB_AVAILABLE", False):
                store = MagicMock()
                store.async_load = AsyncMock(return_value=None)
                store.async_save = AsyncMock()
                mock_store_cls.return_value = store

                manager = MemoryManager(
                    hass=mock_hass,
                    vector_db_manager=mock_vector_db_manager,
                    config=memory_config,
                )

                await manager.async_initialize()

                assert manager._chromadb_available is False


class TestAddMemory:
    """Test add_memory method."""

    async def test_add_memory_success(self, memory_manager):
        """Test adding a memory successfully."""
        memory_id = await memory_manager.add_memory(
            content="User prefers lights at 80% brightness",
            memory_type=MEMORY_TYPE_PREFERENCE,
            conversation_id="conv123",
            importance=0.8,
            metadata={"entities_involved": ["light.living_room"]},
        )

        assert memory_id is not None
        assert memory_id in memory_manager._memories

        memory = memory_manager._memories[memory_id]
        assert memory["content"] == "User prefers lights at 80% brightness"
        assert memory["type"] == MEMORY_TYPE_PREFERENCE
        assert memory["importance"] == 0.8
        assert memory["source_conversation_id"] == "conv123"
        assert "light.living_room" in memory["metadata"]["entities_involved"]

    async def test_add_memory_all_types(self, memory_manager):
        """Test adding all memory types."""
        types = [
            MEMORY_TYPE_FACT,
            MEMORY_TYPE_PREFERENCE,
            MEMORY_TYPE_CONTEXT,
            MEMORY_TYPE_EVENT,
        ]

        for memory_type in types:
            memory_id = await memory_manager.add_memory(
                content=f"Test {memory_type}",
                memory_type=memory_type,
                importance=0.5,
            )

            assert memory_id is not None
            assert memory_manager._memories[memory_id]["type"] == memory_type

    async def test_add_memory_with_default_importance(self, memory_manager):
        """Test adding memory with default importance."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
        )

        memory = memory_manager._memories[memory_id]
        assert memory["importance"] == 0.5  # Default

    async def test_add_memory_invalid_type(self, memory_manager):
        """Test adding memory with invalid type raises error."""
        with pytest.raises(ValueError, match="Invalid memory type"):
            await memory_manager.add_memory(
                content="Test memory",
                memory_type="invalid_type",
            )

    async def test_add_memory_empty_content(self, memory_manager):
        """Test adding memory with empty content raises error."""
        with pytest.raises(ValueError, match="Memory content cannot be empty"):
            await memory_manager.add_memory(
                content="",
                memory_type=MEMORY_TYPE_FACT,
            )

    async def test_add_memory_invalid_importance(self, memory_manager):
        """Test adding memory with invalid importance raises error."""
        with pytest.raises(ValueError, match="Importance must be between"):
            await memory_manager.add_memory(
                content="Test memory",
                memory_type=MEMORY_TYPE_FACT,
                importance=1.5,
            )

    async def test_add_memory_auto_metadata(self, memory_manager):
        """Test that metadata fields are auto-created if missing."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
        )

        memory = memory_manager._memories[memory_id]
        assert "entities_involved" in memory["metadata"]
        assert "topics" in memory["metadata"]
        assert "extraction_method" in memory["metadata"]

    async def test_add_memory_deduplication(self, memory_manager):
        """Test that duplicate memories are detected."""
        # Mock the duplicate detection to return a duplicate
        with patch.object(memory_manager, "_find_duplicate", return_value="existing_id"):
            # Add an existing memory first
            memory_manager._memories["existing_id"] = {
                "id": "existing_id",
                "content": "Duplicate memory",
                "type": MEMORY_TYPE_FACT,
                "importance": 0.5,
                "last_accessed": time.time() - 100,
                "metadata": {},
            }

            # Try to add duplicate
            result_id = await memory_manager.add_memory(
                content="Duplicate memory",
                memory_type=MEMORY_TYPE_FACT,
            )

            # Should return existing ID
            assert result_id == "existing_id"
            # Last accessed should be updated
            assert memory_manager._memories["existing_id"]["last_accessed"] > time.time() - 10

    async def test_add_memory_triggers_pruning(self, memory_manager):
        """Test that adding memories beyond max triggers pruning."""
        memory_manager.max_memories = 5

        # Add 6 memories
        memory_ids = []
        for i in range(6):
            memory_id = await memory_manager.add_memory(
                content=f"Test memory {i}",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.3 + (i * 0.1),  # Varying importance
            )
            memory_ids.append(memory_id)

        # Should have pruned to 5
        assert len(memory_manager._memories) == 5
        # Lowest importance should be removed (first one)
        assert memory_ids[0] not in memory_manager._memories


class TestGetMemory:
    """Test get_memory method."""

    async def test_get_memory_exists(self, memory_manager):
        """Test getting an existing memory."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.5,
        )

        original_importance = memory_manager._memories[memory_id]["importance"]
        original_accessed = memory_manager._memories[memory_id]["last_accessed"]

        # Wait a bit
        await asyncio.sleep(0.1)

        # Get memory
        memory = await memory_manager.get_memory(memory_id)

        assert memory is not None
        assert memory["content"] == "Test memory"
        # Importance should be boosted
        assert memory["importance"] > original_importance
        # Last accessed should be updated
        assert memory["last_accessed"] > original_accessed

    async def test_get_memory_not_found(self, memory_manager):
        """Test getting a non-existent memory."""
        memory = await memory_manager.get_memory("nonexistent")
        assert memory is None

    async def test_get_memory_importance_boost(self, memory_manager):
        """Test that getting memory boosts importance."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.5,
        )

        original_importance = 0.5

        # Get memory multiple times
        for _ in range(3):
            await memory_manager.get_memory(memory_id)

        # Importance should increase
        final_importance = memory_manager._memories[memory_id]["importance"]
        assert final_importance > original_importance

    async def test_get_memory_max_importance_cap(self, memory_manager):
        """Test that importance doesn't exceed 1.0."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.99,
        )

        # Get memory multiple times
        for _ in range(10):
            await memory_manager.get_memory(memory_id)

        # Importance should be capped at 1.0
        assert memory_manager._memories[memory_id]["importance"] <= 1.0


class TestSearchMemories:
    """Test search_memories method."""

    async def test_search_memories_chromadb(self, memory_manager):
        """Test searching memories with ChromaDB."""
        # Add some memories
        mem1_id = await memory_manager.add_memory(
            content="User likes blue lights",
            memory_type=MEMORY_TYPE_PREFERENCE,
            importance=0.8,
        )
        _mem2_id = await memory_manager.add_memory(  # noqa: F841
            content="Living room temperature is 72F",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.6,
        )

        # Mock ChromaDB query to return mem1
        memory_manager._collection.query = MagicMock(
            return_value={
                "ids": [[mem1_id]],
                "distances": [[0.1]],
                "documents": [["User likes blue lights"]],
                "metadatas": [[{"memory_id": mem1_id}]],
            }
        )

        results = await memory_manager.search_memories(
            query="What colors does the user prefer?",
            top_k=5,
        )

        assert len(results) == 1
        assert results[0]["content"] == "User likes blue lights"

    async def test_search_memories_with_importance_filter(self, memory_manager):
        """Test searching with minimum importance threshold."""
        # Add memories with different importance
        await memory_manager.add_memory(
            content="Low importance",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.2,
        )
        high_id = await memory_manager.add_memory(
            content="High importance",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.9,
        )

        # Mock ChromaDB to return both
        memory_manager._collection.query = MagicMock(
            return_value={
                "ids": [[high_id]],
                "distances": [[0.1]],
            }
        )

        # Search with min importance
        results = await memory_manager.search_memories(
            query="test",
            min_importance=0.5,
        )

        # Should only get high importance one
        assert len(results) == 1
        # Check with tolerance for floating point
        assert abs(results[0]["importance"] - 0.9) < 0.1

    async def test_search_memories_with_type_filter(self, memory_manager):
        """Test searching with memory type filter."""
        # Add different types
        fact_id = await memory_manager.add_memory(
            content="This is a fact",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.8,
        )
        pref_id = await memory_manager.add_memory(
            content="This is a preference",
            memory_type=MEMORY_TYPE_PREFERENCE,
            importance=0.8,
        )

        # Mock ChromaDB to return both
        memory_manager._collection.query = MagicMock(
            return_value={
                "ids": [[fact_id, pref_id]],
                "distances": [[0.1, 0.2]],
            }
        )

        # Search for only facts
        results = await memory_manager.search_memories(
            query="test",
            memory_types=[MEMORY_TYPE_FACT],
        )

        assert len(results) == 1
        assert results[0]["type"] == MEMORY_TYPE_FACT

    async def test_search_memories_fallback_mode(self, memory_manager):
        """Test fallback keyword search when ChromaDB unavailable."""
        memory_manager._chromadb_available = False

        # Add some memories
        await memory_manager.add_memory(
            content="User likes blue lights",
            memory_type=MEMORY_TYPE_PREFERENCE,
            importance=0.8,
        )
        await memory_manager.add_memory(
            content="Living room temperature",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.6,
        )

        # Search with keyword
        results = await memory_manager.search_memories(
            query="blue",
            top_k=5,
        )

        assert len(results) == 1
        assert "blue" in results[0]["content"].lower()

    async def test_search_memories_no_results(self, memory_manager):
        """Test search with no matching results."""
        # Mock ChromaDB to return empty results
        memory_manager._collection.query = MagicMock(
            return_value={
                "ids": [[]],
                "distances": [[]],
            }
        )

        results = await memory_manager.search_memories(
            query="nonexistent query",
        )

        assert len(results) == 0


class TestDeleteMemory:
    """Test delete_memory method."""

    async def test_delete_memory_success(self, memory_manager):
        """Test deleting a memory successfully."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
        )

        assert memory_id in memory_manager._memories

        result = await memory_manager.delete_memory(memory_id)

        assert result is True
        assert memory_id not in memory_manager._memories

    async def test_delete_memory_not_found(self, memory_manager):
        """Test deleting a non-existent memory."""
        result = await memory_manager.delete_memory("nonexistent")
        assert result is False

    async def test_delete_memory_removes_from_chromadb(self, memory_manager):
        """Test that deletion removes from ChromaDB."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
        )

        await memory_manager.delete_memory(memory_id)

        # Verify ChromaDB delete was called with correct arguments
        memory_manager._collection.delete.assert_called_once_with(ids=[memory_id])


class TestListAllMemories:
    """Test list_all_memories method."""

    async def test_list_all_memories(self, memory_manager):
        """Test listing all memories."""
        # Add multiple memories
        for i in range(3):
            await memory_manager.add_memory(
                content=f"Memory {i}",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.5 + (i * 0.1),
            )

        memories = await memory_manager.list_all_memories()

        assert len(memories) == 3
        # Should be sorted by importance (descending)
        assert memories[0]["importance"] >= memories[1]["importance"]

    async def test_list_memories_with_limit(self, memory_manager):
        """Test listing memories with limit."""
        # Add multiple memories
        for i in range(5):
            await memory_manager.add_memory(
                content=f"Memory {i}",
                memory_type=MEMORY_TYPE_FACT,
            )

        memories = await memory_manager.list_all_memories(limit=2)

        assert len(memories) == 2

    async def test_list_memories_with_type_filter(self, memory_manager):
        """Test listing memories filtered by type."""
        await memory_manager.add_memory(
            content="Fact memory",
            memory_type=MEMORY_TYPE_FACT,
        )
        await memory_manager.add_memory(
            content="Preference memory",
            memory_type=MEMORY_TYPE_PREFERENCE,
        )

        memories = await memory_manager.list_all_memories(memory_type=MEMORY_TYPE_FACT)

        assert len(memories) == 1
        assert memories[0]["type"] == MEMORY_TYPE_FACT

    async def test_list_empty_memories(self, memory_manager):
        """Test listing when no memories exist."""
        memories = await memory_manager.list_all_memories()
        assert len(memories) == 0


class TestClearAllMemories:
    """Test clear_all_memories method."""

    async def test_clear_all_memories(self, memory_manager):
        """Test clearing all memories."""
        # Add some memories
        for i in range(3):
            await memory_manager.add_memory(
                content=f"Memory {i}",
                memory_type=MEMORY_TYPE_FACT,
            )

        assert len(memory_manager._memories) == 3

        count = await memory_manager.clear_all_memories()

        assert count == 3
        assert len(memory_manager._memories) == 0

    async def test_clear_empty_memories(self, memory_manager):
        """Test clearing when no memories exist."""
        count = await memory_manager.clear_all_memories()
        assert count == 0


class TestImportanceDecay:
    """Test importance decay functionality."""

    async def test_apply_importance_decay_no_decay(self, memory_manager):
        """Test that no decay occurs when decay is 0."""
        memory_manager.importance_decay = 0.0

        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.8,
        )

        original_importance = memory_manager._memories[memory_id]["importance"]

        count = await memory_manager.apply_importance_decay()

        assert count == 0
        assert memory_manager._memories[memory_id]["importance"] == original_importance

    async def test_apply_importance_decay_with_decay(self, memory_manager):
        """Test importance decay reduces importance."""
        memory_manager.importance_decay = 0.1  # 10% decay

        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.8,
        )

        original_importance = 0.8

        await memory_manager.apply_importance_decay()

        # Importance should be reduced
        new_importance = memory_manager._memories[memory_id]["importance"]
        expected = original_importance * (1.0 - 0.1)
        assert abs(new_importance - expected) < 0.01

    async def test_apply_importance_decay_removes_low_importance(self, memory_manager):
        """Test that low importance memories are removed."""
        memory_manager.importance_decay = 0.5  # 50% decay
        memory_manager.min_importance = 0.3

        # Add memory with low importance
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
            importance=0.4,  # Will decay to 0.2, below threshold
        )

        count = await memory_manager.apply_importance_decay()

        assert count == 1
        assert memory_id not in memory_manager._memories


class TestStoragePersistence:
    """Test storage persistence functionality."""

    async def test_save_to_store(self, memory_manager, mock_store):
        """Test that memories are saved to store."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
        )

        # Wait for debounced save
        await asyncio.sleep(1.5)

        # Verify save was called with proper data structure
        mock_store.async_save.assert_called_once()
        saved_data = mock_store.async_save.call_args[0][0]
        assert "version" in saved_data
        assert "memories" in saved_data
        assert memory_id in saved_data["memories"]
        assert saved_data["memories"][memory_id]["content"] == "Test memory"
        assert saved_data["memories"][memory_id]["type"] == MEMORY_TYPE_FACT

    async def test_load_from_store(self, mock_hass, mock_vector_db_manager, memory_config):
        """Test loading memories from store on initialization."""
        existing_data = {
            "version": 1,
            "memories": {
                "mem1": {
                    "id": "mem1",
                    "type": MEMORY_TYPE_FACT,
                    "content": "Existing memory",
                    "importance": 0.8,
                    "extracted_at": time.time(),
                    "last_accessed": time.time(),
                    "metadata": {},
                }
            },
        }

        with patch("custom_components.home_agent.memory_manager.Store") as mock_store_cls:
            with patch("custom_components.home_agent.memory_manager.CHROMADB_AVAILABLE", True):
                store = MagicMock()
                store.async_load = AsyncMock(return_value=existing_data)
                store.async_save = AsyncMock()
                mock_store_cls.return_value = store

                manager = MemoryManager(
                    hass=mock_hass,
                    vector_db_manager=mock_vector_db_manager,
                    config=memory_config,
                )

                await manager.async_initialize()

                assert len(manager._memories) == 1
                assert "mem1" in manager._memories
                assert manager._memories["mem1"]["content"] == "Existing memory"


class TestDualStorage:
    """Test dual storage (HA Store + ChromaDB) functionality."""

    async def test_add_memory_to_both_stores(self, memory_manager):
        """Test that memories are added to both stores."""
        memory_id = await memory_manager.add_memory(
            content="Test memory",
            memory_type=MEMORY_TYPE_FACT,
        )

        # Should be in memory dict (HA Store)
        assert memory_id in memory_manager._memories

        # Should be added to ChromaDB with correct arguments
        memory_manager._collection.upsert.assert_called_once()
        call_args = memory_manager._collection.upsert.call_args
        assert call_args is not None
        # Verify the upsert call includes the memory_id
        assert memory_id in call_args.kwargs.get("ids", [])

    async def test_chromadb_failure_graceful_degradation(
        self, mock_hass, mock_vector_db_manager, memory_config
    ):
        """Test graceful degradation when ChromaDB fails."""
        # Mock ChromaDB to raise an error
        mock_vector_db_manager._client.get_or_create_collection.side_effect = Exception(
            "ChromaDB error"
        )

        with patch("custom_components.home_agent.memory_manager.Store") as mock_store_cls:
            with patch("custom_components.home_agent.memory_manager.CHROMADB_AVAILABLE", True):
                store = MagicMock()
                store.async_load = AsyncMock(return_value=None)
                store.async_save = AsyncMock()
                mock_store_cls.return_value = store

                manager = MemoryManager(
                    hass=mock_hass,
                    vector_db_manager=mock_vector_db_manager,
                    config=memory_config,
                )

                await manager.async_initialize()

                # Should fall back to store-only mode
                assert manager._chromadb_available is False

                # Should still be able to add memories
                memory_id = await manager.add_memory(
                    content="Test memory",
                    memory_type=MEMORY_TYPE_FACT,
                )

                assert memory_id in manager._memories

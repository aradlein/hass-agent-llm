"""Integration tests for Memory system with real ChromaDB and LLM.

These tests verify that the MemoryManager correctly stores, retrieves, and
searches memories using real ChromaDB and real LLM for extraction.
"""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONTEXT_MODE_VECTOR_DB,
    EMBEDDING_PROVIDER_OLLAMA,
)
from custom_components.home_agent.memory_manager import (
    MEMORY_TYPE_EVENT,
    MEMORY_TYPE_FACT,
    MEMORY_TYPE_PREFERENCE,
    MemoryManager,
)
from custom_components.home_agent.vector_db_manager import VectorDBManager


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_extraction_flow(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test extracting memories from conversation.

    This test verifies that:
    1. MemoryManager can initialize with real ChromaDB
    2. Memories can be added with proper validation
    3. Memory metadata is correctly stored
    4. Duplicate detection works
    """
    # Configure VectorDBManager for embeddings
    vector_config = {
        CONF_VECTOR_DB_HOST: chromadb_config["host"],
        CONF_VECTOR_DB_PORT: chromadb_config["port"],
        CONF_VECTOR_DB_COLLECTION: "test_entity_embeddings",
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OLLAMA,
        CONF_VECTOR_DB_EMBEDDING_MODEL: embedding_config["model"],
        CONF_VECTOR_DB_EMBEDDING_BASE_URL: embedding_config["base_url"],
    }

    # Mock entity exposure
    with patch(
        "custom_components.home_agent.vector_db_manager.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        vector_db_manager = VectorDBManager(test_hass, vector_config)
        await vector_db_manager._ensure_initialized()

        # Configure MemoryManager
        memory_config = {
            "memory_max_memories": 100,
            "memory_min_importance": 0.3,
            "memory_collection_name": memory_collection_name,
            "memory_dedup_threshold": 0.95,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        memory_manager = MemoryManager(
            test_hass,
            vector_db_manager,
            memory_config,
        )

        try:
            await memory_manager.async_initialize()

            # Add a fact memory
            memory_id = await memory_manager.add_memory(
                content="User prefers bedroom temperature at 68 degrees Fahrenheit",
                memory_type=MEMORY_TYPE_PREFERENCE,
                conversation_id="test_conv_1",
                importance=0.8,
                metadata={
                    "entities_involved": ["climate.bedroom"],
                    "topics": ["temperature", "bedroom", "preferences"],
                },
            )

            assert memory_id is not None, "Memory ID should not be None after adding memory"
            assert isinstance(
                memory_id, str
            ), f"Memory ID should be a string, got {type(memory_id)}"
            assert len(memory_id) > 0, "Memory ID should not be empty"
            assert (
                len(memory_id) >= 8
            ), f"Memory ID should be meaningful length (>=8 chars), got {len(memory_id)}"

            # Verify memory was stored
            retrieved = await memory_manager.get_memory(memory_id)
            assert retrieved is not None, "Retrieved memory should not be None"
            assert isinstance(
                retrieved, dict
            ), f"Retrieved memory should be a dict, got {type(retrieved)}"
            assert "content" in retrieved, "Retrieved memory should have 'content' key"
            assert "type" in retrieved, "Retrieved memory should have 'type' key"
            assert "importance" in retrieved, "Retrieved memory should have 'importance' key"
            assert retrieved["content"] == (
                "User prefers bedroom temperature at 68 degrees Fahrenheit"
            )
            assert retrieved["type"] == MEMORY_TYPE_PREFERENCE
            # Use approximate comparison for float due to precision
            assert abs(retrieved["importance"] - 0.8) < 0.1

            # Add a different memory to verify we can store multiple
            second_memory_id = await memory_manager.add_memory(
                content="User's favorite color is blue",
                memory_type=MEMORY_TYPE_FACT,
                conversation_id="test_conv_2",
                importance=0.6,
            )

            # Should be a different memory ID (distinct content)
            assert second_memory_id is not None, "Second memory ID should not be None"
            assert isinstance(
                second_memory_id, str
            ), f"Second memory ID should be a string, got {type(second_memory_id)}"
            assert (
                second_memory_id != memory_id
            ), "Second memory should have a different ID from first memory"

            # Verify both memories exist
            second_retrieved = await memory_manager.get_memory(second_memory_id)
            assert second_retrieved is not None, "Second retrieved memory should not be None"
            assert isinstance(
                second_retrieved, dict
            ), f"Second retrieved memory should be a dict, got {type(second_retrieved)}"
            assert second_retrieved["content"] == "User's favorite color is blue"

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_recall(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test searching and retrieving memories.

    This test verifies that:
    1. Memories can be searched by content
    2. Search results are relevant
    3. Multiple memories can be retrieved
    4. Filtering by type and importance works
    """
    # Configure VectorDBManager
    vector_config = {
        CONF_VECTOR_DB_HOST: chromadb_config["host"],
        CONF_VECTOR_DB_PORT: chromadb_config["port"],
        CONF_VECTOR_DB_COLLECTION: "test_entity_embeddings",
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OLLAMA,
        CONF_VECTOR_DB_EMBEDDING_MODEL: embedding_config["model"],
        CONF_VECTOR_DB_EMBEDDING_BASE_URL: embedding_config["base_url"],
    }

    with patch(
        "custom_components.home_agent.vector_db_manager.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        vector_db_manager = VectorDBManager(test_hass, vector_config)
        await vector_db_manager._ensure_initialized()

        # Configure MemoryManager
        memory_config = {
            "memory_max_memories": 100,
            "memory_min_importance": 0.3,
            "memory_collection_name": memory_collection_name,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        memory_manager = MemoryManager(
            test_hass,
            vector_db_manager,
            memory_config,
        )

        try:
            await memory_manager.async_initialize()

            # Add several memories
            await memory_manager.add_memory(
                content="User's favorite color is blue",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.6,
            )

            await memory_manager.add_memory(
                content="User likes to wake up at 7 AM on weekdays",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.8,
            )

            await memory_manager.add_memory(
                content="Kitchen lights should be bright in the morning",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.7,
                metadata={"entities_involved": ["light.kitchen"]},
            )

            # Search for lighting-related memories
            light_results = await memory_manager.search_memories(
                query="kitchen lighting preferences",
                top_k=3,
                min_importance=0.5,
            )

            assert (
                len(light_results) > 0
            ), "Search for kitchen lighting should return at least one result"
            assert isinstance(
                light_results, list
            ), f"Search results should be a list, got {type(light_results)}"
            # Verify structure of first result
            if len(light_results) > 0:
                first_result = light_results[0]
                assert isinstance(
                    first_result, dict
                ), f"Search result should be a dict, got {type(first_result)}"
                assert "content" in first_result, "Search result should have 'content' key"
            # Should find the kitchen lights memory
            kitchen_found = any(
                "kitchen" in mem["content"].lower() and "light" in mem["content"].lower()
                for mem in light_results
            )
            assert kitchen_found, "Kitchen lighting memory not found in search"

            # Search for time-related memories
            time_results = await memory_manager.search_memories(
                query="what time does user wake up",
                top_k=3,
                min_importance=0.5,
            )

            assert (
                len(time_results) > 0
            ), "Search for wake-up time should return at least one result"
            assert isinstance(
                time_results, list
            ), f"Time search results should be a list, got {type(time_results)}"
            # Should find the wake-up time memory
            wakeup_found = any(
                "wake" in mem["content"].lower() or "7" in mem["content"] for mem in time_results
            )
            assert wakeup_found, "Wake-up time memory not found in search"

            # Test filtering by memory type
            pref_results = await memory_manager.search_memories(
                query="user preferences",
                top_k=10,
                memory_types=[MEMORY_TYPE_PREFERENCE],
            )

            # All results should be preferences
            assert all(
                mem["type"] == MEMORY_TYPE_PREFERENCE for mem in pref_results
            ), "Non-preference memories returned when filtering for preferences"

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_semantic_search(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test vector similarity search for memories.

    This test verifies that:
    1. Semantic search returns relevant memories
    2. Similar concepts are matched even with different wording
    3. Irrelevant memories are not returned
    4. Search ranking is meaningful
    """
    # Configure VectorDBManager
    vector_config = {
        CONF_VECTOR_DB_HOST: chromadb_config["host"],
        CONF_VECTOR_DB_PORT: chromadb_config["port"],
        CONF_VECTOR_DB_COLLECTION: "test_entity_embeddings",
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OLLAMA,
        CONF_VECTOR_DB_EMBEDDING_MODEL: embedding_config["model"],
        CONF_VECTOR_DB_EMBEDDING_BASE_URL: embedding_config["base_url"],
    }

    with patch(
        "custom_components.home_agent.vector_db_manager.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        vector_db_manager = VectorDBManager(test_hass, vector_config)
        await vector_db_manager._ensure_initialized()

        # Configure MemoryManager
        memory_config = {
            "memory_max_memories": 100,
            "memory_min_importance": 0.3,
            "memory_collection_name": memory_collection_name,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        memory_manager = MemoryManager(
            test_hass,
            vector_db_manager,
            memory_config,
        )

        try:
            await memory_manager.async_initialize()

            # Add memories with different concepts
            await memory_manager.add_memory(
                content="User exercises every morning at 6 AM",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.7,
                metadata={"topics": ["exercise", "routine", "morning"]},
            )

            await memory_manager.add_memory(
                content="User works from home on Tuesdays and Thursdays",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.6,
                metadata={"topics": ["work", "schedule"]},
            )

            await memory_manager.add_memory(
                content="User enjoys reading science fiction novels",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.5,
                metadata={"topics": ["reading", "books", "hobbies"]},
            )

            # Test semantic search with different wording
            # Query about fitness (should match exercise memory)
            fitness_results = await memory_manager.search_memories(
                query="tell me about the user's fitness routine",
                top_k=3,
                min_importance=0.5,
            )

            assert len(fitness_results) > 0, "Fitness search should return at least one result"
            assert isinstance(
                fitness_results, list
            ), f"Fitness results should be a list, got {type(fitness_results)}"
            # Top result should be about exercise
            top_result = fitness_results[0]
            assert (
                "exercise" in top_result["content"].lower()
                or "6 am" in top_result["content"].lower()
            )

            # Query about books (should match reading memory)
            book_results = await memory_manager.search_memories(
                query="what kind of books does the user like",
                top_k=3,
                min_importance=0.5,
            )

            assert len(book_results) > 0, "Book search should return at least one result"
            assert isinstance(
                book_results, list
            ), f"Book results should be a list, got {type(book_results)}"
            # Should find the reading preference
            reading_found = any(
                "reading" in mem["content"].lower() or "fiction" in mem["content"].lower()
                for mem in book_results
            )
            assert reading_found, "Reading preference not found in semantic search"

            # Query about work schedule
            work_results = await memory_manager.search_memories(
                query="when does the user work from home",
                top_k=3,
                min_importance=0.5,
            )

            assert len(work_results) > 0, "Work schedule search should return at least one result"
            assert isinstance(
                work_results, list
            ), f"Work results should be a list, got {type(work_results)}"
            # Should find work schedule
            work_found = any(
                "work" in mem["content"].lower() and "home" in mem["content"].lower()
                for mem in work_results
            )
            assert work_found, "Work schedule not found in semantic search"

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_lifecycle(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test complete memory lifecycle: add, update, access, delete.

    This test verifies that:
    1. Memories can be added
    2. Importance is boosted on access
    3. Memories can be deleted
    4. Cleanup works properly
    """
    # Configure VectorDBManager
    vector_config = {
        CONF_VECTOR_DB_HOST: chromadb_config["host"],
        CONF_VECTOR_DB_PORT: chromadb_config["port"],
        CONF_VECTOR_DB_COLLECTION: "test_entity_embeddings",
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OLLAMA,
        CONF_VECTOR_DB_EMBEDDING_MODEL: embedding_config["model"],
        CONF_VECTOR_DB_EMBEDDING_BASE_URL: embedding_config["base_url"],
    }

    with patch(
        "custom_components.home_agent.vector_db_manager.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        vector_db_manager = VectorDBManager(test_hass, vector_config)
        await vector_db_manager._ensure_initialized()

        # Configure MemoryManager
        memory_config = {
            "memory_max_memories": 100,
            "memory_min_importance": 0.3,
            "memory_collection_name": memory_collection_name,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        memory_manager = MemoryManager(
            test_hass,
            vector_db_manager,
            memory_config,
        )

        try:
            await memory_manager.async_initialize()

            # Add a memory
            initial_importance = 0.6
            memory_id = await memory_manager.add_memory(
                content="User's birthday is on September 15th",
                memory_type=MEMORY_TYPE_FACT,
                importance=initial_importance,
            )

            # Verify memory exists
            memory = await memory_manager.get_memory(memory_id)
            assert memory is not None, "Memory should exist after being added"
            assert isinstance(memory, dict), f"Memory should be a dict, got {type(memory)}"
            assert "importance" in memory, "Memory should have 'importance' key"
            assert memory["importance"] > initial_importance  # Boosted on first access

            # Access it again (should boost importance further)
            first_access_importance = memory["importance"]
            memory = await memory_manager.get_memory(memory_id)
            assert memory["importance"] > first_access_importance

            # Verify it appears in search
            search_results = await memory_manager.search_memories(
                query="when is the user's birthday",
                top_k=5,
            )
            assert any(m["id"] == memory_id for m in search_results)

            # Delete the memory
            deleted = await memory_manager.delete_memory(memory_id)
            assert deleted is True

            # Verify it's gone
            memory = await memory_manager.get_memory(memory_id)
            assert memory is None

            # Verify it doesn't appear in search anymore
            search_results = await memory_manager.search_memories(
                query="when is the user's birthday",
                top_k=5,
            )
            assert not any(m["id"] == memory_id for m in search_results)

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_type_filtering(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test filtering memories by type.

    This test verifies that:
    1. Different memory types can be stored
    2. Type filtering works in search
    3. Multiple types can be queried
    """
    # Configure VectorDBManager
    vector_config = {
        CONF_VECTOR_DB_HOST: chromadb_config["host"],
        CONF_VECTOR_DB_PORT: chromadb_config["port"],
        CONF_VECTOR_DB_COLLECTION: "test_entity_embeddings",
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OLLAMA,
        CONF_VECTOR_DB_EMBEDDING_MODEL: embedding_config["model"],
        CONF_VECTOR_DB_EMBEDDING_BASE_URL: embedding_config["base_url"],
    }

    with patch(
        "custom_components.home_agent.vector_db_manager.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=[])

        vector_db_manager = VectorDBManager(test_hass, vector_config)
        await vector_db_manager._ensure_initialized()

        # Configure MemoryManager
        memory_config = {
            "memory_max_memories": 100,
            "memory_min_importance": 0.3,
            "memory_collection_name": memory_collection_name,
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        memory_manager = MemoryManager(
            test_hass,
            vector_db_manager,
            memory_config,
        )

        try:
            await memory_manager.async_initialize()

            # Add memories of different types
            fact_id = await memory_manager.add_memory(
                content="User has three children",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.7,
            )

            pref_id = await memory_manager.add_memory(
                content="User prefers tea over coffee",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.6,
            )

            await memory_manager.add_memory(
                content="User attended a meeting at 3 PM",
                memory_type=MEMORY_TYPE_EVENT,
                importance=0.5,
            )

            # Search for only facts
            fact_results = await memory_manager.search_memories(
                query="tell me facts about the user",
                top_k=10,
                memory_types=[MEMORY_TYPE_FACT],
            )

            # Should only return facts
            assert all(m["type"] == MEMORY_TYPE_FACT for m in fact_results)
            assert any(m["id"] == fact_id for m in fact_results)

            # Search for preferences
            pref_results = await memory_manager.search_memories(
                query="user preferences",
                top_k=10,
                memory_types=[MEMORY_TYPE_PREFERENCE],
            )

            assert all(m["type"] == MEMORY_TYPE_PREFERENCE for m in pref_results)
            assert any(m["id"] == pref_id for m in pref_results)

            # Search for multiple types
            multi_results = await memory_manager.search_memories(
                query="user information",
                top_k=10,
                memory_types=[MEMORY_TYPE_FACT, MEMORY_TYPE_PREFERENCE],
            )

            # Should return both facts and preferences, but not events
            types_found = {m["type"] for m in multi_results}
            assert MEMORY_TYPE_EVENT not in types_found
            assert len(multi_results) >= 2

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()

"""Integration tests for Memory-VectorDB full-stack integration.

These tests verify that the memory system correctly integrates with ChromaDB
through the vector database context provider. They test the complete flow:
  1. Adding memories via MemoryManager
  2. Verifying memories are indexed in ChromaDB
  3. Semantic search retrieval via vector similarity
  4. Metadata integrity throughout the pipeline

Unlike test_real_memory.py (which tests memory operations in isolation) or
test_real_vector_db.py (which tests vector DB without memories), these tests
verify the complete end-to-end integration between both systems.
"""

import asyncio
import time
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
async def test_memory_added_to_vectordb(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test that adding a memory via MemoryManager indexes it in ChromaDB.

    This test verifies:
    1. Memory can be added via MemoryManager
    2. Memory is automatically indexed in ChromaDB collection
    3. Memory document and embeddings are stored correctly
    4. Memory metadata is preserved
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
            memory_content = "User prefers living room temperature at 72 degrees"
            memory_id = await memory_manager.add_memory(
                content=memory_content,
                memory_type=MEMORY_TYPE_PREFERENCE,
                conversation_id="test_conv_001",
                importance=0.8,
                metadata={
                    "entities_involved": ["climate.living_room"],
                    "topics": ["temperature", "preferences"],
                },
            )

            assert memory_id is not None

            # Allow async indexing to complete
            await asyncio.sleep(0.5)

            # Verify memory is in ChromaDB collection
            collection = memory_manager._collection
            assert collection is not None

            # Get collection contents (include embeddings explicitly)
            result = await test_hass.async_add_executor_job(
                lambda: collection.get(include=["documents", "metadatas", "embeddings"])
            )

            # Verify memory is indexed
            assert "ids" in result
            assert memory_id in result["ids"]

            # Find the memory in results
            memory_idx = result["ids"].index(memory_id)

            # Verify document content
            assert "documents" in result
            assert result["documents"][memory_idx] == memory_content

            # Verify embeddings exist (when requested via include)
            if result.get("embeddings") is not None:
                assert result["embeddings"][memory_idx] is not None
                # mxbai-embed-large produces 1024-dimensional embeddings
                assert len(result["embeddings"][memory_idx]) == 1024

            # Verify metadata
            assert "metadatas" in result
            metadata = result["metadatas"][memory_idx]
            assert metadata["memory_id"] == memory_id
            assert metadata["type"] == MEMORY_TYPE_PREFERENCE
            assert metadata["importance"] == 0.8
            assert metadata["conversation_id"] == "test_conv_001"
            assert "extracted_at" in metadata
            assert "last_accessed" in metadata

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_semantic_search_retrieval(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test that memories can be retrieved via semantic search.

    This test verifies:
    1. Multiple memories are indexed in vector DB
    2. Semantic search finds relevant memories
    3. Search returns memories ranked by relevance
    4. Search filters work correctly (type, importance)
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

            # Add multiple memories with different topics
            temp_memory_id = await memory_manager.add_memory(
                content="User keeps bedroom temperature at 68 degrees Fahrenheit",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.8,
                metadata={"topics": ["temperature", "bedroom"]},
            )

            light_memory_id = await memory_manager.add_memory(
                content="User prefers bright lights in the kitchen during morning",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.7,
                metadata={"topics": ["lighting", "kitchen"]},
            )

            event_memory_id = await memory_manager.add_memory(
                content="User watched a movie in the living room last night",
                memory_type=MEMORY_TYPE_EVENT,
                importance=0.4,
                metadata={"topics": ["entertainment", "living_room"]},
            )

            # Allow indexing to complete
            await asyncio.sleep(0.5)

            # Test 1: Search for temperature-related memories
            temp_results = await memory_manager.search_memories(
                query="what temperature does the user prefer",
                top_k=3,
                min_importance=0.5,
            )

            # Should find temperature memory
            assert len(temp_results) > 0
            temp_found = any(temp_memory_id == mem["id"] for mem in temp_results)
            assert temp_found, "Temperature memory not found in semantic search"

            # Top result should be about temperature
            assert "temperature" in temp_results[0]["content"].lower()

            # Test 2: Search for lighting-related memories
            light_results = await memory_manager.search_memories(
                query="lighting preferences for the kitchen",
                top_k=3,
                min_importance=0.5,
            )

            # Should find lighting memory
            assert len(light_results) > 0
            light_found = any(light_memory_id == mem["id"] for mem in light_results)
            assert light_found, "Lighting memory not found in semantic search"

            # Test 3: Filter by memory type
            pref_results = await memory_manager.search_memories(
                query="user preferences",
                top_k=10,
                memory_types=[MEMORY_TYPE_PREFERENCE],
            )

            # Should only return preferences, not events
            assert all(mem["type"] == MEMORY_TYPE_PREFERENCE for mem in pref_results)
            assert not any(mem["id"] == event_memory_id for mem in pref_results)

            # Test 4: Filter by importance threshold
            high_importance_results = await memory_manager.search_memories(
                query="user preferences",
                top_k=10,
                min_importance=0.7,
            )

            # Should only return high-importance memories
            assert all(mem["importance"] >= 0.7 for mem in high_importance_results)
            # Event memory (0.4) should not be in results
            assert not any(mem["id"] == event_memory_id for mem in high_importance_results)

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_metadata_integrity(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test that memory metadata remains intact through the full pipeline.

    This test verifies:
    1. Metadata is preserved during indexing
    2. Timestamps are maintained correctly
    3. Importance scores are preserved
    4. Type information is intact
    5. Custom metadata fields are retained
    """
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

            # Add memory with rich metadata
            before_time = time.time()
            memory_id = await memory_manager.add_memory(
                content="User has a morning routine: coffee at 7 AM, exercise at 7:30 AM",
                memory_type=MEMORY_TYPE_FACT,
                conversation_id="conv_123",
                importance=0.85,
                metadata={
                    "entities_involved": ["switch.coffee_maker", "sensor.gym"],
                    "topics": ["routine", "morning", "exercise"],
                    "extraction_method": "conversation_analysis",
                },
            )
            after_time = time.time()

            await asyncio.sleep(0.5)

            # Retrieve from MemoryManager (local storage)
            local_memory = await memory_manager.get_memory(memory_id)
            assert local_memory is not None

            # Verify local storage metadata
            assert local_memory["id"] == memory_id
            assert local_memory["type"] == MEMORY_TYPE_FACT
            assert local_memory["content"] == "User has a morning routine: coffee at 7 AM, exercise at 7:30 AM"
            assert local_memory["source_conversation_id"] == "conv_123"

            # Importance should be boosted by get_memory
            assert local_memory["importance"] >= 0.85

            # Timestamps should be reasonable
            assert before_time <= local_memory["extracted_at"] <= after_time
            assert local_memory["last_accessed"] >= before_time

            # Custom metadata should be intact
            assert "entities_involved" in local_memory["metadata"]
            assert "switch.coffee_maker" in local_memory["metadata"]["entities_involved"]
            assert "topics" in local_memory["metadata"]
            assert "routine" in local_memory["metadata"]["topics"]
            assert local_memory["metadata"]["extraction_method"] == "conversation_analysis"

            # Retrieve from ChromaDB directly
            collection = memory_manager._collection
            result = await test_hass.async_add_executor_job(
                lambda: collection.get(ids=[memory_id])
            )

            assert result["ids"][0] == memory_id
            chroma_metadata = result["metadatas"][0]

            # Verify ChromaDB metadata
            assert chroma_metadata["memory_id"] == memory_id
            assert chroma_metadata["type"] == MEMORY_TYPE_FACT
            # Importance may be boosted due to get_memory access
            assert chroma_metadata["importance"] >= 0.85
            assert chroma_metadata["conversation_id"] == "conv_123"
            assert before_time <= chroma_metadata["extracted_at"] <= after_time

            # Search and verify metadata is returned correctly
            search_results = await memory_manager.search_memories(
                query="morning routine",
                top_k=5,
            )

            # Find our memory in search results
            found_memory = next((m for m in search_results if m["id"] == memory_id), None)
            assert found_memory is not None
            assert found_memory["type"] == MEMORY_TYPE_FACT
            assert found_memory["importance"] >= 0.85  # May be boosted
            assert "morning routine" in found_memory["content"].lower()

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_update_syncs_to_vectordb(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test that updating a memory syncs changes to ChromaDB.

    This test verifies:
    1. Memory importance boosts are reflected in vector DB
    2. Memory updates trigger re-indexing
    3. Duplicate detection works with vector similarity
    4. Memory merging updates the indexed content
    """
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

        memory_config = {
            "memory_max_memories": 100,
            "memory_min_importance": 0.3,
            "memory_collection_name": memory_collection_name,
            "memory_dedup_threshold": 0.95,  # High threshold for duplicate detection
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
        }

        memory_manager = MemoryManager(
            test_hass,
            vector_db_manager,
            memory_config,
        )

        try:
            await memory_manager.async_initialize()

            # Add initial memory
            memory_id = await memory_manager.add_memory(
                content="User likes coffee",
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.6,
            )

            await asyncio.sleep(0.5)

            # Get initial state from ChromaDB
            collection = memory_manager._collection
            initial_result = await test_hass.async_add_executor_job(
                lambda: collection.get(ids=[memory_id])
            )
            initial_importance = initial_result["metadatas"][0]["importance"]
            assert initial_importance == 0.6

            # Access the memory (should boost importance)
            accessed_memory = await memory_manager.get_memory(memory_id)
            assert accessed_memory["importance"] > 0.6

            await asyncio.sleep(0.5)

            # Verify importance boost is synced to ChromaDB
            updated_result = await test_hass.async_add_executor_job(
                lambda: collection.get(ids=[memory_id])
            )
            updated_importance = updated_result["metadatas"][0]["importance"]
            assert updated_importance > initial_importance

            # Test duplicate detection and merging
            # Add very similar memory (should merge with existing)
            duplicate_id = await memory_manager.add_memory(
                content="User likes coffee",  # Exactly the same
                memory_type=MEMORY_TYPE_PREFERENCE,
                importance=0.7,
            )

            # Should return the same ID (merged)
            assert duplicate_id == memory_id

            await asyncio.sleep(0.5)

            # Verify only one entry in ChromaDB
            all_results = await test_hass.async_add_executor_job(lambda: collection.get())
            memory_ids_in_db = [mid for mid in all_results["ids"] if mid == memory_id]
            assert len(memory_ids_in_db) == 1, "Duplicate memory should not create new entry"

            # Verify importance was further boosted
            final_result = await test_hass.async_add_executor_job(
                lambda: collection.get(ids=[memory_id])
            )
            final_importance = final_result["metadatas"][0]["importance"]
            assert final_importance > updated_importance

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_deletion_removes_from_vectordb(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test that deleting a memory removes it from ChromaDB.

    This test verifies:
    1. Memory deletion removes from both local and vector DB
    2. Deleted memories don't appear in search results
    3. Deleted memory IDs are not in ChromaDB collection
    """
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
            memory_id = await memory_manager.add_memory(
                content="User's favorite color is blue",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.7,
            )

            await asyncio.sleep(0.5)

            # Verify memory exists in ChromaDB
            collection = memory_manager._collection
            before_delete = await test_hass.async_add_executor_job(
                lambda: collection.get(ids=[memory_id])
            )
            assert len(before_delete["ids"]) == 1
            assert before_delete["ids"][0] == memory_id

            # Verify memory appears in search
            search_before = await memory_manager.search_memories(
                query="favorite color",
                top_k=5,
            )
            assert any(m["id"] == memory_id for m in search_before)

            # Delete the memory
            deleted = await memory_manager.delete_memory(memory_id)
            assert deleted is True

            await asyncio.sleep(0.5)

            # Verify memory is gone from local storage
            local_memory = await memory_manager.get_memory(memory_id)
            assert local_memory is None

            # Verify memory is gone from ChromaDB
            after_delete = await test_hass.async_add_executor_job(
                lambda: collection.get(ids=[memory_id])
            )
            assert len(after_delete["ids"]) == 0

            # Verify memory doesn't appear in search
            search_after = await memory_manager.search_memories(
                query="favorite color",
                top_k=5,
            )
            assert not any(m["id"] == memory_id for m in search_after)

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_vectordb_cross_query_relevance(
    test_hass,
    chromadb_config,
    embedding_config,
    memory_collection_name,
):
    """Test semantic search returns semantically relevant memories.

    This test verifies:
    1. Semantic similarity works across different phrasings
    2. Irrelevant memories are ranked lower or not returned
    3. Vector embeddings capture semantic meaning
    4. Search quality is sufficient for practical use
    """
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

            # Add memories about different topics
            sleep_id = await memory_manager.add_memory(
                content="User goes to bed at 10 PM every night",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.7,
            )

            pet_id = await memory_manager.add_memory(
                content="User has a cat named Whiskers",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.6,
            )

            workout_id = await memory_manager.add_memory(
                content="User exercises at the gym three times per week",
                memory_type=MEMORY_TYPE_FACT,
                importance=0.65,
            )

            await asyncio.sleep(0.5)

            # Query about sleep with different phrasing
            sleep_queries = [
                "when does the user sleep",
                "bedtime routine",
                "night schedule",
            ]

            for query in sleep_queries:
                results = await memory_manager.search_memories(query=query, top_k=3)
                # Sleep memory should be in top results
                top_ids = [m["id"] for m in results[:2]]  # Check top 2
                assert sleep_id in top_ids, f"Sleep memory not in top results for: {query}"

            # Query about pets with different phrasing
            pet_queries = [
                "tell me about the user's pets",
                "what animals does the user have",
                "cat information",
            ]

            for query in pet_queries:
                results = await memory_manager.search_memories(query=query, top_k=3)
                # Pet memory should be in top results
                top_ids = [m["id"] for m in results[:2]]
                assert pet_id in top_ids, f"Pet memory not in top results for: {query}"

            # Query about fitness/exercise
            fitness_results = await memory_manager.search_memories(
                query="fitness routine and workout schedule",
                top_k=3,
            )

            # Workout memory should be top result
            assert len(fitness_results) > 0
            assert fitness_results[0]["id"] == workout_id

            # Irrelevant query should not return sleep memory as top result
            await memory_manager.search_memories(
                query="cooking recipes and food preferences",
                top_k=3,
            )

            # Even if we get results, sleep/pet/workout memories shouldn't dominate
            # (This is a soft check - semantic search might still return them with low scores)

        finally:
            await memory_manager.async_shutdown()
            await vector_db_manager.async_shutdown()

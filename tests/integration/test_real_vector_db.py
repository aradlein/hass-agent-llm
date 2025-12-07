"""Integration tests for real Vector DB operations.

This test suite validates the complete vector DB integration with real ChromaDB
and embedding services. It tests entity indexing, semantic search, incremental
updates, and collection cleanup.

Requirements:
- ChromaDB running at TEST_CHROMADB_HOST:TEST_CHROMADB_PORT
- Embedding service with mxbai-embed-large model available
"""

import asyncio

import pytest

from custom_components.home_agent.vector_db_manager import VectorDBManager


@pytest.mark.integration
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.usefixtures("socket_enabled")  # Enable real network calls
class TestRealVectorDB:
    """Integration tests for Vector DB with real services."""

    @pytest.fixture(autouse=True)
    async def skip_without_real_services(self, chromadb_config, embedding_config):
        """Skip these tests when real services are not available.

        These are "real" integration tests that validate actual service behavior.
        They should skip when services are unavailable rather than use mocks.
        """
        from tests.integration.health import check_chromadb_health, check_embedding_health

        chromadb_healthy = await check_chromadb_health(
            chromadb_config["host"], chromadb_config["port"]
        )
        embedding_healthy = await check_embedding_health(embedding_config["base_url"])

        if not chromadb_healthy or not embedding_healthy:
            pytest.skip(
                "Real ChromaDB and Embedding services required for these tests. "
                f"ChromaDB ({chromadb_config['host']}:{chromadb_config['port']}): "
                f"{'available' if chromadb_healthy else 'unavailable'}. "
                f"Embedding ({embedding_config['base_url']}): "
                f"{'available' if embedding_healthy else 'unavailable'}."
            )

    @pytest.mark.asyncio
    async def test_entity_indexing_real_embeddings(
        self,
        mock_hass_integration,
        embedding_config,
        test_collection_name,
        sample_entity_states,
    ):
        """Test indexing entities with real mxbai-embed-large embeddings.

        Verifies:
        - VectorDBManager initializes successfully
        - Entities are indexed with real embeddings
        - Collection contains indexed entities
        - Embeddings have correct dimensionality (1024 for mxbai-embed-large)
        """
        # Configure VectorDBManager
        config = {
            "vector_db_host": embedding_config["host"],
            "vector_db_port": embedding_config["port"],
            "vector_db_collection": test_collection_name,
            "vector_db_embedding_model": embedding_config["model"],
            "vector_db_embedding_provider": embedding_config["provider"],
            "vector_db_embedding_base_url": embedding_config["base_url"],
        }

        # Set up mock states
        mock_hass_integration.states.async_all.return_value = sample_entity_states
        mock_hass_integration.states.get.side_effect = lambda entity_id: next(
            (s for s in sample_entity_states if s.entity_id == entity_id), None
        )

        # Create VectorDBManager
        manager = VectorDBManager(mock_hass_integration, config)

        try:
            # Initialize (this should connect to ChromaDB and create collection)
            await manager.async_setup()

            # Wait a moment for async indexing to complete
            await asyncio.sleep(2)

            # Verify collection exists and has data
            assert await manager.async_collection_exists(test_collection_name)

            # Get collection contents
            collection = manager._collection
            assert collection is not None, "Collection should be initialized after setup"
            # Verify collection has the correct name
            assert (
                collection.name == test_collection_name
            ), f"Collection name should be {test_collection_name}, got {collection.name}"

            result = await mock_hass_integration.async_add_executor_job(lambda: collection.get())

            # Verify entities were indexed
            assert "ids" in result, "Result should contain 'ids' key"
            assert isinstance(
                result["ids"], list
            ), f"Result['ids'] should be a list, got {type(result['ids'])}"
            assert len(result["ids"]) > 0, "At least one entity should be indexed"

            # Check that we have entities (should have at least some indexed)
            indexed_ids = result["ids"]
            assert (
                len(indexed_ids) >= 3
            ), f"At least 3 entities should be indexed, got {len(indexed_ids)}"

            # Verify that indexed IDs are valid entity IDs
            for entity_id in indexed_ids[:3]:  # Check first 3
                assert isinstance(
                    entity_id, str
                ), f"Entity ID should be string, got {type(entity_id)}"
                assert (
                    "." in entity_id
                ), f"Entity ID should contain domain separator, got {entity_id}"

            # Verify embedding dimensions (mxbai-embed-large = 1024 dimensions)
            if "embeddings" in result and result["embeddings"]:
                first_embedding = result["embeddings"][0]
                assert len(first_embedding) == 1024

            # Verify metadata is present
            if "metadatas" in result and result["metadatas"]:
                first_metadata = result["metadatas"][0]
                assert "entity_id" in first_metadata
                assert "state" in first_metadata
                assert "friendly_name" in first_metadata

        finally:
            # Cleanup
            await manager.async_shutdown()

    @pytest.mark.asyncio
    async def test_semantic_search_accuracy(
        self,
        mock_hass_integration,
        embedding_config,
        test_collection_name,
        sample_entity_states,
    ):
        """Test that semantic search returns semantically relevant entities.

        Verifies:
        - Query for "lights" returns light entities
        - Query for "temperature" returns sensor/climate entities
        - Results are ranked by semantic similarity
        - Distance/similarity scores are reasonable
        """
        # Configure VectorDBManager
        config = {
            "vector_db_host": embedding_config["host"],
            "vector_db_port": embedding_config["port"],
            "vector_db_collection": test_collection_name,
            "vector_db_embedding_model": embedding_config["model"],
            "vector_db_embedding_provider": embedding_config["provider"],
            "vector_db_embedding_base_url": embedding_config["base_url"],
        }

        # Set up mock states
        mock_hass_integration.states.async_all.return_value = sample_entity_states
        mock_hass_integration.states.get.side_effect = lambda entity_id: next(
            (s for s in sample_entity_states if s.entity_id == entity_id), None
        )

        # Create and initialize manager
        manager = VectorDBManager(mock_hass_integration, config)

        try:
            await manager.async_setup()
            await asyncio.sleep(2)  # Wait for indexing

            # Test 1: Search for lights
            light_query = "turn on the lights in the room"
            light_embedding = await manager._embed_text(light_query)

            collection = manager._collection
            light_results = await mock_hass_integration.async_add_executor_job(
                lambda: collection.query(query_embeddings=[light_embedding], n_results=3)
            )

            # Verify we got results
            assert "ids" in light_results, "Light results should contain 'ids' key"
            assert isinstance(
                light_results["ids"], list
            ), f"Light results['ids'] should be a list, got {type(light_results['ids'])}"
            assert (
                len(light_results["ids"]) > 0
            ), "Light results should have at least one result set"
            assert (
                len(light_results["ids"][0]) > 0
            ), f"Light search should return at least one result, got {len(light_results['ids'][0])}"

            # Check that light entities are in top results
            top_ids = light_results["ids"][0]
            light_count = sum(1 for eid in top_ids if eid.startswith("light."))
            assert light_count >= 1  # At least one light should be in results

            # Test 2: Search for temperature/climate
            temp_query = "what is the temperature"
            temp_embedding = await manager._embed_text(temp_query)

            temp_results = await mock_hass_integration.async_add_executor_job(
                lambda: collection.query(query_embeddings=[temp_embedding], n_results=3)
            )

            # Check for temperature-related entities
            temp_ids = temp_results["ids"][0]
            temp_count = sum(
                1
                for eid in temp_ids
                if eid.startswith("sensor.temperature") or eid.startswith("climate.")
            )
            assert temp_count >= 1  # At least one temp/climate entity

            # Test 3: Verify distances are reasonable (closer = more similar)
            if "distances" in light_results:
                distances = light_results["distances"][0]
                # For L2 distance, values should be non-negative
                # The actual range depends on embedding normalization
                # With mxbai-embed-large, distances can be larger than 2.0
                assert all(d >= 0 for d in distances)
                # Results should be sorted by distance (ascending)
                assert distances == sorted(distances)

        finally:
            await manager.async_shutdown()

    @pytest.mark.asyncio
    async def test_incremental_indexing(
        self,
        mock_hass_integration,
        embedding_config,
        test_collection_name,
        sample_entity_states,
    ):
        """Test that incremental updates to entities work correctly.

        Verifies:
        - Initial indexing works
        - Re-indexing same entity updates the embedding
        - Entity state changes are reflected in metadata
        - No duplicate entries are created
        """
        # Configure VectorDBManager
        config = {
            "vector_db_host": embedding_config["host"],
            "vector_db_port": embedding_config["port"],
            "vector_db_collection": test_collection_name,
            "vector_db_embedding_model": embedding_config["model"],
            "vector_db_embedding_provider": embedding_config["provider"],
            "vector_db_embedding_base_url": embedding_config["base_url"],
        }

        # Set up mock states
        mock_hass_integration.states.async_all.return_value = sample_entity_states
        mock_hass_integration.states.get.side_effect = lambda entity_id: next(
            (s for s in sample_entity_states if s.entity_id == entity_id), None
        )

        # Create and initialize manager
        manager = VectorDBManager(mock_hass_integration, config)

        try:
            await manager.async_setup()
            await asyncio.sleep(2)  # Wait for initial indexing

            collection = manager._collection

            # Get initial count
            initial_result = await mock_hass_integration.async_add_executor_job(
                lambda: collection.get()
            )
            initial_count = len(initial_result["ids"])

            # Update one entity's state
            from homeassistant.core import State

            updated_light = State(
                "light.living_room",
                "off",  # Changed from "on" to "off"
                {"brightness": 0, "friendly_name": "Living Room Light"},
            )

            # Update mock to return new state
            def new_get_state(entity_id):
                if entity_id == "light.living_room":
                    return updated_light
                return next((s for s in sample_entity_states if s.entity_id == entity_id), None)

            mock_hass_integration.states.get.side_effect = new_get_state

            # Re-index the updated entity
            await manager.async_index_entity("light.living_room")
            await asyncio.sleep(1)

            # Get updated results
            updated_result = await mock_hass_integration.async_add_executor_job(
                lambda: collection.get()
            )

            # Verify no duplicates (count should be same)
            assert len(updated_result["ids"]) == initial_count

            # Find the updated entity metadata
            living_room_idx = updated_result["ids"].index("light.living_room")
            living_room_metadata = updated_result["metadatas"][living_room_idx]

            # Verify state was updated
            assert living_room_metadata["state"] == "off"

        finally:
            await manager.async_shutdown()

    @pytest.mark.asyncio
    async def test_collection_cleanup(
        self,
        mock_hass_integration,
        embedding_config,
        chromadb_client,
    ):
        """Test that collections are properly cleaned up between tests.

        Verifies:
        - Unique collection names prevent test interference
        - Collections can be created and deleted
        - Cleanup doesn't affect other collections
        """
        import uuid

        # Create two unique collection names
        collection1_name = f"test_cleanup_{uuid.uuid4().hex[:8]}"
        collection2_name = f"test_cleanup_{uuid.uuid4().hex[:8]}"

        try:
            # Create first collection
            collection1 = chromadb_client.get_or_create_collection(name=collection1_name)
            collection1.add(
                ids=["test1"],
                documents=["test document 1"],
                embeddings=[[0.1] * 1024],
            )

            # Create second collection
            collection2 = chromadb_client.get_or_create_collection(name=collection2_name)
            collection2.add(
                ids=["test2"],
                documents=["test document 2"],
                embeddings=[[0.2] * 1024],
            )

            # Verify both exist
            result1 = collection1.get()
            assert len(result1["ids"]) == 1
            result2 = collection2.get()
            assert len(result2["ids"]) == 1

            # Delete first collection
            chromadb_client.delete_collection(name=collection1_name)

            # Verify second collection still exists and has data
            collection2_reloaded = chromadb_client.get_collection(name=collection2_name)
            result2_after = collection2_reloaded.get()
            assert len(result2_after["ids"]) == 1

            # Verify first collection is gone
            # ChromaDB raises ValueError when collection doesn't exist
            with pytest.raises(
                (ValueError, Exception), match=".*[Cc]ollection.*|.*not found.*|.*does not exist.*"
            ):
                chromadb_client.get_collection(name=collection1_name)

        finally:
            # Cleanup both collections
            try:
                chromadb_client.delete_collection(name=collection1_name)
            except Exception:
                pass
            try:
                chromadb_client.delete_collection(name=collection2_name)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_embedding_generation_consistency(
        self,
        mock_hass_integration,
        embedding_config,
        test_collection_name,
    ):
        """Test that embedding generation is consistent and cacheable.

        Verifies:
        - Same text produces same embedding
        - Embedding cache works correctly
        - Different text produces different embeddings
        """
        # Configure VectorDBManager
        config = {
            "vector_db_host": embedding_config["host"],
            "vector_db_port": embedding_config["port"],
            "vector_db_collection": test_collection_name,
            "vector_db_embedding_model": embedding_config["model"],
            "vector_db_embedding_provider": embedding_config["provider"],
            "vector_db_embedding_base_url": embedding_config["base_url"],
        }

        mock_hass_integration.states.async_all.return_value = []

        manager = VectorDBManager(mock_hass_integration, config)

        try:
            await manager._ensure_initialized()

            # Generate embedding for same text twice
            text1 = "turn on the living room light"
            embedding1_first = await manager._embed_text(text1)
            embedding1_second = await manager._embed_text(text1)

            # Should be identical (from cache)
            assert embedding1_first == embedding1_second
            assert len(embedding1_first) == 1024  # mxbai-embed-large dimension

            # Generate embedding for different text
            text2 = "what is the temperature"
            embedding2 = await manager._embed_text(text2)

            # Should be different
            assert embedding2 != embedding1_first
            assert len(embedding2) == 1024

            # Calculate similarity (cosine similarity via dot product for normalized vectors)
            import numpy as np

            vec1 = np.array(embedding1_first)
            vec2 = np.array(embedding2)

            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)

            # Calculate cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)

            # Different texts should have lower similarity (< 0.9)
            assert similarity < 0.9

        finally:
            await manager.async_shutdown()

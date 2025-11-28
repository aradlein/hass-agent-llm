"""Unit tests for VectorDBManager entity indexing.

Tests to validate that only exposed entities are indexed in ChromaDB,
not all entities in the system.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import State

from custom_components.home_agent.const import (
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    DEFAULT_VECTOR_DB_COLLECTION,
    DEFAULT_VECTOR_DB_HOST,
    DEFAULT_VECTOR_DB_PORT,
)
from custom_components.home_agent.vector_db_manager import VectorDBManager


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    mock = MagicMock()
    mock.data = {}
    # Fix async_add_executor_job to handle both callable and positional args
    mock.async_add_executor_job = AsyncMock(
        side_effect=lambda func, *args, **kwargs: (
            func(*args, **kwargs) if args or kwargs else func()
        )
    )
    mock.bus = MagicMock()
    mock.bus.async_listen = MagicMock(return_value=lambda: None)

    # Create mock states for testing
    # Some exposed, some not exposed
    mock_states = [
        State("light.living_room", "on", {"friendly_name": "Living Room Light"}),
        State("light.bedroom", "off", {"friendly_name": "Bedroom Light"}),
        State("sensor.temperature", "22", {"friendly_name": "Temperature"}),
        State("sensor.internal_metric", "100", {"friendly_name": "Internal Metric"}),
        State("switch.fan", "on", {"friendly_name": "Fan"}),
    ]

    mock.states = MagicMock()
    mock.states.async_all = MagicMock(return_value=mock_states)
    mock.states.get = MagicMock(
        side_effect=lambda entity_id: next(
            (s for s in mock_states if s.entity_id == entity_id), None
        )
    )
    mock.states.async_entity_ids = MagicMock(return_value=[s.entity_id for s in mock_states])

    return mock


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client."""
    with patch("custom_components.home_agent.vector_db_manager.chromadb") as mock:
        client = MagicMock()
        collection = MagicMock()
        collection.upsert = MagicMock()
        collection.delete = MagicMock()
        collection.get = MagicMock(return_value={"ids": []})
        client.get_or_create_collection = MagicMock(return_value=collection)
        mock.HttpClient = MagicMock(return_value=client)
        yield mock


@pytest.fixture
def vector_db_config():
    """Create test configuration for VectorDBManager."""
    return {
        CONF_VECTOR_DB_HOST: DEFAULT_VECTOR_DB_HOST,
        CONF_VECTOR_DB_PORT: DEFAULT_VECTOR_DB_PORT,
        CONF_VECTOR_DB_COLLECTION: DEFAULT_VECTOR_DB_COLLECTION,
        CONF_VECTOR_DB_EMBEDDING_PROVIDER: "ollama",
        CONF_VECTOR_DB_EMBEDDING_MODEL: "nomic-embed-text",
    }


@pytest.fixture
def mock_async_should_expose():
    """Mock the async_should_expose function."""

    def should_expose(hass, domain, entity_id):
        """Only expose certain entities for testing."""
        # Simulate exposure settings:
        # - light.living_room: EXPOSED
        # - light.bedroom: NOT EXPOSED
        # - sensor.temperature: EXPOSED
        # - sensor.internal_metric: NOT EXPOSED
        # - switch.fan: EXPOSED
        exposed_entities = {
            "light.living_room",
            "sensor.temperature",
            "switch.fan",
        }
        return entity_id in exposed_entities

    return should_expose


@pytest.mark.asyncio
async def test_reindex_indexes_all_entities_bug(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that demonstrates the bug: ALL entities are indexed regardless of exposure.

    This test should FAIL initially, confirming the bug exists.
    After the fix, it should PASS.
    """
    # Patch CHROMADB_AVAILABLE and the embedding method
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Mock the embedding method to return a dummy vector
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        # Initialize the manager
        await manager._ensure_initialized()

        # Track which entities got indexed
        indexed_entities = []
        original_upsert = manager._collection.upsert

        def track_upsert(ids, embeddings, metadatas, documents):
            indexed_entities.extend(ids)
            return original_upsert(ids, embeddings, metadatas, documents)

        manager._collection.upsert = track_upsert

        # Patch async_should_expose at the module where it's used
        with patch(
            "custom_components.home_agent.vector_db_manager.async_should_expose",
            mock_async_should_expose,
        ):
            # Run the reindex
            result = await manager.async_reindex_all_entities()

        # BUG: Currently, the code indexes ALL entities (except skipped ones)
        # After fix: Should only index exposed entities

        # These entities should be indexed (they are exposed)
        assert "light.living_room" in indexed_entities
        assert "sensor.temperature" in indexed_entities
        assert "switch.fan" in indexed_entities

        # BUG: These entities should NOT be indexed (they are not exposed)
        # This assertion will FAIL before the fix, confirming the bug
        assert (
            "light.bedroom" not in indexed_entities
        ), "BUG: light.bedroom should NOT be indexed because it's not exposed"
        assert (
            "sensor.internal_metric" not in indexed_entities
        ), "BUG: sensor.internal_metric should NOT be indexed because it's not exposed"

        # Should only index 3 entities (the exposed ones)
        assert result["indexed"] == 3, (
            f"Expected 3 entities to be indexed, but got {result['indexed']}. "
            f"Bug: indexing non-exposed entities"
        )


@pytest.mark.asyncio
async def test_state_change_respects_exposure(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that state changes only trigger indexing for exposed entities.

    This test validates that the incremental update mechanism also respects
    entity exposure settings.
    """
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        await manager._ensure_initialized()

        # Track indexed entities
        indexed_entities = []

        def track_upsert(ids, embeddings, metadatas, documents):
            indexed_entities.extend(ids)

        manager._collection.upsert = track_upsert

        with patch(
            "custom_components.home_agent.vector_db_manager.async_should_expose",
            mock_async_should_expose,
        ):
            # Simulate state change for an exposed entity
            await manager.async_index_entity("light.living_room")
            assert "light.living_room" in indexed_entities

            # Reset tracking
            indexed_entities.clear()

            # Simulate state change for a non-exposed entity
            # BUG: This will currently index it, but shouldn't
            await manager.async_index_entity("light.bedroom")

            # After fix, this should not be indexed
            assert (
                "light.bedroom" not in indexed_entities
            ), "BUG: Non-exposed entity should not be indexed on state change"


@pytest.mark.asyncio
async def test_should_skip_entity_includes_non_exposed(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that _should_skip_entity considers entity exposure."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Patch at the module where it's imported and used
        with patch(
            "custom_components.home_agent.vector_db_manager.async_should_expose",
            mock_async_should_expose,
        ):
            # Exposed entities should NOT be skipped
            assert not manager._should_skip_entity("light.living_room")
            assert not manager._should_skip_entity("sensor.temperature")

            # Non-exposed entities SHOULD be skipped
            assert manager._should_skip_entity(
                "light.bedroom"
            ), "Non-exposed entities should be skipped"
            assert manager._should_skip_entity(
                "sensor.internal_metric"
            ), "Non-exposed entities should be skipped"

            # Internal entities should still be skipped
            assert manager._should_skip_entity("group.all_lights")
            assert manager._should_skip_entity("sun.sun")

"""Unit tests for VectorDBManager entity indexing.

Tests to validate that only exposed entities are indexed in ChromaDB,
not all entities in the system.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import aiohttp
import pytest
from homeassistant.const import EVENT_STATE_CHANGED
from homeassistant.core import Event, State

from custom_components.home_agent.const import (
    CONF_EMBEDDING_KEEP_ALIVE,
    CONF_OPENAI_API_KEY,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    DEFAULT_EMBEDDING_KEEP_ALIVE,
    DEFAULT_VECTOR_DB_COLLECTION,
    DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL,
    DEFAULT_VECTOR_DB_HOST,
    DEFAULT_VECTOR_DB_PORT,
    EMBEDDING_PROVIDER_OLLAMA,
    EMBEDDING_PROVIDER_OPENAI,
)
from custom_components.home_agent.exceptions import ContextInjectionError
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
    mock.async_create_background_task = MagicMock(
        side_effect=lambda coro, name: asyncio.ensure_future(coro)
    )

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


# ============================================================================
# LIFECYCLE MANAGEMENT TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_async_setup_performs_initial_indexing(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that async_setup performs initial entity indexing."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        # Track indexed entities
        indexed_entities = []

        def track_upsert(ids, embeddings, metadatas, documents):
            indexed_entities.extend(ids)

        # We need to mock the collection before setup
        await manager._ensure_initialized()
        manager._collection.upsert = track_upsert

        with patch(
            "custom_components.home_agent.vector_db_manager.async_should_expose",
            mock_async_should_expose,
        ):
            # Mock the reindex method to track that it was called
            with patch.object(manager, "async_reindex_all_entities", AsyncMock()) as mock_reindex:
                await manager.async_setup()

                # Verify initial indexing was scheduled as a background task
                mock_reindex.assert_called_once()
                mock_hass.async_create_background_task.assert_called_once()


@pytest.mark.asyncio
async def test_async_setup_registers_state_listeners(mock_hass, mock_chromadb, vector_db_config):
    """Test that async_setup registers state change listeners."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        await manager._ensure_initialized()

        # Mock async_track_time_interval
        with patch(
            "custom_components.home_agent.vector_db_manager.async_track_time_interval"
        ) as mock_track_time, patch.object(manager, "async_reindex_all_entities", AsyncMock()):
            await manager.async_setup()

            # Verify state change listener was registered
            mock_hass.bus.async_listen.assert_called_once_with(
                EVENT_STATE_CHANGED, manager._async_handle_state_change
            )

            # Verify maintenance listener was registered
            mock_track_time.assert_called_once()
            assert manager._state_listener is not None
            assert manager._maintenance_listener is not None


@pytest.mark.asyncio
async def test_async_setup_handles_chromadb_failure(mock_hass, vector_db_config):
    """Test that async_setup handles ChromaDB initialization failure."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Make _ensure_initialized raise an error
        with patch.object(
            manager, "_ensure_initialized", side_effect=ContextInjectionError("ChromaDB unavailable")
        ):
            with pytest.raises(ContextInjectionError, match="ChromaDB unavailable"):
                await manager.async_setup()


@pytest.mark.asyncio
async def test_async_shutdown_cleans_up_listeners(mock_hass, mock_chromadb, vector_db_config):
    """Test that async_shutdown properly cleans up listeners and resources."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        await manager._ensure_initialized()

        # Set up listeners
        state_listener_mock = MagicMock()
        maintenance_listener_mock = MagicMock()
        manager._state_listener = state_listener_mock
        manager._maintenance_listener = maintenance_listener_mock

        # Add some items to embedding cache
        manager._embedding_cache["test_key"] = [0.1] * 384

        # Shutdown
        await manager.async_shutdown()

        # Verify listeners were called (to unregister)
        state_listener_mock.assert_called_once()
        maintenance_listener_mock.assert_called_once()

        # Verify listeners were cleared
        assert manager._state_listener is None
        assert manager._maintenance_listener is None

        # Verify cache was cleared
        assert len(manager._embedding_cache) == 0


@pytest.mark.asyncio
async def test_async_shutdown_handles_no_listeners(mock_hass, mock_chromadb, vector_db_config):
    """Test that async_shutdown handles case where no listeners are registered."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Shutdown without setting up listeners
        await manager.async_shutdown()

        # Should not raise any errors
        assert manager._state_listener is None
        assert manager._maintenance_listener is None


# ============================================================================
# EMBEDDING GENERATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_embed_text_uses_cache(mock_hass, mock_chromadb, vector_db_config):
    """Test that _embed_text uses cached embeddings when available."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Pre-populate cache
        test_text = "test entity"
        cached_embedding = [0.5] * 384
        import hashlib

        cache_key = hashlib.md5(test_text.encode()).hexdigest()
        manager._embedding_cache[cache_key] = cached_embedding

        # Mock the actual embedding methods to ensure they're not called
        manager._embed_with_openai = AsyncMock()
        manager._embed_with_ollama = AsyncMock()

        # Get embedding
        result = await manager._embed_text(test_text)

        # Verify cache was used
        assert result == cached_embedding
        manager._embed_with_openai.assert_not_called()
        manager._embed_with_ollama.assert_not_called()


@pytest.mark.asyncio
async def test_embed_text_cache_miss_generates_new(mock_hass, mock_chromadb, vector_db_config):
    """Test that _embed_text generates new embedding on cache miss."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OLLAMA

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, config)

        test_text = "new entity"
        new_embedding = [0.7] * 384

        # Mock the Ollama embedding method
        manager._embed_with_ollama = AsyncMock(return_value=new_embedding)

        # Get embedding (cache miss)
        result = await manager._embed_text(test_text)

        # Verify new embedding was generated
        assert result == new_embedding
        manager._embed_with_ollama.assert_called_once_with(test_text)

        # Verify it was cached
        import hashlib

        cache_key = hashlib.md5(test_text.encode()).hexdigest()
        assert cache_key in manager._embedding_cache
        assert manager._embedding_cache[cache_key] == new_embedding


@pytest.mark.asyncio
async def test_embed_with_openai_success(mock_hass, mock_chromadb, vector_db_config):
    """Test successful embedding generation with OpenAI."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OPENAI
    config[CONF_OPENAI_API_KEY] = "test-api-key"

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True), patch(
        "custom_components.home_agent.vector_db_manager.OPENAI_AVAILABLE", True
    ):
        manager = VectorDBManager(mock_hass, config)

        test_text = "test entity"
        expected_embedding = [0.3] * 1536

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data = [MagicMock()]
        mock_response.data[0].embedding = expected_embedding

        # Mock the OpenAI client
        with patch("custom_components.home_agent.vector_db_manager.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            mock_openai.AsyncOpenAI.return_value = mock_client

            # Mock retry_async to await and call the function
            async def mock_retry(func, **kwargs):
                return await func()

            with patch(
                "custom_components.home_agent.vector_db_manager.retry_async",
                side_effect=mock_retry,
            ):
                result = await manager._embed_with_openai(test_text)

                assert result == expected_embedding
                mock_client.embeddings.create.assert_called_once()


@pytest.mark.asyncio
async def test_embed_with_openai_api_error(mock_hass, mock_chromadb, vector_db_config):
    """Test that OpenAI API errors are properly handled."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OPENAI
    config[CONF_OPENAI_API_KEY] = "test-api-key"

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True), patch(
        "custom_components.home_agent.vector_db_manager.OPENAI_AVAILABLE", True
    ):
        manager = VectorDBManager(mock_hass, config)

        test_text = "test entity"

        # Mock OpenAI to raise an error
        with patch("custom_components.home_agent.vector_db_manager.openai") as mock_openai:
            mock_client = AsyncMock()
            mock_error = Exception("API Error")
            mock_client.embeddings.create = AsyncMock(side_effect=mock_error)
            mock_openai.AsyncOpenAI.return_value = mock_client

            # Mock retry_async to await and call the function
            async def mock_retry(func, **kwargs):
                return await func()

            with patch(
                "custom_components.home_agent.vector_db_manager.retry_async",
                side_effect=mock_retry,
            ):
                with pytest.raises(Exception, match="API Error"):
                    await manager._embed_with_openai(test_text)


@pytest.mark.asyncio
async def test_embed_with_openai_missing_api_key(mock_hass, mock_chromadb, vector_db_config):
    """Test that missing OpenAI API key raises error."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OPENAI
    # Don't set API key

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True), patch(
        "custom_components.home_agent.vector_db_manager.OPENAI_AVAILABLE", True
    ):
        manager = VectorDBManager(mock_hass, config)

        with pytest.raises(ContextInjectionError, match="OpenAI API key not configured"):
            await manager._embed_with_openai("test text")


@pytest.mark.asyncio
async def test_embed_with_openai_library_not_available(mock_hass, mock_chromadb, vector_db_config):
    """Test that missing OpenAI library raises error."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OPENAI
    config[CONF_OPENAI_API_KEY] = "test-api-key"

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True), patch(
        "custom_components.home_agent.vector_db_manager.OPENAI_AVAILABLE", False
    ):
        manager = VectorDBManager(mock_hass, config)

        with pytest.raises(ContextInjectionError, match="OpenAI library not installed"):
            await manager._embed_with_openai("test text")


@pytest.mark.asyncio
async def test_embed_with_ollama_success(mock_hass, mock_chromadb, vector_db_config):
    """Test successful embedding generation with Ollama."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OLLAMA
    config[CONF_VECTOR_DB_EMBEDDING_BASE_URL] = "http://localhost:11434"

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, config)

        test_text = "test entity"
        expected_embedding = [0.4] * 768

        # Mock aiohttp response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"embedding": expected_embedding})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Mock retry_async to await and call the function
        async def mock_retry(func, **kwargs):
            return await func()

        # Set the shared session directly instead of patching ClientSession
        manager._aiohttp_session = mock_session

        with patch(
            "custom_components.home_agent.vector_db_manager.retry_async",
            side_effect=mock_retry,
        ):
            result = await manager._embed_with_ollama(test_text)

            assert result == expected_embedding


@pytest.mark.asyncio
async def test_embed_with_ollama_timeout(mock_hass, mock_chromadb, vector_db_config):
    """Test that Ollama timeout errors are properly handled."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OLLAMA

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, config)

        test_text = "test entity"

        # Mock aiohttp to raise ClientError (which includes timeout errors)
        mock_response = MagicMock()
        mock_response.__aenter__ = AsyncMock(
            side_effect=aiohttp.ClientError("Connection timeout")
        )
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Mock retry_async to await and call the function
        async def mock_retry(func, **kwargs):
            return await func()

        # Set the shared session directly
        manager._aiohttp_session = mock_session

        with patch(
            "custom_components.home_agent.vector_db_manager.retry_async",
            side_effect=mock_retry,
        ):
            with pytest.raises(ContextInjectionError, match="Failed to connect to Ollama"):
                await manager._embed_with_ollama(test_text)


@pytest.mark.asyncio
async def test_embed_with_ollama_api_error(mock_hass, mock_chromadb, vector_db_config):
    """Test that Ollama API errors are properly handled."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = EMBEDDING_PROVIDER_OLLAMA

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, config)

        test_text = "test entity"

        # Mock aiohttp response with error
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Mock retry_async to await and call the function
        async def mock_retry(func, **kwargs):
            return await func()

        # Set the shared session directly
        manager._aiohttp_session = mock_session

        with patch(
            "custom_components.home_agent.vector_db_manager.retry_async",
            side_effect=mock_retry,
        ):
            with pytest.raises(ContextInjectionError, match="Ollama API error 500"):
                await manager._embed_with_ollama(test_text)


@pytest.mark.asyncio
async def test_embed_text_unknown_provider(mock_hass, mock_chromadb, vector_db_config):
    """Test that unknown embedding provider raises error."""
    config = vector_db_config.copy()
    config[CONF_VECTOR_DB_EMBEDDING_PROVIDER] = "unknown_provider"

    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, config)

        with pytest.raises(ContextInjectionError, match="Unknown embedding provider"):
            await manager._embed_text("test text")


# ============================================================================
# ENTITY OPERATIONS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_async_remove_entity_success(mock_hass, mock_chromadb, vector_db_config):
    """Test successful entity removal from ChromaDB."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Mock the collection delete method
        delete_mock = MagicMock()
        manager._collection.delete = delete_mock

        # Remove an entity
        await manager.async_remove_entity("light.living_room")

        # Verify delete was called
        mock_hass.async_add_executor_job.assert_called()


@pytest.mark.asyncio
async def test_async_remove_entity_handles_error(mock_hass, mock_chromadb, vector_db_config):
    """Test that entity removal handles errors gracefully."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Make delete raise an error
        mock_hass.async_add_executor_job = AsyncMock(side_effect=Exception("Delete failed"))

        # Remove should not raise, just log error
        await manager.async_remove_entity("light.living_room")
        # No exception should be raised


@pytest.mark.asyncio
async def test_async_collection_exists_returns_true(mock_hass, mock_chromadb, vector_db_config):
    """Test that async_collection_exists returns True for existing collection."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Mock successful get_collection call
        mock_collection = MagicMock()
        mock_hass.async_add_executor_job = AsyncMock(return_value=mock_collection)

        result = await manager.async_collection_exists("test_collection")

        assert result is True


@pytest.mark.asyncio
async def test_async_collection_exists_returns_false(mock_hass, mock_chromadb, vector_db_config):
    """Test that async_collection_exists returns False for non-existent collection."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Mock get_collection to raise exception (collection not found)
        mock_hass.async_add_executor_job = AsyncMock(side_effect=Exception("Not found"))

        result = await manager.async_collection_exists("nonexistent_collection")

        assert result is False


@pytest.mark.asyncio
async def test_async_collection_exists_no_client(mock_hass, mock_chromadb, vector_db_config):
    """Test that async_collection_exists returns False when client is None."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Patch _ensure_initialized to not actually initialize, keeping client as None
        async def mock_ensure_initialized():
            pass  # Don't initialize, keep client as None

        with patch.object(manager, "_ensure_initialized", side_effect=mock_ensure_initialized):
            # Client should remain None
            result = await manager.async_collection_exists("test_collection")

            assert result is False


# ============================================================================
# EVENT HANDLING & BACKGROUND TASKS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_async_handle_state_change_triggers_reindex(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that state changes trigger debounced entity reindexing."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        await manager._ensure_initialized()

        with patch.object(manager, "async_index_entity", AsyncMock()) as mock_index:
            # Create state change event
            event_data = {"entity_id": "light.living_room"}
            event = Event(EVENT_STATE_CHANGED, event_data)

            with patch(
                "custom_components.home_agent.vector_db_manager.async_should_expose",
                mock_async_should_expose,
            ), patch(
                "custom_components.home_agent.vector_db_manager.REINDEX_DEBOUNCE_DELAY", 0.05
            ):
                # Handle the event
                manager._async_handle_state_change(event)

                # Entity should be in pending reindex
                assert "light.living_room" in manager._pending_reindex

                # Wait for debounce to fire
                await asyncio.sleep(0.2)

                # Verify reindexing was called
                mock_index.assert_called_once_with("light.living_room")


@pytest.mark.asyncio
async def test_async_handle_state_change_skips_non_exposed(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that state changes for non-exposed entities are ignored."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        # Mock the reindex method
        with patch.object(manager, "async_index_entity", AsyncMock()) as mock_index:
            # Create state change event for non-exposed entity
            event_data = {"entity_id": "light.bedroom"}
            event = Event(EVENT_STATE_CHANGED, event_data)

            with patch(
                "custom_components.home_agent.vector_db_manager.async_should_expose",
                mock_async_should_expose,
            ):
                # Handle the event
                manager._async_handle_state_change(event)

                # Wait a bit
                await asyncio.sleep(0.1)

                # Indexing should not be called for non-exposed entities
                mock_index.assert_not_called()


@pytest.mark.asyncio
async def test_debounced_reindex_batches_multiple_entities(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that debounced reindex batches multiple entity changes."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)
        manager._embed_text = AsyncMock(return_value=[0.1] * 384)

        await manager._ensure_initialized()

        with patch.object(manager, "async_index_entity", AsyncMock()) as mock_index, patch(
            "custom_components.home_agent.vector_db_manager.async_should_expose",
            mock_async_should_expose,
        ), patch(
            "custom_components.home_agent.vector_db_manager.REINDEX_DEBOUNCE_DELAY", 0.05
        ):
            # Fire multiple state changes rapidly (use exposed entities)
            for entity_id in ["light.living_room", "sensor.temperature", "switch.fan"]:
                event = Event(EVENT_STATE_CHANGED, {"entity_id": entity_id})
                manager._async_handle_state_change(event)

            # Wait for debounce
            await asyncio.sleep(0.2)

            # All three entities should have been reindexed
            assert mock_index.call_count == 3


@pytest.mark.asyncio
async def test_debounced_reindex_handles_error_gracefully(
    mock_hass, mock_chromadb, vector_db_config, mock_async_should_expose
):
    """Test that debounced reindexing handles errors gracefully."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Make async_index_entity raise an error
        with patch.object(
            manager, "async_index_entity", AsyncMock(side_effect=Exception("Index failed"))
        ), patch(
            "custom_components.home_agent.vector_db_manager.async_should_expose",
            mock_async_should_expose,
        ), patch(
            "custom_components.home_agent.vector_db_manager.REINDEX_DEBOUNCE_DELAY", 0.05
        ):
            event = Event(EVENT_STATE_CHANGED, {"entity_id": "light.living_room"})
            manager._async_handle_state_change(event)

            # Should not raise, just log
            await asyncio.sleep(0.2)


@pytest.mark.asyncio
async def test_async_run_maintenance_removes_stale_entities(
    mock_hass, mock_chromadb, vector_db_config
):
    """Test that maintenance task removes stale entities."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Mock collection.get() to return entities including stale ones
        stale_ids = ["light.deleted", "sensor.removed"]
        current_ids = ["light.living_room", "sensor.temperature"]
        all_indexed_ids = stale_ids + current_ids

        mock_hass.async_add_executor_job = AsyncMock(
            return_value={"ids": all_indexed_ids, "documents": [], "metadatas": []}
        )

        # Mock current entity IDs (only current ones, not stale)
        mock_hass.states.async_entity_ids = MagicMock(return_value=current_ids)

        # Mock async_remove_entity
        with patch.object(manager, "async_remove_entity", AsyncMock()) as mock_remove:
            await manager._async_run_maintenance(None)

            # Verify stale entities were removed
            assert mock_remove.call_count == 2
            # Check that both stale IDs were removed
            removed_ids = [call[0][0] for call in mock_remove.call_args_list]
            assert set(removed_ids) == set(stale_ids)


@pytest.mark.asyncio
async def test_async_run_maintenance_handles_no_stale_entities(
    mock_hass, mock_chromadb, vector_db_config
):
    """Test maintenance task when no stale entities exist."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Mock collection.get() to return only current entities
        current_ids = ["light.living_room", "sensor.temperature"]

        mock_hass.async_add_executor_job = AsyncMock(
            return_value={"ids": current_ids, "documents": [], "metadatas": []}
        )

        mock_hass.states.async_entity_ids = MagicMock(return_value=current_ids)

        # Mock async_remove_entity
        with patch.object(manager, "async_remove_entity", AsyncMock()) as mock_remove:
            await manager._async_run_maintenance(None)

            # No entities should be removed
            mock_remove.assert_not_called()


@pytest.mark.asyncio
async def test_async_run_maintenance_handles_error(mock_hass, mock_chromadb, vector_db_config):
    """Test that maintenance task handles errors gracefully."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Make collection.get() raise an error
        mock_hass.async_add_executor_job = AsyncMock(side_effect=Exception("ChromaDB error"))

        # Should not raise, just log warning
        await manager._async_run_maintenance(None)


@pytest.mark.asyncio
async def test_async_run_maintenance_handles_empty_result(
    mock_hass, mock_chromadb, vector_db_config
):
    """Test maintenance task when collection.get() returns empty result."""
    with patch("custom_components.home_agent.vector_db_manager.CHROMADB_AVAILABLE", True):
        manager = VectorDBManager(mock_hass, vector_db_config)

        await manager._ensure_initialized()

        # Mock collection.get() to return empty result
        mock_hass.async_add_executor_job = AsyncMock(return_value={})

        # Should not raise
        await manager._async_run_maintenance(None)

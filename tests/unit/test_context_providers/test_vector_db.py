"""Unit tests for Vector DB context provider.

This module tests the VectorDBContextProvider which integrates with ChromaDB
for semantic entity search and intelligent context injection.
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from homeassistant.core import State

from custom_components.home_agent.context_providers.vector_db import (
    VectorDBContextProvider,
    CHROMADB_AVAILABLE,
    OPENAI_AVAILABLE,
)
from custom_components.home_agent.exceptions import ContextInjectionError
from custom_components.home_agent.const import (
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_TOP_K,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_OPENAI_API_KEY,
)


class TestVectorDBContextProviderInit:
    """Tests for VectorDBContextProvider initialization."""

    def test_vector_db_provider_init_success(self, mock_hass):
        """Test initializing vector DB provider with valid config."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {
            CONF_VECTOR_DB_HOST: "localhost",
            CONF_VECTOR_DB_PORT: 8000,
            CONF_VECTOR_DB_COLLECTION: "test_collection",
            CONF_VECTOR_DB_EMBEDDING_MODEL: "text-embedding-3-small",
            CONF_VECTOR_DB_TOP_K: 5,
            CONF_VECTOR_DB_SIMILARITY_THRESHOLD: 0.7,
            CONF_OPENAI_API_KEY: "sk-test-key"
        }
        provider = VectorDBContextProvider(mock_hass, config)

        assert provider.hass == mock_hass
        assert provider.config == config
        assert provider.host == "localhost"
        assert provider.port == 8000
        assert provider.collection_name == "test_collection"
        assert provider.embedding_model == "text-embedding-3-small"
        assert provider.top_k == 5
        assert provider.similarity_threshold == 0.7
        assert provider.openai_api_key == "sk-test-key"
        assert provider._client is None
        assert provider._collection is None

    def test_vector_db_provider_init_defaults(self, mock_hass):
        """Test initialization with default values."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        assert provider.host == "localhost"
        assert provider.port == 8000
        assert provider.collection_name == "home_entities"
        assert provider.embedding_model == "text-embedding-3-small"
        assert provider.top_k == 5
        assert provider.similarity_threshold == 250.0

    def test_vector_db_provider_init_chromadb_not_available(self, mock_hass):
        """Test initialization fails when ChromaDB is not installed."""
        with patch(
            "custom_components.home_agent.context_providers.vector_db.CHROMADB_AVAILABLE",
            False
        ):
            config = {CONF_OPENAI_API_KEY: "sk-test"}
            with pytest.raises(ContextInjectionError) as exc_info:
                VectorDBContextProvider(mock_hass, config)

            assert "ChromaDB not installed" in str(exc_info.value)

    def test_vector_db_provider_init_state(self, mock_hass):
        """Test initial state of provider."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        assert provider._client is None
        assert provider._collection is None
        assert provider._embedding_cache == {}


class TestGetContext:
    """Tests for get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_success(self, mock_hass):
        """Test successful context retrieval."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test", CONF_VECTOR_DB_TOP_K: 3}
        provider = VectorDBContextProvider(mock_hass, config)

        # Mock initialization
        provider._initialized = True
        provider._collection = Mock()

        # Mock embedding
        with patch.object(provider, '_embed_query', return_value=[0.1, 0.2, 0.3]):
            # Mock vector DB query
            with patch.object(provider, '_query_vector_db', return_value={
                "ids": [["light.living_room", "sensor.temp"]],
                "distances": [[0.1, 0.2]],
                "documents": [["doc1", "doc2"]],
                "metadatas": [[
                    {"entity_id": "light.living_room"},
                    {"entity_id": "sensor.temp"}
                ]]
            }):
                # Mock entity state retrieval
                light_state = Mock(spec=State)
                light_state.state = "on"
                light_state.attributes = {"brightness": 128}

                sensor_state = Mock(spec=State)
                sensor_state.state = "72"
                sensor_state.attributes = {"unit_of_measurement": "°F"}

                def get_state_side_effect(entity_id):
                    if entity_id == "light.living_room":
                        return light_state
                    elif entity_id == "sensor.temp":
                        return sensor_state
                    return None

                mock_hass.states.get.side_effect = get_state_side_effect

                result = await provider.get_context("turn on the lights")

                # Verify result
                assert isinstance(result, str)
                parsed = json.loads(result)
                assert "query" in parsed
                assert "relevant_entities" in parsed
                assert "count" in parsed
                assert parsed["query"] == "turn on the lights"
                assert parsed["count"] == 2

    @pytest.mark.asyncio
    async def test_get_context_initializes_collection(self, mock_hass):
        """Test that get_context initializes collection if needed."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        assert not provider._initialized

        with patch.object(provider, '_initialize_collection') as mock_init:
            mock_init.return_value = None
            provider._initialized = True

            with patch.object(provider, '_embed_query', return_value=[0.1, 0.2]):
                with patch.object(provider, '_query_vector_db', return_value={
                    "ids": [[]],
                    "distances": [[]],
                    "documents": [[]],
                    "metadatas": [[]]
                }):
                    await provider.get_context("test query")
                    mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_no_results(self, mock_hass):
        """Test context retrieval when no results found."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        with patch.object(provider, '_embed_query', return_value=[0.1, 0.2]):
            with patch.object(provider, '_query_vector_db', return_value={
                "ids": [[]],
                "distances": [[]],
                "documents": [[]],
                "metadatas": [[]]
            }):
                result = await provider.get_context("test query")

                assert "No relevant entities found" in result

    @pytest.mark.asyncio
    async def test_get_context_error_handling(self, mock_hass):
        """Test error handling in get_context."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        with patch.object(
            provider,
            '_embed_query',
            side_effect=Exception("Embedding failed")
        ):
            with pytest.raises(ContextInjectionError) as exc_info:
                await provider.get_context("test query")

            assert "Vector DB query failed" in str(exc_info.value)
            assert "Embedding failed" in str(exc_info.value)


class TestInitializeCollection:
    """Tests for _initialize_collection method."""

    @pytest.mark.asyncio
    async def test_initialize_collection_success(self, mock_hass):
        """Test successful collection initialization."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.HttpClient", return_value=mock_client):
            if OPENAI_AVAILABLE:
                with patch("custom_components.home_agent.context_providers.vector_db.AsyncOpenAI"):
                    await provider._initialize_collection()
            else:
                await provider._initialize_collection()

            assert provider._initialized
            assert provider._client == mock_client
            assert provider._collection == mock_collection

    @pytest.mark.asyncio
    async def test_initialize_collection_retry_logic(self, mock_hass):
        """Test retry logic on initialization failure."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        # Fail first attempt, succeed on second
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            mock_client = Mock()
            mock_client.get_or_create_collection.return_value = Mock()
            return mock_client

        with patch("chromadb.HttpClient", side_effect=side_effect):
            if OPENAI_AVAILABLE:
                with patch("custom_components.home_agent.context_providers.vector_db.AsyncOpenAI"):
                    await provider._initialize_collection()
            else:
                await provider._initialize_collection()

            assert call_count == 2
            assert provider._initialized

    @pytest.mark.asyncio
    async def test_initialize_collection_max_retries_exceeded(self, mock_hass):
        """Test initialization fails after max retries."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        with patch(
            "chromadb.HttpClient",
            side_effect=ConnectionError("Connection failed")
        ):
            with pytest.raises(ContextInjectionError) as exc_info:
                await provider._initialize_collection()

            assert "Failed to initialize ChromaDB" in str(exc_info.value)
            assert not provider._initialized

    @pytest.mark.asyncio
    async def test_initialize_collection_no_api_key(self, mock_hass):
        """Test initialization fails without API key."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {"use_local_embeddings": False}  # No API key provided
        provider = VectorDBContextProvider(mock_hass, config)

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.HttpClient", return_value=mock_client):
            with pytest.raises(ContextInjectionError) as exc_info:
                await provider._initialize_collection()

            assert "OpenAI API key required" in str(exc_info.value)


class TestEmbedQuery:
    """Tests for _embed_query method."""

    @pytest.mark.asyncio
    async def test_embed_query_success(self, mock_hass):
        """Test successful query embedding."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        mock_openai = AsyncMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_openai.embeddings.create.return_value = mock_response
        provider._openai_client = mock_openai

        result = await provider._embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        assert "test query" in provider._embedding_cache
        assert provider._embedding_cache["test query"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_query_uses_cache(self, mock_hass):
        """Test that cached embeddings are reused."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        # Populate cache
        provider._embedding_cache["cached query"] = [0.5, 0.6, 0.7]

        mock_openai = AsyncMock()
        provider._openai_client = mock_openai

        result = await provider._embed_query("cached query")

        # Should use cached value, not call API
        assert result == [0.5, 0.6, 0.7]
        mock_openai.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_query_no_client(self, mock_hass):
        """Test embedding fails when client not initialized."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._openai_client = None

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider._embed_query("test")

        assert "OpenAI client not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_query_local_not_implemented(self, mock_hass):
        """Test local embeddings raise NotImplementedError."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {"use_local_embeddings": True}
        provider = VectorDBContextProvider(mock_hass, config)

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider._embed_query("test")

        assert "not yet implemented" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_embed_query_api_error(self, mock_hass):
        """Test error handling when API call fails."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        mock_openai = AsyncMock()
        mock_openai.embeddings.create.side_effect = Exception("API error")
        provider._openai_client = mock_openai

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider._embed_query("test")

        assert "Embedding generation failed" in str(exc_info.value)


class TestQueryVectorDB:
    """Tests for _query_vector_db method."""

    @pytest.mark.asyncio
    async def test_query_vector_db_success(self, mock_hass):
        """Test successful vector DB query."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test", CONF_VECTOR_DB_TOP_K: 5}
        provider = VectorDBContextProvider(mock_hass, config)

        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["entity1", "entity2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"meta1": "val1"}, {"meta2": "val2"}]]
        }
        provider._collection = mock_collection

        embedding = [0.1, 0.2, 0.3]
        result = await provider._query_vector_db(embedding, 5)

        assert result["ids"] == [["entity1", "entity2"]]
        assert result["distances"] == [[0.1, 0.2]]
        mock_collection.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_vector_db_no_collection(self, mock_hass):
        """Test query fails when collection not initialized."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._collection = None

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider._query_vector_db([0.1, 0.2], 5)

        assert "collection not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_query_vector_db_error(self, mock_hass):
        """Test error handling during query."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Query failed")
        provider._collection = mock_collection

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider._query_vector_db([0.1, 0.2], 5)

        assert "Vector DB query failed" in str(exc_info.value)


class TestFormatResults:
    """Tests for _format_results method."""

    def test_format_results_success(self, mock_hass):
        """Test successful results formatting."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test", CONF_VECTOR_DB_SIMILARITY_THRESHOLD: 0.5}
        provider = VectorDBContextProvider(mock_hass, config)

        results = {
            "ids": [["light.living_room", "sensor.temp"]],
            "distances": [[0.2, 0.3]],  # Similarity: 0.8, 0.7
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "val1"}, {"key": "val2"}]]
        }

        light_state = Mock(spec=State)
        light_state.state = "on"
        light_state.attributes = {"brightness": 128}

        sensor_state = Mock(spec=State)
        sensor_state.state = "72"
        sensor_state.attributes = {"unit": "°F"}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light_state
            elif entity_id == "sensor.temp":
                return sensor_state
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        result = provider._format_results(results, "test query")

        parsed = json.loads(result)
        assert parsed["query"] == "test query"
        assert parsed["count"] == 2
        assert len(parsed["relevant_entities"]) == 2
        assert parsed["relevant_entities"][0]["entity_id"] == "light.living_room"
        assert "similarity_score" in parsed["relevant_entities"][0]

    def test_format_results_empty(self, mock_hass):
        """Test formatting empty results."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        results = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]]
        }

        result = provider._format_results(results, "test query")

        assert "No relevant entities found" in result

    def test_format_results_below_threshold(self, mock_hass):
        """Test filtering results below similarity threshold."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test", CONF_VECTOR_DB_SIMILARITY_THRESHOLD: 0.8}
        provider = VectorDBContextProvider(mock_hass, config)

        results = {
            "ids": [["entity1", "entity2"]],
            "distances": [[0.5, 0.9]],  # Similarity: 0.5, 0.1 (both below 0.8)
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{}, {}]]
        }

        result = provider._format_results(results, "test query")

        assert "No entities similar enough" in result
        assert "0.8" in result  # Should mention threshold


class TestIndexEntities:
    """Tests for _index_entities method."""

    @pytest.mark.asyncio
    async def test_index_entities_all(self, mock_hass):
        """Test indexing all entities."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        mock_collection = Mock()
        provider._collection = mock_collection

        # Mock entity IDs
        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "sensor.temp"
        ]

        # Mock entity states
        light_state = Mock(spec=State)
        light_state.state = "on"
        light_state.attributes = {"brightness": 128}

        sensor_state = Mock(spec=State)
        sensor_state.state = "72"
        sensor_state.attributes = {"unit": "°F"}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light_state
            elif entity_id == "sensor.temp":
                return sensor_state
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        # Mock embedding batch
        with patch.object(
            provider,
            '_embed_batch',
            return_value=[[0.1, 0.2], [0.3, 0.4]]
        ):
            count = await provider._index_entities()

            assert count == 2
            mock_collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_entities_with_patterns(self, mock_hass):
        """Test indexing entities matching patterns."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True
        provider._collection = Mock()

        # Mock pattern matching
        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom",
            "sensor.temp"
        ]

        light1 = Mock(spec=State)
        light1.state = "on"
        light1.attributes = {}

        light2 = Mock(spec=State)
        light2.state = "off"
        light2.attributes = {}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light1
            elif entity_id == "light.bedroom":
                return light2
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        with patch.object(provider, '_embed_batch', return_value=[[0.1], [0.2]]):
            count = await provider._index_entities(["light.*"])

            assert count == 2  # Only lights indexed

    @pytest.mark.asyncio
    async def test_index_entities_no_entities(self, mock_hass):
        """Test indexing when no entities found."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        mock_hass.states.async_entity_ids.return_value = []

        count = await provider._index_entities()

        assert count == 0


class TestCreateSearchableDocument:
    """Tests for _create_searchable_document method."""

    def test_create_searchable_document_light(self, mock_hass):
        """Test creating document for light entity."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        state_data = {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "friendly_name": "Living Room Light",
                "brightness": 128
            }
        }

        doc = provider._create_searchable_document(state_data)

        assert "Living Room Light" in doc
        assert "light" in doc
        assert "on" in doc
        assert "light that can be controlled" in doc

    def test_create_searchable_document_sensor(self, mock_hass):
        """Test creating document for sensor entity."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        state_data = {
            "entity_id": "sensor.temperature",
            "state": "72",
            "attributes": {
                "unit_of_measurement": "°F",
                "device_class": "temperature"
            }
        }

        doc = provider._create_searchable_document(state_data)

        assert "Temperature" in doc
        assert "sensor" in doc
        assert "72" in doc
        assert "°F" in doc
        assert "temperature" in doc
        assert "sensor that measures" in doc


class TestUpdateEntity:
    """Tests for update_entity method."""

    @pytest.mark.asyncio
    async def test_update_entity_success(self, mock_hass):
        """Test successful entity update."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        mock_collection = Mock()
        provider._collection = mock_collection

        # Mock entity state
        state = Mock(spec=State)
        state.state = "on"
        state.attributes = {"brightness": 200}
        mock_hass.states.get.return_value = state

        with patch.object(provider, '_embed_query', return_value=[0.1, 0.2, 0.3]):
            result = await provider.update_entity("light.living_room")

            assert result is True
            mock_collection.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entity_not_found(self, mock_hass):
        """Test updating non-existent entity."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        mock_hass.states.get.return_value = None

        result = await provider.update_entity("light.nonexistent")

        assert result is False


class TestClearCollection:
    """Tests for clear_collection method."""

    @pytest.mark.asyncio
    async def test_clear_collection_success(self, mock_hass):
        """Test successful collection clearing."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.delete_collection = Mock()
        mock_client.create_collection.return_value = mock_collection
        provider._client = mock_client
        provider._collection = Mock()

        await provider.clear_collection()

        mock_client.delete_collection.assert_called_once()
        mock_client.create_collection.assert_called_once()
        assert provider._collection == mock_collection

    @pytest.mark.asyncio
    async def test_clear_collection_error(self, mock_hass):
        """Test error handling when clearing collection."""
        if not CHROMADB_AVAILABLE:
            pytest.skip("ChromaDB not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)
        provider._initialized = True

        mock_client = Mock()
        mock_client.delete_collection.side_effect = Exception("Delete failed")
        provider._client = mock_client

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider.clear_collection()

        assert "Collection clear failed" in str(exc_info.value)

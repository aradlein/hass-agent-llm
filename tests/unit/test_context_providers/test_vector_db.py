"""Unit tests for Vector DB context provider.

This module tests the VectorDBContextProvider which integrates with ChromaDB
for semantic entity search and intelligent context injection.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from homeassistant.core import State

from custom_components.home_agent.const import (
    CONF_OPENAI_API_KEY,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
    CONF_VECTOR_DB_TOP_K,
    EMBEDDING_PROVIDER_OPENAI,
)
from custom_components.home_agent.context_providers.vector_db import (
    CHROMADB_AVAILABLE,
    OPENAI_AVAILABLE,
    VectorDBContextProvider,
)
from custom_components.home_agent.exceptions import ContextInjectionError


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
            CONF_OPENAI_API_KEY: "sk-test-key",
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
            "custom_components.home_agent.context_providers.vector_db.CHROMADB_AVAILABLE", False
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

        # Mock ChromaDB client and collection to prevent real connections
        provider._client = Mock()
        provider._collection = Mock()

        # Mock embedding
        with patch.object(provider, "_embed_query", return_value=[0.1, 0.2, 0.3]):
            # Mock vector DB query - returns list of dicts with entity_id and distance
            with patch.object(
                provider,
                "_query_vector_db",
                return_value=[
                    {"entity_id": "light.living_room", "distance": 0.1},
                    {"entity_id": "sensor.temp", "distance": 0.2},
                ],
            ):
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
                assert "entities" in parsed
                assert "count" in parsed
                assert parsed["count"] == 2
                assert len(parsed["entities"]) == 2
                # Verify entity data
                assert parsed["entities"][0]["entity_id"] == "light.living_room"
                assert parsed["entities"][0]["state"] == "on"
                assert parsed["entities"][1]["entity_id"] == "sensor.temp"
                assert parsed["entities"][1]["state"] == "72"

    @pytest.mark.asyncio
    async def test_get_context_no_results(self, mock_hass):
        """Test context retrieval when no results found."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        # Mock ChromaDB client and collection
        provider._client = Mock()
        provider._collection = Mock()

        with patch.object(provider, "_embed_query", return_value=[0.1, 0.2]):
            # Mock query returning empty list (no results)
            with patch.object(provider, "_query_vector_db", return_value=[]):
                result = await provider.get_context("test query")

                assert "No relevant context found" in result

    @pytest.mark.asyncio
    async def test_get_context_error_handling(self, mock_hass):
        """Test error handling in get_context."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {CONF_OPENAI_API_KEY: "sk-test"}
        provider = VectorDBContextProvider(mock_hass, config)

        # Mock ChromaDB client and collection
        provider._client = Mock()
        provider._collection = Mock()

        with patch.object(provider, "_embed_query", side_effect=Exception("Embedding failed")):
            with pytest.raises(ContextInjectionError) as exc_info:
                await provider.get_context("test query")

            assert "Vector DB query failed" in str(exc_info.value)
            assert "Embedding failed" in str(exc_info.value)


class TestEmbedQuery:
    """Tests for _embed_query method."""

    @pytest.mark.asyncio
    async def test_embed_query_success(self, mock_hass):
        """Test successful query embedding."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {
            CONF_OPENAI_API_KEY: "sk-test",
            CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OPENAI,
        }
        provider = VectorDBContextProvider(mock_hass, config)

        # Mock the new OpenAI API client.embeddings.create call
        mock_embedding_data = MagicMock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_data]

        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client
            result = await provider._embed_query("test query")

        assert result == [0.1, 0.2, 0.3]
        # Cache key is MD5 hash of the text
        import hashlib

        cache_key = hashlib.md5("test query".encode()).hexdigest()
        assert cache_key in provider._embedding_cache
        assert provider._embedding_cache[cache_key] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_query_uses_cache(self, mock_hass):
        """Test that cached embeddings are reused."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {
            CONF_OPENAI_API_KEY: "sk-test",
            CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OPENAI,
        }
        provider = VectorDBContextProvider(mock_hass, config)

        # Populate cache with MD5 hash key
        import hashlib

        cache_key = hashlib.md5("cached query".encode()).hexdigest()
        provider._embedding_cache[cache_key] = [0.5, 0.6, 0.7]

        # Mock openai.Embedding.create - should not be called due to cache
        with patch("openai.Embedding.create") as mock_create:
            result = await provider._embed_query("cached query")

            # Should use cached value, not call API
            assert result == [0.5, 0.6, 0.7]
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_query_no_client(self, mock_hass):
        """Test embedding fails when API key not configured."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {
            CONF_OPENAI_API_KEY: "",  # Empty API key
            CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OPENAI,
        }
        provider = VectorDBContextProvider(mock_hass, config)

        with pytest.raises(ContextInjectionError) as exc_info:
            await provider._embed_query("test")

        assert "OpenAI API key not configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_query_api_error(self, mock_hass):
        """Test error handling when API call fails."""
        if not CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
            pytest.skip("ChromaDB or OpenAI not available")

        config = {
            CONF_OPENAI_API_KEY: "sk-test",
            CONF_VECTOR_DB_EMBEDDING_PROVIDER: EMBEDDING_PROVIDER_OPENAI,
        }
        provider = VectorDBContextProvider(mock_hass, config)

        # Mock openai.Embedding.create to raise an error
        with patch("openai.Embedding.create", side_effect=Exception("API error")):
            with pytest.raises(ContextInjectionError) as exc_info:
                await provider._embed_query("test")

            assert "Embedding failed" in str(exc_info.value)

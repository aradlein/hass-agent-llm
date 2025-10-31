"""Vector DB (ChromaDB) context provider for Home Agent.

This module provides semantic search-based context injection using ChromaDB
vector database and embedding models.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import (
    CONF_OPENAI_API_KEY,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
    CONF_VECTOR_DB_TOP_K,
    DEFAULT_VECTOR_DB_COLLECTION,
    DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL,
    DEFAULT_VECTOR_DB_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER,
    DEFAULT_VECTOR_DB_HOST,
    DEFAULT_VECTOR_DB_PORT,
    DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_DB_TOP_K,
    EMBEDDING_PROVIDER_OLLAMA,
    EMBEDDING_PROVIDER_OPENAI,
)
from ..exceptions import ContextInjectionError
from .base import ContextProvider

# Conditional imports for ChromaDB
try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Conditional imports for OpenAI embeddings
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)


class VectorDBContextProvider(ContextProvider):
    """Context provider using ChromaDB for semantic entity search."""

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the Vector DB context provider."""
        super().__init__(hass, config)

        if not CHROMADB_AVAILABLE:
            raise ContextInjectionError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        self.host = config.get(CONF_VECTOR_DB_HOST, DEFAULT_VECTOR_DB_HOST)
        self.port = config.get(CONF_VECTOR_DB_PORT, DEFAULT_VECTOR_DB_PORT)
        self.collection_name = config.get(
            CONF_VECTOR_DB_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION
        )
        self.embedding_model = config.get(
            CONF_VECTOR_DB_EMBEDDING_MODEL, DEFAULT_VECTOR_DB_EMBEDDING_MODEL
        )
        self.embedding_provider = config.get(
            CONF_VECTOR_DB_EMBEDDING_PROVIDER, DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER
        )
        self.embedding_base_url = config.get(
            CONF_VECTOR_DB_EMBEDDING_BASE_URL, DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL
        )
        self.openai_api_key = config.get(CONF_OPENAI_API_KEY, "")
        self.top_k = config.get(CONF_VECTOR_DB_TOP_K, DEFAULT_VECTOR_DB_TOP_K)
        self.similarity_threshold = config.get(
            CONF_VECTOR_DB_SIMILARITY_THRESHOLD, DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD
        )

        self._client = None
        self._collection = None
        self._embedding_cache = {}

        _LOGGER.info(
            "Vector DB provider initialized (host=%s:%s, collection=%s)",
            self.host,
            self.port,
            self.collection_name,
        )

    async def get_context(self, user_input: str) -> str:
        """Get relevant context via semantic search."""
        try:
            await self._ensure_initialized()
            query_embedding = await self._embed_query(user_input)
            results = await self._query_vector_db(query_embedding, self.top_k)

            # ChromaDB uses L2 distance - smaller distances mean more similar
            # Filter to keep only results with distance below threshold
            # (lower threshold = more strict, only very similar results)
            filtered_results = [
                r for r in results if r.get("distance", float("inf")) <= self.similarity_threshold
            ]

            if not filtered_results:
                return "No relevant context found."

            entity_ids = [r["entity_id"] for r in filtered_results]
            entities = []

            for entity_id in entity_ids:
                try:
                    entity_state = self._get_entity_state(entity_id)
                    if entity_state:
                        # Add available services for this entity
                        entity_state["available_services"] = self._get_entity_services(
                            entity_id
                        )
                        entities.append(entity_state)
                except Exception as err:
                    _LOGGER.warning("Failed to get state for %s: %s", entity_id, err)

            if not entities:
                return "No entity states available."

            return json.dumps({"entities": entities, "count": len(entities)}, indent=2)

        except Exception as err:
            _LOGGER.error("Vector DB context retrieval failed: %s", err, exc_info=True)
            raise ContextInjectionError(f"Vector DB query failed: {err}") from err

    async def _ensure_initialized(self) -> None:
        """Ensure ChromaDB client and collection are initialized."""
        if self._client is None:
            try:
                # Create ChromaDB client in executor to avoid blocking the event loop
                # ChromaDB's HttpClient does SSL setup and file I/O during init
                self._client = await self.hass.async_add_executor_job(
                    chromadb.HttpClient,
                    self.host,
                    self.port,
                )
                _LOGGER.debug("ChromaDB client connected")
            except Exception as err:
                raise ContextInjectionError(
                    f"Failed to connect to ChromaDB: {err}"
                ) from err

        if self._collection is None:
            try:
                # Collection operations should also be in executor as they may do I/O
                from functools import partial

                get_collection = partial(
                    self._client.get_or_create_collection,
                    name=self.collection_name,
                    metadata={"description": "Home Assistant entity embeddings"},
                )
                self._collection = await self.hass.async_add_executor_job(get_collection)
                _LOGGER.debug("ChromaDB collection ready")
            except Exception as err:
                raise ContextInjectionError(
                    f"Failed to access collection: {err}"
                ) from err

    async def _embed_query(self, text: str) -> list[float]:
        """Embed text using configured embedding model."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            if self.embedding_provider == EMBEDDING_PROVIDER_OPENAI:
                embedding = await self._embed_with_openai(text)
            elif self.embedding_provider == EMBEDDING_PROVIDER_OLLAMA:
                embedding = await self._embed_with_ollama(text)
            else:
                raise ContextInjectionError(
                    f"Unknown embedding provider: {self.embedding_provider}"
                )

            self._embedding_cache[cache_key] = embedding
            return embedding

        except Exception as err:
            raise ContextInjectionError(f"Embedding failed: {err}") from err

    async def _embed_with_openai(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        if not OPENAI_AVAILABLE:
            raise ContextInjectionError(
                "OpenAI library not installed. Install with: pip install openai"
            )

        if not self.openai_api_key:
            raise ContextInjectionError(
                "OpenAI API key not configured. "
                "Please configure it in Vector DB settings."
            )

        # Set API key for OpenAI client
        openai.api_key = self.openai_api_key

        response = await self.hass.async_add_executor_job(
            lambda: openai.Embedding.create(
                model=self.embedding_model,
                input=text,
            )
        )
        return response["data"][0]["embedding"]

    async def _embed_with_ollama(self, text: str) -> list[float]:
        """Generate embedding using Ollama API."""
        import aiohttp

        url = f"{self.embedding_base_url.rstrip('/')}/api/embeddings"
        payload = {"model": self.embedding_model, "prompt": text}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ContextInjectionError(
                            f"Ollama API error {response.status}: {error_text}"
                        )
                    result = await response.json()
                    return result["embedding"]
        except aiohttp.ClientError as err:
            raise ContextInjectionError(
                f"Failed to connect to Ollama at {self.embedding_base_url}: {err}"
            ) from err

    async def _query_vector_db(
        self, embedding: list[float], top_k: int
    ) -> list[dict[str, Any]]:
        """Query ChromaDB with embedding vector."""
        if self._collection is None:
            raise ContextInjectionError("Collection not initialized")

        try:
            results = await self.hass.async_add_executor_job(
                lambda: self._collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                )
            )

            parsed_results = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]

                for i, entity_id in enumerate(ids):
                    parsed_results.append(
                        {
                            "entity_id": entity_id,
                            "distance": distances[i] if i < len(distances) else 0,
                        }
                    )

            return parsed_results

        except Exception as err:
            raise ContextInjectionError(f"Vector DB query failed: {err}") from err

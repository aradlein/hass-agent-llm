"""Vector DB (ChromaDB) context provider for Home Agent.

This module provides semantic search-based context injection using ChromaDB
vector database and embedding models.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Sequence, cast

from homeassistant.core import HomeAssistant

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

from ..const import (
    CONF_ADDITIONAL_COLLECTIONS,
    CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
    CONF_ADDITIONAL_TOP_K,
    CONF_OPENAI_API_KEY,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
    CONF_VECTOR_DB_TOP_K,
    DEFAULT_ADDITIONAL_COLLECTIONS,
    DEFAULT_ADDITIONAL_L2_DISTANCE_THRESHOLD,
    DEFAULT_ADDITIONAL_TOP_K,
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
from ..exceptions import ContextInjectionError, EmbeddingTimeoutError
from .base import ContextProvider
from .direct import DirectContextProvider

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
        self.collection_name = config.get(CONF_VECTOR_DB_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION)
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

        # Additional collections configuration
        self.additional_collections = config.get(
            CONF_ADDITIONAL_COLLECTIONS, DEFAULT_ADDITIONAL_COLLECTIONS
        )
        self.additional_top_k = config.get(CONF_ADDITIONAL_TOP_K, DEFAULT_ADDITIONAL_TOP_K)
        self.additional_threshold = config.get(
            CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD, DEFAULT_ADDITIONAL_L2_DISTANCE_THRESHOLD
        )

        self._client: ClientAPI | None = None
        self._collection: Collection | None = None
        self._embedding_cache: dict[str, list[float]] = {}
        self._fallback_provider: DirectContextProvider | None = None

        _LOGGER.info(
            "Vector DB provider initialized (host=%s:%s, collection=%s)",
            self.host,
            self.port,
            self.collection_name,
        )

    async def get_context(self, user_input: str) -> str:
        """Get relevant context via semantic search.

        Implements two-tier ranking system:
        1. Entity collection (priority) - always queried first
        2. Additional collections (supplementary) - merged and ranked together
        """
        try:
            await self._ensure_initialized()
            query_embedding = await self._embed_query(user_input)

            # Tier 1: Query entity collection (priority)
            entity_results = await self._query_vector_db(query_embedding, self.top_k)

            # ChromaDB uses L2 distance - smaller distances mean more similar
            # Filter to keep only results with distance below threshold
            filtered_entity_results = [
                r
                for r in entity_results
                if r.get("distance", float("inf")) <= self.similarity_threshold
            ]

            # Build entity context
            entity_context = ""
            if filtered_entity_results:
                entity_ids = [r["entity_id"] for r in filtered_entity_results]
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

                if entities:
                    entity_context = json.dumps(
                        {"entities": entities, "count": len(entities)}, indent=2
                    )

            # Tier 2: Query additional collections (supplementary)
            additional_context = ""
            if self.additional_collections and isinstance(self.additional_collections, list):
                additional_results = await self._query_additional_collections(query_embedding)

                if additional_results:
                    # Format additional results as JSON with metadata
                    additional_context = json.dumps(
                        {
                            "additional_context": additional_results,
                            "count": len(additional_results),
                        },
                        indent=2,
                    )

            # Combine contexts
            if entity_context and additional_context:
                return (
                    f"{entity_context}\n\n"
                    "### RELEVANT ADDITIONAL CONTEXT FOR ANSWERING QUESTIONS, NOT CONTROL ###\n"
                    f"{additional_context}"
                )
            elif entity_context:
                return entity_context
            elif additional_context:
                return (
                    "### RELEVANT ADDITIONAL CONTEXT FOR ANSWERING QUESTIONS, NOT CONTROL ###\n"
                    f"{additional_context}"
                )
            else:
                return "No relevant context found."

        except EmbeddingTimeoutError:
            # Timeout during embedding - fall back to direct mode
            return await self._fallback_to_direct(user_input)
        except ContextInjectionError:
            # Vector DB or embedding failure - fall back to direct mode
            return await self._fallback_to_direct(user_input)
        except Exception as err:
            # Unexpected error - fall back to direct mode
            _LOGGER.error("Vector DB context retrieval failed: %s", err, exc_info=True)
            return await self._fallback_to_direct(user_input)

    def _get_fallback_provider(self) -> DirectContextProvider:
        """Lazy-initialize fallback direct context provider."""
        if self._fallback_provider is None:
            self._fallback_provider = DirectContextProvider(self.hass, {"entities": []})
        return self._fallback_provider

    async def _fallback_to_direct(self, user_input: str) -> str:
        """Fall back to direct context when vector DB fails."""
        _LOGGER.warning("Falling back to direct context provider")
        fallback = self._get_fallback_provider()
        context = await fallback.get_context(user_input)
        return f"[Fallback mode - Vector DB unavailable]\n{context}" if context else ""

    async def _ensure_initialized(self) -> None:
        """Ensure ChromaDB client and collection are initialized."""
        if self._client is None:
            try:
                # Create ChromaDB client in executor to avoid blocking the event loop
                # ChromaDB's HttpClient does SSL setup and file I/O during init
                from functools import partial

                create_client = partial(
                    chromadb.HttpClient,
                    host=self.host,
                    port=self.port,
                )
                self._client = await self.hass.async_add_executor_job(create_client)
                _LOGGER.debug("ChromaDB client connected")
            except Exception as err:
                raise ContextInjectionError(f"Failed to connect to ChromaDB: {err}") from err

        if self._collection is None:
            try:
                # Collection operations should also be in executor as they may do I/O
                from functools import partial

                assert self._client is not None  # Type narrowing for mypy
                get_collection = partial(
                    self._client.get_or_create_collection,
                    name=self.collection_name,
                    metadata={"description": "Home Assistant entity embeddings"},
                )
                self._collection = await self.hass.async_add_executor_job(get_collection)
                _LOGGER.debug("ChromaDB collection ready")
            except Exception as err:
                raise ContextInjectionError(f"Failed to access collection: {err}") from err

    async def _embed_query(self, text: str) -> list[float]:
        """Embed text using configured embedding model."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embedding: list[float]
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
                "OpenAI API key not configured. " "Please configure it in Vector DB settings."
            )

        # Create OpenAI client with API key
        client = openai.OpenAI(api_key=self.openai_api_key)

        # Use the new API for embeddings
        response = await self.hass.async_add_executor_job(
            lambda: client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
        )
        embedding: list[float] = response.data[0].embedding
        return embedding

    async def _embed_with_ollama(self, text: str) -> list[float]:
        """Generate embedding using Ollama API."""
        import aiohttp
        import asyncio

        url = f"{self.embedding_base_url.rstrip('/')}/api/embeddings"
        payload = {"model": self.embedding_model, "prompt": text}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ContextInjectionError(
                            f"Ollama API error {response.status}: {error_text}"
                        )
                    result = await response.json()
                    embedding: list[float] = result["embedding"]
                    return embedding
        except asyncio.TimeoutError as err:
            raise EmbeddingTimeoutError(
                f"Ollama embedding timed out after 30s for model {self.embedding_model}"
            ) from err
        except aiohttp.ClientError as err:
            raise ContextInjectionError(
                f"Failed to connect to Ollama at {self.embedding_base_url}: {err}"
            ) from err

    async def _query_vector_db(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Query ChromaDB with embedding vector."""
        if self._collection is None:
            raise ContextInjectionError("Collection not initialized")

        try:
            # Type narrowing assertion for mypy
            assert self._collection is not None
            collection = self._collection

            # Cast to list[Sequence[float]] to satisfy chromadb's type signature
            query_embeddings = cast(list[Sequence[float]], [embedding])

            results = await self.hass.async_add_executor_job(
                lambda: collection.query(
                    query_embeddings=query_embeddings,
                    n_results=top_k,
                )
            )

            parsed_results: list[dict[str, Any]] = []
            if results and "ids" in results and results["ids"]:
                ids_list = results["ids"]
                if ids_list and len(ids_list) > 0:
                    ids = ids_list[0]
                    distances_list: Any = results.get("distances", [[]])
                    distances = (
                        distances_list[0] if distances_list and len(distances_list) > 0 else []
                    )

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

    async def _query_additional_collections(
        self, query_embedding: list[float]
    ) -> list[dict[str, Any]]:
        """Query additional collections and return merged, ranked results.

        Args:
            query_embedding: The embedding vector to query with

        Returns:
            List of merged results from all additional collections, sorted by distance
        """
        if not self.additional_collections:
            return []

        if self._client is None:
            _LOGGER.warning("ChromaDB client not initialized, cannot query additional collections")
            return []

        all_results = []

        # Query each additional collection
        for collection_name in self.additional_collections:
            try:
                # Try to get the collection
                collection = await self.hass.async_add_executor_job(
                    self._client.get_collection,
                    collection_name,
                )

                # Query the collection with extra results for merging
                results = await self.hass.async_add_executor_job(
                    lambda col=collection: col.query(
                        query_embeddings=[query_embedding],
                        n_results=self.additional_top_k * len(self.additional_collections),
                        include=["documents", "metadatas", "distances"],
                    )
                )

                # Parse and add results with collection name
                if results and "ids" in results and results["ids"]:
                    ids = results["ids"][0]
                    distances = results.get("distances", [[]])[0]
                    documents = results.get("documents", [[]])[0]
                    metadatas = results.get("metadatas", [[]])[0]

                    for i in range(len(ids)):
                        result = {
                            "id": ids[i],
                            "distance": distances[i] if i < len(distances) else float("inf"),
                            "document": documents[i] if i < len(documents) else "",
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "collection": collection_name,
                        }
                        all_results.append(result)

            except Exception as err:
                # Log warning but continue with other collections
                _LOGGER.warning(
                    "Collection '%s' not found or inaccessible, skipping. Error: %s",
                    collection_name,
                    str(err),
                )
                continue

        if not all_results:
            return []

        # Sort by distance (ascending - lower is better)
        all_results.sort(key=lambda x: x.get("distance", float("inf")))

        # Filter by threshold
        filtered_results = [
            r for r in all_results if r.get("distance", float("inf")) <= self.additional_threshold
        ]

        # Take top K from merged pool
        top_results = filtered_results[: self.additional_top_k]

        return top_results

"""Memory manager for long-term memory system.

This module handles persistent storage and retrieval of memories extracted
from conversations using dual storage (Home Assistant Store + ChromaDB).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    CONF_MEMORY_COLLECTION_NAME,
    CONF_MEMORY_DEDUP_THRESHOLD,
    CONF_MEMORY_IMPORTANCE_DECAY,
    CONF_MEMORY_MAX_MEMORIES,
    CONF_MEMORY_MIN_IMPORTANCE,
    DEFAULT_MEMORY_COLLECTION_NAME,
    DEFAULT_MEMORY_DEDUP_THRESHOLD,
    DEFAULT_MEMORY_IMPORTANCE_DECAY,
    DEFAULT_MEMORY_MAX_MEMORIES,
    DEFAULT_MEMORY_MIN_IMPORTANCE,
    MEMORY_STORAGE_KEY,
    MEMORY_STORAGE_VERSION,
)
from .exceptions import ContextInjectionError

# Conditional import for ChromaDB
try:
    import chromadb  # noqa: F401

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

# Memory type constants
MEMORY_TYPE_FACT = "fact"
MEMORY_TYPE_PREFERENCE = "preference"
MEMORY_TYPE_CONTEXT = "context"
MEMORY_TYPE_EVENT = "event"

# Access boost amount for importance scoring
IMPORTANCE_ACCESS_BOOST = 0.05


class MemoryManager:
    """Manages long-term memories with dual storage."""

    def __init__(
        self,
        hass: HomeAssistant,
        vector_db_manager: Any,
        config: dict[str, Any],
    ) -> None:
        """Initialize the Memory Manager.

        Args:
            hass: Home Assistant instance
            vector_db_manager: VectorDBManager instance for ChromaDB operations
            config: Configuration dictionary
        """
        self.hass = hass
        self.vector_db_manager = vector_db_manager
        self.config = config

        # Configuration
        self.max_memories = config.get(
            CONF_MEMORY_MAX_MEMORIES, DEFAULT_MEMORY_MAX_MEMORIES
        )
        self.min_importance = config.get(
            CONF_MEMORY_MIN_IMPORTANCE, DEFAULT_MEMORY_MIN_IMPORTANCE
        )
        self.collection_name = config.get(
            CONF_MEMORY_COLLECTION_NAME, DEFAULT_MEMORY_COLLECTION_NAME
        )
        self.importance_decay = config.get(
            CONF_MEMORY_IMPORTANCE_DECAY, DEFAULT_MEMORY_IMPORTANCE_DECAY
        )
        self.dedup_threshold = config.get(
            CONF_MEMORY_DEDUP_THRESHOLD, DEFAULT_MEMORY_DEDUP_THRESHOLD
        )

        # State
        self._store = Store(hass, MEMORY_STORAGE_VERSION, MEMORY_STORAGE_KEY)
        self._memories: dict[str, dict[str, Any]] = {}
        self._collection = None
        self._chromadb_available = False
        self._save_lock = asyncio.Lock()
        self._save_task = None
        self._pending_save = False

        _LOGGER.info(
            "Memory Manager initialized (max=%d, collection=%s)",
            self.max_memories,
            self.collection_name,
        )

    async def async_initialize(self) -> None:
        """Initialize the memory manager.

        Loads existing memories from storage and sets up ChromaDB collection.
        """
        try:
            # Load memories from HA Store
            stored_data = await self._store.async_load()
            if stored_data and "memories" in stored_data:
                self._memories = stored_data["memories"]
                _LOGGER.info("Loaded %d memories from storage", len(self._memories))
            else:
                self._memories = {}
                _LOGGER.info("No existing memories found, starting fresh")

            # Initialize ChromaDB collection if available
            if CHROMADB_AVAILABLE and self.vector_db_manager:
                try:
                    await self._ensure_chromadb_initialized()
                    self._chromadb_available = True
                    _LOGGER.info("ChromaDB collection initialized for memories")

                    # Sync existing memories to ChromaDB if needed
                    await self._sync_to_chromadb()
                except Exception as err:
                    _LOGGER.warning(
                        "ChromaDB not available for memories, using store-only mode: %s",
                        err,
                    )
                    self._chromadb_available = False
            else:
                _LOGGER.info("ChromaDB not available, using store-only mode")
                self._chromadb_available = False

            _LOGGER.info("Memory Manager initialization complete")

        except Exception as err:
            _LOGGER.error("Failed to initialize Memory Manager: %s", err, exc_info=True)
            raise

    async def async_shutdown(self) -> None:
        """Shut down the memory manager.

        Ensures all pending saves are completed.
        """
        # Wait for any pending save
        if self._save_task:
            await self._save_task

        # Final save
        await self._save_to_store()

        _LOGGER.info("Memory Manager shut down")

    async def add_memory(
        self,
        content: str,
        memory_type: str,
        conversation_id: str | None = None,
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> str:
        """Add a new memory to storage.

        Args:
            content: The actual memory text
            memory_type: Type of memory (fact, preference, context, event)
            conversation_id: Origin conversation ID
            importance: Importance score (0.0 - 1.0)
            metadata: Additional metadata

        Returns:
            Memory ID (UUID)
        """
        try:
            # Validate inputs
            if not content or not content.strip():
                raise ValueError("Memory content cannot be empty")

            if memory_type not in [
                MEMORY_TYPE_FACT,
                MEMORY_TYPE_PREFERENCE,
                MEMORY_TYPE_CONTEXT,
                MEMORY_TYPE_EVENT,
            ]:
                raise ValueError(f"Invalid memory type: {memory_type}")

            if not 0.0 <= importance <= 1.0:
                raise ValueError("Importance must be between 0.0 and 1.0")

            # Check for duplicates
            if self._chromadb_available:
                duplicate_id = await self._find_duplicate(content)
                if duplicate_id:
                    _LOGGER.info(
                        "Duplicate memory found, updating existing memory: %s",
                        duplicate_id,
                    )
                    # Update last_accessed and optionally merge metadata
                    existing = self._memories[duplicate_id]
                    existing["last_accessed"] = time.time()
                    if metadata:
                        existing["metadata"].update(metadata)
                    await self._schedule_save()
                    await self._update_chromadb_memory(duplicate_id)
                    return duplicate_id

            # Create new memory
            memory_id = str(uuid.uuid4())
            current_time = time.time()

            memory = {
                "id": memory_id,
                "type": memory_type,
                "content": content,
                "source_conversation_id": conversation_id,
                "extracted_at": current_time,
                "last_accessed": current_time,
                "importance": importance,
                "metadata": metadata or {},
            }

            # Ensure metadata has required fields
            if "entities_involved" not in memory["metadata"]:
                memory["metadata"]["entities_involved"] = []
            if "topics" not in memory["metadata"]:
                memory["metadata"]["topics"] = []
            if "extraction_method" not in memory["metadata"]:
                memory["metadata"]["extraction_method"] = "manual"

            # Store in memory
            self._memories[memory_id] = memory

            # Store in ChromaDB
            if self._chromadb_available:
                await self._add_to_chromadb(memory)

            # Schedule save to HA Store
            await self._schedule_save()

            # Check if we need to prune old memories
            if len(self._memories) > self.max_memories:
                await self._prune_memories()

            _LOGGER.info("Added memory: %s (type=%s, importance=%.2f)", memory_id, memory_type, importance)

            return memory_id

        except Exception as err:
            _LOGGER.error("Failed to add memory: %s", err, exc_info=True)
            raise

    async def get_memory(self, memory_id: str) -> dict | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory dictionary or None if not found
        """
        memory = self._memories.get(memory_id)

        if memory:
            # Update last accessed time and apply importance boost
            memory["last_accessed"] = time.time()
            memory["importance"] = min(
                1.0, memory["importance"] + IMPORTANCE_ACCESS_BOOST
            )
            await self._schedule_save()

            if self._chromadb_available:
                await self._update_chromadb_memory(memory_id)

        return memory

    async def search_memories(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
        memory_types: list[str] | None = None,
    ) -> list[dict]:
        """Search memories using semantic similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_importance: Minimum importance threshold
            memory_types: Optional list of memory types to filter by

        Returns:
            List of memories sorted by relevance
        """
        if not self._chromadb_available:
            _LOGGER.warning("ChromaDB not available, falling back to keyword search")
            return await self._fallback_search(query, top_k, min_importance, memory_types)

        try:
            # Generate embedding for query
            embedding = await self.vector_db_manager._embed_text(query)

            # Query ChromaDB
            results = await self.hass.async_add_executor_job(
                lambda: self._collection.query(
                    query_embeddings=[embedding],
                    n_results=min(top_k * 2, len(self._memories)),  # Get more to filter
                )
            )

            if not results or "ids" not in results or not results["ids"][0]:
                return []

            # Filter and sort results
            memories = []
            for memory_id in results["ids"][0]:
                memory = self._memories.get(memory_id)
                if not memory:
                    continue

                # Apply filters
                if memory["importance"] < min_importance:
                    continue

                if memory_types and memory["type"] not in memory_types:
                    continue

                # Update access tracking
                memory["last_accessed"] = time.time()
                memory["importance"] = min(
                    1.0, memory["importance"] + IMPORTANCE_ACCESS_BOOST
                )

                memories.append(memory)

            # Limit to top_k
            memories = memories[:top_k]

            if memories:
                await self._schedule_save()
                if self._chromadb_available:
                    for memory in memories:
                        await self._update_chromadb_memory(memory["id"])

            return memories

        except Exception as err:
            _LOGGER.error("Memory search failed: %s", err, exc_info=True)
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        if memory_id not in self._memories:
            return False

        try:
            # Remove from memory
            del self._memories[memory_id]

            # Remove from ChromaDB
            if self._chromadb_available:
                await self.hass.async_add_executor_job(
                    lambda: self._collection.delete(ids=[memory_id])
                )

            # Save to store
            await self._schedule_save()

            _LOGGER.info("Deleted memory: %s", memory_id)
            return True

        except Exception as err:
            _LOGGER.error("Failed to delete memory %s: %s", memory_id, err)
            return False

    async def list_all_memories(
        self,
        limit: int | None = None,
        memory_type: str | None = None,
    ) -> list[dict]:
        """List all memories with optional filtering.

        Args:
            limit: Maximum number of memories to return
            memory_type: Filter by memory type

        Returns:
            List of memories
        """
        memories = list(self._memories.values())

        # Filter by type
        if memory_type:
            memories = [m for m in memories if m["type"] == memory_type]

        # Sort by importance (descending) then last_accessed (descending)
        memories.sort(
            key=lambda m: (m["importance"], m["last_accessed"]),
            reverse=True,
        )

        # Apply limit
        if limit:
            memories = memories[:limit]

        return memories

    async def clear_all_memories(self) -> int:
        """Clear all memories.

        Returns:
            Count of deleted memories
        """
        count = len(self._memories)

        try:
            # Clear in-memory storage
            self._memories.clear()

            # Clear ChromaDB
            if self._chromadb_available:
                await self.hass.async_add_executor_job(
                    lambda: self._collection.delete(where={})
                )

            # Save to store
            await self._save_to_store()

            _LOGGER.info("Cleared all memories (count=%d)", count)
            return count

        except Exception as err:
            _LOGGER.error("Failed to clear memories: %s", err)
            raise

    async def apply_importance_decay(self) -> int:
        """Apply importance decay to all memories.

        Returns:
            Count of memories that fell below minimum importance and were removed
        """
        if self.importance_decay == 0.0:
            return 0

        removed_count = 0
        to_remove = []

        for memory_id, memory in self._memories.items():
            # Apply decay
            memory["importance"] *= 1.0 - self.importance_decay

            # Mark for removal if below threshold
            if memory["importance"] < self.min_importance:
                to_remove.append(memory_id)

        # Remove low-importance memories
        for memory_id in to_remove:
            await self.delete_memory(memory_id)
            removed_count += 1

        if removed_count > 0:
            _LOGGER.info(
                "Applied importance decay, removed %d low-importance memories",
                removed_count,
            )
            await self._schedule_save()

        return removed_count

    # Private helper methods

    async def _ensure_chromadb_initialized(self) -> None:
        """Ensure ChromaDB collection is initialized."""
        if self._collection is None:
            if not self.vector_db_manager or not self.vector_db_manager._client:
                raise ContextInjectionError("VectorDBManager client not available")

            from functools import partial

            get_collection = partial(
                self.vector_db_manager._client.get_or_create_collection,
                name=self.collection_name,
                metadata={"description": "Home Agent long-term memories"},
            )
            self._collection = await self.hass.async_add_executor_job(get_collection)
            _LOGGER.debug("Memory ChromaDB collection ready")

    async def _find_duplicate(self, content: str) -> str | None:
        """Find duplicate memory using semantic similarity.

        Args:
            content: Memory content to check

        Returns:
            Memory ID of duplicate or None
        """
        if not self._chromadb_available or not self._memories:
            return None

        try:
            # Generate embedding
            embedding = await self.vector_db_manager._embed_text(content)

            # Query ChromaDB for similar memories
            results = await self.hass.async_add_executor_job(
                lambda: self._collection.query(
                    query_embeddings=[embedding],
                    n_results=1,
                )
            )

            if not results or "ids" not in results or not results["ids"][0]:
                return None

            # Check if similarity is above threshold
            # ChromaDB returns distances, not similarities
            # For L2 distance, lower is more similar
            if results.get("distances") and results["distances"][0]:
                distance = results["distances"][0][0]
                # Convert distance to similarity (approximate)
                # This is a simple heuristic - may need adjustment
                if distance < (1.0 - self.dedup_threshold):
                    return results["ids"][0][0]

            return None

        except Exception as err:
            _LOGGER.warning("Duplicate detection failed: %s", err)
            return None

    async def _add_to_chromadb(self, memory: dict[str, Any]) -> None:
        """Add a memory to ChromaDB.

        Args:
            memory: Memory dictionary
        """
        try:
            await self._ensure_chromadb_initialized()

            # Generate embedding
            embedding = await self.vector_db_manager._embed_text(memory["content"])

            # Prepare metadata (ChromaDB requires simple types)
            metadata = {
                "memory_id": memory["id"],
                "type": memory["type"],
                "importance": memory["importance"],
                "extracted_at": memory["extracted_at"],
                "last_accessed": memory["last_accessed"],
            }

            # Add conversation ID if present
            if memory.get("source_conversation_id"):
                metadata["conversation_id"] = memory["source_conversation_id"]

            # Add to collection
            await self.hass.async_add_executor_job(
                lambda: self._collection.upsert(
                    ids=[memory["id"]],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[memory["content"]],
                )
            )

            _LOGGER.debug("Added memory to ChromaDB: %s", memory["id"])

        except Exception as err:
            _LOGGER.error(
                "Failed to add memory to ChromaDB: %s",
                err,
                exc_info=True,
            )
            # Don't raise - allow graceful degradation to store-only mode

    async def _update_chromadb_memory(self, memory_id: str) -> None:
        """Update a memory in ChromaDB.

        Args:
            memory_id: Memory ID to update
        """
        if not self._chromadb_available:
            return

        memory = self._memories.get(memory_id)
        if not memory:
            return

        # Re-add to ChromaDB (upsert will update)
        await self._add_to_chromadb(memory)

    async def _sync_to_chromadb(self) -> None:
        """Sync all memories to ChromaDB."""
        if not self._chromadb_available or not self._memories:
            return

        _LOGGER.info("Syncing %d memories to ChromaDB", len(self._memories))

        for memory in self._memories.values():
            try:
                await self._add_to_chromadb(memory)
            except Exception as err:
                _LOGGER.warning(
                    "Failed to sync memory %s to ChromaDB: %s",
                    memory["id"],
                    err,
                )

    async def _schedule_save(self) -> None:
        """Schedule a debounced save to HA Store."""
        self._pending_save = True

        # Cancel existing save task if any
        if self._save_task and not self._save_task.done():
            return  # Already scheduled

        # Schedule save after short delay (debounce)
        self._save_task = asyncio.create_task(self._debounced_save())

    async def _debounced_save(self) -> None:
        """Perform debounced save to reduce I/O."""
        await asyncio.sleep(1.0)  # Wait 1 second for more changes

        if self._pending_save:
            await self._save_to_store()
            self._pending_save = False

    async def _save_to_store(self) -> None:
        """Save memories to HA Store."""
        async with self._save_lock:
            try:
                data = {
                    "version": MEMORY_STORAGE_VERSION,
                    "memories": self._memories,
                }
                await self._store.async_save(data)
                _LOGGER.debug("Saved %d memories to store", len(self._memories))
            except Exception as err:
                _LOGGER.error("Failed to save memories to store: %s", err)

    async def _prune_memories(self) -> None:
        """Prune least important memories when limit is exceeded."""
        if len(self._memories) <= self.max_memories:
            return

        # Sort by importance (ascending) and last_accessed (ascending)
        sorted_memories = sorted(
            self._memories.items(),
            key=lambda item: (item[1]["importance"], item[1]["last_accessed"]),
        )

        # Remove oldest/least important memories
        to_remove = len(self._memories) - self.max_memories
        for memory_id, _ in sorted_memories[:to_remove]:
            await self.delete_memory(memory_id)

        _LOGGER.info("Pruned %d memories to stay under limit", to_remove)

    async def _fallback_search(
        self,
        query: str,
        top_k: int,
        min_importance: float,
        memory_types: list[str] | None,
    ) -> list[dict]:
        """Fallback keyword search when ChromaDB is unavailable.

        Args:
            query: Search query
            top_k: Number of results
            min_importance: Minimum importance
            memory_types: Memory types to filter

        Returns:
            List of matching memories
        """
        query_lower = query.lower()
        matches = []

        for memory in self._memories.values():
            # Filter by importance
            if memory["importance"] < min_importance:
                continue

            # Filter by type
            if memory_types and memory["type"] not in memory_types:
                continue

            # Simple keyword matching
            if query_lower in memory["content"].lower():
                matches.append(memory)

        # Sort by importance
        matches.sort(key=lambda m: m["importance"], reverse=True)

        return matches[:top_k]

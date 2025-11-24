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
    CONF_CONTEXT_MODE,
    CONF_MEMORY_CLEANUP_INTERVAL,
    CONF_MEMORY_COLLECTION_NAME,
    CONF_MEMORY_DEDUP_THRESHOLD,
    CONF_MEMORY_EVENT_TTL,
    CONF_MEMORY_FACT_TTL,
    CONF_MEMORY_IMPORTANCE_DECAY,
    CONF_MEMORY_MAX_MEMORIES,
    CONF_MEMORY_MIN_IMPORTANCE,
    CONF_MEMORY_PREFERENCE_TTL,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_MEMORY_CLEANUP_INTERVAL,
    DEFAULT_MEMORY_COLLECTION_NAME,
    DEFAULT_MEMORY_DEDUP_THRESHOLD,
    DEFAULT_MEMORY_EVENT_TTL,
    DEFAULT_MEMORY_FACT_TTL,
    DEFAULT_MEMORY_IMPORTANCE_DECAY,
    DEFAULT_MEMORY_MAX_MEMORIES,
    DEFAULT_MEMORY_MIN_IMPORTANCE,
    DEFAULT_MEMORY_PREFERENCE_TTL,
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
        self.max_memories = config.get(CONF_MEMORY_MAX_MEMORIES, DEFAULT_MEMORY_MAX_MEMORIES)
        self.min_importance = config.get(CONF_MEMORY_MIN_IMPORTANCE, DEFAULT_MEMORY_MIN_IMPORTANCE)
        self.collection_name = config.get(
            CONF_MEMORY_COLLECTION_NAME, DEFAULT_MEMORY_COLLECTION_NAME
        )
        self.importance_decay = config.get(
            CONF_MEMORY_IMPORTANCE_DECAY, DEFAULT_MEMORY_IMPORTANCE_DECAY
        )
        self.dedup_threshold = config.get(
            CONF_MEMORY_DEDUP_THRESHOLD, DEFAULT_MEMORY_DEDUP_THRESHOLD
        )
        self.event_ttl = config.get(CONF_MEMORY_EVENT_TTL, DEFAULT_MEMORY_EVENT_TTL)
        self.fact_ttl = config.get(CONF_MEMORY_FACT_TTL, DEFAULT_MEMORY_FACT_TTL)
        self.preference_ttl = config.get(CONF_MEMORY_PREFERENCE_TTL, DEFAULT_MEMORY_PREFERENCE_TTL)
        self.cleanup_interval = config.get(
            CONF_MEMORY_CLEANUP_INTERVAL, DEFAULT_MEMORY_CLEANUP_INTERVAL
        )

        # State
        self._store: Store[dict[str, Any]] = Store(hass, MEMORY_STORAGE_VERSION, MEMORY_STORAGE_KEY)
        self._memories: dict[str, dict[str, Any]] = {}
        self._collection: Any = None
        self._chromadb_available = False
        self._save_lock = asyncio.Lock()
        self._save_task: asyncio.Task[None] | None = None
        self._pending_save = False
        self._cleanup_task: asyncio.Task[None] | None = None

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

            # Start periodic cleanup task
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            _LOGGER.info("Started periodic cleanup task (interval=%ds)", self.cleanup_interval)

            _LOGGER.info("Memory Manager initialization complete")

        except Exception as err:
            _LOGGER.error("Failed to initialize Memory Manager: %s", err, exc_info=True)
            raise

    async def async_shutdown(self) -> None:
        """Shut down the memory manager.

        Ensures all pending saves are completed.
        """
        # Cancel periodic cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

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
                        "Duplicate memory found, merging with existing memory: %s",
                        duplicate_id,
                    )
                    # Merge with existing memory
                    existing = self._memories[duplicate_id]
                    current_time = time.time()
                    existing["last_accessed"] = current_time

                    # If new content is more specific (longer), update it
                    if len(content) > len(existing["content"]):
                        _LOGGER.info(
                            "New content is more specific, updating: %s -> %s",
                            existing["content"][:50],
                            content[:50],
                        )
                        existing["content"] = content

                    # Boost importance score (reinforcement learning)
                    old_importance = existing["importance"]
                    existing["importance"] = min(1.0, old_importance + (importance * 0.1))
                    _LOGGER.debug(
                        "Boosted importance: %.2f -> %.2f",
                        old_importance,
                        existing["importance"],
                    )

                    # Merge metadata
                    if metadata:
                        existing["metadata"].update(metadata)

                    # Recalculate expiration if memory type changed
                    if existing["type"] != memory_type:
                        existing["type"] = memory_type
                        existing["expires_at"] = self._calculate_expires_at(
                            memory_type, current_time
                        )

                    await self._schedule_save()
                    await self._update_chromadb_memory(duplicate_id)
                    return duplicate_id

            # Create new memory
            memory_id = str(uuid.uuid4())
            current_time = time.time()

            # Calculate expiration time based on memory type
            expires_at = self._calculate_expires_at(memory_type, current_time)

            # Detect if this is transient state
            is_transient = self._is_transient_state(content)

            # Log warning if transient state detected
            if is_transient:
                _LOGGER.warning(
                    "Transient state detected in memory content: %s. "
                    "Consider using event type instead of fact.",
                    content[:50],
                )

            # Ensure metadata has required fields
            memory_metadata: dict[str, Any] = metadata or {}
            if "entities_involved" not in memory_metadata:
                memory_metadata["entities_involved"] = []
            if "topics" not in memory_metadata:
                memory_metadata["topics"] = []
            if "extraction_method" not in memory_metadata:
                memory_metadata["extraction_method"] = "manual"

            memory = {
                "id": memory_id,
                "type": memory_type,
                "content": content,
                "source_conversation_id": conversation_id,
                "extracted_at": current_time,
                "last_accessed": current_time,
                "importance": importance,
                "metadata": memory_metadata,
                "expires_at": expires_at,
                "is_transient": is_transient,
            }

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

            _LOGGER.info(
                "Added memory: %s (type=%s, importance=%.2f)", memory_id, memory_type, importance
            )

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
            memory["importance"] = min(1.0, memory["importance"] + IMPORTANCE_ACCESS_BOOST)
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
        # Check context mode to determine search strategy
        context_mode = self.config.get(CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE)

        if context_mode == CONTEXT_MODE_VECTOR_DB:
            # Vector DB mode: Query ChromaDB directly
            return await self._search_memories_chromadb(
                query, top_k, min_importance, memory_types
            )
        else:
            # Direct mode: Use local memory storage
            return await self._search_memories_local(
                query, top_k, min_importance, memory_types
            )

    async def _search_memories_chromadb(
        self,
        query: str,
        top_k: int,
        min_importance: float,
        memory_types: list[str] | None,
    ) -> list[dict]:
        """Search memories directly from ChromaDB.

        Used when context mode is set to vector_db.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_importance: Minimum importance threshold
            memory_types: Optional list of memory types to filter by

        Returns:
            List of memories sorted by relevance
        """
        if not self._chromadb_available:
            _LOGGER.warning("ChromaDB not available for memory search")
            return []

        if self._collection is None:
            _LOGGER.warning("ChromaDB collection not initialized")
            return []

        try:
            # Generate embedding for query
            embedding = await self.vector_db_manager._embed_text(query)

            # Query ChromaDB with include for documents and metadata
            results = await self.hass.async_add_executor_job(
                lambda: self._collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k * 2,  # Get more to filter
                    include=["documents", "metadatas"],
                )
            )

            if not results or "ids" not in results or not results["ids"][0]:
                return []

            # Build memories from ChromaDB results
            memories = []
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, memory_id in enumerate(ids):
                # Get metadata for this memory
                metadata = metadatas[i] if i < len(metadatas) else {}
                document = documents[i] if i < len(documents) else ""

                # Extract importance from metadata
                importance = float(metadata.get("importance", 0.5))
                if importance < min_importance:
                    continue

                # Extract type from metadata
                memory_type = metadata.get("type", MEMORY_TYPE_FACT)
                if memory_types and memory_type not in memory_types:
                    continue

                # Build memory object from ChromaDB data
                memory = {
                    "id": memory_id,
                    "content": document,
                    "type": memory_type,
                    "importance": importance,
                    "last_accessed": float(metadata.get("last_accessed", time.time())),
                    "extracted_at": float(metadata.get("extracted_at", time.time())),
                }

                # Also update local cache if available
                if memory_id in self._memories:
                    self._memories[memory_id]["last_accessed"] = time.time()
                    self._memories[memory_id]["importance"] = min(
                        1.0, importance + IMPORTANCE_ACCESS_BOOST
                    )

                memories.append(memory)

            # Limit to top_k
            memories = memories[:top_k]

            _LOGGER.debug(
                "ChromaDB memory search returned %d results for query: %s",
                len(memories),
                query[:50],
            )

            return memories

        except Exception as err:
            _LOGGER.error("ChromaDB memory search failed: %s", err, exc_info=True)
            return []

    async def _search_memories_local(
        self,
        query: str,
        top_k: int,
        min_importance: float,
        memory_types: list[str] | None,
    ) -> list[dict]:
        """Search memories from local storage.

        Used when context mode is set to direct.

        Args:
            query: Search query text
            top_k: Number of results to return
            min_importance: Minimum importance threshold
            memory_types: Optional list of memory types to filter by

        Returns:
            List of memories sorted by relevance
        """
        # Early return if no memories exist in local storage
        if not self._memories:
            _LOGGER.debug("No memories in local storage to search")
            return []

        if not self._chromadb_available:
            _LOGGER.warning("ChromaDB not available, falling back to keyword search")
            return await self._fallback_search(query, top_k, min_importance, memory_types)

        try:
            # Generate embedding for query
            embedding = await self.vector_db_manager._embed_text(query)

            # Query ChromaDB - ensure n_results is at least 1
            n_results = min(top_k * 2, len(self._memories))

            # Query ChromaDB
            if self._collection is None:
                _LOGGER.warning("ChromaDB collection not initialized")
                return []

            results = await self.hass.async_add_executor_job(
                lambda: self._collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results,  # Get more to filter
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
                memory["importance"] = min(1.0, memory["importance"] + IMPORTANCE_ACCESS_BOOST)

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
            _LOGGER.error("Local memory search failed: %s", err, exc_info=True)
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
            if self._chromadb_available and self._collection is not None:
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
            if self._chromadb_available and self._collection is not None:
                await self.hass.async_add_executor_job(lambda: self._collection.delete(where={}))

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
            return None

    def _calculate_expires_at(self, memory_type: str, current_time: float) -> float | None:
        """Calculate expiration timestamp for a memory based on its type.

        Args:
            memory_type: Type of memory (fact, event, preference, context)
            current_time: Current timestamp

        Returns:
            Expiration timestamp or None if no expiration
        """
        ttl_map = {
            MEMORY_TYPE_EVENT: self.event_ttl,
            MEMORY_TYPE_FACT: self.fact_ttl,
            MEMORY_TYPE_PREFERENCE: self.preference_ttl,
            MEMORY_TYPE_CONTEXT: self.event_ttl,  # Context expires like events
        }

        ttl = ttl_map.get(memory_type)
        if ttl is None:
            return None

        return float(current_time + ttl)

    def _is_transient_state(self, content: str) -> bool:
        """Detect if content represents a transient state or low-quality content.

        This includes both device state patterns and conversational meta-information
        that should not be stored as memories.

        Args:
            content: Memory content to check

        Returns:
            True if content appears to be transient state or low-quality
        """
        # Patterns that indicate transient device state
        transient_patterns = [
            "is on",
            "is off",
            "are on",
            "are off",
            "is currently",
            "are currently",
            "is now",
            "are now",
            "was on",
            "was off",
            "were on",
            "were off",
            "temperature is",
            "humidity is",
            "status is",
            "state is",
            "is open",
            "is closed",
            "is locked",
            "is unlocked",
            "lights are",
            "light is",
            "is playing",
            "is paused",
            "is stopped",
            "is running",
        ]

        # Patterns that indicate low-value conversational meta-information
        low_value_patterns = [
            "conversation occurred",
            "conversation took place",
            "we discussed",
            "we talked about",
            "user asked about",
            "i mentioned",
            "there is no",
            "there are no",
            "no specific",
            "does not have",
            "doesn't have",
            "do not have",
            "don't have",
            "at the time",
            "during the conversation",
            "in the conversation",
            "during our conversation",
            "we were discussing",
        ]

        content_lower = content.lower()

        # Check low-value patterns first (always reject these)
        if any(pattern in content_lower for pattern in low_value_patterns):
            return True

        # Check transient state patterns with context awareness
        # Avoid false positives like "birthday is on"
        for pattern in transient_patterns:
            if pattern in content_lower:
                # Check if this is a false positive (e.g., "is on" in "birthday is on")
                # by looking at what comes before the pattern
                idx = content_lower.find(pattern)
                if idx > 0:
                    # Get the word before the pattern
                    before = content_lower[:idx].strip().split()
                    if before:
                        last_word = before[-1]
                        # "birthday is on", "event is on", etc. are temporal, not state
                        if last_word in ["birthday", "event", "date", "day", "anniversary"]:
                            continue
                # This appears to be a genuine transient state pattern
                return True

        return False

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

            # Query ChromaDB for similar memories (check top 5 to catch more duplicates)
            if self._collection is None:
                return None

            results = await self.hass.async_add_executor_job(
                lambda: self._collection.query(
                    query_embeddings=[embedding],
                    n_results=min(5, len(self._memories)),
                )
            )

            if not results or "ids" not in results or not results["ids"][0]:
                return None

            # Check if similarity is above threshold
            # ChromaDB uses L2 (squared Euclidean) distance by default
            # For normalized embeddings, L2 distance ranges from 0 (identical) to 2 (opposite)
            # Convert to similarity: similarity = 1 - (distance / 2)
            # We want similarity >= dedup_threshold
            # So: distance <= 2 * (1 - dedup_threshold)
            if results.get("distances") and results["distances"][0]:
                for i, distance in enumerate(results["distances"][0]):
                    # Calculate similarity score
                    similarity = 1.0 - (distance / 2.0)

                    _LOGGER.debug(
                        "Checking duplicate candidate: dist=%.3f, sim=%.3f, thresh=%.3f",
                        distance,
                        similarity,
                        self.dedup_threshold,
                    )

                    if similarity >= self.dedup_threshold:
                        duplicate_id: str = results["ids"][0][i]
                        _LOGGER.info(
                            "Duplicate found with similarity=%.3f (threshold=%.3f)",
                            similarity,
                            self.dedup_threshold,
                        )
                        return duplicate_id

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
            if self._collection is None:
                _LOGGER.warning("ChromaDB collection not initialized, skipping upsert")
                return

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

    async def _cleanup_expired_memories(self) -> int:
        """Remove expired memories based on TTL.

        Returns:
            Number of memories removed
        """
        current_time = time.time()
        expired_ids = []

        # Find expired memories
        for memory_id, memory in self._memories.items():
            expires_at = memory.get("expires_at")
            if expires_at is not None and current_time >= expires_at:
                expired_ids.append(memory_id)

        # Delete expired memories
        for memory_id in expired_ids:
            await self.delete_memory(memory_id)

        if expired_ids:
            _LOGGER.info(
                "Cleaned up %d expired memories (types: %s)",
                len(expired_ids),
                ", ".join(
                    set(self._memories.get(mid, {}).get("type", "unknown") for mid in expired_ids)
                ),
            )

        return len(expired_ids)

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task that runs at configured interval."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_memories()
            except asyncio.CancelledError:
                _LOGGER.info("Periodic cleanup task cancelled")
                break
            except Exception as err:
                _LOGGER.error("Error in periodic cleanup: %s", err)

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

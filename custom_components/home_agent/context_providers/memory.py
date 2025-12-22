"""Memory context provider for Home Agent.

This module provides the MemoryContextProvider class that retrieves
relevant memories based on user input using semantic search.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from ..const import (
    CONF_MEMORY_CONTEXT_TOP_K,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MIN_IMPORTANCE,
    DEFAULT_MEMORY_CONTEXT_TOP_K,
    DEFAULT_MEMORY_ENABLED,
    DEFAULT_MEMORY_MIN_IMPORTANCE,
)
from .base import ContextProvider

if TYPE_CHECKING:
    from ..memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)


class MemoryContextProvider(ContextProvider):
    """Provide relevant memories as conversation context.

    This context provider searches stored memories for information
    relevant to the current user input and formats them for injection
    into the LLM's system prompt.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any],
        memory_manager: MemoryManager,
    ) -> None:
        """Initialize memory context provider.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary
            memory_manager: MemoryManager instance for memory operations
        """
        super().__init__(hass, config)
        self.memory_manager = memory_manager

    async def get_context(
        self,
        user_input: str,
        conversation_id: str | None = None,
    ) -> str:
        """Get relevant memories for this conversation.

        Uses semantic search to find memories related to user input.

        Args:
            user_input: User's message/query
            conversation_id: Optional conversation ID (unused for now)

        Returns:
            Formatted memory context string
        """
        # Check if memory is enabled
        if not self.config.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED):
            return ""

        try:
            # Search for relevant memories
            top_k = self.config.get(CONF_MEMORY_CONTEXT_TOP_K, DEFAULT_MEMORY_CONTEXT_TOP_K)
            min_importance = self.config.get(
                CONF_MEMORY_MIN_IMPORTANCE, DEFAULT_MEMORY_MIN_IMPORTANCE
            )

            relevant_memories = await self.memory_manager.search_memories(
                query=user_input,
                top_k=top_k,
                min_importance=min_importance,
            )

            if not relevant_memories:
                self._logger.debug("No relevant memories found for user input")
                return ""

            # Format memories for LLM context
            memory_context = self._format_memories(relevant_memories)

            self._logger.debug(
                "Retrieved %d relevant memories for context",
                len(relevant_memories),
            )

            return memory_context

        except Exception as err:
            self._logger.error("Error retrieving memory context: %s", err)
            return ""

    def _format_memories(self, memories: list[dict[str, Any]]) -> str:
        """Format memories for LLM context injection.

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted context string
        """
        if not memories:
            return ""

        context = "## Relevant Information from Past Conversations\n\n"

        for memory in memories:
            # Format each memory with type indicator
            memory_type = memory.get("type", "fact").title()
            content = memory.get("content", "")

            context += f"- [{memory_type}] {content}\n"

        context += "\n"
        return context

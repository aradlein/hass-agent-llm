"""Base context provider interface for home_agent.

This module defines the abstract base class for all context providers.
Context providers are responsible for gathering and formatting relevant
entity and state information to be injected into LLM prompts.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class ContextProvider(ABC):
    """Abstract base class for context providers.

    Context providers gather entity state information and format it
    for consumption by LLMs. Different implementations can use different
    strategies (direct entity listing, vector DB retrieval, etc.).
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the context provider.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary for this provider
        """
        self.hass = hass
        self.config = config
        self._logger = _LOGGER.getChild(self.__class__.__name__)

    @abstractmethod
    async def get_context(self, user_input: str) -> str:
        """Get formatted context for LLM based on user input.

        This method should be implemented by subclasses to provide
        their specific context gathering and formatting logic.

        Args:
            user_input: The user's query or message

        Returns:
            Formatted context string ready for LLM consumption

        Raises:
            ContextProviderError: If context gathering fails
        """
        raise NotImplementedError

    def _format_entity_state(
        self,
        entity_id: str,
        state: Any,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format entity state into a structured dictionary.

        Args:
            entity_id: The entity ID
            state: The entity's current state
            attributes: Optional dictionary of attributes to include

        Returns:
            Dictionary containing formatted entity information
        """
        formatted = {
            "entity_id": entity_id,
            "state": str(state),
        }

        if attributes:
            formatted["attributes"] = attributes

        return formatted

    def _get_entity_state(
        self,
        entity_id: str,
        attribute_filter: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Get current state and attributes for an entity.

        Args:
            entity_id: The entity ID to query
            attribute_filter: Optional list of specific attributes to include

        Returns:
            Dictionary with entity state and attributes, or None if entity not found
        """
        state_obj = self.hass.states.get(entity_id)

        if state_obj is None:
            self._logger.warning("Entity not found: %s", entity_id)
            return None

        result = {
            "entity_id": entity_id,
            "state": state_obj.state,
            "attributes": {},
        }

        # Include filtered attributes or all attributes
        if attribute_filter:
            result["attributes"] = {
                key: value
                for key, value in state_obj.attributes.items()
                if key in attribute_filter
            }
        else:
            result["attributes"] = dict(state_obj.attributes)

        return result

    def _get_entities_matching_pattern(self, pattern: str) -> list[str]:
        """Get entity IDs matching a pattern (supports wildcards).

        Args:
            pattern: Entity ID pattern (e.g., "light.*", "sensor.temperature_*")

        Returns:
            List of matching entity IDs
        """
        if "*" not in pattern:
            # No wildcard, return as-is if entity exists
            if self.hass.states.get(pattern):
                return [pattern]
            return []

        # Handle wildcards
        import fnmatch
        all_entity_ids = self.hass.states.async_entity_ids()
        matching = [
            entity_id
            for entity_id in all_entity_ids
            if fnmatch.fnmatch(entity_id, pattern)
        ]

        self._logger.debug(
            "Pattern '%s' matched %d entities",
            pattern,
            len(matching)
        )

        return matching

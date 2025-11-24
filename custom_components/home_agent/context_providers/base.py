"""Base context provider interface for home_agent.

This module defines the abstract base class for all context providers.
Context providers are responsible for gathering and formatting relevant
entity and state information to be injected into LLM prompts.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

_LOGGER = logging.getLogger(__name__)

# Bloat attributes to filter out from entity context
BLOAT_ATTRIBUTES = {
    "supported_features",  # Internal bitmask
    "icon",  # UI metadata
    "entity_picture",  # Image URL
    "entity_picture_local",  # Local image path
    "context_id",  # Internal HA tracking ID
    "attribution",  # Data source attribution
    "assumed_state",  # UI flag
    "restore",  # Restart behavior flag
    "editable",  # UI editability flag
}


def _make_json_serializable(value: Any) -> Any:
    """Convert a value to a JSON-serializable format.

    Handles datetime objects and other non-serializable types.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(item) for item in value]
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    # For other types, try to convert to string if not already serializable
    try:
        import json
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


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
        formatted: dict[str, Any] = {
            "entity_id": entity_id,
            "state": str(state),
        }

        if attributes is not None:
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

        # Get aliases from entity registry
        aliases = []
        try:
            entity_registry = er.async_get(self.hass)
            entity_entry = entity_registry.async_get(entity_id)
            if entity_entry and entity_entry.aliases:
                aliases = list(entity_entry.aliases)
        except (AttributeError, RuntimeError):
            # Entity registry not available (e.g., in tests or early startup)
            pass

        result["aliases"] = aliases

        # Include filtered attributes or all attributes, ensuring JSON serializability
        if attribute_filter is not None:
            result["attributes"] = {
                key: _make_json_serializable(value)
                for key, value in state_obj.attributes.items()
                if key in attribute_filter
            }
        else:
            # Filter out bloat attributes and internal attributes (starting with _)
            result["attributes"] = {
                key: _make_json_serializable(value)
                for key, value in state_obj.attributes.items()
                if key not in BLOAT_ATTRIBUTES and not key.startswith("_")
            }

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
            entity_id for entity_id in all_entity_ids if fnmatch.fnmatch(entity_id, pattern)
        ]

        self._logger.debug("Pattern '%s' matched %d entities", pattern, len(matching))

        return matching

    def _get_entity_services(self, entity_id: str) -> list[str]:
        """Get available services for an entity based on its domain.

        Args:
            entity_id: The entity ID to get services for

        Returns:
            List of available service names for this entity
        """
        domain = entity_id.split(".")[0]

        # Get all services for the entity's domain
        services = self.hass.services.async_services().get(domain, {})

        # Common service mapping by domain
        service_list = list(services.keys())

        # Add homeassistant domain services that work on all entities
        homeassistant_services = self.hass.services.async_services().get("homeassistant", {})
        common_services = [
            "turn_on",
            "turn_off",
            "toggle",
            "update_entity",
            "reload_config_entry",
        ]

        for service in common_services:
            if service in homeassistant_services:
                service_list.append(f"homeassistant.{service}")

        return service_list

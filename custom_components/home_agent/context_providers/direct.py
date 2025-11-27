"""Direct context provider for home_agent.

This provider directly fetches and formats configured entities and their
attributes for injection into LLM prompts. It supports both JSON and
natural language formatting.
"""

import json
import logging
from typing import Any, Literal

from homeassistant.core import HomeAssistant

from .base import ContextProvider

_LOGGER = logging.getLogger(__name__)


class DirectContextProvider(ContextProvider):
    """Context provider that directly fetches configured entities.

    This provider takes a list of entity configurations and fetches their
    current state from Home Assistant, formatting the results as JSON or
    natural language for LLM consumption.

    Configuration format:
        {
            "entities": [
                {
                    "entity_id": "light.living_room",
                    "attributes": ["brightness", "color_temp"]
                },
                {
                    "entity_id": "sensor.*",  # Supports wildcards
                    "attributes": null  # Include all attributes
                }
            ],
            "format": "json"  # or "natural_language"
        }
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the direct context provider.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary containing entity list and format
        """
        super().__init__(hass, config)
        self.entities_config = config.get("entities", [])
        self.format_type: Literal["json", "natural_language"] = config.get("format", "json")

    async def get_context(self, user_input: str) -> str:
        """Get formatted context for configured entities.

        Note: user_input is not used in direct mode as we always return
        the same configured entities, but it's part of the base interface.

        Args:
            user_input: The user's query (not used in direct mode)

        Returns:
            Formatted context string containing entity states

        Raises:
            ValueError: If format type is invalid
        """
        self._logger.debug(
            "Getting direct context for %d entity configurations",
            len(self.entities_config),
        )

        # Gather all entity states
        entity_states = await self._gather_entity_states()

        # Format based on configured format type
        if self.format_type == "json":
            return self._format_as_json(entity_states)
        elif self.format_type == "natural_language":
            return self._format_as_natural_language(entity_states)
        elif self.format_type == "hybrid":
            return self._format_as_hybrid(entity_states)
        else:
            raise ValueError(f"Invalid format type: {self.format_type}")

    async def _gather_entity_states(self) -> list[dict[str, Any]]:
        """Gather states for all configured entities.

        If no entities are configured, defaults to all entities exposed
        to the voice assistant (respecting Home Assistant's exposure settings).

        Returns:
            List of dictionaries containing entity state information
        """
        entity_states = []

        # If no entities configured, use all exposed entities
        if not self.entities_config:
            self._logger.debug(
                "No entities configured, using all exposed entities from voice assistant"
            )
            entity_states = self._get_all_exposed_entities()
            self._logger.debug("Gathered state for %d exposed entities", len(entity_states))
            return entity_states

        for entity_config in self.entities_config:
            # Handle both dict format {"entity_id": "...", "attributes": [...]}
            # and simple string format "entity_id"
            if isinstance(entity_config, dict):
                entity_id = entity_config.get("entity_id")
                attributes_filter = entity_config.get("attributes")
            else:
                # Simple string format - just the entity_id
                entity_id = str(entity_config)
                attributes_filter = None

            if not entity_id:
                self._logger.warning("Entity configuration missing entity_id")
                continue

            # Handle wildcard patterns
            matching_entities = self._get_entities_matching_pattern(entity_id)

            for matched_entity_id in matching_entities:
                state_data = self._get_entity_state(matched_entity_id, attributes_filter)

                if state_data:
                    # Add available services for consistency with vector DB mode
                    state_data["available_services"] = self._get_entity_services(matched_entity_id)
                    entity_states.append(state_data)

        self._logger.debug("Gathered state for %d entities", len(entity_states))

        return entity_states

    def _get_all_exposed_entities(self) -> list[dict[str, Any]]:
        """Get all entities exposed to the voice assistant.

        Returns:
            List of entity state dictionaries for all exposed entities
        """
        from homeassistant.components import conversation as ha_conversation
        from homeassistant.components.homeassistant.exposed_entities import async_should_expose

        entity_states = []

        # Get all entities that should be exposed to conversation
        for state in self.hass.states.async_all():
            if async_should_expose(self.hass, ha_conversation.DOMAIN, state.entity_id):
                state_data = self._get_entity_state(state.entity_id)
                if state_data:
                    # Add available services for consistency with vector DB mode
                    state_data["available_services"] = self._get_entity_services(state.entity_id)
                    entity_states.append(state_data)

        return entity_states

    def _format_as_json(self, entity_states: list[dict[str, Any]]) -> str:
        """Format entity states as JSON.

        Args:
            entity_states: List of entity state dictionaries

        Returns:
            JSON formatted string
        """
        result = {"entities": entity_states, "count": len(entity_states)}

        return json.dumps(result, indent=2, default=str)

    def _format_as_natural_language(self, entity_states: list[dict[str, Any]]) -> str:
        """Format entity states as natural language description.

        Args:
            entity_states: List of entity state dictionaries

        Returns:
            Natural language formatted string
        """
        if not entity_states:
            return "No entities currently configured for context."

        lines = ["Current Home State:"]

        for entity_data in entity_states:
            entity_id = entity_data["entity_id"]
            state = entity_data["state"]
            attributes = entity_data.get("attributes", {})

            # Extract domain and friendly name
            domain = entity_id.split(".")[0]
            friendly_name = attributes.get(
                "friendly_name", entity_id.split(".")[1].replace("_", " ").title()
            )

            # Format based on domain for better readability
            line = self._format_entity_natural_language(domain, friendly_name, state, attributes)

            lines.append(f"- {line}")

        return "\n".join(lines)

    def _format_entity_natural_language(
        self, domain: str, friendly_name: str, state: str, attributes: dict[str, Any]
    ) -> str:
        """Format a single entity in natural language.

        Args:
            domain: Entity domain (e.g., "light", "sensor")
            friendly_name: Human-readable entity name
            state: Current state
            attributes: Entity attributes

        Returns:
            Natural language description of the entity
        """
        # Handle different domains with domain-specific formatting
        if domain == "light":
            return self._format_light(friendly_name, state, attributes)
        elif domain == "sensor":
            return self._format_sensor(friendly_name, state, attributes)
        elif domain == "binary_sensor":
            return self._format_binary_sensor(friendly_name, state, attributes)
        elif domain == "climate":
            return self._format_climate(friendly_name, state, attributes)
        elif domain == "switch":
            return self._format_switch(friendly_name, state, attributes)
        elif domain == "lock":
            return self._format_lock(friendly_name, state, attributes)
        else:
            # Generic formatting for unknown domains
            return f"{friendly_name} is {state}"

    def _format_light(self, name: str, state: str, attributes: dict[str, Any]) -> str:
        """Format light entity in natural language."""
        if state == "off":
            return f"{name} is off"

        parts = [f"{name} is on"]

        if "brightness" in attributes:
            # Convert brightness (0-255) to percentage
            brightness_pct = int((attributes["brightness"] / 255) * 100)
            parts.append(f"at {brightness_pct}% brightness")

        if "color_temp" in attributes:
            parts.append(f"with color temperature {attributes['color_temp']}K")

        return " ".join(parts)

    def _format_sensor(self, name: str, state: str, attributes: dict[str, Any]) -> str:
        """Format sensor entity in natural language."""
        unit = attributes.get("unit_of_measurement", "")
        device_class = attributes.get("device_class", "")

        if unit:
            return f"{name} is {state} {unit}"
        elif device_class:
            return f"{name} ({device_class}) is {state}"
        else:
            return f"{name} is {state}"

    def _format_binary_sensor(self, name: str, state: str, attributes: dict[str, Any]) -> str:
        """Format binary sensor entity in natural language."""
        device_class = attributes.get("device_class", "")

        # Use device class to provide better context
        if device_class == "door":
            return f"{name} is {'open' if state == 'on' else 'closed'}"
        elif device_class == "window":
            return f"{name} is {'open' if state == 'on' else 'closed'}"
        elif device_class == "motion":
            return f"{name} {'detects motion' if state == 'on' else 'no motion'}"
        elif device_class == "occupancy":
            return f"{name} is {'occupied' if state == 'on' else 'not occupied'}"
        else:
            return f"{name} is {state}"

    def _format_climate(self, name: str, state: str, attributes: dict[str, Any]) -> str:
        """Format climate entity in natural language."""
        parts = [f"{name} is {state}"]

        if "current_temperature" in attributes:
            temp = attributes["current_temperature"]
            unit = attributes.get("temperature_unit", "")
            parts.append(f"at {temp}{unit}")

        if "target_temperature" in attributes:
            target = attributes["target_temperature"]
            unit = attributes.get("temperature_unit", "")
            parts.append(f"(target: {target}{unit})")

        return " ".join(parts)

    def _format_switch(self, name: str, state: str, attributes: dict[str, Any]) -> str:
        """Format switch entity in natural language."""
        return f"{name} is {state}"

    def _format_lock(self, name: str, state: str, attributes: dict[str, Any]) -> str:
        """Format lock entity in natural language."""
        return f"{name} is {state}"

    def _format_as_hybrid(self, entity_states: list[dict[str, Any]]) -> str:
        """Format entity states as hybrid (JSON structure + natural language summary).

        Combines structured JSON data with a natural language summary for optimal
        LLM understanding.

        Args:
            entity_states: List of entity state dictionaries

        Returns:
            Hybrid formatted string with both JSON and natural language
        """
        if not entity_states:
            return "No entities currently configured for context."

        # Start with natural language summary
        nl_summary = self._format_as_natural_language(entity_states)

        # Add JSON structure for precise data
        json_data = self._format_as_json(entity_states)

        return f"{nl_summary}\n\n--- Detailed Entity Data ---\n{json_data}"

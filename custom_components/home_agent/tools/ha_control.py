"""Home Assistant control tool for the Home Agent integration.

This module provides the HomeAssistantControlTool for executing control
actions on Home Assistant entities (turn_on, turn_off, toggle, set_value).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TOGGLE,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
)
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from ..const import (
    ACTION_SET_VALUE,
    ACTION_TOGGLE,
    ACTION_TURN_OFF,
    ACTION_TURN_ON,
    TOOL_HA_CONTROL,
)
from ..exceptions import PermissionDenied, ToolExecutionError, ValidationError
from .registry import BaseTool

if TYPE_CHECKING:
    from homeassistant.helpers.entity import Entity

_LOGGER = logging.getLogger(__name__)


class HomeAssistantControlTool(BaseTool):
    """Tool for controlling Home Assistant entities.

    This tool allows the LLM to control devices and entities in Home Assistant
    by executing service calls. It supports common actions like turn_on,
    turn_off, toggle, and set_value.

    Supported actions:
        - turn_on: Turn on an entity (lights, switches, etc.)
        - turn_off: Turn off an entity
        - toggle: Toggle an entity's state
        - set_value: Set specific values (brightness, temperature, etc.)

    Example tool calls:
        # Turn on a light at 50% brightness
        {
            "action": "turn_on",
            "entity_id": "light.living_room",
            "parameters": {"brightness_pct": 50}
        }

        # Set thermostat temperature
        {
            "action": "set_value",
            "entity_id": "climate.thermostat",
            "parameters": {"temperature": 72}
        }

        # Turn off a switch
        {
            "action": "turn_off",
            "entity_id": "switch.fan"
        }
    """

    def __init__(
        self,
        hass: HomeAssistant,
        exposed_entities: set[str] | None = None,
    ) -> None:
        """Initialize the Home Assistant control tool.

        Args:
            hass: Home Assistant instance
            exposed_entities: Optional set of entity IDs that are exposed
                for control. If None, all entities are accessible (not
                recommended for production).
        """
        super().__init__(hass)
        self._exposed_entities = exposed_entities

    @property
    def name(self) -> str:
        """Return the tool name."""
        return TOOL_HA_CONTROL

    @property
    def description(self) -> str:
        """Return the tool description."""
        return (
            "Control Home Assistant devices and services. Use this to turn on/off "
            "lights, adjust brightness, set thermostat temperatures, lock doors, "
            "control switches, and perform other device control actions. "
            "Always use ha_query first to check the current state before making "
            "changes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "The action to perform on the entity. "
                        "Use 'turn_on' to turn on devices, 'turn_off' to turn them off, "
                        "'toggle' to switch between states, or 'set_value' to set "
                        "specific attributes like brightness or temperature."
                    ),
                    "enum": [ACTION_TURN_ON, ACTION_TURN_OFF, ACTION_TOGGLE, ACTION_SET_VALUE],
                },
                "entity_id": {
                    "type": "string",
                    "description": (
                        "The entity ID to control in the format 'domain.entity_name'. "
                        "Examples: 'light.living_room', 'switch.fan', 'climate.thermostat', "
                        "'lock.front_door'. Use ha_query with wildcards if unsure of exact ID."
                    ),
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "Additional parameters for the action. Common parameters: "
                        "brightness_pct (0-100 for lights), temperature (for climate), "
                        "rgb_color ([R, G, B] for lights), hvac_mode (for climate). "
                        "Parameters depend on the entity domain and action."
                    ),
                },
            },
            "required": ["action", "entity_id"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a control action on a Home Assistant entity.

        Args:
            action: Action to perform (turn_on, turn_off, toggle, set_value)
            entity_id: Entity ID to control
            parameters: Optional additional parameters for the action

        Returns:
            Dict containing:
                - success: bool indicating if execution succeeded
                - entity_id: The controlled entity ID
                - action: The action performed
                - new_state: The entity's new state
                - message: Human-readable result message

        Raises:
            ValidationError: If parameters are invalid
            PermissionDenied: If entity is not accessible
            ToolExecutionError: If execution fails
        """
        action = kwargs.get("action")
        entity_id = kwargs.get("entity_id")
        parameters = kwargs.get("parameters", {})

        # Validate required parameters
        if not action:
            raise ValidationError("Parameter 'action' is required")

        if not entity_id:
            raise ValidationError("Parameter 'entity_id' is required")

        # Validate action
        valid_actions = [ACTION_TURN_ON, ACTION_TURN_OFF, ACTION_TOGGLE, ACTION_SET_VALUE]
        if action not in valid_actions:
            raise ValidationError(
                f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"
            )

        # Validate entity ID format
        if "." not in entity_id:
            raise ValidationError(
                f"Invalid entity_id format: '{entity_id}'. "
                f"Expected format: 'domain.entity_name' (e.g., 'light.living_room')"
            )

        # Check entity access permissions
        self._validate_entity_access(entity_id)

        # Verify entity exists
        entity_registry = er.async_get(self.hass)
        if not entity_registry.async_get(entity_id):
            # Entity might not be in registry but still exist in state machine
            state = self.hass.states.get(entity_id)
            if not state:
                raise ValidationError(
                    f"Entity '{entity_id}' does not exist. "
                    f"Use ha_query with wildcards to search for entities."
                )

        try:
            # Execute the action
            await self._execute_action(action, entity_id, parameters)

            # Get the new state
            new_state = self.hass.states.get(entity_id)
            state_value = new_state.state if new_state else "unknown"

            # Build success response
            result = {
                "success": True,
                "entity_id": entity_id,
                "action": action,
                "new_state": state_value,
                "message": self._build_success_message(action, entity_id, state_value),
            }

            # Include relevant attributes in the response
            if new_state and new_state.attributes:
                result["attributes"] = self._extract_relevant_attributes(
                    entity_id, new_state.attributes
                )

            _LOGGER.info(
                "Successfully executed %s on %s, new state: %s",
                action,
                entity_id,
                state_value,
            )

            return result

        except HomeAssistantError as error:
            _LOGGER.error(
                "Failed to execute %s on %s: %s",
                action,
                entity_id,
                error,
                exc_info=True,
            )
            raise ToolExecutionError(
                f"Failed to execute {action} on {entity_id}: {error}"
            ) from error

    def _validate_entity_access(self, entity_id: str) -> None:
        """Validate that the entity is accessible.

        Args:
            entity_id: Entity ID to validate

        Raises:
            PermissionDenied: If entity is not accessible
        """
        # If no exposed entities set is provided, allow all access
        # (not recommended for production, but useful for testing)
        if self._exposed_entities is None:
            return

        if entity_id not in self._exposed_entities:
            _LOGGER.warning(
                "Attempted access to unexposed entity: %s",
                entity_id,
            )
            raise PermissionDenied(
                f"Entity '{entity_id}' is not accessible. "
                f"Ensure it is exposed in the integration configuration or "
                f"voice assistant settings."
            )

    async def _execute_action(
        self,
        action: str,
        entity_id: str,
        parameters: dict[str, Any],
    ) -> None:
        """Execute the specified action on the entity.

        Args:
            action: Action to perform
            entity_id: Entity ID to control
            parameters: Additional parameters for the action

        Raises:
            ToolExecutionError: If action execution fails
        """
        domain = entity_id.split(".")[0]

        # Build service data
        service_data = {ATTR_ENTITY_ID: entity_id}
        service_data.update(parameters)

        # Map action to service
        if action == ACTION_TURN_ON:
            service = SERVICE_TURN_ON
        elif action == ACTION_TURN_OFF:
            service = SERVICE_TURN_OFF
        elif action == ACTION_TOGGLE:
            service = SERVICE_TOGGLE
        elif action == ACTION_SET_VALUE:
            # For set_value, we use the domain-specific service
            # This typically requires custom handling per domain
            service = self._get_set_value_service(domain, parameters)
        else:
            raise ToolExecutionError(f"Unknown action: {action}")

        # Execute the service call
        _LOGGER.debug(
            "Calling service %s.%s with data: %s",
            domain,
            service,
            service_data,
        )

        await self.hass.services.async_call(
            domain,
            service,
            service_data,
            blocking=True,
        )

    def _get_set_value_service(
        self,
        domain: str,
        parameters: dict[str, Any],
    ) -> str:
        """Determine the appropriate service for set_value action.

        Different domains use different services for setting values.
        This method maps domain and parameters to the correct service.

        Args:
            domain: Entity domain (e.g., 'light', 'climate')
            parameters: Parameters being set

        Returns:
            Service name to call

        Raises:
            ToolExecutionError: If no appropriate service found
        """
        # For lights, use turn_on with parameters
        if domain == "light":
            return SERVICE_TURN_ON

        # For climate, use set_temperature, set_hvac_mode, etc.
        if domain == "climate":
            if "temperature" in parameters or "target_temp_high" in parameters:
                return "set_temperature"
            if "hvac_mode" in parameters:
                return "set_hvac_mode"
            if "fan_mode" in parameters:
                return "set_fan_mode"

        # For covers, use set_cover_position
        if domain == "cover" and "position" in parameters:
            return "set_cover_position"

        # For input_number, use set_value
        if domain == "input_number" and "value" in parameters:
            return "set_value"

        # For input_select, use select_option
        if domain == "input_select" and "option" in parameters:
            return "select_option"

        # For fans, use set_percentage or set_preset_mode
        if domain == "fan":
            if "percentage" in parameters:
                return "set_percentage"
            if "preset_mode" in parameters:
                return "set_preset_mode"

        # Default to turn_on for most domains
        # This works for lights with brightness, color, etc.
        _LOGGER.warning(
            "No specific set_value service found for domain %s, using turn_on",
            domain,
        )
        return SERVICE_TURN_ON

    def _build_success_message(
        self,
        action: str,
        entity_id: str,
        new_state: str,
    ) -> str:
        """Build a human-readable success message.

        Args:
            action: Action that was performed
            entity_id: Entity that was controlled
            new_state: New state of the entity

        Returns:
            Human-readable success message
        """
        # Extract entity name from ID
        entity_name = entity_id.split(".")[1].replace("_", " ").title()

        action_descriptions = {
            ACTION_TURN_ON: f"Turned on {entity_name}",
            ACTION_TURN_OFF: f"Turned off {entity_name}",
            ACTION_TOGGLE: f"Toggled {entity_name}",
            ACTION_SET_VALUE: f"Updated {entity_name}",
        }

        message = action_descriptions.get(action, f"Executed {action} on {entity_name}")
        message += f". Current state: {new_state}"

        return message

    def _extract_relevant_attributes(
        self,
        entity_id: str,
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract relevant attributes to include in the response.

        Different entity types have different important attributes.
        This method extracts the most relevant ones for the LLM.

        Args:
            entity_id: Entity ID
            attributes: All entity attributes

        Returns:
            Dict of relevant attributes
        """
        domain = entity_id.split(".")[0]
        relevant_attrs = {}

        # Common attributes for all entities
        if "friendly_name" in attributes:
            relevant_attrs["friendly_name"] = attributes["friendly_name"]

        # Domain-specific attributes
        if domain == "light":
            for attr in ["brightness", "color_temp", "rgb_color", "effect"]:
                if attr in attributes:
                    relevant_attrs[attr] = attributes[attr]

        elif domain == "climate":
            for attr in [
                "temperature",
                "target_temp_high",
                "target_temp_low",
                "current_temperature",
                "hvac_mode",
                "fan_mode",
            ]:
                if attr in attributes:
                    relevant_attrs[attr] = attributes[attr]

        elif domain == "cover":
            for attr in ["current_position", "current_tilt_position"]:
                if attr in attributes:
                    relevant_attrs[attr] = attributes[attr]

        elif domain == "fan":
            for attr in ["percentage", "preset_mode", "oscillating"]:
                if attr in attributes:
                    relevant_attrs[attr] = attributes[attr]

        elif domain == "media_player":
            for attr in [
                "volume_level",
                "is_volume_muted",
                "media_title",
                "media_artist",
                "source",
            ]:
                if attr in attributes:
                    relevant_attrs[attr] = attributes[attr]

        return relevant_attrs

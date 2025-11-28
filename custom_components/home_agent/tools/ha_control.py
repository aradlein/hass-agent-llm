"""Home Assistant control tool for the Home Agent integration.

This module provides the HomeAssistantControlTool for executing control
actions on Home Assistant entities (turn_on, turn_off, toggle, set_value).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.components.cover import CoverEntityFeature
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TOGGLE, SERVICE_TURN_OFF, SERVICE_TURN_ON
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from ..const import (
    ACTION_SET_VALUE,
    ACTION_TOGGLE,
    ACTION_TURN_OFF,
    ACTION_TURN_ON,
    DOMAIN_SERVICE_MAPPINGS,
    TOOL_HA_CONTROL,
)
from ..exceptions import PermissionDenied, ToolExecutionError, ValidationError
from .registry import BaseTool

if TYPE_CHECKING:
    pass

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
            "control switches, operate covers/blinds, and perform other device control actions. "
            "Check the entity's available_services field to see what actions are supported. "
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
                        "Check the entity's available_services field to see supported actions. "
                        "The action to perform on the entity. "
                        "Use 'turn_on' to turn on devices, 'turn_off' to turn them off, "
                        "'toggle' to switch between states, or 'set_value' to set "
                        "specific attributes like brightness or temperature."
                    ),
                    "enum": [
                        ACTION_TURN_ON,
                        ACTION_TURN_OFF,
                        ACTION_TOGGLE,
                        ACTION_SET_VALUE,
                    ],
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
                        "Additional parameters for the action. Common parameters by domain: "
                        "Lights: brightness_pct (0-100), rgb_color ([R, G, B]); "
                        "Climate: temperature, hvac_mode, fan_mode; "
                        "Covers: position (0-100, accepts 'current_position' too), tilt_position; "
                        "Media Players: volume_level (0.0-1.0), source; "
                        "Humidifiers: humidity (0-100); "
                        "Fans: percentage (0-100), preset_mode. "
                        "Note: Both attribute names (e.g., 'current_position') and service "
                        "parameter names (e.g., 'position') are accepted and automatically normalized."
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
        valid_actions = [
            ACTION_TURN_ON,
            ACTION_TURN_OFF,
            ACTION_TOGGLE,
            ACTION_SET_VALUE,
        ]
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

        # Normalize parameters (e.g., current_position -> position)
        normalized_parameters = self._normalize_parameters(domain, parameters)

        # Build service data
        service_data = {ATTR_ENTITY_ID: entity_id}
        service_data.update(normalized_parameters)

        # Map action to service using the domain service mappings
        service = self._get_service_for_action(action, domain, entity_id, normalized_parameters)
        if service is None:
            raise ToolExecutionError(
                f"Action '{action}' is not supported for domain '{domain}'"
            )

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

    def _normalize_parameters(
        self,
        domain: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize attribute names to service parameter names.

        Users/LLMs often provide attribute names (e.g., 'current_position')
        when they should use service parameter names (e.g., 'position').
        This method normalizes common mismatches to improve usability.

        Args:
            domain: Entity domain (e.g., 'light', 'climate', 'cover')
            parameters: Original parameters dict

        Returns:
            Normalized parameters dict with corrected parameter names
        """
        normalized = parameters.copy()

        # Cover domain: Normalize position-related attributes
        if domain == "cover":
            # Normalize current_position -> position
            if "current_position" in normalized and "position" not in normalized:
                _LOGGER.debug(
                    "Normalizing cover parameter 'current_position' -> 'position'"
                )
                normalized["position"] = normalized.pop("current_position")

            # Normalize current_tilt_position -> tilt_position
            if "current_tilt_position" in normalized and "tilt_position" not in normalized:
                _LOGGER.debug(
                    "Normalizing cover parameter 'current_tilt_position' -> 'tilt_position'"
                )
                normalized["tilt_position"] = normalized.pop("current_tilt_position")

        # Climate domain: Warn about current_temperature confusion
        elif domain == "climate":
            if "current_temperature" in normalized and "temperature" not in normalized:
                _LOGGER.warning(
                    "User provided 'current_temperature' but likely meant 'temperature' "
                    "(target temperature). Normalizing to 'temperature'."
                )
                normalized["temperature"] = normalized.pop("current_temperature")

        # Light domain: Convert brightness_pct (0-100) to brightness (0-255)
        elif domain == "light":
            if "brightness_pct" in normalized:
                brightness_pct = normalized.pop("brightness_pct")
                if brightness_pct is not None:
                    # Convert percentage to 0-255 range
                    normalized["brightness"] = int(brightness_pct * 255 / 100)
                    _LOGGER.debug(
                        "Converting brightness_pct=%s to brightness=%s",
                        brightness_pct,
                        normalized["brightness"],
                    )

        return normalized

    def _entity_supports_feature(
        self,
        entity_id: str,
        feature: int,
    ) -> bool:
        """Check if an entity supports a specific feature.

        Args:
            entity_id: The entity to check
            feature: The feature bitmask to check for

        Returns:
            True if entity supports the feature, False otherwise
        """
        state = self.hass.states.get(entity_id)
        if not state:
            return False

        supported_features = state.attributes.get("supported_features", 0)
        return bool(supported_features & feature)

    def _get_service_for_action(
        self,
        action: str,
        domain: str,
        entity_id: str,
        parameters: dict[str, Any],
    ) -> str | None:
        """Determine the appropriate service for an action using domain mappings.

        Uses the DOMAIN_SERVICE_MAPPINGS to map actions to services,
        validating entity capabilities when necessary.

        Args:
            action: The action to perform (turn_on, turn_off, toggle, set_value)
            domain: Entity domain (e.g., 'light', 'climate')
            entity_id: Entity ID being controlled (for capability checks)
            parameters: Parameters being set (used for set_value action)

        Returns:
            Service name to call, or None if action not supported

        Raises:
            ToolExecutionError: If entity doesn't support requested capability
        """
        # Get the domain's service mapping
        domain_mapping = DOMAIN_SERVICE_MAPPINGS.get(domain)
        if not domain_mapping:
            # Fallback to generic mapping for unmapped domains
            _LOGGER.warning(
                "No service mapping found for domain '%s', using generic mapping",
                domain,
            )
            if action == ACTION_TURN_ON:
                return SERVICE_TURN_ON
            elif action == ACTION_TURN_OFF:
                return SERVICE_TURN_OFF
            elif action == ACTION_TOGGLE:
                return SERVICE_TOGGLE
            elif action == ACTION_SET_VALUE:
                # Default to turn_on for set_value on unknown domains
                return SERVICE_TURN_ON
            else:
                return None

        # Get the action mapping for this domain
        action_map = domain_mapping.get("action_service_map", {})
        service_or_map = action_map.get(action)

        if service_or_map is None:
            return None

        # Handle SET_VALUE action (requires parameter-based mapping)
        if action == ACTION_SET_VALUE:
            if isinstance(service_or_map, dict):
                # Parameter-based mapping (e.g., {"position": "set_cover_position"})
                # Find which parameter is present and map to the correct service
                for param_name, service_name in service_or_map.items():
                    if param_name in parameters:
                        # Check if this service requires a specific feature
                        service = self._validate_feature_for_service(
                            entity_id, domain, service_name
                        )
                        return service
                # No matching parameter found
                _LOGGER.warning(
                    "No matching parameter found for %s set_value with params %s",
                    domain,
                    list(parameters.keys()),
                )
                return None
            else:
                # Simple service name (e.g., "set_datetime")
                return service_or_map  # type: ignore[return-value]
        else:
            # Direct action mapping (turn_on, turn_off, toggle)
            return service_or_map  # type: ignore[return-value]

    def _validate_feature_for_service(
        self,
        entity_id: str,
        domain: str,
        service_name: str,
    ) -> str:
        """Validate that an entity supports the feature required for a service.

        Args:
            entity_id: The entity to validate
            domain: Entity domain
            service_name: The service being called

        Returns:
            The service name if validation passes

        Raises:
            ToolExecutionError: If entity doesn't support the required feature
        """
        # Cover domain feature validation
        if domain == "cover":
            if service_name == "set_cover_position":
                if not self._entity_supports_feature(entity_id, CoverEntityFeature.SET_POSITION):
                    _LOGGER.warning(
                        "Entity %s does not support position control (binary cover only)",
                        entity_id,
                    )
                    raise ToolExecutionError(
                        f"Entity '{entity_id}' does not support position control. "
                        f"This is a binary cover (open/close only). "
                        f"Use action='turn_on' to open or action='turn_off' to close."
                    )
            elif service_name == "set_cover_tilt_position":
                if not self._entity_supports_feature(entity_id, CoverEntityFeature.SET_TILT_POSITION):
                    raise ToolExecutionError(
                        f"Entity '{entity_id}' does not support tilt position control."
                    )

        # Add other domain-specific feature validations here as needed
        # For now, most other domains don't require strict feature validation

        return service_name

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

        # Convert brightness (0-255) to brightness_pct (0-100) for light entities
        if domain == "light" and "brightness" in relevant_attrs:
            brightness = relevant_attrs.pop("brightness")
            if brightness is not None:
                relevant_attrs["brightness_pct"] = int(brightness / 255 * 100)

        return relevant_attrs

"""Base context provider interface for home_agent.

This module defines the abstract base class for all context providers.
Context providers are responsible for gathering and formatting relevant
entity and state information to be injected into LLM prompts.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from homeassistant.core import HomeAssistant, State
from homeassistant.helpers import entity_registry as er

from ..const import DOMAIN_SERVICE_MAPPINGS

_LOGGER = logging.getLogger(__name__)

# Bloat attributes to filter out from entity context
BLOAT_ATTRIBUTES = {
    # UI/Internal metadata
    "supported_features",  # Internal bitmask
    "icon",  # UI metadata
    "entity_picture",  # Image URL
    "entity_picture_local",  # Local image path
    "context_id",  # Internal HA tracking ID
    "attribution",  # Data source attribution
    "assumed_state",  # UI flag
    "restore",  # Restart behavior flag
    "editable",  # UI editability flag
    # Color-related attributes (rarely needed)
    "min_color_temp_kelvin",
    "max_color_temp_kelvin",
    "min_mireds",
    "max_mireds",
    "supported_color_modes",
    "color_mode",
    "color_temp_kelvin",
    "color_temp",
    "hs_color",
    "rgb_color",
    "xy_color",
    "rgbw_color",
    "rgbww_color",
    "effect_list",
    "effect",
    # Technical metadata
    "last_changed",
    "last_updated",
    "device_id",
    "unique_id",
    "platform",
    "integration",
    "linkquality",
    "update_available",
}


# Hardcoded parameter hints for services where HA doesn't mark params as required
# but they're practically required for the service to work
CRITICAL_SERVICE_PARAMS = {
    "media_player": {
        "play_media": ["media_content_id", "media_content_type"],
        "volume_set": ["volume_level"],
    },
    "cover": {
        "set_cover_position": ["position"],
        "set_cover_tilt_position": ["tilt_position"],
    },
    "climate": {
        "set_temperature": ["temperature"],
        "set_hvac_mode": ["hvac_mode"],
    },
    "fan": {
        "set_percentage": ["percentage"],
    },
    "humidifier": {
        "set_humidity": ["humidity"],
    },
    "input_number": {
        "set_value": ["value"],
    },
    "input_select": {
        "select_option": ["option"],
    },
    "input_text": {
        "set_value": ["value"],
    },
    "number": {
        "set_value": ["value"],
    },
    "select": {
        "select_option": ["option"],
    },
}


def _add_parameter_hints_to_services(
    hass: HomeAssistant,
    domain: str,
    services: list[str],
) -> list[str]:
    """Add parameter hints to service names showing required parameters.

    Args:
        hass: Home Assistant instance
        domain: The domain to get service schemas for
        services: List of service names

    Returns:
        List of service names with parameter hints (e.g., "play_media[media_content_id,media_content_type]")
    """
    # Get all service schemas for the domain
    try:
        all_schemas = hass.services.async_services()
        domain_schemas = all_schemas.get(domain, {})
    except Exception:
        # If we can't get schemas (e.g., during testing with mocks), return services as-is
        return services

    services_with_hints = []
    for service in services:
        # Check hardcoded critical parameters first
        if domain in CRITICAL_SERVICE_PARAMS and service in CRITICAL_SERVICE_PARAMS[domain]:
            params = CRITICAL_SERVICE_PARAMS[domain][service]
            params_str = ",".join(params[:3])  # Limit to 3 params
            services_with_hints.append(f"{service}[{params_str}]")
            continue

        if service in domain_schemas:
            try:
                # Get the service fields/parameters
                service_data = domain_schemas[service]
                fields = service_data.get("fields", {})

                # Check if fields is a dict (not a Mock or other object)
                if not isinstance(fields, dict):
                    services_with_hints.append(service)
                    continue

                # Extract required parameters only
                required_params = [
                    param_name
                    for param_name, param_def in fields.items()
                    if isinstance(param_def, dict) and param_def.get("required", False)
                ]

                # Add parameter hints if there are required parameters
                if required_params:
                    # Limit to first 3 required params to avoid bloat
                    params_str = ",".join(required_params[:3])
                    services_with_hints.append(f"{service}[{params_str}]")
                else:
                    services_with_hints.append(service)
            except Exception:
                # If we encounter any error processing this service, just use the name as-is
                services_with_hints.append(service)
        else:
            # Service not in schema, return as-is
            services_with_hints.append(service)

    return services_with_hints


def get_entity_available_services(
    hass: HomeAssistant,
    entity_id: str,
    state: State | None = None,
    include_parameter_hints: bool = True,
) -> list[str]:
    """Get available services for an entity based on its domain and supported features.

    This is the unified function used by both context providers and ha_control
    to determine which services are actually available for a specific entity.

    Args:
        hass: Home Assistant instance
        entity_id: The entity ID to get services for
        state: Optional state object (if not provided, will be fetched)
        include_parameter_hints: Whether to include parameter hints like "service[param1,param2]"

    Returns:
        List of available service names for this entity, filtered by supported_features.
        If include_parameter_hints is True, services with required parameters will have
        hints like "play_media[media_content_id,media_content_type]".
    """
    domain = entity_id.split(".")[0]

    # Get the domain's service mapping
    domain_mapping = DOMAIN_SERVICE_MAPPINGS.get(domain)
    if not domain_mapping:
        # Fallback: return all services for the domain if no mapping exists
        _LOGGER.debug(
            "No service mapping found for domain '%s', falling back to all domain services",
            domain,
        )
        services = hass.services.async_services().get(domain, {})
        service_list = list(services.keys())

        # Add parameter hints if requested
        if include_parameter_hints:
            return _add_parameter_hints_to_services(hass, domain, service_list)
        return service_list

    # Start with base services (always available)
    available_services = list(domain_mapping.get("base_services", []))

    # Get entity state to check supported_features
    if state is None:
        state = hass.states.get(entity_id)

    if state is None:
        _LOGGER.warning("Could not find state for entity %s", entity_id)
        # Add parameter hints if requested
        if include_parameter_hints:
            return _add_parameter_hints_to_services(hass, domain, available_services)
        return available_services

    # Get supported_features bitmask
    supported_features = state.attributes.get("supported_features", 0)

    # Add feature-based services using bitwise checks
    feature_services = domain_mapping.get("feature_services", {})
    for feature_flag, services in feature_services.items():
        # Check if entity supports this feature using bitwise AND
        if supported_features & feature_flag:
            available_services.extend(services)

    # Remove duplicates while preserving order
    seen = set()
    unique_services = []
    for service in available_services:
        if service not in seen:
            seen.add(service)
            unique_services.append(service)

    _LOGGER.debug(
        "Entity %s (supported_features=%s) has services: %s",
        entity_id,
        supported_features,
        unique_services,
    )

    # Add parameter hints if requested
    if include_parameter_hints:
        unique_services = _add_parameter_hints_to_services(hass, domain, unique_services)

    return unique_services


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

        # Convert brightness (0-255) to brightness_pct (0-100) for light entities
        domain = entity_id.split('.')[0]
        if domain == 'light' and 'brightness' in result["attributes"]:
            brightness = result["attributes"]['brightness']
            if brightness is not None:
                result["attributes"]['brightness_pct'] = int(brightness / 255 * 100)
                del result["attributes"]['brightness']

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
        """Get available services for an entity based on its domain and features.

        This method uses the unified get_entity_available_services function
        to filter services based on the entity's supported_features.

        Args:
            entity_id: The entity ID to get services for

        Returns:
            List of available service names for this entity, filtered by capabilities
        """
        return get_entity_available_services(self.hass, entity_id)

"""Helper utility functions for the Home Agent component.

This module provides utility functions for formatting data, validation,
security, and token estimation used throughout the integration.
"""

from __future__ import annotations

import re
from typing import Any

from homeassistant.const import ATTR_FRIENDLY_NAME, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State
from homeassistant.helpers import entity_registry as er

from .exceptions import ValidationError


def format_entity_state(
    state: State,
    attributes: list[str] | None = None,
    format_type: str = "json",
) -> dict[str, Any] | str:
    """Format entity state for LLM consumption.

    Takes a Home Assistant entity state object and formats it in a way that's
    easy for LLMs to understand and process. Can output as structured JSON
    or natural language description.

    Args:
        state: Home Assistant State object to format
        attributes: List of specific attribute names to include. If None,
            includes all attributes except internal ones (those starting with _)
        format_type: Output format - "json" for structured data or
            "natural_language" for human-readable description

    Returns:
        If format_type is "json": Dict with entity_id, state, and attributes
        If format_type is "natural_language": Formatted string description

    Raises:
        ValidationError: If format_type is not "json" or "natural_language"

    Example:
        >>> state = hass.states.get("light.living_room")
        >>> format_entity_state(state, ["brightness"], "json")
        {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "brightness": 128,
                "friendly_name": "Living Room Light"
            }
        }

        >>> format_entity_state(state, format_type="natural_language")
        "Living Room Light (light.living_room) is on with brightness 128"
    """
    if format_type not in ("json", "natural_language"):
        raise ValidationError(
            f"Invalid format_type: {format_type}. "
            f"Must be 'json' or 'natural_language'"
        )

    # Get attributes to include
    if attributes is None:
        # Include all non-internal attributes
        attrs = {
            k: v for k, v in state.attributes.items() if not k.startswith("_")
        }
    else:
        # Include only specified attributes, plus friendly_name if available
        attrs = {}
        for attr in attributes:
            if attr in state.attributes:
                attrs[attr] = state.attributes[attr]
        if ATTR_FRIENDLY_NAME in state.attributes:
            attrs[ATTR_FRIENDLY_NAME] = state.attributes[ATTR_FRIENDLY_NAME]

    if format_type == "json":
        return {
            "entity_id": state.entity_id,
            "state": state.state,
            "attributes": attrs,
        }

    # Natural language format
    friendly_name = attrs.get(ATTR_FRIENDLY_NAME, state.entity_id)
    description = f"{friendly_name} ({state.entity_id}) is {state.state}"

    # Add key attributes in natural language
    attr_parts = []
    for key, value in attrs.items():
        if key == ATTR_FRIENDLY_NAME:
            continue
        # Format the attribute nicely
        attr_parts.append(f"{key.replace('_', ' ')}: {value}")

    if attr_parts:
        description += " with " + ", ".join(attr_parts)

    return description


def validate_entity_id(entity_id: str) -> str:
    """Validate entity ID format.

    Checks if an entity ID follows the Home Assistant format of domain.entity_name.
    Also validates that the domain and entity name contain only allowed characters.

    Args:
        entity_id: Entity ID string to validate

    Returns:
        The validated entity ID (unchanged if valid)

    Raises:
        ValidationError: If entity ID format is invalid

    Example:
        >>> validate_entity_id("light.living_room")
        "light.living_room"

        >>> validate_entity_id("invalid_entity")
        ValidationError: Invalid entity_id format: invalid_entity
    """
    if not entity_id or not isinstance(entity_id, str):
        raise ValidationError(
            f"Invalid entity_id: {entity_id}. Must be a non-empty string."
        )

    # Check for basic format: domain.entity_name
    if "." not in entity_id:
        raise ValidationError(
            f"Invalid entity_id format: {entity_id}. "
            f"Expected format: domain.entity_name"
        )

    domain, entity_name = entity_id.split(".", 1)

    # Validate domain (letters, numbers, underscores)
    if not re.match(r"^[a-z0-9_]+$", domain):
        raise ValidationError(
            f"Invalid domain in entity_id: {entity_id}. "
            f"Domain must contain only lowercase letters, numbers, and underscores."
        )

    # Validate entity name (letters, numbers, underscores)
    if not re.match(r"^[a-z0-9_]+$", entity_name):
        raise ValidationError(
            f"Invalid entity name in entity_id: {entity_id}. "
            f"Entity name must contain only lowercase letters, numbers, and underscores."
        )

    return entity_id


def redact_sensitive_data(text: str, sensitive_values: list[str]) -> str:
    """Redact sensitive information from text for logging.

    Replaces sensitive values (like API keys, tokens, passwords) with a
    redacted placeholder to prevent them from appearing in logs or error messages.

    Args:
        text: Text that may contain sensitive information
        sensitive_values: List of sensitive strings to redact (e.g., API keys)

    Returns:
        Text with sensitive values replaced with "***REDACTED***"

    Example:
        >>> api_key = "sk-1234567890abcdef"
        >>> text = f"Error calling API with key {api_key}"
        >>> redact_sensitive_data(text, [api_key])
        "Error calling API with key ***REDACTED***"
    """
    if not text or not sensitive_values:
        return text

    redacted = text
    for value in sensitive_values:
        if value and isinstance(value, str) and value in redacted:
            redacted = redacted.replace(value, "***REDACTED***")

    return redacted


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Provides a rough estimation of how many tokens a piece of text will consume
    when sent to an LLM. This is not perfectly accurate (actual tokenization
    depends on the specific model), but provides a reasonable approximation.

    The estimation uses a simple heuristic:
    - ~4 characters per token on average for English text
    - Accounts for whitespace and punctuation

    For accurate token counting, consider using tiktoken library, but this
    estimation is sufficient for most use cases and doesn't require additional
    dependencies.

    Args:
        text: Text to estimate token count for

    Returns:
        Estimated number of tokens (minimum 1 if text is not empty)

    Example:
        >>> estimate_tokens("Hello, world!")
        4
        >>> estimate_tokens("This is a longer sentence with more words.")
        11
    """
    if not text:
        return 0

    # Rough estimation: ~4 characters per token
    # This is a conservative estimate that works reasonably well for English
    char_count = len(text)
    estimated = max(1, char_count // 4)

    return estimated


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Truncates text to fit within a maximum length, adding a suffix to indicate
    truncation. Useful for limiting log message size or preparing text for
    display in UI.

    Args:
        text: Text to potentially truncate
        max_length: Maximum allowed length (including suffix)
        suffix: String to append when truncating (default: "...")

    Returns:
        Original text if within max_length, otherwise truncated text with suffix

    Example:
        >>> truncate_text("This is a very long message", 15)
        "This is a ve..."

        >>> truncate_text("Short", 100)
        "Short"
    """
    if not text or len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length]

    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


def safe_get_state(
    state: State | None,
    default: str = STATE_UNAVAILABLE,
) -> str:
    """Safely get state value with fallback.

    Gets the state value from a State object, handling None states and
    unknown/unavailable states gracefully.

    Args:
        state: Home Assistant State object or None
        default: Default value to return if state is None or unavailable

    Returns:
        State value, or default if state is invalid

    Example:
        >>> state = hass.states.get("sensor.temperature")
        >>> safe_get_state(state)
        "23.5"

        >>> state = hass.states.get("sensor.nonexistent")
        >>> safe_get_state(state, "unknown")
        "unknown"
    """
    if state is None:
        return default

    if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        return default

    return state.state


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Converts a duration in seconds to a readable format like "2h 15m" or "45s".
    Useful for logging execution times and displaying durations to users.

    Args:
        seconds: Duration in seconds (can be fractional)

    Returns:
        Formatted duration string

    Example:
        >>> format_duration(45)
        "45s"

        >>> format_duration(3665)
        "1h 1m 5s"

        >>> format_duration(0.5)
        "0.5s"
    """
    if seconds < 1:
        return f"{seconds:.2f}s"

    parts = []
    remaining = int(seconds)

    hours = remaining // 3600
    if hours > 0:
        parts.append(f"{hours}h")
        remaining %= 3600

    minutes = remaining // 60
    if minutes > 0:
        parts.append(f"{minutes}m")
        remaining %= 60

    if remaining > 0 or not parts:
        parts.append(f"{remaining}s")

    return " ".join(parts)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries with override taking precedence.

    Deep merges two dictionaries, with values from override replacing those
    in base. Nested dictionaries are also merged recursively.

    Args:
        base: Base dictionary
        override: Dictionary with values that should override base

    Returns:
        New dictionary with merged values

    Example:
        >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
        >>> override = {"b": {"c": 10}, "e": 4}
        >>> merge_dicts(base, override)
        {"a": 1, "b": {"c": 10, "d": 3}, "e": 4}
    """
    result = base.copy()

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result

"""Helper utility functions for the Home Agent component.

This module provides utility functions for formatting data, validation,
security, and token estimation used throughout the integration.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import aiohttp

from homeassistant.const import ATTR_FRIENDLY_NAME, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State

from .exceptions import ValidationError

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Pre-compiled regex pattern for stripping thinking blocks from reasoning models
# Matches <think>...</think> blocks including newlines (DOTALL flag)
_THINKING_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking_blocks(text: str | None) -> str | None:
    """Remove <think>...</think> blocks from reasoning model output.

    Reasoning models (Qwen3, DeepSeek R1, o1/o3, etc.) output their reasoning
    process in <think>...</think> blocks. These should be filtered out before:
    - Displaying responses to users
    - Parsing JSON content (to avoid corruption)
    - Storing in conversation history

    Args:
        text: The text that may contain thinking blocks

    Returns:
        Text with thinking blocks removed and stripped of leading/trailing
        whitespace. Returns None if input is None.

    Examples:
        >>> strip_thinking_blocks("<think>Let me think...</think>Hello!")
        'Hello!'
        >>> strip_thinking_blocks("Answer<think>reasoning</think>here")
        'Answerhere'
        >>> strip_thinking_blocks(None)
        None
    """
    if text is None:
        return None
    if not text:
        return ""

    # Remove all <think>...</think> blocks
    result = _THINKING_BLOCK_PATTERN.sub("", text)

    # Strip leading/trailing whitespace
    return result.strip()


async def retry_async(
    func: Callable[[], Any],
    max_retries: int = 3,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    non_retryable_exceptions: tuple[type[Exception], ...] = (),
) -> Any:
    """Retry an async function on transient failures.

    Args:
        func: Async callable to retry (no arguments)
        max_retries: Maximum number of retry attempts (default: 3)
        retryable_exceptions: Tuple of exception types to catch and retry
        non_retryable_exceptions: Tuple of exception types to never retry

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail, or immediately if non-retryable

    Example:
        async def make_request():
            async with session.get(url) as resp:
                return await resp.json()

        result = await retry_async(
            make_request,
            max_retries=3,
            retryable_exceptions=(aiohttp.ClientError, asyncio.TimeoutError),
        )
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except non_retryable_exceptions:
            # Don't retry these exceptions
            raise
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Not the last attempt, log and retry
                delay = 1.0  # Simple fixed delay
                _LOGGER.debug(
                    "Attempt %d/%d failed: %s, retrying in %ss",
                    attempt + 1,
                    max_retries,
                    str(e),
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                # Last attempt failed
                _LOGGER.warning(
                    "All %d attempts failed: %s",
                    max_retries,
                    str(e),
                )

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry logic failed without capturing exception")


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
            f"Invalid format_type: {format_type}. Must be 'json' or 'natural_language'"
        )

    # Get attributes to include
    if attributes is None:
        # Include all non-internal attributes
        attrs = {k: v for k, v in state.attributes.items() if not k.startswith("_")}
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
        raise ValidationError(f"Invalid entity_id: {entity_id}. Must be a non-empty string.")

    # Check for basic format: domain.entity_name
    if "." not in entity_id:
        raise ValidationError(
            f"Invalid entity_id format: {entity_id}. " f"Expected format: domain.entity_name"
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

    return str(state.state)


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
    if seconds == 0:
        return "0s"

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
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


async def check_chromadb_health(host: str, port: int, timeout: int = 5) -> tuple[bool, str]:
    """Check if ChromaDB is reachable and healthy.

    Attempts to connect to ChromaDB's heartbeat endpoint to verify
    the service is running and responsive.

    Args:
        host: ChromaDB host address
        port: ChromaDB port number
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_healthy: bool, message: str)
        - (True, "ChromaDB healthy") on success
        - (False, "error description") on failure

    Example:
        healthy, msg = await check_chromadb_health("localhost", 8000)
        if not healthy:
            _LOGGER.warning("ChromaDB health check failed: %s", msg)
    """
    endpoints = [
        f"http://{host}:{port}/api/v2/heartbeat",
        f"http://{host}:{port}/api/v1/heartbeat",
    ]

    try:
        async with aiohttp.ClientSession() as session:
            for url in endpoints:
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as response:
                        if response.status == 200:
                            return True, f"ChromaDB healthy at {host}:{port}"
                except aiohttp.ClientError:
                    continue

            return False, f"ChromaDB not responding at {host}:{port}"

    except Exception as err:
        return False, f"ChromaDB health check error: {err}"


async def check_ollama_health(base_url: str, timeout: int = 5) -> tuple[bool, str]:
    """Check if Ollama embedding service is reachable.

    Attempts to connect to Ollama's API endpoint to verify
    the service is running and responsive.

    Args:
        base_url: Ollama base URL (e.g., "http://localhost:11434")
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_healthy: bool, message: str)
        - (True, "Ollama healthy") on success
        - (False, "error description") on failure

    Example:
        healthy, msg = await check_ollama_health("http://localhost:11434")
        if not healthy:
            _LOGGER.warning("Ollama health check failed: %s", msg)
    """
    endpoints = ["/api/tags", "/api/version", ""]

    try:
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                url = f"{base_url.rstrip('/')}{endpoint}"
                try:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as response:
                        if response.status == 200:
                            return True, f"Ollama healthy at {base_url}"
                except aiohttp.ClientError:
                    continue

            return False, f"Ollama not responding at {base_url}"

    except Exception as err:
        return False, f"Ollama health check error: {err}"

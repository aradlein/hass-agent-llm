"""Lightweight validation helpers for the MCP server configuration."""

from __future__ import annotations

import logging
from typing import Any

from ..const import (
    CONF_MCP_HEADERS,
    CONF_MCP_NAME,
    CONF_MCP_TYPE,
    CONF_MCP_URL,
    DEFAULT_MCP_TYPE,
    MCP_TYPE_HTTP,
    MCP_TYPE_SSE,
)
from ..exceptions import ValidationError

_LOGGER = logging.getLogger(__name__)

SUPPORTED_MCP_TYPES = {MCP_TYPE_HTTP, MCP_TYPE_SSE}


def validate_mcp_server_config(server: Any) -> dict[str, Any]:
    """Validate and normalize a single MCP server configuration entry.

    Args:
        server: Raw configuration value from the YAML ``mcp_servers`` list.

    Returns:
        A normalized dictionary with ``name``, ``type``, ``url``, and
        optionally ``headers`` and ``timeout`` keys.

    Raises:
        ValidationError: If the entry is missing required fields or has an
            unsupported transport type.
    """
    if not isinstance(server, dict):
        raise ValidationError(f"MCP server entry must be a mapping, got {type(server).__name__}")

    name = server.get(CONF_MCP_NAME)
    if not name or not isinstance(name, str):
        raise ValidationError("MCP server entry must have a non-empty 'name' string")

    server_type = server.get(CONF_MCP_TYPE, DEFAULT_MCP_TYPE)
    if isinstance(server_type, str):
        server_type = server_type.lower().strip()
    if server_type not in SUPPORTED_MCP_TYPES:
        raise ValidationError(
            f"MCP server '{name}' type must be one of {sorted(SUPPORTED_MCP_TYPES)}, "
            f"got '{server_type}'"
        )

    url = server.get(CONF_MCP_URL)
    if not url or not isinstance(url, str):
        raise ValidationError(f"MCP server '{name}' must have a non-empty 'url' string")

    headers = server.get(CONF_MCP_HEADERS)
    if headers is not None and not isinstance(headers, dict):
        raise ValidationError(
            f"MCP server '{name}' 'headers' must be a mapping, got {type(headers).__name__}"
        )

    timeout = server.get("timeout", None)
    if timeout is not None:
        try:
            timeout = int(timeout)
        except (TypeError, ValueError) as err:
            raise ValidationError(f"MCP server '{name}' 'timeout' must be an integer") from err

    return {
        "name": name,
        "type": server_type,
        "url": url,
        "headers": headers or {},
        "timeout": timeout,
    }


def validate_mcp_servers_config(servers: Any) -> list[dict[str, Any]]:
    """Validate a list of MCP server configurations.

    Invalid entries are logged and skipped rather than raising, so that a
    single bad server does not prevent the integration from starting.

    Args:
        servers: Raw ``mcp_servers`` configuration value.

    Returns:
        A list of validated and normalized server dictionaries.
    """
    if not servers:
        return []

    if not isinstance(servers, list):
        _LOGGER.warning("MCP server configuration must be a list, got %s", type(servers).__name__)
        return []

    valid: list[dict[str, Any]] = []
    for server in servers:
        try:
            valid.append(validate_mcp_server_config(server))
        except ValidationError as err:
            _LOGGER.warning("Skipping invalid MCP server entry: %s", err)

    return valid

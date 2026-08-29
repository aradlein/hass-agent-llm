"""Manager that discovers and instantiates tools from configured MCP servers."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant

from ..const import CONF_MCP_SERVERS, DEFAULT_MCP_TIMEOUT
from .client import McpServerClient
from .tool import MCPTool
from .validation import validate_mcp_servers_config

_LOGGER = logging.getLogger(__name__)


def _normalize_json_schema(parameters: Any) -> dict[str, Any]:
    """Ensure a parameter definition is a valid OpenAI-compatible JSON Schema.

    Args:
        parameters: The raw parameters description returned by the MCP server.

    Returns:
        A JSON Schema object.  If the server returned an invalid or empty
        value, a permissive default schema is returned.
    """
    if isinstance(parameters, dict) and parameters.get("type") == "object":
        return parameters

    if isinstance(parameters, dict):
        return {
            "type": "object",
            "properties": parameters,
            "additionalProperties": True,
        }

    return {"type": "object", "properties": {}, "additionalProperties": True}


class McpServerManager:
    """Discovers and loads MCP tools from the YAML configuration.

    This class is intentionally decoupled from ``CustomToolHandler`` and the
    existing custom tools framework.  It only reads from ``CONF_MCP_SERVERS``,
    talks to the configured remote servers, and emits ``MCPTool`` instances.
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the MCP server manager.

        Args:
            hass: Home Assistant instance.
            config: Runtime configuration dictionary (data merged with options).
        """
        self.hass = hass
        self.config = config
        self._tools: list[MCPTool] = []

    async def load_tools(self) -> list[MCPTool]:
        """Load and return all tools advertised by configured MCP servers.

        Invalid or unreachable servers are logged and skipped; a failure in
        one server does not prevent other servers from being loaded.

        Returns:
            A list of ``MCPTool`` instances ready for registration.
        """
        raw_servers = self.config.get(CONF_MCP_SERVERS, [])
        servers = validate_mcp_servers_config(raw_servers)
        if not servers:
            return []

        loaded: list[MCPTool] = []
        for server in servers:
            try:
                tools = await self._load_server_tools(server)
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.warning("Failed to load MCP server %s: %s", server.get("name"), err)
                continue

            loaded.extend(tools)

        _LOGGER.info("Loaded %d MCP tool(s) from %d server(s)", len(loaded), len(servers))
        self._tools = loaded
        return loaded

    async def _load_server_tools(self, server: dict[str, Any]) -> list[MCPTool]:
        """Discover tools from a validated server configuration."""
        name = server["name"]
        server_type = server["type"]
        url = server["url"]
        headers = server["headers"]
        timeout = server.get("timeout") or DEFAULT_MCP_TIMEOUT

        client = McpServerClient(
            self.hass,
            server_name=name,
            server_type=server_type,
            base_url=url,
            headers=headers,
            timeout=timeout,
        )

        raw_tools = await client.list_tools()
        tools: list[MCPTool] = []
        for raw in raw_tools:
            if not isinstance(raw, dict) or "name" not in raw:
                _LOGGER.warning("Skipping invalid tool definition from MCP server '%s'", name)
                continue

            mcp_name = raw["name"]
            description = raw.get("description", "") or ""
            parameters = _normalize_json_schema(raw.get("parameters"))

            tools.append(
                MCPTool(
                    self.hass,
                    client,
                    server_name=name,
                    mcp_name=mcp_name,
                    description=description,
                    parameters=parameters,
                )
            )

        _LOGGER.info("MCP server '%s' (%s) provided %d tool(s)", name, server_type, len(tools))
        return tools

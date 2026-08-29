"""MCP tool wrapper exposing remote server tools to the Home Agent tool registry."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.core import HomeAssistant

from ..exceptions import ToolExecutionError
from ..tools.registry import BaseTool
from .client import McpServerClient

_LOGGER = logging.getLogger(__name__)


class MCPTool(BaseTool):
    """Adapter that wraps a remote MCP server tool as a Home Agent tool.

    The tool name is namespaced with the server name to avoid collisions with
    native or custom tools (e.g. ``github__search_issues``).
    """

    def __init__(
        self,
        hass: HomeAssistant,
        client: McpServerClient,
        server_name: str,
        mcp_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Initialize the MCP tool wrapper.

        Args:
            hass: Home Assistant instance.
            client: The MCP server client used to invoke the tool.
            server_name: The configured name of the MCP server.
            mcp_name: The original tool name as advertised by the MCP server.
            description: The tool description.
            parameters: JSON Schema describing the tool parameters.
        """
        super().__init__(hass)
        self._client = client
        self._server_name = server_name
        self._mcp_name = mcp_name
        self._description = description
        self._parameters = parameters

    @property
    def name(self) -> str:
        """Return the namespaced tool name."""
        return f"{self._server_name}__{self._mcp_name}"

    @property
    def description(self) -> str:
        """Return the tool description."""
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return self._parameters

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the remote MCP tool and normalize the result.

        Args:
            **kwargs: The parameters to pass to the remote tool.

        Returns:
            A dictionary with at minimum a ``success`` key and either a
            ``result`` or ``error`` field, matching the format expected by
            ``ToolHandler.execute_tool``.
        """
        try:
            raw = await self._client.call_tool(self._mcp_name, kwargs)
        except ToolExecutionError as err:
            _LOGGER.warning(
                "MCP tool '%s' on server '%s' failed: %s",
                self._mcp_name,
                self._server_name,
                err,
            )
            return {"success": False, "result": None, "error": str(err)}

        if not isinstance(raw, dict):
            return {"success": True, "result": raw}

        if "success" not in raw:
            return {"success": True, "result": raw}

        if raw["success"]:
            return {"success": True, "result": raw.get("result")}

        error = raw.get("error") or raw.get("message") or "MCP tool reported failure"
        return {"success": False, "result": None, "error": error}

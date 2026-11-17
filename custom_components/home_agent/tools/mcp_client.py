"""Dynamic MCP tool wrappers for Home Agent.

The MCP manager discovers tools from the MCP server and builds instances of
MCPDiscoveredTool so the Home Agent can register them like native BaseTool
implementations.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from ..exceptions import ValidationError
from .registry import BaseTool


class MCPDiscoveredTool(BaseTool):
    """BaseTool wrapper around one discovered MCP tool definition."""

    def __init__(
        self,
        hass: HomeAssistant,
        manager: Any,
        tool_name: str,
        description: str,
        parameters: dict[str, Any] | None,
    ) -> None:
        """Initialize discovered tool metadata."""
        super().__init__(hass)
        self._manager = manager
        self._name = tool_name
        self._description = description or "Discovered MCP tool"
        self._parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @property
    def name(self) -> str:
        """Return tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Return tool description."""
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        """Return tool parameter schema."""
        return self._parameters

    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        """Validate required arguments declared by MCP input schema."""
        required = self._parameters.get("required", [])
        if isinstance(required, list):
            missing = [key for key in required if key not in parameters]
            if missing:
                raise ValidationError(
                    f"Missing required parameters for MCP tool '{self._name}': {', '.join(missing)}"
                )

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute this discovered MCP tool through the shared manager."""
        if self._manager is None:
            raise ValidationError("MCP manager is not available")

        result = await self._manager.async_call_tool(self._name, kwargs)

        return {
            "success": True,
            "tool_name": self._name,
            "result": result,
        }

"""Thin adapter around the official MCP Python SDK Client.

This module intentionally stays small: discovery and invocation are delegated to
``mcp.Client``.  Only ``http`` (Streamable HTTP) and ``sse`` transports are
supported; ``stdio`` is explicitly out of scope.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from homeassistant.core import HomeAssistant

from ..const import DEFAULT_MCP_TIMEOUT, MCP_TYPE_HTTP, MCP_TYPE_SSE
from ..exceptions import ToolExecutionError

_LOGGER = logging.getLogger(__name__)


def _http_client_for_server(
    url: str,
    headers: dict[str, str] | None,
    timeout: float,
) -> httpx2.AsyncClient:
    """Create a ``httpx2.AsyncClient`` configured for a Streamable HTTP server.

    ``httpx2`` is the HTTP client bundled with the official MCP SDK.
    """
    return httpx2.AsyncClient(
        headers=headers or {},
        timeout=httpx2.Timeout(timeout, read=timeout * 10),
        follow_redirects=True,
    )


class McpServerClient:
    """Adapter that wraps ``mcp.Client`` for a single remote MCP server."""

    def __init__(
        self,
        hass: HomeAssistant,
        server_name: str,
        server_type: str,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = DEFAULT_MCP_TIMEOUT,
    ) -> None:
        """Initialize the MCP server client.

        Args:
            hass: Home Assistant instance.
            server_name: User-defined name for this server (used for namespacing).
            server_type: Transport type, either ``http`` or ``sse``.
            base_url: Base URL of the MCP server.
            headers: Optional static HTTP headers (e.g. Authorization).
            timeout: Request timeout in seconds.

        Raises:
            ToolExecutionError: If the server type is not supported.
        """
        self.hass = hass
        self.server_name = server_name
        self.server_type = server_type.lower().strip()
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout

        if self.server_type not in {MCP_TYPE_HTTP, MCP_TYPE_SSE}:
            raise ToolExecutionError(
                f"MCP server '{server_name}' has unsupported type '{server_type}'. "
                f"Only '{MCP_TYPE_HTTP}' and '{MCP_TYPE_SSE}' are supported."
            )

    @asynccontextmanager
    async def _client(self):
        """Enter and exit the official MCP ``Client`` for the configured server."""
        from mcp import Client

        if self.server_type == MCP_TYPE_HTTP:
            http_client = _http_client_for_server(self.base_url, self.headers, self.timeout)
            try:
                async with http_client:
                    from mcp.client.streamable_http import streamable_http_client

                    transport = streamable_http_client(self.base_url, http_client=http_client)
                    async with Client(transport) as client:
                        yield client
            except Exception:
                await http_client.aclose()
                raise
        else:
            from mcp.client.sse import sse_client

            # SSE client has native headers/timeout support.
            transport = sse_client(
                self.base_url,
                headers=self.headers,
                timeout=float(self.timeout),
                sse_read_timeout=float(self.timeout),
            )
            async with Client(transport) as client:
                yield client

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover the tools advertised by the MCP server.

        Returns:
            A list of tool definitions.  Each item has ``name``, ``description``
            and ``parameters`` keys, where ``parameters`` is a JSON Schema object.

        Raises:
            ToolExecutionError: If discovery fails.
        """
        _LOGGER.debug("Listing MCP tools for server '%s'", self.server_name)
        try:
            async with self._client() as client:
                result = await client.list_tools()
        except Exception as err:  # pylint: disable=broad-except
            raise ToolExecutionError(
                f"Failed to list tools from MCP server '{self.server_name}': {err}"
            ) from err

        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            }
            for tool in result.tools
        ]

        _LOGGER.debug("Discovered %d tool(s) from MCP server '%s'", len(tools), self.server_name)
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Invoke a tool on the remote MCP server.

        Args:
            tool_name: The un-prefixed tool name as advertised by the server.
            arguments: The arguments to pass to the tool.

        Returns:
            A normalized dictionary with ``success`` and either ``result`` or
            ``error`` keys, matching the format expected by ``ToolHandler``.

        Raises:
            ToolExecutionError: If the request fails at the transport/JSON-RPC
                layer.  Tool-level failures are returned as ``success: False``.
        """
        _LOGGER.debug(
            "Calling MCP tool '%s' on server '%s' with arguments %s",
            tool_name,
            self.server_name,
            arguments,
        )

        try:
            async with self._client() as client:
                result = await client.call_tool(tool_name, arguments or {})
        except Exception as err:  # pylint: disable=broad-except
            raise ToolExecutionError(
                f"MCP server '{self.server_name}' tool '{tool_name}' failed: {err}"
            ) from err

        if result.structured_content is not None:
            return {"success": True, "result": result.structured_content}

        text = [block.text for block in (result.content or []) if block.type == "text"]
        text = " ".join(text) if text else "MCP tool returned no usable content"

        if result.is_error:
            return {"success": False, "result": None, "error": text}

        # Fall back for non-error, non-structured content (e.g. plain text result).
        return {"success": True, "result": text}

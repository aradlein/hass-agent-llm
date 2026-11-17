"""MCP client manager for shared transport lifecycle.

This module centralizes MCP transport setup/teardown so tools can reuse
one long-lived client/session instead of reconnecting on every call.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from contextlib import suppress
from typing import Any

try:
    import httpx2
except ImportError:  # pragma: no cover - runtime dependency guard
    httpx2 = None  # type: ignore[assignment]
from homeassistant.core import HomeAssistant

from .const import (
    CONF_MCP_AUTH_HEADER,
    CONF_MCP_AUTH_TYPE,
    CONF_MCP_AUTH_VALUE,
    CONF_MCP_HEADERS,
    CONF_MCP_TOOL_ALLOWLIST,
    CONF_MCP_TRANSPORT,
    CONF_MCP_URL,
    DEFAULT_MCP_AUTH_HEADER,
    DEFAULT_MCP_AUTH_TYPE,
    DEFAULT_MCP_TOOL_ALLOWLIST,
    DEFAULT_MCP_TRANSPORT,
)
from .exceptions import ToolExecutionError, ValidationError
from .tool_handler import ToolHandler
from .tools.mcp_client import MCPDiscoveredTool
from .tools.registry import BaseTool

try:
    from mcp import ClientSession
except ImportError:  # pragma: no cover - runtime dependency guard
    ClientSession = None  # type: ignore[assignment]

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover - fallback for older SDK layouts
    try:
        from mcp.client.streamable_http import streamablehttp_client as streamable_http_client
    except ImportError:
        streamable_http_client = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)
_MAX_MCP_DESCRIPTION_CHARS = 220


class MCPClientManager:
    """Shared MCP client lifecycle manager.

    The manager is initialized once in async_setup_entry and shut down once
    in async_unload_entry.
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the manager from integration config."""
        self.hass = hass
        self.config = config

        self.transport = str(config.get(CONF_MCP_TRANSPORT, DEFAULT_MCP_TRANSPORT)).lower()
        self.url = str(config.get(CONF_MCP_URL, "")).strip()
        self.auth_type = str(config.get(CONF_MCP_AUTH_TYPE, DEFAULT_MCP_AUTH_TYPE)).lower()
        self.auth_value = str(config.get(CONF_MCP_AUTH_VALUE, ""))
        self.auth_header = str(config.get(CONF_MCP_AUTH_HEADER, DEFAULT_MCP_AUTH_HEADER)).strip()

        raw_headers = config.get(CONF_MCP_HEADERS, {})
        self.headers = raw_headers if isinstance(raw_headers, dict) else {}

        raw_allowlist = config.get(CONF_MCP_TOOL_ALLOWLIST, DEFAULT_MCP_TOOL_ALLOWLIST)
        if isinstance(raw_allowlist, str):
            allowlist_values = [item.strip() for item in raw_allowlist.split(",") if item.strip()]
        elif isinstance(raw_allowlist, list):
            allowlist_values = [str(item).strip() for item in raw_allowlist if str(item).strip()]
        else:
            allowlist_values = list(DEFAULT_MCP_TOOL_ALLOWLIST)
        self._tool_allowlist = set(allowlist_values)

        self._transport_cm: Any = None
        self._transport_ctx: Any = None
        self._http_client_cm: Any = None
        self._http_client: Any = None
        self._client_session_cm: Any = None
        self._client_session: Any = None
        self._discovered_tools: list[BaseTool] = []
        self._registered_tool_names: set[str] = set()
        self._connected = False
        self._last_discovery_error: str | None = None

    async def async_setup(self) -> None:
        """Create reusable SDK transport/session and discover available tools."""
        if ClientSession is None or streamable_http_client is None:
            raise ValidationError(
                "The 'mcp' Python SDK is required for MCP support. "
                "Add it to integration requirements and restart Home Assistant."
            )

        # Accept legacy aliases but normalize to streamable_http.
        if self.transport == "http":
            self.transport = "streamable_http"

        if self.transport != "streamable_http":
            raise ValidationError(
                "Unsupported MCP transport. This integration currently supports only "
                "'streamable_http' (JSON-RPC 2.0 over Streamable HTTP)."
            )

        if not self.url:
            raise ValidationError("MCP URL is required for streamable_http transport")

        request_headers = self._build_headers()

        try:
            self._transport_cm = await self._create_transport_context(request_headers)
            self._transport_ctx = await self._transport_cm.__aenter__()

            read_stream, write_stream = self._extract_stream_pair(self._transport_ctx)

            self._client_session_cm = ClientSession(read_stream, write_stream)
            self._client_session = await self._client_session_cm.__aenter__()

            # Initialize MCP session handshake if available in installed SDK version.
            initialize = getattr(self._client_session, "initialize", None)
            if callable(initialize):
                await initialize()

            self._connected = True

            try:
                await self._discover_tools()
                self._last_discovery_error = None
            except Exception as err:
                # Keep manager connected so status service can report discovery issues.
                self._last_discovery_error = str(err)
                _LOGGER.error("MCP tool discovery failed: %s", err, exc_info=True)
        except Exception as err:
            await self.async_shutdown()
            raise ToolExecutionError(f"Failed to initialize MCP manager: {err}") from err

        _LOGGER.info(
            "MCP client manager initialized (transport=%s, discovered_tools=%d)",
            self.transport,
            len(self._discovered_tools),
        )

    async def async_shutdown(self) -> None:
        """Close transport resources."""
        self._discovered_tools = []
        self._registered_tool_names = set()
        self._connected = False

        if self._client_session_cm is not None:
            with suppress(Exception):
                await self._client_session_cm.__aexit__(None, None, None)

        if self._transport_cm is not None:
            with suppress(Exception):
                await self._transport_cm.__aexit__(None, None, None)

        if self._http_client_cm is not None:
            with suppress(Exception):
                await self._http_client_cm.__aexit__(None, None, None)

        self._client_session = None
        self._client_session_cm = None
        self._transport_ctx = None
        self._transport_cm = None
        self._http_client = None
        self._http_client_cm = None

        _LOGGER.info("MCP client manager shut down")

    def register_discovered_tools(self, tool_handler: ToolHandler) -> int:
        """Register all discovered MCP tools in the active ToolHandler."""
        self._registered_tool_names = set()
        count = 0
        for tool in self._discovered_tools:
            tool_handler.register_tool(tool)
            self._registered_tool_names.add(tool.name)
            count += 1

        return count

    async def async_refresh_tools(self, tool_handler: ToolHandler | None = None) -> dict[str, Any]:
        """Re-run MCP discovery and optionally refresh ToolHandler registrations."""
        if self._client_session is None:
            raise ToolExecutionError("MCP manager is not initialized")

        try:
            await self._discover_tools()
            self._last_discovery_error = None
        except Exception as err:
            self._last_discovery_error = str(err)
            raise ToolExecutionError(f"MCP discovery refresh failed: {err}") from err

        if tool_handler is not None:
            # Remove previously registered MCP tools before re-registering current discovery.
            for old_name in list(self._registered_tool_names):
                tool_handler.unregister_tool(old_name)

            self.register_discovered_tools(tool_handler)

        return self.get_status()

    def get_discovered_tools(self) -> list[BaseTool]:
        """Return a copy of discovered MCP tool wrappers."""
        return list(self._discovered_tools)

    def is_connected(self) -> bool:
        """Return whether the MCP session is currently connected."""
        return self._connected and self._client_session is not None

    def get_tool_names(self) -> list[str]:
        """Return discovered MCP tool names."""
        return [tool.name for tool in self._discovered_tools]

    def get_last_discovery_error(self) -> str | None:
        """Return last discovery error, if any."""
        return self._last_discovery_error

    def get_status(self) -> dict[str, Any]:
        """Return current manager status for diagnostics services."""
        return {
            "connected": self.is_connected(),
            "discovered_tools": self.get_tool_names(),
            "last_discovery_error": self.get_last_discovery_error(),
        }

    async def async_call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one MCP tool through the connected SDK session."""
        if not tool_name:
            raise ValidationError("MCP tool_name cannot be empty")

        if not isinstance(arguments, dict):
            raise ValidationError("MCP arguments must be an object")

        if self._client_session is None:
            raise ToolExecutionError("MCP manager is not initialized")

        try:
            call_tool = getattr(self._client_session, "call_tool")
            raw_result = await call_tool(tool_name, arguments)
            return self._to_jsonable(raw_result)
        except Exception as err:
            raise ToolExecutionError(f"MCP tool call failed for '{tool_name}': {err}") from err

    async def _discover_tools(self) -> None:
        """Fetch tools/list and build dynamic BaseTool wrappers."""
        if self._client_session is None:
            raise ToolExecutionError("MCP session is not initialized")

        list_tools = getattr(self._client_session, "list_tools", None)
        if not callable(list_tools):
            raise ToolExecutionError("Installed MCP SDK does not expose list_tools()")

        raw_tools = await list_tools()
        tool_defs = self._extract_tool_defs(raw_tools)

        discovered: list[BaseTool] = []
        skipped_by_allowlist = 0
        for tool_def in tool_defs:
            name = str(tool_def.get("name", "")).strip()
            if not name:
                continue

            if self._tool_allowlist and name not in self._tool_allowlist:
                skipped_by_allowlist += 1
                continue

            description = str(tool_def.get("description", "")).strip()
            if len(description) > _MAX_MCP_DESCRIPTION_CHARS:
                description = f"{description[:_MAX_MCP_DESCRIPTION_CHARS - 1]}..."
            schema = tool_def.get("inputSchema") or tool_def.get("input_schema")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}, "required": []}

            discovered.append(
                MCPDiscoveredTool(
                    hass=self.hass,
                    manager=self,
                    tool_name=name,
                    description=description,
                    parameters=schema,
                )
            )

        self._discovered_tools = discovered
        _LOGGER.info(
            "MCP discovery filtered tools (allowed=%d, discovered=%d, skipped=%d)",
            len(self._tool_allowlist),
            len(discovered),
            skipped_by_allowlist,
        )

    def _extract_stream_pair(self, transport_ctx: Any) -> tuple[Any, Any]:
        """Extract read/write streams from SDK transport context."""
        if isinstance(transport_ctx, tuple) and len(transport_ctx) >= 2:
            return transport_ctx[0], transport_ctx[1]

        read_stream = getattr(transport_ctx, "read_stream", None)
        write_stream = getattr(transport_ctx, "write_stream", None)
        if read_stream is not None and write_stream is not None:
            return read_stream, write_stream

        raise ToolExecutionError("Unable to extract read/write streams from MCP transport")

    def _extract_tool_defs(self, raw_tools: Any) -> list[dict[str, Any]]:
        """Normalize SDK list_tools response into plain dictionaries."""
        if raw_tools is None:
            return []

        if isinstance(raw_tools, dict) and isinstance(raw_tools.get("tools"), list):
            return [self._to_jsonable(item) for item in raw_tools["tools"]]

        tools_attr = getattr(raw_tools, "tools", None)
        if isinstance(tools_attr, list):
            return [self._to_jsonable(item) for item in tools_attr]

        if isinstance(raw_tools, list):
            return [self._to_jsonable(item) for item in raw_tools]

        return []

    def _to_jsonable(self, value: Any) -> Any:
        """Convert SDK/Pydantic objects into JSON-serializable primitives."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]

        if dataclasses.is_dataclass(value):
            return self._to_jsonable(dataclasses.asdict(value))

        if hasattr(value, "model_dump") and callable(value.model_dump):
            return self._to_jsonable(value.model_dump())

        if hasattr(value, "dict") and callable(value.dict):
            return self._to_jsonable(value.dict())

        if hasattr(value, "__dict__"):
            return self._to_jsonable(value.__dict__)

        return str(value)

    def _build_headers(self) -> dict[str, str]:
        """Build transport headers including optional auth."""
        headers = {str(k): str(v) for k, v in self.headers.items()}

        if self.auth_type == "bearer" and self.auth_value:
            headers[self.auth_header or "Authorization"] = f"Bearer {self.auth_value}"
        elif self.auth_type == "api_key" and self.auth_value:
            headers[self.auth_header or "X-API-Key"] = self.auth_value

        return headers

    async def _create_transport_context(self, request_headers: dict[str, str]) -> Any:
        """Create streamable transport context manager for installed SDK variant."""
        assert streamable_http_client is not None

        params = inspect.signature(streamable_http_client).parameters

        # MCP SDK 2.x: headers removed; pass custom headers via http_client.
        if "http_client" in params:
            if httpx2 is None:
                raise ValidationError(
                    "MCP SDK requires 'httpx2' for streamable_http transport in this environment."
                )

            self._http_client_cm = httpx2.AsyncClient(
                headers=request_headers,
                timeout=httpx2.Timeout(30.0, read=300.0),
            )
            self._http_client = await self._http_client_cm.__aenter__()
            return streamable_http_client(self.url, http_client=self._http_client)

        # Legacy MCP SDK: headers passed directly to streamable_http_client.
        if "headers" in params:
            return streamable_http_client(self.url, headers=request_headers)

        # Last-resort compatibility: no known knobs exposed.
        _LOGGER.warning(
            "Unknown streamable_http_client signature (%s). Falling back to URL-only call.",
            ", ".join(params.keys()),
        )
        return streamable_http_client(self.url)

"""Unit tests for MCP server support.

These tests target the SDK-based ``McpServerClient``.  The client is an async
context manager around ``mcp.Client``, so unit tests patch at the
``mcp.Client`` boundary rather than making real network calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.const import (
    CONF_MCP_SERVERS,
    DEFAULT_MCP_TIMEOUT,
    MCP_TYPE_HTTP,
)
from custom_components.home_agent.exceptions import ToolExecutionError
from custom_components.home_agent.mcp.client import McpServerClient
from custom_components.home_agent.mcp.manager import McpServerManager, _normalize_json_schema
from custom_components.home_agent.mcp.tool import MCPTool
from custom_components.home_agent.mcp.validation import (
    validate_mcp_server_config,
    validate_mcp_servers_config,
)


class TestNormalizeJsonSchema:
    """Tests for the JSON schema normalizer."""

    def test_keeps_valid_object_schema(self):
        """A valid object schema is returned as-is."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        assert _normalize_json_schema(schema) is schema

    def test_wraps_plain_properties(self):
        """A dict without a type is wrapped as an object schema."""
        schema = {"query": {"type": "string"}}
        result = _normalize_json_schema(schema)
        assert result["type"] == "object"
        assert result["properties"] == schema
        assert result["additionalProperties"] is True

    def test_uses_default_for_non_dict(self):
        """A non-dict value is replaced with a permissive default."""
        result = _normalize_json_schema(None)
        assert result == {"type": "object", "properties": {}, "additionalProperties": True}


class TestValidateMcpServerConfig:
    """Tests for single MCP server config validation."""

    def test_valid_http_config(self):
        """A minimal valid HTTP server is accepted."""
        result = validate_mcp_server_config("github", {"type": "http", "url": "http://example.com"})
        assert result == {
            "name": "github",
            "type": "http",
            "url": "http://example.com",
            "headers": {},
        }

    def test_defaults_type_and_headers(self):
        """Missing type and headers are filled with defaults."""
        result = validate_mcp_server_config("local", {"url": "http://localhost:3000"})
        assert result["type"] == MCP_TYPE_HTTP
        assert result["headers"] == {}

    def test_invalid_type_raises(self):
        """An unsupported transport type raises ValidationError."""
        from custom_components.home_agent.exceptions import ValidationError

        with pytest.raises(ValidationError):
            validate_mcp_server_config("bad", {"type": "stdio", "url": "http://localhost"})

    def test_missing_url_raises(self):
        """A server without a URL raises ValidationError."""
        from custom_components.home_agent.exceptions import ValidationError

        with pytest.raises(ValidationError):
            validate_mcp_server_config("bad", {"type": "http"})

    def test_invalid_headers_raise(self):
        """Non-dict headers raise ValidationError."""
        from custom_components.home_agent.exceptions import ValidationError

        with pytest.raises(ValidationError):
            validate_mcp_server_config(
                "bad", {"type": "http", "url": "http://localhost", "headers": "token"}
            )

    def test_timeout_is_preserved(self):
        """A custom timeout is preserved."""
        result = validate_mcp_server_config(
            "slow", {"type": "sse", "url": "http://localhost", "timeout": 120}
        )
        assert result["timeout"] == 120


class TestValidateMcpServersConfig:
    """Tests for the bulk validator."""

    def test_returns_valid_servers(self):
        """Only valid, validated servers are returned."""
        servers = [
            {"name": "ok", "type": "http", "url": "http://example.com"},
            {"name": "bad", "type": "stdio"},
        ]
        result = validate_mcp_servers_config(servers)
        assert len(result) == 1
        assert result[0]["name"] == "ok"

    def test_returns_empty_for_non_list(self):
        """Non-list input returns an empty list."""
        assert validate_mcp_servers_config(None) == []
        assert validate_mcp_servers_config("string") == []

    def test_skips_missing_name(self):
        """Servers without a name are skipped."""
        result = validate_mcp_servers_config([{"type": "http", "url": "http://example.com"}])
        assert result == []


class TestMcpServerClient:
    """Tests for the SDK-backed MCP client."""

    @patch("custom_components.home_agent.mcp.client.Client")
    @patch("custom_components.home_agent.mcp.client.streamable_http_client")
    @patch("custom_components.home_agent.mcp.client.httpx2.AsyncClient")
    def test_list_tools_uses_sdk(
        self,
        mock_httpx2,
        mock_transport_factory,
        mock_client_cls,
        hass,
    ):
        """list_tools delegates to ``mcp.Client.list_tools``."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools = AsyncMock(
            return_value=MagicMock(
                tools=[
                    MagicMock(
                        name="search",
                        description="Search issues",
                        input_schema={"type": "object", "properties": {}},
                    )
                ]
            )
        )
        mock_client.call_tool = AsyncMock()
        mock_client_cls.return_value = mock_client

        client = McpServerClient(
            hass,
            server_name="github",
            server_type=MCP_TYPE_HTTP,
            base_url="http://example.com/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=DEFAULT_MCP_TIMEOUT,
        )
        result = hass.loop.run_until_complete(client.list_tools())

        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search issues"
        assert result[0]["parameters"] == {"type": "object", "properties": {}}
        mock_client.list_tools.assert_awaited_once()

    @patch("custom_components.home_agent.mcp.client.Client")
    @patch("custom_components.home_agent.mcp.client.streamable_http_client")
    @patch("custom_components.home_agent.mcp.client.httpx2.AsyncClient")
    def test_call_tool_uses_sdk_structured(
        self,
        mock_httpx2,
        mock_transport_factory,
        mock_client_cls,
        hass,
    ):
        """call_tool returns structured content when available."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools = AsyncMock()
        mock_client.call_tool = AsyncMock(
            return_value=MagicMock(
                structured_content={"ok": True},
                content=None,
                is_error=False,
            )
        )
        mock_client_cls.return_value = mock_client

        client = McpServerClient(
            hass,
            server_name="github",
            server_type=MCP_TYPE_HTTP,
            base_url="http://example.com/mcp",
            headers={},
            timeout=DEFAULT_MCP_TIMEOUT,
        )
        result = hass.loop.run_until_complete(client.call_tool("search", {"query": "hass"}))

        assert result == {"success": True, "result": {"ok": True}}
        mock_client.call_tool.assert_awaited_once_with("search", {"query": "hass"})

    @patch("custom_components.home_agent.mcp.client.Client")
    @patch("custom_components.home_agent.mcp.client.streamable_http_client")
    @patch("custom_components.home_agent.mcp.client.httpx2.AsyncClient")
    def test_call_tool_returns_text_content(
        self,
        mock_httpx2,
        mock_transport_factory,
        mock_client_cls,
        hass,
    ):
        """call_tool falls back to text content when not structured."""
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.list_tools = AsyncMock()
        mock_client.call_tool = AsyncMock(
            return_value=MagicMock(
                structured_content=None,
                content=[MagicMock(type="text", text="hello")],
                is_error=False,
            )
        )
        mock_client_cls.return_value = mock_client

        client = McpServerClient(
            hass,
            server_name="github",
            server_type=MCP_TYPE_HTTP,
            base_url="http://example.com/mcp",
            headers={},
            timeout=DEFAULT_MCP_TIMEOUT,
        )
        result = hass.loop.run_until_complete(client.call_tool("echo"))

        assert result == {"success": True, "result": "hello"}

    def test_unsupported_type_raises(self, hass):
        """An unsupported server type raises ToolExecutionError."""
        with pytest.raises(ToolExecutionError):
            McpServerClient(
                hass,
                server_name="bad",
                server_type="stdio",
                base_url="http://example.com",
                headers={},
                timeout=DEFAULT_MCP_TIMEOUT,
            )


class TestMCPTool:
    """Tests for the MCPTool wrapper."""

    def test_name_is_namespaced(self, hass):
        """The tool name is prefixed with the server name."""
        client = AsyncMock()
        tool = MCPTool(
            hass,
            client,
            server_name="github",
            mcp_name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "github__search"

    def test_execute_returns_success(self, hass):
        """A successful call is normalized."""
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value={"success": True, "result": "ok"})
        tool = MCPTool(
            hass,
            client,
            server_name="github",
            mcp_name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )
        result = hass.loop.run_until_complete(tool.execute(query="hass"))
        assert result == {"success": True, "result": "ok"}

    def test_execute_wraps_non_dict(self, hass):
        """Non-dict responses are wrapped as a result."""
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value="ok")
        tool = MCPTool(
            hass,
            client,
            server_name="github",
            mcp_name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )
        result = hass.loop.run_until_complete(tool.execute(query="hass"))
        assert result == {"success": True, "result": "ok"}

    def test_execute_propagates_tool_error(self, hass):
        """A failed call is propagated as a structured error."""
        client = AsyncMock()
        client.call_tool = AsyncMock(return_value={"success": False, "error": "nope"})
        tool = MCPTool(
            hass,
            client,
            server_name="github",
            mcp_name="search",
            description="Search",
            parameters={"type": "object", "properties": {}},
        )
        result = hass.loop.run_until_complete(tool.execute(query="hass"))
        assert result == {"success": False, "result": None, "error": "nope"}


class TestMcpServerManager:
    """Tests for the MCP server manager."""

    def test_loads_tools_from_single_server(self, hass):
        """Manager loads and wraps tools from configured servers."""
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(
            return_value=[{"name": "search", "description": "Search", "parameters": {}}]
        )

        config = {CONF_MCP_SERVERS: [{"name": "github", "url": "http://example.com"}]}
        manager = McpServerManager(hass, config)

        with patch(
            "custom_components.home_agent.mcp.manager.McpServerClient",
            return_value=mock_client,
        ):
            tools = hass.loop.run_until_complete(manager.load_tools())

        assert len(tools) == 1
        assert tools[0].name == "github__search"

    def test_skips_invalid_server_gracefully(self, hass):
        """A bad server does not block other servers from loading."""
        bad_client = AsyncMock()
        bad_client.list_tools = AsyncMock(side_effect=Exception("timeout"))

        good_client = AsyncMock()
        good_client.list_tools = AsyncMock(
            return_value=[{"name": "ok", "description": "OK", "parameters": {}}]
        )

        config = {
            CONF_MCP_SERVERS: [
                {"name": "bad", "url": "http://bad.example"},
                {"name": "good", "url": "http://good.example"},
            ]
        }
        manager = McpServerManager(hass, config)

        with patch(
            "custom_components.home_agent.mcp.manager.McpServerClient",
            side_effect=[bad_client, good_client],
        ):
            tools = hass.loop.run_until_complete(manager.load_tools())

        assert len(tools) == 1
        assert tools[0].name == "good__ok"

"""Unit tests for RestCustomTool execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.home_agent.exceptions import ValidationError
from custom_components.home_agent.tools.custom import RestCustomTool


class TestRestCustomToolExecution:
    """Test RestCustomTool execute method."""

    @pytest.fixture
    def simple_get_tool(self, mock_hass):
        """Create a simple GET REST tool."""
        config = {
            "name": "weather_api",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            "handler": {"type": "rest", "url": "https://api.weather.com/forecast", "method": "GET"},
        }
        return RestCustomTool(mock_hass, config)

    @pytest.fixture
    def tool_with_query_params(self, mock_hass):
        """Create a REST tool with query parameters."""
        config = {
            "name": "search_api",
            "description": "Search for items",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            "handler": {
                "type": "rest",
                "url": "https://api.example.com/search",
                "method": "GET",
                "query_params": {"q": "{{ query }}", "format": "json"},
            },
        }
        return RestCustomTool(mock_hass, config)

    @pytest.fixture
    def tool_with_headers(self, mock_hass):
        """Create a REST tool with headers."""
        config = {
            "name": "api_with_auth",
            "description": "API with authentication",
            "parameters": {"type": "object", "properties": {"data": {"type": "string"}}},
            "handler": {
                "type": "rest",
                "url": "https://api.example.com/data",
                "method": "GET",
                "headers": {
                    "Authorization": "Bearer test-token",
                    "Content-Type": "application/json",
                },
            },
        }
        return RestCustomTool(mock_hass, config)

    @pytest.fixture
    def post_tool_with_body(self, mock_hass):
        """Create a POST REST tool with body."""
        config = {
            "name": "create_item",
            "description": "Create an item",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
            },
            "handler": {
                "type": "rest",
                "url": "https://api.example.com/items",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {"name": "{{ name }}", "value": "{{ value }}"},
            },
        }
        return RestCustomTool(mock_hass, config)

    @pytest.mark.asyncio
    async def test_successful_get_request_json_response(self, simple_get_tool):
        """Test successful GET request with JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"temperature": 72, "condition": "sunny"})
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)

        with patch.object(simple_get_tool, "_ensure_session", return_value=mock_session):
            result = await simple_get_tool.execute(location="San Francisco")

        assert result["success"] is True
        assert result["result"] == {"temperature": 72, "condition": "sunny"}
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_successful_get_request_text_response(self, simple_get_tool):
        """Test successful GET request with text response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.json = AsyncMock(side_effect=ValueError("Not JSON"))
        mock_response.text = AsyncMock(return_value="Plain text response")
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)

        with patch.object(simple_get_tool, "_ensure_session", return_value=mock_session):
            result = await simple_get_tool.execute(location="San Francisco")

        assert result["success"] is True
        assert result["result"] == "Plain text response"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_request_with_query_params(self, tool_with_query_params, mock_hass):
        """Test GET request with query parameters."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"results": []})
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)

        # Mock template rendering
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(side_effect=lambda x: x.get("query", "json"))
            mock_template_class.return_value = mock_template

            with patch.object(tool_with_query_params, "_ensure_session", return_value=mock_session):
                result = await tool_with_query_params.execute(query="test search")

            # Verify request was made with correct params
            mock_session.request.assert_called_once()
            call_kwargs = mock_session.request.call_args[1]
            assert call_kwargs["params"]["format"] == "json"

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_request_with_headers(self, tool_with_headers):
        """Test request with custom headers."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"data": "test"})
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)

        with patch.object(tool_with_headers, "_ensure_session", return_value=mock_session):
            result = await tool_with_headers.execute(data="test")

        # Verify headers were passed
        call_kwargs = mock_session.request.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_post_request_with_body(self, post_tool_with_body, mock_hass):
        """Test POST request with request body."""
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"id": "123", "created": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=mock_response)

        # Mock template rendering
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                side_effect=lambda x: (
                    x.get("name", "test") if "name" in x else x.get("value", "123")
                )
            )
            mock_template_class.return_value = mock_template

            with patch.object(post_tool_with_body, "_ensure_session", return_value=mock_session):
                result = await post_tool_with_body.execute(name="test_item", value="test_value")

            # Verify POST method was used
            call_kwargs = mock_session.request.call_args[1]
            assert call_kwargs["method"] == "POST"
            assert "json" in call_kwargs
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_http_error_404(self, simple_get_tool):
        """Test handling of HTTP 404 error."""
        # Mock the _make_request method directly to raise the error
        with patch.object(
            simple_get_tool,
            "_make_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=404, message="Not Found"
            ),
        ):
            result = await simple_get_tool.execute(location="Unknown")

        assert result["success"] is False
        assert result["result"] is None
        assert "404" in result["error"]

    @pytest.mark.asyncio
    async def test_http_error_500(self, simple_get_tool):
        """Test handling of HTTP 500 error."""
        # Mock the _make_request method directly to raise the error
        with patch.object(
            simple_get_tool,
            "_make_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=500, message="Internal Server Error"
            ),
        ):
            result = await simple_get_tool.execute(location="Test")

        assert result["success"] is False
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_network_error(self, simple_get_tool):
        """Test handling of network errors."""
        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))

        with patch.object(simple_get_tool, "_ensure_session", return_value=mock_session):
            result = await simple_get_tool.execute(location="Test")

        assert result["success"] is False
        assert result["result"] is None
        assert "network error" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_timeout_error(self, simple_get_tool):
        """Test handling of timeout errors."""

        # Mock _make_request to take longer than timeout
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(10)

        # Set a short timeout in config
        simple_get_tool._config["tools_timeout"] = 0.1

        with patch.object(simple_get_tool, "_make_request", side_effect=slow_request):
            result = await simple_get_tool.execute(location="Test")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_template_rendering_error(self, tool_with_query_params):
        """Test handling of template rendering errors."""
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(side_effect=Exception("Template error"))
            mock_template_class.return_value = mock_template

            result = await tool_with_query_params.execute(query="test")

        assert result["success"] is False
        assert "error" in result["error"].lower()


class TestRestCustomToolTemplateRendering:
    """Test template rendering in RestCustomTool."""

    @pytest.fixture
    def tool_with_templates(self, mock_hass):
        """Create a tool with template values."""
        config = {
            "name": "templated_api",
            "description": "API with templates",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "units": {"type": "string"}},
            },
            "handler": {
                "type": "rest",
                "url": "https://api.example.com/{{ location }}/weather",
                "method": "GET",
                "headers": {"X-Units": "{{ units }}"},
                "query_params": {"location": "{{ location }}", "format": "json"},
            },
        }
        return RestCustomTool(mock_hass, config)

    @pytest.mark.asyncio
    async def test_render_template_success(self, tool_with_templates):
        """Test successful template rendering."""
        variables = {"location": "San Francisco", "units": "metric"}

        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(return_value="San Francisco")
            mock_template_class.return_value = mock_template

            result = await tool_with_templates._render_template("{{ location }}", variables)

        assert result == "San Francisco"

    @pytest.mark.asyncio
    async def test_render_template_no_template(self, tool_with_templates):
        """Test rendering string without template."""
        result = await tool_with_templates._render_template("static_string", {})

        assert result == "static_string"

    @pytest.mark.asyncio
    async def test_render_template_failure(self, tool_with_templates):
        """Test template rendering failure."""
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(side_effect=Exception("Render error"))
            mock_template_class.return_value = mock_template

            with pytest.raises(ValidationError) as exc_info:
                await tool_with_templates._render_template("{{ bad_var }}", {})

            assert "failed to render template" in str(exc_info.value).lower()

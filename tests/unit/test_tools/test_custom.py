"""Unit tests for CustomToolHandler."""

from unittest.mock import MagicMock

import pytest

from custom_components.home_agent.const import CUSTOM_TOOL_HANDLER_SERVICE
from custom_components.home_agent.exceptions import ValidationError
from custom_components.home_agent.tools.custom import (
    CustomToolHandler,
    RestCustomTool,
    ServiceCustomTool,
)


class TestCustomToolHandler:
    """Test the CustomToolHandler factory class."""

    def test_create_rest_tool_success(self, mock_hass):
        """Test successful creation of REST custom tool."""
        config = {
            "name": "check_weather",
            "description": "Get weather forecast",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
            "handler": {
                "type": "rest",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET",
            },
        }

        tool = CustomToolHandler.create_tool_from_config(mock_hass, config)

        assert isinstance(tool, RestCustomTool)
        assert tool.name == "check_weather"
        assert tool.description == "Get weather forecast"

    def test_create_tool_missing_required_keys(self, mock_hass):
        """Test that creation fails when required keys are missing."""
        # Missing 'description'
        config = {"name": "test_tool", "parameters": {}, "handler": {"type": "rest"}}

        with pytest.raises(ValidationError) as exc_info:
            CustomToolHandler.create_tool_from_config(mock_hass, config)

        assert "missing required keys" in str(exc_info.value).lower()
        assert "description" in str(exc_info.value).lower()

    def test_create_tool_missing_handler_type(self, mock_hass):
        """Test that creation fails when handler type is missing."""
        config = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {},
            "handler": {},  # Missing 'type'
        }

        with pytest.raises(ValidationError) as exc_info:
            CustomToolHandler.create_tool_from_config(mock_hass, config)

        assert "type" in str(exc_info.value).lower()

    def test_create_tool_unsupported_handler_type(self, mock_hass):
        """Test that creation fails for unsupported handler types."""
        config = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {},
            "handler": {"type": "unknown_handler"},
        }

        with pytest.raises(ValidationError) as exc_info:
            CustomToolHandler.create_tool_from_config(mock_hass, config)

        assert "unknown handler type" in str(exc_info.value).lower()

    def test_create_tool_service_handler(self, mock_hass):
        """Test that service handler creates ServiceCustomTool."""
        # Mock has_service to return True
        mock_hass.services.has_service = MagicMock(return_value=True)

        config = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {},
            "handler": {
                "type": CUSTOM_TOOL_HANDLER_SERVICE,
                "service": "light.turn_on",
            },
        }

        tool = CustomToolHandler.create_tool_from_config(mock_hass, config)

        assert isinstance(tool, ServiceCustomTool)
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"


class TestRestCustomToolInitialization:
    """Test RestCustomTool initialization."""

    def test_initialization_success(self, mock_hass):
        """Test successful initialization of REST custom tool."""
        config = {
            "name": "weather_api",
            "description": "Get weather data",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "handler": {"type": "rest", "url": "https://api.example.com/weather", "method": "GET"},
        }

        tool = RestCustomTool(mock_hass, config)

        assert tool.hass == mock_hass
        assert tool._config == config
        assert tool._handler_config == config["handler"]

    def test_initialization_missing_url(self, mock_hass):
        """Test initialization fails when URL is missing."""
        config = {
            "name": "test_tool",
            "description": "Test",
            "parameters": {},
            "handler": {
                "type": "rest",
                "method": "GET",
                # Missing 'url'
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            RestCustomTool(mock_hass, config)

        assert "missing required keys" in str(exc_info.value).lower()
        assert "url" in str(exc_info.value).lower()

    def test_initialization_missing_method(self, mock_hass):
        """Test initialization fails when HTTP method is missing."""
        config = {
            "name": "test_tool",
            "description": "Test",
            "parameters": {},
            "handler": {
                "type": "rest",
                "url": "https://api.example.com",
                # Missing 'method'
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            RestCustomTool(mock_hass, config)

        assert "missing required keys" in str(exc_info.value).lower()
        assert "method" in str(exc_info.value).lower()

    def test_initialization_invalid_method(self, mock_hass):
        """Test initialization fails with invalid HTTP method."""
        config = {
            "name": "test_tool",
            "description": "Test",
            "parameters": {},
            "handler": {"type": "rest", "url": "https://api.example.com", "method": "INVALID"},
        }

        with pytest.raises(ValidationError) as exc_info:
            RestCustomTool(mock_hass, config)

        assert "invalid http method" in str(exc_info.value).lower()


class TestRestCustomToolProperties:
    """Test RestCustomTool properties."""

    @pytest.fixture
    def rest_tool(self, mock_hass):
        """Create a REST custom tool for testing."""
        config = {
            "name": "weather_api",
            "description": "Get weather forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
            "handler": {
                "type": "rest",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET",
            },
        }
        return RestCustomTool(mock_hass, config)

    def test_tool_name(self, rest_tool):
        """Test that tool name is correct."""
        assert rest_tool.name == "weather_api"

    def test_tool_description(self, rest_tool):
        """Test that tool description is correct."""
        assert rest_tool.description == "Get weather forecast for a location"

    def test_tool_parameters(self, rest_tool):
        """Test that tool parameters are correct."""
        params = rest_tool.parameters

        assert params["type"] == "object"
        assert "location" in params["properties"]
        assert "location" in params["required"]

    def test_get_definition(self, rest_tool):
        """Test get_definition returns correct format."""
        definition = rest_tool.get_definition()

        assert definition["name"] == "weather_api"
        assert definition["description"] == "Get weather forecast for a location"
        assert "parameters" in definition

    def test_to_openai_format(self, rest_tool):
        """Test conversion to OpenAI format."""
        openai_format = rest_tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert "function" in openai_format
        assert openai_format["function"]["name"] == "weather_api"

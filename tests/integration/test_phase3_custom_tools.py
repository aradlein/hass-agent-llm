"""Integration tests for Phase 3: Custom Tool Framework.

This test suite validates the complete custom tool integration flow:
- Custom tool registration from configuration
- REST handler execution with real HTTP calls (mocked)
- Template rendering with parameters
- Error handling and validation
- Integration with Home Agent's tool system
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp

from homeassistant.core import HomeAssistant

from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_TOOLS_CUSTOM,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
)
from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.tools.custom import RestCustomTool, ServiceCustomTool


@pytest.fixture
def custom_tools_config():
    """Provide configuration with custom tools."""
    return {
        # Primary LLM config
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        # Tool configuration
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
        CONF_TOOLS_TIMEOUT: 30,
        # Custom tools configuration
        CONF_TOOLS_CUSTOM: [
            {
                "name": "check_weather",
                "description": "Get weather forecast for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location"
                        }
                    },
                    "required": ["location"]
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.weather.com/v1/forecast",
                    "method": "GET",
                    "headers": {
                        "Authorization": "Bearer test-token"
                    },
                    "query_params": {
                        "location": "{{ location }}",
                        "format": "json"
                    }
                }
            },
            {
                "name": "create_task",
                "description": "Create a new task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["title"]
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.tasks.com/v1/tasks",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "body": {
                        "title": "{{ title }}",
                        "description": "{{ description }}"
                    }
                }
            }
        ]
    }


@pytest.fixture
def mock_hass_for_custom_tools():
    """Create a mock Home Assistant instance for integration tests."""
    mock = MagicMock(spec=HomeAssistant)
    mock.data = {}

    # Mock states
    mock.states = MagicMock()
    mock.states.async_all = MagicMock(return_value=[])
    mock.states.get = MagicMock(return_value=None)

    # Mock services
    mock.services = MagicMock()
    mock.services.async_call = AsyncMock()

    # Mock config
    mock.config = MagicMock()
    mock.config.config_dir = "/config"
    mock.config.location_name = "Test Home"

    # Mock bus
    mock.bus = MagicMock()
    mock.bus.async_fire = AsyncMock()

    return mock


@pytest.mark.asyncio
async def test_custom_tools_registration(mock_hass_for_custom_tools, custom_tools_config):
    """Test that custom tools are registered from configuration."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)

        # Trigger lazy tool registration
        agent._ensure_tools_registered()

        # Verify custom tools are registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert "check_weather" in registered_tool_names
        assert "create_task" in registered_tool_names


@pytest.mark.asyncio
async def test_custom_tool_has_correct_properties(mock_hass_for_custom_tools, custom_tools_config):
    """Test that registered custom tools have correct properties."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)
        agent._ensure_tools_registered()

        # Get the weather tool
        weather_tool = agent.tool_handler.tools.get("check_weather")

        assert weather_tool is not None
        assert isinstance(weather_tool, RestCustomTool)
        assert weather_tool.name == "check_weather"
        assert weather_tool.description == "Get weather forecast for a location"
        assert "location" in weather_tool.parameters["properties"]


@pytest.mark.asyncio
async def test_custom_tool_appears_in_llm_tools_list(mock_hass_for_custom_tools, custom_tools_config):
    """Test that custom tools appear in the tools list for LLM."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)
        agent._ensure_tools_registered()

        # Get tools formatted for LLM
        llm_tools = agent.tool_handler.get_tool_definitions()

        # Find custom tools in the list
        tool_names = [tool["function"]["name"] for tool in llm_tools]

        assert "check_weather" in tool_names
        assert "create_task" in tool_names


@pytest.mark.asyncio
async def test_custom_rest_tool_execution_success(mock_hass_for_custom_tools, custom_tools_config):
    """Test successful execution of a custom REST tool."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)
        agent._ensure_tools_registered()

        # Get the tool and mock its _make_request method
        weather_tool = agent.tool_handler.tools["check_weather"]

        async def mock_make_request(*args, **kwargs):
            return {
                "location": "San Francisco",
                "temperature": 72,
                "condition": "sunny"
            }

        with patch.object(weather_tool, "_make_request", side_effect=mock_make_request):
            # Execute the custom tool
            result = await agent.tool_handler.execute_tool(
                "check_weather",
                {"location": "San Francisco"}
            )

        # Tool handler wraps the result, so check the outer success first
        assert result["success"] is True
        # Then check the tool's actual response
        tool_result = result["result"]
        assert tool_result["success"] is True
        assert tool_result["result"]["temperature"] == 72
        assert tool_result["error"] is None


@pytest.mark.asyncio
async def test_custom_rest_tool_execution_with_post(mock_hass_for_custom_tools, custom_tools_config):
    """Test POST request execution of a custom REST tool."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)
        agent._ensure_tools_registered()

        # Get the tool and mock its _make_request method
        task_tool = agent.tool_handler.tools["create_task"]

        async def mock_make_request(*args, **kwargs):
            return {
                "id": "task-123",
                "title": "Test Task",
                "created": True
            }

        with patch.object(task_tool, "_make_request", side_effect=mock_make_request):
            # Execute the custom tool
            result = await agent.tool_handler.execute_tool(
                "create_task",
                {"title": "Test Task", "description": "Test description"}
            )

        # Tool handler wraps the result
        assert result["success"] is True
        tool_result = result["result"]
        assert tool_result["success"] is True
        assert tool_result["result"]["id"] == "task-123"


@pytest.mark.asyncio
async def test_custom_tool_registration_with_validation_error(mock_hass_for_custom_tools):
    """Test that invalid custom tool configuration is handled gracefully."""
    config_with_invalid_tool = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                # Missing required 'description' field
                "name": "invalid_tool",
                "parameters": {},
                "handler": {
                    "type": "rest",
                    "url": "https://api.example.com",
                    "method": "GET"
                }
            }
        ]
    }

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        # Should not raise exception - just log error and continue
        agent = HomeAgent(mock_hass_for_custom_tools, config_with_invalid_tool)
        agent._ensure_tools_registered()

        # Invalid tool should not be registered
        registered_tool_names = agent.tool_handler.get_registered_tools()
        assert "invalid_tool" not in registered_tool_names


@pytest.mark.asyncio
async def test_multiple_custom_tools_registration(mock_hass_for_custom_tools, custom_tools_config):
    """Test that multiple custom tools can be registered simultaneously."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)
        agent._ensure_tools_registered()

        # Verify both custom tools are registered alongside core tools
        registered_tool_names = agent.tool_handler.get_registered_tools()

        # Core tools
        assert "ha_control" in registered_tool_names
        assert "ha_query" in registered_tool_names

        # Custom tools
        assert "check_weather" in registered_tool_names
        assert "create_task" in registered_tool_names

        # Total count should be core + custom
        assert len(registered_tool_names) >= 4


@pytest.mark.asyncio
async def test_custom_tool_error_propagation(mock_hass_for_custom_tools, custom_tools_config):
    """Test that custom tool errors are properly propagated."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config)
        agent._ensure_tools_registered()

        # Get the tool and mock its _make_request method to raise an error
        weather_tool = agent.tool_handler.tools["check_weather"]

        with patch.object(
            weather_tool,
            "_make_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Not Found"
            )
        ):
            # Execute the custom tool - should return error, not raise
            result = await agent.tool_handler.execute_tool(
                "check_weather",
                {"location": "Unknown"}
            )

        # Tool handler wraps the result
        assert result["success"] is True  # Tool handler success (tool executed)
        tool_result = result["result"]
        # But the tool itself reports failure
        assert tool_result["success"] is False
        assert tool_result["result"] is None
        assert "404" in tool_result["error"]


@pytest.fixture
def service_tools_config():
    """Provide configuration with service-based custom tools."""
    return {
        # Primary LLM config
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        # Tool configuration
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
        CONF_TOOLS_TIMEOUT: 30,
        # Custom tools configuration
        CONF_TOOLS_CUSTOM: [
            {
                "name": "trigger_morning_routine",
                "description": "Trigger the morning routine automation",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "handler": {
                    "type": "service",
                    "service": "automation.trigger",
                    "data": {
                        "entity_id": "automation.morning_routine"
                    }
                }
            },
            {
                "name": "notify_arrival",
                "description": "Send arrival notification",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {"type": "string"},
                        "location": {"type": "string"}
                    },
                    "required": ["person"]
                },
                "handler": {
                    "type": "service",
                    "service": "script.arrival_notification",
                    "data": {
                        "person": "{{ person }}",
                        "location": "{{ location }}"
                    }
                }
            },
            {
                "name": "set_movie_scene",
                "description": "Activate movie watching scene",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "handler": {
                    "type": "service",
                    "service": "scene.turn_on",
                    "target": {
                        "entity_id": "scene.movie_time"
                    }
                }
            }
        ]
    }


@pytest.mark.asyncio
async def test_service_tools_registration(mock_hass_for_custom_tools, service_tools_config):
    """Test that service-based custom tools are registered from configuration."""
    # Mock has_service to return True for all services
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)

        # Trigger lazy tool registration
        agent._ensure_tools_registered()

        # Verify custom service tools are registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert "trigger_morning_routine" in registered_tool_names
        assert "notify_arrival" in registered_tool_names
        assert "set_movie_scene" in registered_tool_names


@pytest.mark.asyncio
async def test_service_tool_has_correct_properties(mock_hass_for_custom_tools, service_tools_config):
    """Test that registered service tools have correct properties."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)
        agent._ensure_tools_registered()

        # Get the automation trigger tool
        automation_tool = agent.tool_handler.tools.get("trigger_morning_routine")

        assert automation_tool is not None
        assert isinstance(automation_tool, ServiceCustomTool)
        assert automation_tool.name == "trigger_morning_routine"
        assert automation_tool.description == "Trigger the morning routine automation"
        assert automation_tool.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_service_tool_appears_in_llm_tools_list(mock_hass_for_custom_tools, service_tools_config):
    """Test that service tools appear in the tools list for LLM."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)
        agent._ensure_tools_registered()

        # Get tools formatted for LLM
        llm_tools = agent.tool_handler.get_tool_definitions()

        # Find service tools in the list
        tool_names = [tool["function"]["name"] for tool in llm_tools]

        assert "trigger_morning_routine" in tool_names
        assert "notify_arrival" in tool_names
        assert "set_movie_scene" in tool_names


@pytest.mark.asyncio
async def test_service_tool_execution_success(mock_hass_for_custom_tools, service_tools_config):
    """Test successful execution of a custom service tool."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)
    mock_hass_for_custom_tools.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)
        agent._ensure_tools_registered()

        # Execute the service tool
        result = await agent.tool_handler.execute_tool(
            "trigger_morning_routine",
            {}
        )

        # Verify service was called
        mock_hass_for_custom_tools.services.async_call.assert_called_once_with(
            domain="automation",
            service="trigger",
            service_data={"entity_id": "automation.morning_routine"},
            target=None,
            blocking=True
        )

        # Tool handler wraps the result
        assert result["success"] is True
        tool_result = result["result"]
        assert tool_result["success"] is True
        assert "successfully" in tool_result["result"].lower()
        assert tool_result["error"] is None


@pytest.mark.asyncio
async def test_service_tool_execution_with_parameters(mock_hass_for_custom_tools, service_tools_config):
    """Test service tool execution with templated parameters."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)
    mock_hass_for_custom_tools.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)
        agent._ensure_tools_registered()

        # Execute the service tool with parameters
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                side_effect=lambda x: x.get("person", "John") if "person" in x else x.get("location", "Home")
            )
            mock_template_class.return_value = mock_template

            result = await agent.tool_handler.execute_tool(
                "notify_arrival",
                {"person": "John", "location": "Home"}
            )

        # Verify service was called
        mock_hass_for_custom_tools.services.async_call.assert_called_once()
        call_args = mock_hass_for_custom_tools.services.async_call.call_args
        assert call_args[1]["domain"] == "script"
        assert call_args[1]["service"] == "arrival_notification"

        # Verify result
        assert result["success"] is True
        tool_result = result["result"]
        assert tool_result["success"] is True


@pytest.mark.asyncio
async def test_service_tool_execution_with_target(mock_hass_for_custom_tools, service_tools_config):
    """Test service tool execution with target field."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)
    mock_hass_for_custom_tools.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)
        agent._ensure_tools_registered()

        # Execute the service tool
        result = await agent.tool_handler.execute_tool(
            "set_movie_scene",
            {}
        )

        # Verify service was called with target
        mock_hass_for_custom_tools.services.async_call.assert_called_once()
        call_args = mock_hass_for_custom_tools.services.async_call.call_args
        assert call_args[1]["target"] == {"entity_id": "scene.movie_time"}

        # Verify result
        assert result["success"] is True


@pytest.mark.asyncio
async def test_service_tool_error_propagation(mock_hass_for_custom_tools, service_tools_config):
    """Test that service tool errors are properly propagated."""
    from homeassistant.core import ServiceNotFound

    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    # Create ServiceNotFound with message pre-set to avoid translation system
    error = ServiceNotFound("automation", "trigger")
    error._message = "Service automation.trigger not found"
    mock_hass_for_custom_tools.services.async_call = AsyncMock(side_effect=error)

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config)
        agent._ensure_tools_registered()

        # Execute the service tool - should return error, not raise
        result = await agent.tool_handler.execute_tool(
            "trigger_morning_routine",
            {}
        )

        # Tool handler wraps the result
        assert result["success"] is True  # Tool handler success (tool executed)
        tool_result = result["result"]
        # But the tool itself reports failure
        assert tool_result["success"] is False
        assert tool_result["result"] is None
        assert "not found" in tool_result["error"].lower() or "service" in tool_result["error"].lower()


@pytest.mark.asyncio
async def test_mixed_rest_and_service_tools(mock_hass_for_custom_tools):
    """Test that both REST and service tools can be registered together."""
    mixed_config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "check_weather",
                "description": "Get weather forecast",
                "parameters": {"type": "object", "properties": {}},
                "handler": {
                    "type": "rest",
                    "url": "https://api.weather.com/v1/forecast",
                    "method": "GET"
                }
            },
            {
                "name": "trigger_automation",
                "description": "Trigger an automation",
                "parameters": {"type": "object", "properties": {}},
                "handler": {
                    "type": "service",
                    "service": "automation.trigger",
                    "data": {"entity_id": "automation.test"}
                }
            }
        ]
    }

    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, mixed_config)
        agent._ensure_tools_registered()

        # Verify both types of tools are registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert "check_weather" in registered_tool_names
        assert "trigger_automation" in registered_tool_names

        # Verify tool types
        weather_tool = agent.tool_handler.tools["check_weather"]
        automation_tool = agent.tool_handler.tools["trigger_automation"]

        assert isinstance(weather_tool, RestCustomTool)
        assert isinstance(automation_tool, ServiceCustomTool)

"""Integration tests for Phase 3: Custom Tool Framework.

This test suite validates the complete custom tool integration flow:
- Custom tool registration from configuration
- REST handler execution with real HTTP calls (mocked)
- Template rendering with parameters
- Error handling and validation
- Integration with Home Agent's tool system
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from homeassistant.core import HomeAssistant

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_EMIT_EVENTS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_TOOLS_CUSTOM,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
)
from custom_components.home_agent.tools.custom import RestCustomTool, ServiceCustomTool

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


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
                        "location": {"type": "string", "description": "City name or location"}
                    },
                    "required": ["location"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.weather.com/v1/forecast",
                    "method": "GET",
                    "headers": {"Authorization": "Bearer test-token"},
                    "query_params": {"location": "{{ location }}", "format": "json"},
                },
            },
            {
                "name": "create_task",
                "description": "Create a new task",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "description": {"type": "string"}},
                    "required": ["title"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.tasks.com/v1/tasks",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"title": "{{ title }}", "description": "{{ description }}"},
                },
            },
        ],
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
    # async_fire is sync in HA, not actually async
    mock.bus.async_fire = MagicMock(return_value=None)

    return mock


@pytest.mark.asyncio
async def test_custom_tools_registration(
    mock_hass_for_custom_tools, custom_tools_config, session_manager
):
    """Test that custom tools are registered from configuration."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)

        # Trigger lazy tool registration
        agent._ensure_tools_registered()

        # Verify custom tools are registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert "check_weather" in registered_tool_names
        assert "create_task" in registered_tool_names


@pytest.mark.asyncio
async def test_custom_tool_has_correct_properties(
    mock_hass_for_custom_tools, custom_tools_config, session_manager
):
    """Test that registered custom tools have correct properties."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get the weather tool
        weather_tool = agent.tool_handler.tools.get("check_weather")

        assert weather_tool is not None
        assert isinstance(weather_tool, RestCustomTool)
        assert weather_tool.name == "check_weather"
        assert weather_tool.description == "Get weather forecast for a location"
        assert "location" in weather_tool.parameters["properties"]


@pytest.mark.asyncio
async def test_custom_tool_appears_in_llm_tools_list(
    session_manager, mock_hass_for_custom_tools, custom_tools_config
):
    """Test that custom tools appear in the tools list for LLM."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get tools formatted for LLM
        llm_tools = agent.tool_handler.get_tool_definitions()

        # Find custom tools in the list
        tool_names = [tool["function"]["name"] for tool in llm_tools]

        assert "check_weather" in tool_names
        assert "create_task" in tool_names


@pytest.mark.asyncio
async def test_custom_rest_tool_execution_success(
    mock_hass_for_custom_tools, custom_tools_config, session_manager
):
    """Test successful execution of a custom REST tool."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get the tool and mock its _make_request method
        weather_tool = agent.tool_handler.tools["check_weather"]

        async def mock_make_request(*args, **kwargs):
            return {"location": "San Francisco", "temperature": 72, "condition": "sunny"}

        with patch.object(weather_tool, "_make_request", side_effect=mock_make_request):
            # Execute the custom tool
            result = await agent.tool_handler.execute_tool(
                "check_weather", {"location": "San Francisco"}
            )

        # Tool handler wraps the result, so check the outer success first
        assert result["success"] is True
        # Then check the tool's actual response
        tool_result = result["result"]
        assert tool_result["success"] is True
        assert tool_result["result"]["temperature"] == 72
        assert tool_result["error"] is None


@pytest.mark.asyncio
async def test_custom_rest_tool_execution_with_post(
    session_manager, mock_hass_for_custom_tools, custom_tools_config
):
    """Test POST request execution of a custom REST tool."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get the tool and mock its _make_request method
        task_tool = agent.tool_handler.tools["create_task"]

        async def mock_make_request(*args, **kwargs):
            return {"id": "task-123", "title": "Test Task", "created": True}

        with patch.object(task_tool, "_make_request", new=mock_make_request):
            # Execute the custom tool
            result = await agent.tool_handler.execute_tool(
                "create_task", {"title": "Test Task", "description": "Test description"}
            )

        # Tool handler wraps the result
        assert result["success"] is True
        tool_result = result["result"]
        assert tool_result["success"] is True
        assert tool_result["result"]["id"] == "task-123"


@pytest.mark.asyncio
async def test_custom_tool_registration_with_validation_error(
    mock_hass_for_custom_tools, session_manager
):
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
                "handler": {"type": "rest", "url": "https://api.example.com", "method": "GET"},
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        # Should not raise exception - just log error and continue
        agent = HomeAgent(mock_hass_for_custom_tools, config_with_invalid_tool, session_manager)
        agent._ensure_tools_registered()

        # Invalid tool should not be registered
        registered_tool_names = agent.tool_handler.get_registered_tools()
        assert "invalid_tool" not in registered_tool_names


@pytest.mark.asyncio
async def test_multiple_custom_tools_registration(
    mock_hass_for_custom_tools, custom_tools_config, session_manager
):
    """Test that multiple custom tools can be registered simultaneously."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)
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
async def test_custom_tool_error_propagation(
    mock_hass_for_custom_tools, custom_tools_config, session_manager
):
    """Test that custom tool errors are properly propagated."""
    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, custom_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get the tool and mock its _make_request method to raise an error
        weather_tool = agent.tool_handler.tools["check_weather"]

        with patch.object(
            weather_tool,
            "_make_request",
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=404, message="Not Found"
            ),
        ):
            # Execute the custom tool - should return error, not raise
            result = await agent.tool_handler.execute_tool("check_weather", {"location": "Unknown"})

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
                "parameters": {"type": "object", "properties": {}},
                "handler": {
                    "type": "service",
                    "service": "automation.trigger",
                    "data": {"entity_id": "automation.morning_routine"},
                },
            },
            {
                "name": "notify_arrival",
                "description": "Send arrival notification",
                "parameters": {
                    "type": "object",
                    "properties": {"person": {"type": "string"}, "location": {"type": "string"}},
                    "required": ["person"],
                },
                "handler": {
                    "type": "service",
                    "service": "script.arrival_notification",
                    "data": {"person": "{{ person }}", "location": "{{ location }}"},
                },
            },
            {
                "name": "set_movie_scene",
                "description": "Activate movie watching scene",
                "parameters": {"type": "object", "properties": {}},
                "handler": {
                    "type": "service",
                    "service": "scene.turn_on",
                    "target": {"entity_id": "scene.movie_time"},
                },
            },
        ],
    }


@pytest.mark.asyncio
async def test_service_tools_registration(
    mock_hass_for_custom_tools, service_tools_config, session_manager
):
    """Test that service-based custom tools are registered from configuration."""
    # Mock has_service to return True for all services
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)

        # Trigger lazy tool registration
        agent._ensure_tools_registered()

        # Verify custom service tools are registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert "trigger_morning_routine" in registered_tool_names
        assert "notify_arrival" in registered_tool_names
        assert "set_movie_scene" in registered_tool_names


@pytest.mark.asyncio
async def test_service_tool_has_correct_properties(
    session_manager, mock_hass_for_custom_tools, service_tools_config
):
    """Test that registered service tools have correct properties."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get the automation trigger tool
        automation_tool = agent.tool_handler.tools.get("trigger_morning_routine")

        assert automation_tool is not None
        assert isinstance(automation_tool, ServiceCustomTool)
        assert automation_tool.name == "trigger_morning_routine"
        assert automation_tool.description == "Trigger the morning routine automation"
        assert automation_tool.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_service_tool_appears_in_llm_tools_list(
    session_manager, mock_hass_for_custom_tools, service_tools_config
):
    """Test that service tools appear in the tools list for LLM."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Get tools formatted for LLM
        llm_tools = agent.tool_handler.get_tool_definitions()

        # Find service tools in the list
        tool_names = [tool["function"]["name"] for tool in llm_tools]

        assert "trigger_morning_routine" in tool_names
        assert "notify_arrival" in tool_names
        assert "set_movie_scene" in tool_names


@pytest.mark.asyncio
async def test_service_tool_execution_success(
    mock_hass_for_custom_tools, service_tools_config, session_manager
):
    """Test successful execution of a custom service tool."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)
    mock_hass_for_custom_tools.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Execute the service tool
        result = await agent.tool_handler.execute_tool("trigger_morning_routine", {})

        # Verify service was called
        mock_hass_for_custom_tools.services.async_call.assert_called_once_with(
            domain="automation",
            service="trigger",
            service_data={"entity_id": "automation.morning_routine"},
            target=None,
            blocking=True,
        )

        # Tool handler wraps the result
        assert result["success"] is True
        tool_result = result["result"]
        assert tool_result["success"] is True
        assert "successfully" in tool_result["result"].lower()
        assert tool_result["error"] is None


@pytest.mark.asyncio
async def test_service_tool_execution_with_parameters(
    session_manager, mock_hass_for_custom_tools, service_tools_config
):
    """Test service tool execution with templated parameters."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)
    mock_hass_for_custom_tools.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Execute the service tool with parameters
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                side_effect=lambda x: (
                    x.get("person", "John") if "person" in x else x.get("location", "Home")
                )
            )
            mock_template_class.return_value = mock_template

            result = await agent.tool_handler.execute_tool(
                "notify_arrival", {"person": "John", "location": "Home"}
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
async def test_service_tool_execution_with_target(
    mock_hass_for_custom_tools, service_tools_config, session_manager
):
    """Test service tool execution with target field."""
    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)
    mock_hass_for_custom_tools.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Execute the service tool
        result = await agent.tool_handler.execute_tool("set_movie_scene", {})

        # Verify service was called with target
        mock_hass_for_custom_tools.services.async_call.assert_called_once()
        call_args = mock_hass_for_custom_tools.services.async_call.call_args
        assert call_args[1]["target"] == {"entity_id": "scene.movie_time"}

        # Verify result
        assert result["success"] is True


@pytest.mark.asyncio
async def test_service_tool_error_propagation(
    mock_hass_for_custom_tools, service_tools_config, session_manager
):
    """Test that service tool errors are properly propagated."""
    from homeassistant.core import ServiceNotFound

    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    # Create ServiceNotFound with message pre-set to avoid translation system
    error = ServiceNotFound("automation", "trigger")
    error._message = "Service automation.trigger not found"
    mock_hass_for_custom_tools.services.async_call = AsyncMock(side_effect=error)

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, service_tools_config, session_manager)
        agent._ensure_tools_registered()

        # Execute the service tool - should return error, not raise
        result = await agent.tool_handler.execute_tool("trigger_morning_routine", {})

        # Tool handler wraps the result
        assert result["success"] is True  # Tool handler success (tool executed)
        tool_result = result["result"]
        # But the tool itself reports failure
        assert tool_result["success"] is False
        assert tool_result["result"] is None
        assert (
            "not found" in tool_result["error"].lower() or "service" in tool_result["error"].lower()
        )


@pytest.mark.asyncio
async def test_mixed_rest_and_service_tools(mock_hass_for_custom_tools, session_manager):
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
                    "method": "GET",
                },
            },
            {
                "name": "trigger_automation",
                "description": "Trigger an automation",
                "parameters": {"type": "object", "properties": {}},
                "handler": {
                    "type": "service",
                    "service": "automation.trigger",
                    "data": {"entity_id": "automation.test"},
                },
            },
        ],
    }

    mock_hass_for_custom_tools.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_custom_tools, mixed_config, session_manager)
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


# ============================================================================
# TIMEOUT HANDLING TESTS
# ============================================================================


class SlowTestTool:
    """Test tool that simulates slow execution for timeout testing."""

    def __init__(self, hass, delay_seconds=2):
        """Initialize the slow test tool."""
        self.hass = hass
        self.name = "slow_test_tool"
        self.delay_seconds = delay_seconds

    def get_definition(self):
        """Return tool definition."""
        return {
            "name": self.name,
            "description": "A test tool that is intentionally slow",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    def to_openai_format(self):
        """Return tool definition in OpenAI format."""
        return {
            "type": "function",
            "function": self.get_definition(),
        }

    async def execute(self, **kwargs):
        """Execute the tool with a delay."""
        import asyncio

        await asyncio.sleep(self.delay_seconds)
        return {"success": True, "result": "completed", "error": None}


@pytest.mark.asyncio
async def test_tool_timeout_is_triggered_when_execution_exceeds_limit(
    mock_hass_for_custom_tools, session_manager
):
    """Test that tool execution timeout is triggered when tool exceeds max_timeout_seconds.

    This test verifies that when a tool takes longer than the configured timeout,
    a timeout error is raised and properly handled.
    """
    from custom_components.home_agent.exceptions import ToolExecutionError
    from custom_components.home_agent.tool_handler import ToolHandler

    # Create a tool handler with a 1 second timeout
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 1,  # 1 second timeout
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: False,  # Disable events to simplify test
        },
    )

    # Register a slow tool that takes 3 seconds
    slow_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=3)
    tool_handler.register_tool(slow_tool)

    # Execute the tool - should timeout
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool_handler.execute_tool("slow_test_tool", {})

    # Verify the error message mentions timeout
    assert "timed out" in str(exc_info.value).lower()
    assert "1s" in str(exc_info.value) or "1 s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_timeout_does_not_crash_agent(mock_hass_for_custom_tools, session_manager):
    """Test that tool timeout triggers proper error handling without crashing.

    This test verifies that the tool handler gracefully handles timeouts by:
    - Catching the timeout exception
    - Recording failure metrics
    - Raising a proper ToolExecutionError (not crashing)
    """
    from custom_components.home_agent.exceptions import ToolExecutionError
    from custom_components.home_agent.tool_handler import ToolHandler

    # Create a tool handler with a 1 second timeout
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 1,
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: False,
        },
    )

    # Reset metrics to track this test's execution
    tool_handler.reset_metrics()

    # Register a slow tool that takes 2 seconds
    slow_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=2)
    tool_handler.register_tool(slow_tool)

    # Execute and expect proper exception handling
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool_handler.execute_tool("slow_test_tool", {})

    # Verify the error is a ToolExecutionError, not a raw asyncio.TimeoutError
    assert isinstance(exc_info.value, ToolExecutionError)
    assert "timed out" in str(exc_info.value).lower()

    # Verify metrics show the failure was tracked
    metrics = tool_handler.get_metrics()
    assert metrics["total_executions"] == 1
    assert metrics["failed_executions"] == 1
    assert metrics["successful_executions"] == 0


@pytest.mark.asyncio
async def test_tool_timeout_returns_proper_error_message(
    mock_hass_for_custom_tools, session_manager
):
    """Test that a proper error message is returned when timeout occurs.

    This test verifies that the timeout error message:
    - Clearly indicates a timeout occurred
    - Includes the timeout duration
    - Is informative for debugging
    """
    from custom_components.home_agent.exceptions import ToolExecutionError
    from custom_components.home_agent.tool_handler import ToolHandler

    # Set a 2 second timeout for clarity in error message
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 2,
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: False,
        },
    )

    # Register a slow tool that takes 4 seconds
    slow_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=4)
    tool_handler.register_tool(slow_tool)

    with pytest.raises(ToolExecutionError) as exc_info:
        await tool_handler.execute_tool("slow_test_tool", {})

    error_message = str(exc_info.value)

    # Verify error message content
    assert "timed out" in error_message.lower()
    assert "2" in error_message  # Should mention the 2 second timeout
    assert "s" in error_message.lower()  # Should include time unit


@pytest.mark.asyncio
async def test_tool_timeout_fires_proper_events(mock_hass_for_custom_tools, session_manager):
    """Test that timeout triggers proper event emission.

    This test verifies that when a timeout occurs:
    - A "started" event is fired
    - A "failed" event is fired with timeout information
    - The failed event includes error_type: "TimeoutError"
    """
    from custom_components.home_agent.const import EVENT_TOOL_PROGRESS
    from custom_components.home_agent.exceptions import ToolExecutionError
    from custom_components.home_agent.tool_handler import ToolHandler

    # Track events
    events_fired = []

    def capture_event(event_type, event_data=None):
        events_fired.append({"type": event_type, "data": event_data})
        # Return None to avoid coroutine warning (async_fire is sync in HA)
        return None

    mock_hass_for_custom_tools.bus.async_fire = MagicMock(side_effect=capture_event)

    # Create tool handler with events enabled and 1 second timeout
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 1,
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: True,  # Enable events
        },
    )

    # Register a slow tool
    slow_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=2)
    tool_handler.register_tool(slow_tool)

    with pytest.raises(ToolExecutionError):
        await tool_handler.execute_tool("slow_test_tool", {}, tool_call_id="test_call_123")

    # Find the events
    progress_events = [e for e in events_fired if e["type"] == EVENT_TOOL_PROGRESS]

    # Should have started and failed events
    assert len(progress_events) >= 2

    # Verify started event
    started_events = [e for e in progress_events if e["data"].get("status") == "started"]
    assert len(started_events) == 1
    assert started_events[0]["data"]["tool_name"] == "slow_test_tool"

    # Verify failed event with timeout info
    failed_events = [e for e in progress_events if e["data"].get("status") == "failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["data"]["tool_name"] == "slow_test_tool"
    assert failed_events[0]["data"]["error_type"] == "TimeoutError"
    assert "timed out" in failed_events[0]["data"]["error"].lower()
    assert failed_events[0]["data"]["success"] is False


@pytest.mark.asyncio
async def test_agent_continues_working_after_tool_timeout(
    mock_hass_for_custom_tools, session_manager
):
    """Test that the agent continues working after a tool timeout.

    This test verifies that:
    - After a timeout occurs, the tool handler is still functional
    - Subsequent tool calls can execute successfully
    - Metrics are properly tracked across timeout and success
    """
    from custom_components.home_agent.exceptions import ToolExecutionError
    from custom_components.home_agent.tool_handler import ToolHandler

    # Create tool handler with 1 second timeout
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 1,
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: False,
        },
    )

    tool_handler.reset_metrics()

    # Register a slow tool (2 seconds) and a fast tool (0.1 seconds)
    slow_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=2)
    slow_tool.name = "slow_tool"
    tool_handler.register_tool(slow_tool)

    fast_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=0.1)
    fast_tool.name = "fast_tool"
    tool_handler.register_tool(fast_tool)

    # First call: timeout
    with pytest.raises(ToolExecutionError):
        await tool_handler.execute_tool("slow_tool", {})

    # Verify first call timed out
    metrics_after_timeout = tool_handler.get_metrics()
    assert metrics_after_timeout["total_executions"] == 1
    assert metrics_after_timeout["failed_executions"] == 1

    # Second call: should succeed
    result = await tool_handler.execute_tool("fast_tool", {})

    # Verify second call succeeded
    assert result["success"] is True

    # Verify metrics show both executions
    final_metrics = tool_handler.get_metrics()
    assert final_metrics["total_executions"] == 2
    assert final_metrics["failed_executions"] == 1
    assert final_metrics["successful_executions"] == 1
    assert final_metrics["success_rate"] == 50.0


@pytest.mark.asyncio
async def test_tool_completes_successfully_within_timeout(
    mock_hass_for_custom_tools, session_manager
):
    """Test that tools completing within timeout work normally.

    This test verifies that the timeout mechanism doesn't interfere with
    normal tool execution when the tool completes quickly.
    """
    from custom_components.home_agent.tool_handler import ToolHandler

    # Configure a reasonable timeout
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 5,
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: False,
        },
    )

    tool_handler.reset_metrics()

    # Register a fast tool that completes in 0.5 seconds (well within 5 second timeout)
    fast_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=0.5)
    tool_handler.register_tool(fast_tool)

    # Execute the tool
    result = await tool_handler.execute_tool("slow_test_tool", {})

    # Verify successful execution
    assert result["success"] is True
    assert result["result"]["success"] is True
    assert result["result"]["result"] == "completed"

    # Verify metrics
    metrics = tool_handler.get_metrics()
    assert metrics["total_executions"] == 1
    assert metrics["successful_executions"] == 1
    assert metrics["failed_executions"] == 0
    assert metrics["success_rate"] == 100.0


@pytest.mark.asyncio
async def test_different_timeout_configurations(mock_hass_for_custom_tools, session_manager):
    """Test that different timeout configurations are respected.

    This test verifies that:
    - Custom timeout values are properly read from config
    - The timeout is actually applied during execution
    - Different timeout values work as expected
    """
    from custom_components.home_agent.exceptions import ToolExecutionError
    from custom_components.home_agent.tool_handler import ToolHandler

    # Test with a very short timeout (0.5 seconds)
    tool_handler = ToolHandler(
        mock_hass_for_custom_tools,
        {
            CONF_TOOLS_TIMEOUT: 0.5,  # Very short timeout
            CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
            CONF_EMIT_EVENTS: False,
        },
    )

    # Verify timeout is set correctly
    assert tool_handler.timeout == 0.5

    # Register a tool that takes 1 second (longer than 0.5 second timeout)
    slow_tool = SlowTestTool(mock_hass_for_custom_tools, delay_seconds=1)
    tool_handler.register_tool(slow_tool)

    # Should timeout with 0.5s timeout
    with pytest.raises(ToolExecutionError) as exc_info:
        await tool_handler.execute_tool("slow_test_tool", {})

    assert "0.5" in str(exc_info.value)

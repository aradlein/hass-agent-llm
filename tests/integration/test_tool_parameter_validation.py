"""Integration tests for tool parameter validation.

This test suite validates that tools properly handle invalid parameters at call time:
- Wrong parameter types (string when number expected)
- Missing required parameters
- Empty/invalid parameter values
- Nested object structure validation
- Template rendering failures
- Extra/unexpected parameters

These tests complement test_phase3_custom_tools.py by focusing specifically on
parameter validation rather than tool configuration or execution flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_TOOLS_CUSTOM,
)
from custom_components.home_agent.exceptions import ToolExecutionError, ValidationError

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance for parameter validation tests."""
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


# ============================================================================
# Type Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_parameter_wrong_type_string_instead_of_number(mock_hass, session_manager):
    """Test tool with wrong parameter type - string when number expected.

    Note: The current implementation uses template rendering which coerces types.
    This test documents the actual behavior - type validation is not enforced
    at the parameter validation layer, relying instead on the external API to
    reject invalid types.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "set_temperature",
                "description": "Set temperature for a device",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string", "description": "Device identifier"},
                        "temperature": {
                            "type": "number",
                            "description": "Temperature in celsius",
                            "minimum": 10,
                            "maximum": 30,
                        },
                    },
                    "required": ["device_id", "temperature"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.devices.com/v1/temperature",
                    "method": "POST",
                    "body": {"device_id": "{{ device_id }}", "temperature": "{{ temperature }}"},
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        temp_tool = agent.tool_handler.tools["set_temperature"]

        async def mock_make_request(*args, **kwargs):
            return {"status": "ok"}

        with patch.object(temp_tool, "_make_request", side_effect=mock_make_request):
            # Call with string instead of number
            result = await agent.tool_handler.execute_tool(
                "set_temperature", {"device_id": "dev_123", "temperature": "not_a_number"}
            )

            # Currently accepts the parameter and renders it in template
            # Type validation is delegated to the external API
            assert result["success"] is True


@pytest.mark.asyncio
async def test_tool_parameter_invalid_parameters_not_dict(mock_hass, session_manager):
    """Test tool call with parameters that aren't a dictionary.

    This should raise ValidationError because tool_handler.validate_tool_call()
    explicitly checks that parameters must be a dict.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "simple_tool",
                "description": "A simple tool",
                "parameters": {"type": "object", "properties": {}},
                "handler": {
                    "type": "rest",
                    "url": "https://api.example.com",
                    "method": "GET",
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Try to call with non-dict parameters
        with pytest.raises(ValidationError) as exc_info:
            await agent.tool_handler.execute_tool("simple_tool", "not_a_dict")

        assert "must be a dictionary" in str(exc_info.value)
        assert "str" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_parameter_parameters_is_list(mock_hass, session_manager):
    """Test tool call with parameters as list instead of dict."""
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "simple_tool",
                "description": "A simple tool",
                "parameters": {"type": "object", "properties": {}},
                "handler": {"type": "rest", "url": "https://api.example.com", "method": "GET"},
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Try to call with list parameters
        with pytest.raises(ValidationError) as exc_info:
            await agent.tool_handler.execute_tool("simple_tool", ["param1", "param2"])

        assert "must be a dictionary" in str(exc_info.value)


# ============================================================================
# Missing Parameter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_parameter_missing_required_parameter(mock_hass, session_manager):
    """Test tool execution when required parameter is missing at call time.

    Note: The current implementation doesn't enforce JSON schema validation
    for missing parameters. Template rendering handles missing variables by
    rendering them as empty strings or 'None'. This test documents that behavior.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "send_notification",
                "description": "Send a notification message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The notification message"},
                        "recipient": {"type": "string", "description": "Who to notify"},
                    },
                    "required": ["message", "recipient"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.notify.com/v1/send",
                    "method": "POST",
                    "body": {"message": "{{ message }}", "recipient": "{{ recipient }}"},
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        notify_tool = agent.tool_handler.tools["send_notification"]

        async def mock_make_request(*args, **kwargs):
            return {"status": "sent"}

        with patch.object(notify_tool, "_make_request", side_effect=mock_make_request):
            # Call with missing required 'recipient' parameter
            result = await agent.tool_handler.execute_tool(
                "send_notification",
                {"message": "Hello"},  # Missing 'recipient'
            )

            # Currently passes - template rendering handles missing vars
            assert result["success"] is True


@pytest.mark.asyncio
async def test_service_tool_parameter_missing_required(mock_hass, session_manager):
    """Test service tool with missing required parameter at call time."""
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "notify_person",
                "description": "Send notification to a person",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "person": {"type": "string", "description": "Person to notify"},
                        "message": {"type": "string", "description": "Notification message"},
                    },
                    "required": ["person", "message"],
                },
                "handler": {
                    "type": "service",
                    "service": "notify.mobile_app",
                    "data": {"message": "{{ message }}", "target": "{{ person }}"},
                },
            }
        ],
    }

    mock_hass.services.has_service = MagicMock(return_value=True)
    mock_hass.services.async_call = AsyncMock()

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Call with missing 'message' parameter
        result = await agent.tool_handler.execute_tool(
            "notify_person",
            {"person": "john"},  # Missing 'message'
        )

        # Service will be called with None/empty for missing template variable
        assert result["success"] is True
        assert mock_hass.services.async_call.called


# ============================================================================
# Value Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_parameter_empty_string_when_non_empty_required(mock_hass, session_manager):
    """Test tool parameter with empty string when non-empty required.

    Note: The current implementation doesn't validate JSON schema constraints
    like minLength at execution time. This test documents that behavior.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "search_query",
                "description": "Execute a search query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text",
                            "minLength": 1,
                        }
                    },
                    "required": ["query"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.search.com/v1/search",
                    "method": "GET",
                    "query_params": {"q": "{{ query }}"},
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        search_tool = agent.tool_handler.tools["search_query"]

        async def mock_make_request(*args, **kwargs):
            return {"results": []}

        with patch.object(search_tool, "_make_request", side_effect=mock_make_request):
            # Call with empty string for query
            result = await agent.tool_handler.execute_tool(
                "search_query",
                {"query": ""},  # Empty string
            )

            # Currently passes - no minLength validation at execution time
            assert result["success"] is True


@pytest.mark.asyncio
async def test_tool_execution_with_none_parameter_value(mock_hass, session_manager):
    """Test tool execution when parameter value is None."""
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "optional_param_tool",
                "description": "Tool with optional parameter",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "required_param": {"type": "string"},
                        "optional_param": {"type": "string"},
                    },
                    "required": ["required_param"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.example.com",
                    "method": "POST",
                    "body": {
                        "required": "{{ required_param }}",
                        "optional": "{{ optional_param }}",
                    },
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        tool = agent.tool_handler.tools["optional_param_tool"]

        async def mock_make_request(*args, **kwargs):
            return {"status": "ok"}

        with patch.object(tool, "_make_request", side_effect=mock_make_request):
            # Call with None for optional parameter
            result = await agent.tool_handler.execute_tool(
                "optional_param_tool",
                {"required_param": "value", "optional_param": None},
            )

            # Should handle None gracefully
            assert result["success"] is True


# ============================================================================
# Nested Object Validation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_parameter_nested_object_invalid_structure(mock_hass, session_manager):
    """Test tool with nested object that has invalid structure.

    This tests what happens when a parameter expects an object but receives
    a different type (like a string).
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "create_user",
                "description": "Create a new user account",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {"type": "string"},
                        "profile": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "age": {"type": "number"},
                            },
                            "required": ["email"],
                        },
                    },
                    "required": ["username", "profile"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.users.com/v1/create",
                    "method": "POST",
                    "body": {
                        "username": "{{ username }}",
                        "email": "{{ profile.email }}",
                        "age": "{{ profile.age }}",
                    },
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        user_tool = agent.tool_handler.tools["create_user"]

        async def mock_make_request(*args, **kwargs):
            return {"user_id": "123"}

        with patch.object(user_tool, "_make_request", side_effect=mock_make_request):
            # profile is not an object (it's a string)
            # This will likely cause issues during template rendering
            await agent.tool_handler.execute_tool(
                "create_user",
                {
                    "username": "john_doe",
                    "profile": "invalid",  # Should be object, not string
                },
            )
            # Template rendering may fail or produce unexpected results
            # but system doesn't reject upfront


# ============================================================================
# Additional Parameters Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_parameter_validation_with_additional_properties(mock_hass, session_manager):
    """Test tool called with extra parameters not in schema.

    The system currently allows additional properties and doesn't validate
    against them. This test documents that behavior.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "simple_action",
                "description": "Perform a simple action",
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.example.com/action",
                    "method": "POST",
                    "body": {"action": "{{ action }}"},
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        tool = agent.tool_handler.tools["simple_action"]

        async def mock_make_request(*args, **kwargs):
            return {"status": "ok"}

        with patch.object(tool, "_make_request", side_effect=mock_make_request):
            # Call with extra parameters not defined in schema
            result = await agent.tool_handler.execute_tool(
                "simple_action",
                {
                    "action": "start",
                    "extra_param": "should_be_ignored",
                    "another_extra": 123,
                },
            )

            # Currently accepts extra parameters without validation
            assert result["success"] is True


# ============================================================================
# Template Rendering Failure Tests
# ============================================================================


@pytest.mark.asyncio
async def test_rest_tool_template_rendering_failure(mock_hass, session_manager):
    """Test REST tool when template rendering fails due to invalid template syntax.

    When template rendering fails, the tool's execute() method catches the exception
    and returns it as part of the tool result. The tool_handler wraps this in another
    layer, so we need to check the nested result structure.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "broken_template_tool",
                "description": "Tool with broken template",
                "parameters": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
                "handler": {
                    "type": "rest",
                    "url": "https://api.example.com",
                    "method": "GET",
                    "query_params": {
                        "value": "{{ invalid template syntax"  # Missing closing braces
                    },
                },
            }
        ],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Execute tool - should fail during template rendering
        result = await agent.tool_handler.execute_tool("broken_template_tool", {"param": "test"})

        # Tool handler reports success (tool executed), but tool itself reports failure
        assert result["success"] is True
        # Check the nested tool result
        tool_result = result["result"]
        assert tool_result["success"] is False
        assert tool_result["error"] is not None
        assert (
            "template" in tool_result["error"].lower() or "render" in tool_result["error"].lower()
        )


@pytest.mark.asyncio
async def test_service_tool_template_rendering_failure(mock_hass, session_manager):
    """Test service tool when template rendering fails.

    Similar to REST tools, service tools catch template rendering errors
    and return them as part of the tool result.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [
            {
                "name": "service_broken_template",
                "description": "Service tool with broken template",
                "parameters": {
                    "type": "object",
                    "properties": {"entity": {"type": "string"}},
                    "required": ["entity"],
                },
                "handler": {
                    "type": "service",
                    "service": "light.turn_on",
                    "data": {"entity_id": "{{ invalid.template.syntax"},  # Broken template
                },
            }
        ],
    }

    mock_hass.services.has_service = MagicMock(return_value=True)

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Execute tool - should fail during template rendering
        result = await agent.tool_handler.execute_tool(
            "service_broken_template", {"entity": "light.living_room"}
        )

        # Tool handler reports success (tool executed), but tool itself reports failure
        assert result["success"] is True
        # Check the nested tool result
        tool_result = result["result"]
        assert tool_result["success"] is False
        assert tool_result["error"] is not None
        assert (
            "template" in tool_result["error"].lower() or "render" in tool_result["error"].lower()
        )


# ============================================================================
# Nonexistent Tool Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tool_nonexistent_tool_call(mock_hass, session_manager):
    """Test calling a tool that doesn't exist.

    This should raise ToolExecutionError with a helpful message listing
    available tools.
    """
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Try to call non-existent tool
        with pytest.raises(ToolExecutionError) as exc_info:
            await agent.tool_handler.execute_tool("nonexistent_tool", {})

        error_message = str(exc_info.value)
        assert "not found" in error_message.lower()
        assert "nonexistent_tool" in error_message
        # Should list available tools in error message
        assert "available" in error_message.lower() or "ha_control" in error_message


@pytest.mark.asyncio
async def test_tool_call_empty_tool_name(mock_hass, session_manager):
    """Test calling a tool with empty string as name."""
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "test-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_TOOLS_CUSTOM: [],
    }

    with patch("custom_components.home_agent.agent.core.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass, config, session_manager)
        agent._ensure_tools_registered()

        # Try to call with empty tool name
        with pytest.raises(ToolExecutionError) as exc_info:
            await agent.tool_handler.execute_tool("", {})

        error_message = str(exc_info.value)
        assert "not found" in error_message.lower()

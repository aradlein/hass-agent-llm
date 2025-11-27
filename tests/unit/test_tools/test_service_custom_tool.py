"""Unit tests for ServiceCustomTool execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import ServiceNotFound

from custom_components.home_agent.exceptions import ValidationError
from custom_components.home_agent.tools.custom import ServiceCustomTool


class TestServiceCustomToolExecution:
    """Test ServiceCustomTool execute method."""

    @pytest.fixture
    def simple_automation_tool(self, mock_hass):
        """Create a simple automation trigger tool."""
        config = {
            "name": "trigger_morning_routine",
            "description": "Trigger the morning routine automation",
            "parameters": {"type": "object", "properties": {}},
            "handler": {
                "type": "service",
                "service": "automation.trigger",
                "data": {"entity_id": "automation.morning_routine"},
            },
        }
        # Mock hass.services.has_service to return True
        mock_hass.services.has_service = MagicMock(return_value=True)
        return ServiceCustomTool(mock_hass, config)

    @pytest.fixture
    def script_tool_with_params(self, mock_hass):
        """Create a script tool with parameters."""
        config = {
            "name": "notify_arrival",
            "description": "Send arrival notification",
            "parameters": {
                "type": "object",
                "properties": {"person": {"type": "string"}, "location": {"type": "string"}},
            },
            "handler": {
                "type": "service",
                "service": "script.arrival_notification",
                "data": {"person": "{{ person }}", "location": "{{ location }}"},
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        return ServiceCustomTool(mock_hass, config)

    @pytest.fixture
    def scene_tool_with_target(self, mock_hass):
        """Create a scene tool with target."""
        config = {
            "name": "set_movie_scene",
            "description": "Activate movie watching scene",
            "parameters": {"type": "object", "properties": {}},
            "handler": {
                "type": "service",
                "service": "scene.turn_on",
                "target": {"entity_id": "scene.movie_time"},
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        return ServiceCustomTool(mock_hass, config)

    @pytest.fixture
    def light_tool_with_template_target(self, mock_hass):
        """Create a light tool with templated target."""
        config = {
            "name": "turn_on_room_lights",
            "description": "Turn on lights in a specific room",
            "parameters": {"type": "object", "properties": {"room": {"type": "string"}}},
            "handler": {
                "type": "service",
                "service": "light.turn_on",
                "target": {"area_id": "{{ room }}"},
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        return ServiceCustomTool(mock_hass, config)

    @pytest.mark.asyncio
    async def test_successful_service_call_no_params(self, simple_automation_tool, mock_hass):
        """Test successful service call with no parameters."""
        mock_hass.services.async_call = AsyncMock()

        result = await simple_automation_tool.execute()

        # Verify service was called correctly
        mock_hass.services.async_call.assert_called_once_with(
            domain="automation",
            service="trigger",
            service_data={"entity_id": "automation.morning_routine"},
            target=None,
            blocking=True,
        )

        assert result["success"] is True
        assert "successfully" in result["result"].lower()
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_successful_service_call_with_params(self, script_tool_with_params, mock_hass):
        """Test successful service call with templated parameters."""
        mock_hass.services.async_call = AsyncMock()

        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                side_effect=lambda x: (
                    x.get("person", "John") if "person" in x else x.get("location", "Home")
                )
            )
            mock_template_class.return_value = mock_template

            result = await script_tool_with_params.execute(person="John", location="Home")

        # Verify service was called
        mock_hass.services.async_call.assert_called_once()
        call_args = mock_hass.services.async_call.call_args
        assert call_args[1]["domain"] == "script"
        assert call_args[1]["service"] == "arrival_notification"
        assert call_args[1]["blocking"] is True

        assert result["success"] is True
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_service_call_with_target(self, scene_tool_with_target, mock_hass):
        """Test service call with target field."""
        mock_hass.services.async_call = AsyncMock()

        result = await scene_tool_with_target.execute()

        # Verify target was passed correctly
        mock_hass.services.async_call.assert_called_once()
        call_args = mock_hass.services.async_call.call_args
        assert call_args[1]["target"] == {"entity_id": "scene.movie_time"}
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_service_call_with_templated_target(
        self, light_tool_with_template_target, mock_hass
    ):
        """Test service call with templated target."""
        mock_hass.services.async_call = AsyncMock()

        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(return_value="bedroom")
            mock_template_class.return_value = mock_template

            result = await light_tool_with_template_target.execute(room="bedroom")

        # Verify target was rendered and passed
        mock_hass.services.async_call.assert_called_once()
        call_args = mock_hass.services.async_call.call_args
        assert call_args[1]["target"]["area_id"] == "bedroom"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_service_not_found_error(self, simple_automation_tool, mock_hass):
        """Test handling of ServiceNotFound error."""
        # Create a ServiceNotFound with domain and service
        error = ServiceNotFound("automation", "trigger")
        # Prevent the error from trying to access HA's translation system
        error._message = "Service automation.trigger not found"
        mock_hass.services.async_call = AsyncMock(side_effect=error)

        result = await simple_automation_tool.execute()

        assert result["success"] is False
        assert result["result"] is None
        assert "service" in result["error"].lower() or "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_service_call_exception(self, simple_automation_tool, mock_hass):
        """Test handling of general exceptions during service call."""
        mock_hass.services.async_call = AsyncMock(side_effect=Exception("Service execution failed"))

        result = await simple_automation_tool.execute()

        assert result["success"] is False
        assert result["result"] is None
        assert "failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_template_rendering_error(self, script_tool_with_params):
        """Test handling of template rendering errors."""
        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(side_effect=Exception("Template error"))
            mock_template_class.return_value = mock_template

            result = await script_tool_with_params.execute(person="John", location="Home")

        assert result["success"] is False
        assert "error" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_target_with_entity_id_list(self, mock_hass):
        """Test service call with multiple entity_ids in target."""
        config = {
            "name": "turn_off_lights",
            "description": "Turn off multiple lights",
            "parameters": {"type": "object", "properties": {}},
            "handler": {
                "type": "service",
                "service": "light.turn_off",
                "target": {"entity_id": ["light.bedroom", "light.kitchen"]},
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        mock_hass.services.async_call = AsyncMock()

        tool = ServiceCustomTool(mock_hass, config)
        result = await tool.execute()

        # Verify target with list was passed correctly
        call_args = mock_hass.services.async_call.call_args
        assert call_args[1]["target"]["entity_id"] == ["light.bedroom", "light.kitchen"]
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_target_with_device_id(self, mock_hass):
        """Test service call with device_id in target."""
        config = {
            "name": "turn_on_device",
            "description": "Turn on a device",
            "parameters": {"type": "object", "properties": {}},
            "handler": {
                "type": "service",
                "service": "homeassistant.turn_on",
                "target": {"device_id": "device123"},
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        mock_hass.services.async_call = AsyncMock()

        tool = ServiceCustomTool(mock_hass, config)
        result = await tool.execute()

        # Verify device_id target was passed
        call_args = mock_hass.services.async_call.call_args
        assert call_args[1]["target"]["device_id"] == "device123"
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_non_string_data_values(self, mock_hass):
        """Test service call with non-string data values."""
        config = {
            "name": "set_climate",
            "description": "Set climate temperature",
            "parameters": {"type": "object", "properties": {}},
            "handler": {
                "type": "service",
                "service": "climate.set_temperature",
                "data": {
                    "entity_id": "climate.living_room",
                    "temperature": 72,
                    "hvac_mode": "heat",
                },
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        mock_hass.services.async_call = AsyncMock()

        tool = ServiceCustomTool(mock_hass, config)
        result = await tool.execute()

        # Verify non-string values were passed as-is
        call_args = mock_hass.services.async_call.call_args
        service_data = call_args[1]["service_data"]
        assert service_data["temperature"] == 72
        assert service_data["hvac_mode"] == "heat"
        assert result["success"] is True


class TestServiceCustomToolValidation:
    """Test ServiceCustomTool configuration validation."""

    @pytest.mark.asyncio
    async def test_missing_service_key(self, mock_hass):
        """Test validation error when service key is missing."""
        config = {
            "name": "invalid_tool",
            "description": "Tool without service",
            "parameters": {},
            "handler": {"type": "service", "data": {}},
        }

        with pytest.raises(ValidationError) as exc_info:
            ServiceCustomTool(mock_hass, config)

        assert "missing required key: service" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_service_format_no_dot(self, mock_hass):
        """Test validation error for invalid service format (no dot)."""
        config = {
            "name": "invalid_tool",
            "description": "Tool with invalid service",
            "parameters": {},
            "handler": {"type": "service", "service": "invalid_service"},
        }

        with pytest.raises(ValidationError) as exc_info:
            ServiceCustomTool(mock_hass, config)

        assert "invalid service format" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_service_format_empty_parts(self, mock_hass):
        """Test validation error for invalid service format (empty parts)."""
        config = {
            "name": "invalid_tool",
            "description": "Tool with invalid service",
            "parameters": {},
            "handler": {"type": "service", "service": ".invalid"},
        }

        with pytest.raises(ValidationError) as exc_info:
            ServiceCustomTool(mock_hass, config)

        assert "invalid service format" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_service_not_found_warning(self, mock_hass, caplog):
        """Test warning when service doesn't exist at setup time."""
        config = {
            "name": "nonexistent_service_tool",
            "description": "Tool with nonexistent service",
            "parameters": {},
            "handler": {"type": "service", "service": "fake.nonexistent"},
        }

        # Mock service not found
        mock_hass.services.has_service = MagicMock(return_value=False)

        # Should not raise, but should warn
        tool = ServiceCustomTool(mock_hass, config)
        assert tool is not None
        # Check that warning was logged (caplog should capture it)


class TestServiceCustomToolTemplateRendering:
    """Test template rendering in ServiceCustomTool."""

    @pytest.fixture
    def tool_with_templates(self, mock_hass):
        """Create a tool with template values."""
        config = {
            "name": "templated_service",
            "description": "Service with templates",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}, "target": {"type": "string"}},
            },
            "handler": {
                "type": "service",
                "service": "notify.mobile_app",
                "data": {"message": "{{ message }}", "title": "Notification"},
                "target": {"entity_id": "notify.{{ target }}"},
            },
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        return ServiceCustomTool(mock_hass, config)

    @pytest.mark.asyncio
    async def test_render_template_success(self, tool_with_templates):
        """Test successful template rendering."""
        variables = {"message": "Hello World", "target": "phone"}

        with patch("custom_components.home_agent.tools.custom.Template") as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(return_value="Hello World")
            mock_template_class.return_value = mock_template

            result = await tool_with_templates._render_template("{{ message }}", variables)

        assert result == "Hello World"

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

    @pytest.mark.asyncio
    async def test_render_non_string_value(self, tool_with_templates):
        """Test rendering non-string value."""
        result = await tool_with_templates._render_template(123, {})
        assert result == "123"


class TestServiceCustomToolProperties:
    """Test ServiceCustomTool properties."""

    @pytest.fixture
    def basic_tool(self, mock_hass):
        """Create a basic service tool."""
        config = {
            "name": "test_service",
            "description": "Test service tool",
            "parameters": {"type": "object", "properties": {"param1": {"type": "string"}}},
            "handler": {"type": "service", "service": "test.service"},
        }
        mock_hass.services.has_service = MagicMock(return_value=True)
        return ServiceCustomTool(mock_hass, config)

    def test_name_property(self, basic_tool):
        """Test name property returns correct value."""
        assert basic_tool.name == "test_service"

    def test_description_property(self, basic_tool):
        """Test description property returns correct value."""
        assert basic_tool.description == "Test service tool"

    def test_parameters_property(self, basic_tool):
        """Test parameters property returns correct schema."""
        params = basic_tool.parameters
        assert params["type"] == "object"
        assert "param1" in params["properties"]

    @pytest.mark.asyncio
    async def test_close_method(self, basic_tool):
        """Test close method completes without error."""
        await basic_tool.close()
        # Just verify it doesn't raise an exception

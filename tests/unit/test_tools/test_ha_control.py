"""Unit tests for the HomeAssistantControlTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TOGGLE, SERVICE_TURN_OFF, SERVICE_TURN_ON
from homeassistant.core import State
from homeassistant.exceptions import HomeAssistantError

from custom_components.home_agent.const import (
    ACTION_SET_VALUE,
    ACTION_TOGGLE,
    ACTION_TURN_OFF,
    ACTION_TURN_ON,
    TOOL_HA_CONTROL,
)
from custom_components.home_agent.exceptions import (
    PermissionDenied,
    ToolExecutionError,
    ValidationError,
)
from custom_components.home_agent.tools.ha_control import HomeAssistantControlTool


class TestHomeAssistantControlTool:
    """Test the HomeAssistantControlTool class."""

    def test_tool_initialization(self, mock_hass):
        """Test that tool initializes correctly."""
        tool = HomeAssistantControlTool(mock_hass)
        assert tool.hass == mock_hass
        assert tool._exposed_entities is None

    def test_tool_initialization_with_exposed_entities(self, mock_hass, exposed_entities):
        """Test initialization with exposed entities."""
        tool = HomeAssistantControlTool(mock_hass, exposed_entities)
        assert tool._exposed_entities == exposed_entities

    def test_tool_name(self, mock_hass):
        """Test that tool name is correct."""
        tool = HomeAssistantControlTool(mock_hass)
        assert tool.name == TOOL_HA_CONTROL

    def test_tool_description(self, mock_hass):
        """Test that tool has a description."""
        tool = HomeAssistantControlTool(mock_hass)
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0
        assert "control" in tool.description.lower()

    def test_tool_parameters_schema(self, mock_hass):
        """Test that parameter schema is valid."""
        tool = HomeAssistantControlTool(mock_hass)
        params = tool.parameters

        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        # Check required parameters
        assert "action" in params["required"]
        assert "entity_id" in params["required"]

        # Check properties
        assert "action" in params["properties"]
        assert "entity_id" in params["properties"]
        assert "parameters" in params["properties"]

        # Check action enum
        action_enum = params["properties"]["action"]["enum"]
        assert ACTION_TURN_ON in action_enum
        assert ACTION_TURN_OFF in action_enum
        assert ACTION_TOGGLE in action_enum
        assert ACTION_SET_VALUE in action_enum

    def test_to_openai_format(self, mock_hass):
        """Test conversion to OpenAI format."""
        tool = HomeAssistantControlTool(mock_hass)
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == TOOL_HA_CONTROL
        assert "description" in openai_format["function"]
        assert "parameters" in openai_format["function"]

    @pytest.mark.asyncio
    async def test_execute_turn_on_light(self, mock_hass, sample_light_state):
        """Test turning on a light."""
        tool = HomeAssistantControlTool(mock_hass)

        # Mock entity registry
        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            # Mock state
            mock_hass.states.get.return_value = sample_light_state

            # Mock service call
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="light.living_room")

            assert result["success"] is True
            assert result["entity_id"] == "light.living_room"
            assert result["action"] == ACTION_TURN_ON
            assert result["new_state"] == "on"
            assert "message" in result

            # Verify service was called
            mock_hass.services.async_call.assert_called_once()
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][0] == "light"  # domain
            assert call_args[0][1] == SERVICE_TURN_ON  # service
            assert call_args[0][2][ATTR_ENTITY_ID] == "light.living_room"

    @pytest.mark.asyncio
    async def test_execute_turn_off_switch(self, mock_hass, sample_switch_state):
        """Test turning off a switch."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_switch_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_OFF, entity_id="switch.fan")

            assert result["success"] is True
            assert result["action"] == ACTION_TURN_OFF

            # Verify correct service was called
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][0] == "switch"
            assert call_args[0][1] == SERVICE_TURN_OFF

    @pytest.mark.asyncio
    async def test_execute_toggle(self, mock_hass, sample_light_state):
        """Test toggling an entity."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TOGGLE, entity_id="light.living_room")

            assert result["success"] is True
            assert result["action"] == ACTION_TOGGLE

            # Verify correct service was called
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][1] == SERVICE_TOGGLE

    @pytest.mark.asyncio
    async def test_execute_with_parameters(self, mock_hass, sample_light_state):
        """Test executing with additional parameters."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(
                action=ACTION_TURN_ON,
                entity_id="light.living_room",
                parameters={"brightness_pct": 50, "color_temp": 370},
            )

            assert result["success"] is True

            # Verify parameters were passed to service call (brightness_pct converted to brightness)
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["brightness"] == int(50 * 255 / 100)  # Converted from brightness_pct
            assert service_data["color_temp"] == 370

    @pytest.mark.asyncio
    async def test_execute_set_value_light(self, mock_hass, sample_light_state):
        """Test set_value action for light."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(
                action=ACTION_SET_VALUE,
                entity_id="light.living_room",
                parameters={"brightness_pct": 75},
            )

            assert result["success"] is True

            # For lights, set_value uses turn_on service
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][1] == SERVICE_TURN_ON

    @pytest.mark.asyncio
    async def test_execute_set_value_climate(self, mock_hass, sample_climate_state):
        """Test set_value action for climate."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(
                action=ACTION_SET_VALUE,
                entity_id="climate.thermostat",
                parameters={"temperature": 72},
            )

            assert result["success"] is True

            # For climate with temperature, uses set_temperature service
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][0] == "climate"
            assert call_args[0][1] == "set_temperature"

    @pytest.mark.asyncio
    async def test_execute_set_hvac_mode(self, mock_hass, sample_climate_state):
        """Test setting HVAC mode."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(
                action=ACTION_SET_VALUE,
                entity_id="climate.thermostat",
                parameters={"hvac_mode": "cool"},
            )

            assert result["success"] is True

            # Uses set_hvac_mode service
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][1] == "set_hvac_mode"

    @pytest.mark.asyncio
    async def test_missing_action_raises_validation_error(self, mock_hass):
        """Test that missing action raises ValidationError."""
        tool = HomeAssistantControlTool(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(entity_id="light.living_room")

        assert "action" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_entity_id_raises_validation_error(self, mock_hass):
        """Test that missing entity_id raises ValidationError."""
        tool = HomeAssistantControlTool(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(action=ACTION_TURN_ON)

        assert "entity_id" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_action_raises_validation_error(self, mock_hass):
        """Test that invalid action raises ValidationError."""
        tool = HomeAssistantControlTool(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(action="invalid_action", entity_id="light.living_room")

        assert "invalid action" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_entity_id_format_raises_validation_error(self, mock_hass):
        """Test that invalid entity_id format raises ValidationError."""
        tool = HomeAssistantControlTool(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(action=ACTION_TURN_ON, entity_id="invalid_format")  # Missing dot

        assert "invalid entity_id format" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_entity_access_validation_allowed(
        self, mock_hass, exposed_entities, sample_light_state
    ):
        """Test entity access validation when entity is allowed."""
        tool = HomeAssistantControlTool(mock_hass, exposed_entities)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            # Should succeed - light.living_room is in exposed_entities
            result = await tool.execute(action=ACTION_TURN_ON, entity_id="light.living_room")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_entity_access_validation_denied(self, mock_hass, exposed_entities):
        """Test entity access validation when entity is not allowed."""
        tool = HomeAssistantControlTool(mock_hass, exposed_entities)

        with pytest.raises(PermissionDenied) as exc_info:
            await tool.execute(
                action=ACTION_TURN_ON, entity_id="light.secret_room"  # Not in exposed_entities
            )

        assert "not accessible" in str(exc_info.value).lower()
        assert "light.secret_room" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_entity_access_validation_none_allows_all(self, mock_hass, sample_light_state):
        """Test that None exposed_entities allows all access."""
        tool = HomeAssistantControlTool(mock_hass, exposed_entities=None)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            # Should succeed even though entity is not in any list
            result = await tool.execute(action=ACTION_TURN_ON, entity_id="light.any_entity")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_nonexistent_entity_raises_validation_error(self, mock_hass):
        """Test that nonexistent entity raises ValidationError."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = None  # Not in registry
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = None  # Not in state machine

            with pytest.raises(ValidationError) as exc_info:
                await tool.execute(action=ACTION_TURN_ON, entity_id="light.nonexistent")

            assert "does not exist" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_entity_not_in_registry_but_in_state_machine(self, mock_hass, sample_light_state):
        """Test handling entity not in registry but in state machine."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = None  # Not in registry
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state  # But in state machine
            mock_hass.services.async_call = AsyncMock()

            # Should still work
            result = await tool.execute(action=ACTION_TURN_ON, entity_id="light.living_room")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_service_call_failure_raises_tool_execution_error(
        self, mock_hass, sample_light_state
    ):
        """Test that service call failure raises ToolExecutionError."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock(
                side_effect=HomeAssistantError("Service call failed")
            )

            with pytest.raises(ToolExecutionError) as exc_info:
                await tool.execute(action=ACTION_TURN_ON, entity_id="light.living_room")

            assert "failed to execute" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_response_includes_attributes(self, mock_hass, sample_light_state):
        """Test that response includes relevant attributes."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="light.living_room")

            assert "attributes" in result
            assert "friendly_name" in result["attributes"]
            assert "brightness_pct" in result["attributes"]  # Light-specific attribute (converted from brightness)

    @pytest.mark.asyncio
    async def test_climate_attributes_extraction(self, mock_hass, sample_climate_state):
        """Test that climate-specific attributes are extracted."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="climate.thermostat")

            assert "attributes" in result
            assert "temperature" in result["attributes"]
            assert "hvac_mode" in result["attributes"]

    def test_build_success_message(self, mock_hass):
        """Test building success messages."""
        tool = HomeAssistantControlTool(mock_hass)

        message = tool._build_success_message(ACTION_TURN_ON, "light.living_room", "on")

        assert "Living Room" in message
        assert "on" in message.lower()

    def test_get_set_value_service_for_light(self, mock_hass):
        """Test determining set_value service for light domain."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "light", "light.living_room", {"brightness_pct": 50})
        assert service == SERVICE_TURN_ON

    def test_get_set_value_service_for_climate_temperature(self, mock_hass):
        """Test determining set_value service for climate temperature."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "climate", "climate.thermostat", {"temperature": 72})
        assert service == "set_temperature"

    def test_get_set_value_service_for_climate_hvac_mode(self, mock_hass):
        """Test determining set_value service for climate HVAC mode."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "climate", "climate.thermostat", {"hvac_mode": "cool"})
        assert service == "set_hvac_mode"

    def test_get_set_value_service_for_cover(self, mock_hass):
        """Test determining set_value service for cover."""
        tool = HomeAssistantControlTool(mock_hass)

        # Mock a cover with position support
        mock_state = MagicMock()
        mock_state.attributes = {"supported_features": 4}  # CoverEntityFeature.SET_POSITION
        mock_hass.states.get.return_value = mock_state

        service = tool._get_service_for_action(ACTION_SET_VALUE, "cover", "cover.living_room", {"position": 50})
        assert service == "set_cover_position"

    def test_get_set_value_service_for_input_number(self, mock_hass):
        """Test determining set_value service for input_number."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "input_number", "input_number.slider", {"value": 42})
        assert service == "set_value"

    def test_get_set_value_service_for_fan_percentage(self, mock_hass):
        """Test determining set_value service for fan percentage."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "fan", "fan.bedroom", {"percentage": 75})
        assert service == "set_percentage"

    def test_get_set_value_service_unknown_domain(self, mock_hass):
        """Test determining set_value service for unknown domain defaults to turn_on."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "unknown_domain", "unknown.entity", {"param": "value"})
        assert service == SERVICE_TURN_ON

    def test_extract_relevant_attributes_light(self, mock_hass):
        """Test extracting relevant attributes for light."""
        tool = HomeAssistantControlTool(mock_hass)

        attrs = {
            "friendly_name": "Living Room",
            "brightness": 128,
            "color_temp": 370,
            "rgb_color": [255, 200, 150],
            "supported_features": 63,
            "irrelevant_attr": "should not be included",
        }

        relevant = tool._extract_relevant_attributes("light.living_room", attrs)

        assert "friendly_name" in relevant
        assert "brightness_pct" in relevant  # Converted from brightness
        assert relevant["brightness_pct"] == int(128 / 255 * 100)  # Should be 50
        assert "color_temp" in relevant
        assert "rgb_color" in relevant
        assert "irrelevant_attr" not in relevant

    def test_extract_relevant_attributes_climate(self, mock_hass):
        """Test extracting relevant attributes for climate."""
        tool = HomeAssistantControlTool(mock_hass)

        attrs = {
            "friendly_name": "Thermostat",
            "temperature": 72,
            "current_temperature": 71,
            "hvac_mode": "heat",
            "fan_mode": "auto",
            "supported_features": 91,
            "irrelevant_attr": "should not be included",
        }

        relevant = tool._extract_relevant_attributes("climate.thermostat", attrs)

        assert "friendly_name" in relevant
        assert "temperature" in relevant
        assert "current_temperature" in relevant
        assert "hvac_mode" in relevant
        assert "irrelevant_attr" not in relevant

    @pytest.mark.asyncio
    async def test_blocking_service_call(self, mock_hass, sample_light_state):
        """Test that service calls are blocking."""
        tool = HomeAssistantControlTool(mock_hass)

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = sample_light_state
            mock_hass.services.async_call = AsyncMock()

            await tool.execute(action=ACTION_TURN_ON, entity_id="light.living_room")

            # Verify blocking=True was passed
            call_kwargs = mock_hass.services.async_call.call_args[1]
            assert call_kwargs.get("blocking") is True

    # Parameter normalization tests
    def test_normalize_parameters_cover_current_position(self, mock_hass):
        """Test normalizing current_position to position for cover."""
        tool = HomeAssistantControlTool(mock_hass)

        normalized = tool._normalize_parameters("cover", {"current_position": 50})
        assert "position" in normalized
        assert normalized["position"] == 50
        assert "current_position" not in normalized

    def test_normalize_parameters_cover_current_tilt_position(self, mock_hass):
        """Test normalizing current_tilt_position to tilt_position for cover."""
        tool = HomeAssistantControlTool(mock_hass)

        normalized = tool._normalize_parameters("cover", {"current_tilt_position": 75})
        assert "tilt_position" in normalized
        assert normalized["tilt_position"] == 75
        assert "current_tilt_position" not in normalized

    def test_normalize_parameters_cover_preserves_position(self, mock_hass):
        """Test that existing position parameter is preserved."""
        tool = HomeAssistantControlTool(mock_hass)

        # If position already exists, don't normalize current_position
        normalized = tool._normalize_parameters("cover", {"position": 30, "current_position": 50})
        assert normalized["position"] == 30  # Original preserved
        assert "current_position" in normalized  # Not removed

    def test_normalize_parameters_climate_current_temperature(self, mock_hass):
        """Test normalizing current_temperature to temperature for climate."""
        tool = HomeAssistantControlTool(mock_hass)

        normalized = tool._normalize_parameters("climate", {"current_temperature": 72})
        assert "temperature" in normalized
        assert normalized["temperature"] == 72
        assert "current_temperature" not in normalized

    def test_normalize_parameters_converts_brightness_pct_for_lights(self, mock_hass):
        """Test that brightness_pct is converted to brightness for light domain."""
        tool = HomeAssistantControlTool(mock_hass)

        params = {"brightness_pct": 50}
        normalized = tool._normalize_parameters("light", params)
        assert "brightness_pct" not in normalized
        assert normalized["brightness"] == int(50 * 255 / 100)  # Should be 127

    @pytest.mark.asyncio
    async def test_execute_set_value_cover_with_current_position(self, mock_hass):
        """Test set_value with current_position parameter gets normalized and uses correct service."""
        tool = HomeAssistantControlTool(mock_hass)

        # Create a mock cover state with position support
        cover_state = MagicMock()
        cover_state.state = "open"
        cover_state.attributes = {
            "friendly_name": "Kitchen Window",
            "current_position": 0,
            "supported_features": 4,  # CoverEntityFeature.SET_POSITION
        }

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = cover_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(
                action=ACTION_SET_VALUE,
                entity_id="cover.kitchen_window",
                parameters={"current_position": 50},  # Using attribute name
            )

            assert result["success"] is True

            # Verify the correct service was called
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][0] == "cover"
            assert call_args[0][1] == "set_cover_position"  # Not turn_on!

            # Verify parameter was normalized
            service_data = call_args[0][2]
            assert "position" in service_data
            assert service_data["position"] == 50
            assert "current_position" not in service_data

    def test_get_set_value_service_for_media_player_volume(self, mock_hass):
        """Test determining set_value service for media_player volume."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "media_player", "media_player.tv", {"volume_level": 0.5})
        assert service == "volume_set"

    def test_get_set_value_service_for_media_player_source(self, mock_hass):
        """Test determining set_value service for media_player source."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "media_player", "media_player.tv", {"source": "TV"})
        assert service == "select_source"

    def test_get_set_value_service_for_humidifier(self, mock_hass):
        """Test determining set_value service for humidifier."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "humidifier", "humidifier.bedroom", {"humidity": 60})
        assert service == "set_humidity"

    def test_get_set_value_service_for_water_heater(self, mock_hass):
        """Test determining set_value service for water_heater."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "water_heater", "water_heater.tank", {"temperature": 120})
        assert service == "set_temperature"

    def test_get_set_value_service_for_number(self, mock_hass):
        """Test determining set_value service for number."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "number", "number.value", {"value": 42})
        assert service == "set_value"

    def test_get_set_value_service_for_select(self, mock_hass):
        """Test determining set_value service for select."""
        tool = HomeAssistantControlTool(mock_hass)

        service = tool._get_service_for_action(ACTION_SET_VALUE, "select", "select.option", {"option": "option1"})
        assert service == "select_option"

    def test_get_set_value_service_for_cover_tilt(self, mock_hass):
        """Test determining set_value service for cover tilt."""
        tool = HomeAssistantControlTool(mock_hass)

        # Mock a cover with tilt support
        mock_state = MagicMock()
        mock_state.attributes = {"supported_features": 128}  # CoverEntityFeature.SET_TILT_POSITION
        mock_hass.states.get.return_value = mock_state

        service = tool._get_service_for_action(ACTION_SET_VALUE, "cover", "cover.window", {"tilt_position": 45})
        assert service == "set_cover_tilt_position"

    def test_cover_binary_position_not_supported(self, mock_hass):
        """Test that binary covers raise error when trying to set position."""
        from custom_components.home_agent.exceptions import ToolExecutionError
        tool = HomeAssistantControlTool(mock_hass)

        # Mock a binary cover (no position support)
        mock_state = MagicMock()
        mock_state.attributes = {"supported_features": 3}  # OPEN + CLOSE only
        mock_hass.states.get.return_value = mock_state

        with pytest.raises(ToolExecutionError) as exc_info:
            tool._get_service_for_action(ACTION_SET_VALUE, "cover", "cover.kitchen_window", {"position": 50})

        assert "does not support position control" in str(exc_info.value)
        assert "binary cover" in str(exc_info.value)

    def test_cover_tilt_not_supported(self, mock_hass):
        """Test that covers without tilt raise error when trying to set tilt."""
        from custom_components.home_agent.exceptions import ToolExecutionError
        tool = HomeAssistantControlTool(mock_hass)

        # Mock cover with position but no tilt support
        mock_state = MagicMock()
        mock_state.attributes = {"supported_features": 4}  # Only SET_POSITION
        mock_hass.states.get.return_value = mock_state

        with pytest.raises(ToolExecutionError) as exc_info:
            tool._get_service_for_action(ACTION_SET_VALUE, "cover", "cover.window", {"tilt_position": 30})

        assert "does not support tilt" in str(exc_info.value)

    # Climate domain auto-injection tests
    @pytest.mark.asyncio
    async def test_climate_turn_off_auto_injects_hvac_mode_off(self, mock_hass):
        """Test that turn_off auto-injects hvac_mode='off' for climate entities."""
        tool = HomeAssistantControlTool(mock_hass)

        # Create climate state with available hvac_modes
        climate_state = State(
            "climate.thermostat",
            "heat",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "heat",
                "hvac_modes": ["off", "heat", "cool", "auto"],
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_OFF, entity_id="climate.thermostat")

            assert result["success"] is True
            assert result["action"] == ACTION_TURN_OFF

            # Verify set_hvac_mode service was called with hvac_mode="off"
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][0] == "climate"
            assert call_args[0][1] == "set_hvac_mode"
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "off"
            assert service_data[ATTR_ENTITY_ID] == "climate.thermostat"

    @pytest.mark.asyncio
    async def test_climate_turn_off_preserves_explicit_hvac_mode(self, mock_hass):
        """Test that explicit hvac_mode parameter is not overwritten on turn_off."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "heat",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "heat",
                "hvac_modes": ["off", "heat", "cool", "auto"],
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            # Explicitly provide hvac_mode (even though it's unusual for turn_off)
            result = await tool.execute(
                action=ACTION_TURN_OFF,
                entity_id="climate.thermostat",
                parameters={"hvac_mode": "cool"},  # Explicit mode should be preserved
            )

            assert result["success"] is True

            # Verify the explicit hvac_mode was preserved
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "cool"  # Not overwritten to "off"

    @pytest.mark.asyncio
    async def test_climate_turn_on_auto_selects_heat_cool(self, mock_hass):
        """Test that turn_on auto-selects heat_cool mode when available."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "off",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "off",
                "hvac_modes": ["off", "heat", "cool", "heat_cool"],
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="climate.thermostat")

            assert result["success"] is True

            # Verify set_hvac_mode was called with heat_cool (preferred mode)
            call_args = mock_hass.services.async_call.call_args
            assert call_args[0][0] == "climate"
            assert call_args[0][1] == "set_hvac_mode"
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "heat_cool"

    @pytest.mark.asyncio
    async def test_climate_turn_on_auto_selects_auto_when_no_heat_cool(self, mock_hass):
        """Test that turn_on selects auto mode when heat_cool is not available."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "off",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "off",
                "hvac_modes": ["off", "heat", "cool", "auto"],  # No heat_cool, but has auto
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="climate.thermostat")

            assert result["success"] is True

            # Verify set_hvac_mode was called with auto (second preference)
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "auto"

    @pytest.mark.asyncio
    async def test_climate_turn_on_auto_selects_heat_when_no_heat_cool(self, mock_hass):
        """Test that turn_on falls back to heat when heat_cool and auto are not available."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "off",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "off",
                "hvac_modes": ["off", "heat", "cool"],  # No heat_cool or auto
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="climate.thermostat")

            assert result["success"] is True

            # Verify set_hvac_mode was called with heat (third preference)
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "heat"

    @pytest.mark.asyncio
    async def test_climate_turn_on_auto_selects_cool_when_only_cool_available(self, mock_hass):
        """Test that turn_on falls back to cool when only cool mode is available."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "off",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "off",
                "hvac_modes": ["off", "cool"],  # Only cool mode
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="climate.thermostat")

            assert result["success"] is True

            # Verify set_hvac_mode was called with cool (fourth preference)
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "cool"

    @pytest.mark.asyncio
    async def test_climate_turn_on_uses_first_non_off_mode(self, mock_hass):
        """Test that turn_on uses first non-off mode when preferred modes unavailable."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "off",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "off",
                "hvac_modes": ["off", "dry", "fan_only"],  # No standard modes
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            result = await tool.execute(action=ACTION_TURN_ON, entity_id="climate.thermostat")

            assert result["success"] is True

            # Verify set_hvac_mode was called with first non-off mode
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "dry"  # First non-off mode

    @pytest.mark.asyncio
    async def test_climate_turn_on_preserves_explicit_hvac_mode(self, mock_hass):
        """Test that explicit hvac_mode parameter is not overwritten on turn_on."""
        tool = HomeAssistantControlTool(mock_hass)

        climate_state = State(
            "climate.thermostat",
            "off",
            attributes={
                "friendly_name": "Thermostat",
                "temperature": 72,
                "hvac_mode": "off",
                "hvac_modes": ["off", "heat", "cool", "heat_cool", "auto"],
                "supported_features": 1,
            },
        )

        with patch("custom_components.home_agent.tools.ha_control.er.async_get") as mock_er:
            mock_registry = MagicMock()
            mock_registry.async_get.return_value = MagicMock()
            mock_er.return_value = mock_registry

            mock_hass.states.get.return_value = climate_state
            mock_hass.services.async_call = AsyncMock()

            # Explicitly provide hvac_mode
            result = await tool.execute(
                action=ACTION_TURN_ON,
                entity_id="climate.thermostat",
                parameters={"hvac_mode": "cool"},  # Explicit mode
            )

            assert result["success"] is True

            # Verify the explicit hvac_mode was preserved (not auto-selected)
            call_args = mock_hass.services.async_call.call_args
            service_data = call_args[0][2]
            assert service_data["hvac_mode"] == "cool"  # Not auto-selected heat_cool

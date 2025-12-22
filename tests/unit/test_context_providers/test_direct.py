"""Unit tests for direct context provider.

This module tests the DirectContextProvider which directly fetches
configured entities and formats them for LLM consumption.
"""

import json
from unittest.mock import Mock

import pytest
from homeassistant.core import State

from custom_components.home_agent.context_providers.direct import DirectContextProvider


class TestDirectContextProviderInit:
    """Tests for DirectContextProvider initialization."""

    def test_direct_context_provider_init(self, mock_hass):
        """Test initializing direct context provider."""
        config = {
            "entities": [{"entity_id": "light.living_room", "attributes": ["brightness"]}],
            "format": "json",
        }
        provider = DirectContextProvider(mock_hass, config)

        assert provider.hass == mock_hass
        assert provider.config == config
        assert provider.entities_config == config["entities"]
        assert provider.format_type == "json"

    def test_direct_context_provider_default_format(self, mock_hass):
        """Test default format is json."""
        config = {"entities": []}
        provider = DirectContextProvider(mock_hass, config)

        assert provider.format_type == "json"

    def test_direct_context_provider_natural_language_format(self, mock_hass):
        """Test natural_language format."""
        config = {"entities": [], "format": "natural_language"}
        provider = DirectContextProvider(mock_hass, config)

        assert provider.format_type == "natural_language"

    def test_direct_context_provider_empty_entities(self, mock_hass):
        """Test initialization with empty entities list."""
        config = {"format": "json"}
        provider = DirectContextProvider(mock_hass, config)

        assert provider.entities_config == []


class TestGetContext:
    """Tests for get_context method."""

    @pytest.mark.asyncio
    async def test_get_context_json_single_entity(self, mock_hass):
        """Test getting context in JSON format for single entity."""
        config = {
            "entities": [{"entity_id": "light.living_room", "attributes": ["brightness"]}],
            "format": "json",
        }
        provider = DirectContextProvider(mock_hass, config)

        # Mock entity state
        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {"brightness": 128, "friendly_name": "Living Room Light"}
        mock_hass.states.get.return_value = state

        result = await provider.get_context("turn on the lights")

        # Parse JSON result
        parsed = json.loads(result)
        assert "entities" in parsed
        assert "count" in parsed
        assert parsed["count"] == 1
        assert parsed["entities"][0]["entity_id"] == "light.living_room"
        assert parsed["entities"][0]["state"] == "on"

    @pytest.mark.asyncio
    async def test_get_context_json_multiple_entities(self, mock_hass):
        """Test getting context for multiple entities."""
        config = {
            "entities": [{"entity_id": "light.living_room"}, {"entity_id": "sensor.temperature"}],
            "format": "json",
        }
        provider = DirectContextProvider(mock_hass, config)

        # Mock entity states
        light_state = Mock(spec=State)
        light_state.entity_id = "light.living_room"
        light_state.state = "on"
        light_state.attributes = {"brightness": 128}

        sensor_state = Mock(spec=State)
        sensor_state.entity_id = "sensor.temperature"
        sensor_state.state = "72"
        sensor_state.attributes = {"unit_of_measurement": "°F"}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light_state
            elif entity_id == "sensor.temperature":
                return sensor_state
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        result = await provider.get_context("what's the temperature?")

        parsed = json.loads(result)
        assert parsed["count"] == 2
        assert len(parsed["entities"]) == 2

    @pytest.mark.asyncio
    async def test_get_context_natural_language(self, mock_hass):
        """Test getting context in natural language format."""
        config = {"entities": [{"entity_id": "light.living_room"}], "format": "natural_language"}
        provider = DirectContextProvider(mock_hass, config)

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {"brightness": 128, "friendly_name": "Living Room Light"}
        mock_hass.states.get.return_value = state

        result = await provider.get_context("status")

        assert isinstance(result, str)
        assert "Current Home State:" in result
        assert "Living Room Light" in result
        assert "on" in result

    @pytest.mark.asyncio
    async def test_get_context_entity_not_found(self, mock_hass):
        """Test getting context when entity doesn't exist."""
        config = {"entities": [{"entity_id": "light.nonexistent"}], "format": "json"}
        provider = DirectContextProvider(mock_hass, config)
        mock_hass.states.get.return_value = None

        result = await provider.get_context("test")

        parsed = json.loads(result)
        assert parsed["count"] == 0
        assert len(parsed["entities"]) == 0

    @pytest.mark.asyncio
    async def test_get_context_wildcard_entities(self, mock_hass):
        """Test getting context with wildcard pattern."""
        config = {"entities": [{"entity_id": "light.*"}], "format": "json"}
        provider = DirectContextProvider(mock_hass, config)

        # Mock entity list
        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom",
            "sensor.temperature",
        ]

        # Mock entity states
        light1 = Mock(spec=State)
        light1.entity_id = "light.living_room"
        light1.state = "on"
        light1.attributes = {}

        light2 = Mock(spec=State)
        light2.entity_id = "light.bedroom"
        light2.state = "off"
        light2.attributes = {}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light1
            elif entity_id == "light.bedroom":
                return light2
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        result = await provider.get_context("test")

        parsed = json.loads(result)
        assert parsed["count"] == 2

    @pytest.mark.asyncio
    async def test_get_context_invalid_format_type(self, mock_hass):
        """Test that invalid format type raises ValueError."""
        config = {"entities": [], "format": "invalid"}
        provider = DirectContextProvider(mock_hass, config)

        with pytest.raises(ValueError) as exc_info:
            await provider.get_context("test")

        assert "Invalid format type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_context_no_entities_configured(self, mock_hass):
        """Test getting context with no entities configured."""
        config = {"entities": [], "format": "json"}
        provider = DirectContextProvider(mock_hass, config)

        result = await provider.get_context("test")

        parsed = json.loads(result)
        assert parsed["count"] == 0
        assert parsed["entities"] == []

    @pytest.mark.asyncio
    async def test_get_context_empty_natural_language(self, mock_hass):
        """Test natural language format with no entities."""
        config = {"entities": [], "format": "natural_language"}
        provider = DirectContextProvider(mock_hass, config)

        result = await provider.get_context("test")

        assert "No entities currently configured" in result

    @pytest.mark.asyncio
    async def test_get_context_with_attribute_filter(self, mock_hass):
        """Test getting context with attribute filter."""
        config = {
            "entities": [{"entity_id": "light.living_room", "attributes": ["brightness"]}],
            "format": "json",
        }
        provider = DirectContextProvider(mock_hass, config)

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {
            "brightness": 128,
            "color_temp": 370,
            "friendly_name": "Living Room Light",
        }
        mock_hass.states.get.return_value = state

        result = await provider.get_context("test")

        parsed = json.loads(result)
        entity = parsed["entities"][0]
        assert "brightness_pct" in entity["attributes"]  # Converted from brightness
        # color_temp should not be included due to filter
        assert "color_temp" not in entity["attributes"]

    @pytest.mark.asyncio
    async def test_get_context_missing_entity_id(self, mock_hass):
        """Test handling entity config missing entity_id."""
        config = {
            "entities": [{"attributes": ["brightness"]}],  # Missing entity_id
            "format": "json",
        }
        provider = DirectContextProvider(mock_hass, config)

        result = await provider.get_context("test")

        parsed = json.loads(result)
        assert parsed["count"] == 0  # Should skip invalid config


class TestFormatAsJson:
    """Tests for _format_as_json method."""

    def test_format_as_json_empty(self, mock_hass):
        """Test formatting empty entity list as JSON."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_as_json([])

        parsed = json.loads(result)
        assert parsed["entities"] == []
        assert parsed["count"] == 0

    def test_format_as_json_single_entity(self, mock_hass):
        """Test formatting single entity as JSON."""
        provider = DirectContextProvider(mock_hass, {})
        entity_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {"brightness": 128}}
        ]

        result = provider._format_as_json(entity_states)

        parsed = json.loads(result)
        assert parsed["count"] == 1
        assert len(parsed["entities"]) == 1
        assert parsed["entities"][0]["entity_id"] == "light.living_room"

    def test_format_as_json_multiple_entities(self, mock_hass):
        """Test formatting multiple entities as JSON."""
        provider = DirectContextProvider(mock_hass, {})
        entity_states = [
            {"entity_id": "light.living_room", "state": "on", "attributes": {}},
            {"entity_id": "sensor.temp", "state": "72", "attributes": {}},
        ]

        result = provider._format_as_json(entity_states)

        parsed = json.loads(result)
        assert parsed["count"] == 2
        assert len(parsed["entities"]) == 2

    def test_format_as_json_pretty_printed(self, mock_hass):
        """Test that JSON is formatted with indentation."""
        provider = DirectContextProvider(mock_hass, {})
        entity_states = [{"entity_id": "test", "state": "on", "attributes": {}}]

        result = provider._format_as_json(entity_states)

        # Should contain newlines from indentation
        assert "\n" in result
        assert "  " in result  # Indentation


class TestFormatAsNaturalLanguage:
    """Tests for _format_as_natural_language method."""

    def test_format_natural_language_empty(self, mock_hass):
        """Test formatting empty entity list as natural language."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_as_natural_language([])

        assert "No entities currently configured" in result

    def test_format_natural_language_single_entity(self, mock_hass):
        """Test formatting single entity as natural language."""
        provider = DirectContextProvider(mock_hass, {})
        entity_states = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "Living Room Light"},
            }
        ]

        result = provider._format_as_natural_language(entity_states)

        assert "Current Home State:" in result
        assert "Living Room Light" in result
        assert "on" in result

    def test_format_natural_language_no_friendly_name(self, mock_hass):
        """Test formatting without friendly_name."""
        provider = DirectContextProvider(mock_hass, {})
        entity_states = [{"entity_id": "sensor.temperature", "state": "72", "attributes": {}}]

        result = provider._format_as_natural_language(entity_states)

        # Should use formatted entity name
        assert "Temperature" in result

    def test_format_natural_language_multiple_entities(self, mock_hass):
        """Test formatting multiple entities."""
        provider = DirectContextProvider(mock_hass, {})
        entity_states = [
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "Living Room Light"},
            },
            {
                "entity_id": "sensor.temp",
                "state": "72",
                "attributes": {"friendly_name": "Temperature"},
            },
        ]

        result = provider._format_as_natural_language(entity_states)

        assert "Living Room Light" in result
        assert "Temperature" in result
        assert result.count("-") >= 2  # Each entity should have a bullet


class TestFormatEntityNaturalLanguage:
    """Tests for domain-specific natural language formatting."""

    def test_format_light_on(self, mock_hass):
        """Test formatting light entity that is on."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_light("Living Room Light", "on", {"brightness": 128})

        assert "Living Room Light is on" in result
        assert "50%" in result  # 128/255 * 100 ≈ 50%

    def test_format_light_off(self, mock_hass):
        """Test formatting light entity that is off."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_light("Bedroom Light", "off", {})

        assert "Bedroom Light is off" in result

    def test_format_light_with_color_temp(self, mock_hass):
        """Test formatting light with color temperature."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_light(
            "Kitchen Light", "on", {"brightness": 255, "color_temp": 370}
        )

        assert "Kitchen Light is on" in result
        assert "100%" in result
        assert "370K" in result

    def test_format_sensor_with_unit(self, mock_hass):
        """Test formatting sensor with unit of measurement."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_sensor("Temperature", "72", {"unit_of_measurement": "°F"})

        assert "Temperature is 72 °F" in result

    def test_format_sensor_with_device_class(self, mock_hass):
        """Test formatting sensor with device class."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_sensor("Battery", "95", {"device_class": "battery"})

        assert "Battery (battery) is 95" in result

    def test_format_binary_sensor_door_open(self, mock_hass):
        """Test formatting binary sensor for open door."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_binary_sensor("Front Door", "on", {"device_class": "door"})

        assert "Front Door is open" in result

    def test_format_binary_sensor_door_closed(self, mock_hass):
        """Test formatting binary sensor for closed door."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_binary_sensor("Front Door", "off", {"device_class": "door"})

        assert "Front Door is closed" in result

    def test_format_binary_sensor_motion(self, mock_hass):
        """Test formatting binary sensor for motion."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_binary_sensor(
            "Living Room Motion", "on", {"device_class": "motion"}
        )

        assert "Living Room Motion detects motion" in result

    def test_format_climate(self, mock_hass):
        """Test formatting climate entity."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_climate(
            "Thermostat",
            "heat",
            {"current_temperature": 68, "target_temperature": 70, "temperature_unit": "°F"},
        )

        assert "Thermostat is heat" in result
        assert "68°F" in result
        assert "70°F" in result

    def test_format_switch(self, mock_hass):
        """Test formatting switch entity."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_switch("Outlet", "on", {})

        assert "Outlet is on" in result

    def test_format_lock(self, mock_hass):
        """Test formatting lock entity."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_lock("Front Door", "locked", {})

        assert "Front Door is locked" in result

    def test_format_unknown_domain(self, mock_hass):
        """Test formatting entity with unknown domain."""
        provider = DirectContextProvider(mock_hass, {})

        result = provider._format_entity_natural_language(
            "unknown_domain", "Test Entity", "active", {}
        )

        assert "Test Entity is active" in result


class TestGatherEntityStates:
    """Tests for _gather_entity_states method."""

    @pytest.mark.asyncio
    async def test_gather_entity_states_single(self, mock_hass):
        """Test gathering states for single entity."""
        provider = DirectContextProvider(mock_hass, {})
        provider.entities_config = [{"entity_id": "light.living_room"}]

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {}
        mock_hass.states.get.return_value = state

        result = await provider._gather_entity_states()

        assert len(result) == 1
        assert result[0]["entity_id"] == "light.living_room"

    @pytest.mark.asyncio
    async def test_gather_entity_states_multiple(self, mock_hass):
        """Test gathering states for multiple entities."""
        provider = DirectContextProvider(mock_hass, {})
        provider.entities_config = [
            {"entity_id": "light.living_room"},
            {"entity_id": "sensor.temperature"},
        ]

        light_state = Mock(spec=State)
        light_state.entity_id = "light.living_room"
        light_state.state = "on"
        light_state.attributes = {}

        sensor_state = Mock(spec=State)
        sensor_state.entity_id = "sensor.temperature"
        sensor_state.state = "72"
        sensor_state.attributes = {}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light_state
            elif entity_id == "sensor.temperature":
                return sensor_state
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        result = await provider._gather_entity_states()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_gather_entity_states_with_wildcards(self, mock_hass):
        """Test gathering states with wildcard patterns."""
        provider = DirectContextProvider(mock_hass, {})
        provider.entities_config = [{"entity_id": "light.*"}]

        mock_hass.states.async_entity_ids.return_value = ["light.living_room", "light.bedroom"]

        light1 = Mock(spec=State)
        light1.entity_id = "light.living_room"
        light1.state = "on"
        light1.attributes = {}

        light2 = Mock(spec=State)
        light2.entity_id = "light.bedroom"
        light2.state = "off"
        light2.attributes = {}

        def get_state_side_effect(entity_id):
            if entity_id == "light.living_room":
                return light1
            elif entity_id == "light.bedroom":
                return light2
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        result = await provider._gather_entity_states()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_gather_entity_states_skip_not_found(self, mock_hass):
        """Test that non-existent entities are skipped."""
        provider = DirectContextProvider(mock_hass, {})
        provider.entities_config = [
            {"entity_id": "light.existing"},
            {"entity_id": "light.nonexistent"},
        ]

        existing_state = Mock(spec=State)
        existing_state.entity_id = "light.existing"
        existing_state.state = "on"
        existing_state.attributes = {}

        def get_state_side_effect(entity_id):
            if entity_id == "light.existing":
                return existing_state
            return None

        mock_hass.states.get.side_effect = get_state_side_effect

        result = await provider._gather_entity_states()

        assert len(result) == 1
        assert result[0]["entity_id"] == "light.existing"

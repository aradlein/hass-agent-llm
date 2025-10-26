"""Unit tests for base context provider.

This module tests the abstract base class for context providers,
including helper methods and the abstract interface.
"""

import pytest
from unittest.mock import MagicMock, Mock

from homeassistant.core import HomeAssistant, State

from custom_components.home_agent.context_providers.base import ContextProvider


class ConcreteContextProvider(ContextProvider):
    """Concrete implementation of ContextProvider for testing."""

    async def get_context(self, user_input: str) -> str:
        """Concrete implementation of get_context."""
        return f"Context for: {user_input}"


class TestContextProviderInit:
    """Tests for ContextProvider initialization."""

    def test_context_provider_init(self, mock_hass):
        """Test initializing context provider."""
        config = {"test": "value"}
        provider = ConcreteContextProvider(mock_hass, config)

        assert provider.hass == mock_hass
        assert provider.config == config
        assert provider._logger is not None

    def test_context_provider_init_empty_config(self, mock_hass):
        """Test initializing with empty config."""
        provider = ConcreteContextProvider(mock_hass, {})

        assert provider.hass == mock_hass
        assert provider.config == {}

    def test_context_provider_logger_name(self, mock_hass):
        """Test that logger includes class name."""
        provider = ConcreteContextProvider(mock_hass, {})

        assert "ConcreteContextProvider" in provider._logger.name


class TestContextProviderAbstractMethod:
    """Tests for abstract method enforcement."""

    def test_cannot_instantiate_base_class(self, mock_hass):
        """Test that base ContextProvider cannot be instantiated."""
        with pytest.raises(TypeError):
            # Should fail because get_context is abstract
            ContextProvider(mock_hass, {})

    def test_concrete_implementation_required(self, mock_hass):
        """Test that concrete class must implement get_context."""
        class IncompleteProvider(ContextProvider):
            """Provider missing get_context implementation."""
            pass

        with pytest.raises(TypeError):
            IncompleteProvider(mock_hass, {})

    @pytest.mark.asyncio
    async def test_concrete_implementation_works(self, mock_hass):
        """Test that concrete implementation can be used."""
        provider = ConcreteContextProvider(mock_hass, {})
        result = await provider.get_context("test input")

        assert result == "Context for: test input"


class TestFormatEntityState:
    """Tests for _format_entity_state helper method."""

    def test_format_entity_state_basic(self, mock_hass):
        """Test formatting basic entity state."""
        provider = ConcreteContextProvider(mock_hass, {})

        result = provider._format_entity_state(
            entity_id="light.living_room",
            state="on"
        )

        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert "attributes" not in result

    def test_format_entity_state_with_attributes(self, mock_hass):
        """Test formatting entity state with attributes."""
        provider = ConcreteContextProvider(mock_hass, {})
        attributes = {
            "brightness": 128,
            "color_temp": 370
        }

        result = provider._format_entity_state(
            entity_id="light.living_room",
            state="on",
            attributes=attributes
        )

        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["attributes"] == attributes
        assert result["attributes"]["brightness"] == 128

    def test_format_entity_state_none_state(self, mock_hass):
        """Test formatting with None state."""
        provider = ConcreteContextProvider(mock_hass, {})

        result = provider._format_entity_state(
            entity_id="sensor.temperature",
            state=None
        )

        assert result["entity_id"] == "sensor.temperature"
        assert result["state"] == "None"

    def test_format_entity_state_numeric_state(self, mock_hass):
        """Test formatting with numeric state."""
        provider = ConcreteContextProvider(mock_hass, {})

        result = provider._format_entity_state(
            entity_id="sensor.temperature",
            state=72.5
        )

        assert result["entity_id"] == "sensor.temperature"
        assert result["state"] == "72.5"

    def test_format_entity_state_empty_attributes(self, mock_hass):
        """Test formatting with empty attributes dict."""
        provider = ConcreteContextProvider(mock_hass, {})

        result = provider._format_entity_state(
            entity_id="switch.outlet",
            state="off",
            attributes={}
        )

        assert result["entity_id"] == "switch.outlet"
        assert result["state"] == "off"
        assert result["attributes"] == {}


class TestGetEntityState:
    """Tests for _get_entity_state helper method."""

    def test_get_entity_state_success(self, mock_hass):
        """Test getting entity state successfully."""
        provider = ConcreteContextProvider(mock_hass, {})

        # Mock state object
        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {
            "brightness": 128,
            "friendly_name": "Living Room Light"
        }
        mock_hass.states.get.return_value = state

        result = provider._get_entity_state("light.living_room")

        assert result is not None
        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["attributes"]["brightness"] == 128
        assert result["attributes"]["friendly_name"] == "Living Room Light"
        mock_hass.states.get.assert_called_once_with("light.living_room")

    def test_get_entity_state_not_found(self, mock_hass):
        """Test getting entity state when entity doesn't exist."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.get.return_value = None

        result = provider._get_entity_state("light.nonexistent")

        assert result is None
        mock_hass.states.get.assert_called_once_with("light.nonexistent")

    def test_get_entity_state_with_attribute_filter(self, mock_hass):
        """Test getting entity state with attribute filter."""
        provider = ConcreteContextProvider(mock_hass, {})

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {
            "brightness": 128,
            "color_temp": 370,
            "friendly_name": "Living Room Light",
            "other_attr": "value"
        }
        mock_hass.states.get.return_value = state

        result = provider._get_entity_state(
            "light.living_room",
            attribute_filter=["brightness", "color_temp"]
        )

        assert result is not None
        assert result["attributes"]["brightness"] == 128
        assert result["attributes"]["color_temp"] == 370
        assert "friendly_name" not in result["attributes"]
        assert "other_attr" not in result["attributes"]

    def test_get_entity_state_with_empty_filter(self, mock_hass):
        """Test getting entity state with empty attribute filter."""
        provider = ConcreteContextProvider(mock_hass, {})

        state = Mock(spec=State)
        state.entity_id = "sensor.temperature"
        state.state = "72"
        state.attributes = {"unit": "°F"}
        mock_hass.states.get.return_value = state

        result = provider._get_entity_state(
            "sensor.temperature",
            attribute_filter=[]
        )

        assert result is not None
        assert result["attributes"] == {}

    def test_get_entity_state_no_attributes(self, mock_hass):
        """Test getting entity state with no attributes."""
        provider = ConcreteContextProvider(mock_hass, {})

        state = Mock(spec=State)
        state.entity_id = "binary_sensor.door"
        state.state = "on"
        state.attributes = {}
        mock_hass.states.get.return_value = state

        result = provider._get_entity_state("binary_sensor.door")

        assert result is not None
        assert result["attributes"] == {}

    def test_get_entity_state_filter_nonexistent_attribute(self, mock_hass):
        """Test attribute filter with nonexistent attribute."""
        provider = ConcreteContextProvider(mock_hass, {})

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {"brightness": 128}
        mock_hass.states.get.return_value = state

        result = provider._get_entity_state(
            "light.living_room",
            attribute_filter=["nonexistent", "brightness"]
        )

        assert result is not None
        assert "nonexistent" not in result["attributes"]
        assert result["attributes"]["brightness"] == 128


class TestGetEntitiesMatchingPattern:
    """Tests for _get_entities_matching_pattern helper method."""

    def test_get_entities_exact_match(self, mock_hass):
        """Test getting entities with exact match (no wildcard)."""
        provider = ConcreteContextProvider(mock_hass, {})

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        mock_hass.states.get.return_value = state

        result = provider._get_entities_matching_pattern("light.living_room")

        assert result == ["light.living_room"]
        mock_hass.states.get.assert_called_once_with("light.living_room")

    def test_get_entities_exact_match_not_found(self, mock_hass):
        """Test exact match when entity doesn't exist."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.get.return_value = None

        result = provider._get_entities_matching_pattern("light.nonexistent")

        assert result == []

    def test_get_entities_wildcard_domain(self, mock_hass):
        """Test getting entities with domain wildcard."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom",
            "sensor.temperature",
            "switch.outlet"
        ]

        result = provider._get_entities_matching_pattern("light.*")

        assert len(result) == 2
        assert "light.living_room" in result
        assert "light.bedroom" in result
        assert "sensor.temperature" not in result

    def test_get_entities_wildcard_entity_name(self, mock_hass):
        """Test getting entities with entity name wildcard."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.async_entity_ids.return_value = [
            "sensor.living_room_temperature",
            "sensor.bedroom_temperature",
            "sensor.living_room_humidity",
            "light.living_room"
        ]

        result = provider._get_entities_matching_pattern("sensor.*_temperature")

        assert len(result) == 2
        assert "sensor.living_room_temperature" in result
        assert "sensor.bedroom_temperature" in result
        assert "sensor.living_room_humidity" not in result

    def test_get_entities_wildcard_no_matches(self, mock_hass):
        """Test wildcard pattern with no matches."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "sensor.temperature"
        ]

        result = provider._get_entities_matching_pattern("switch.*")

        assert result == []

    def test_get_entities_wildcard_all(self, mock_hass):
        """Test getting all entities with full wildcard."""
        provider = ConcreteContextProvider(mock_hass, {})
        all_entities = [
            "light.living_room",
            "sensor.temperature",
            "switch.outlet"
        ]
        mock_hass.states.async_entity_ids.return_value = all_entities

        result = provider._get_entities_matching_pattern("*")

        assert len(result) == 3
        assert set(result) == set(all_entities)

    def test_get_entities_wildcard_middle(self, mock_hass):
        """Test wildcard in middle of pattern."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.async_entity_ids.return_value = [
            "sensor.living_room_temp",
            "sensor.bedroom_temp",
            "sensor.living_room_humidity"
        ]

        result = provider._get_entities_matching_pattern("sensor.*_temp")

        assert len(result) == 2
        assert "sensor.living_room_temp" in result
        assert "sensor.bedroom_temp" in result

    def test_get_entities_complex_wildcard(self, mock_hass):
        """Test complex wildcard pattern."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.async_entity_ids.return_value = [
            "binary_sensor.door_1",
            "binary_sensor.door_2",
            "binary_sensor.window_1",
            "sensor.door_1"
        ]

        result = provider._get_entities_matching_pattern("binary_sensor.door_*")

        assert len(result) == 2
        assert "binary_sensor.door_1" in result
        assert "binary_sensor.door_2" in result
        assert "binary_sensor.window_1" not in result

    def test_get_entities_empty_entity_list(self, mock_hass):
        """Test when no entities exist."""
        provider = ConcreteContextProvider(mock_hass, {})
        mock_hass.states.async_entity_ids.return_value = []

        result = provider._get_entities_matching_pattern("light.*")

        assert result == []


class TestContextProviderIntegration:
    """Integration tests for ContextProvider functionality."""

    def test_full_workflow_single_entity(self, mock_hass):
        """Test complete workflow for getting single entity context."""
        provider = ConcreteContextProvider(mock_hass, {})

        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {"brightness": 128}
        mock_hass.states.get.return_value = state

        # Get entity state
        entity_state = provider._get_entity_state("light.living_room")

        assert entity_state is not None
        assert entity_state["entity_id"] == "light.living_room"

    def test_full_workflow_wildcard_entities(self, mock_hass):
        """Test complete workflow for getting multiple entities via wildcard."""
        provider = ConcreteContextProvider(mock_hass, {})

        mock_hass.states.async_entity_ids.return_value = [
            "light.living_room",
            "light.bedroom"
        ]

        # Mock individual entity states
        light1 = Mock(spec=State)
        light1.entity_id = "light.living_room"
        light1.state = "on"
        light1.attributes = {"brightness": 128}

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

        # Get matching entities
        matching = provider._get_entities_matching_pattern("light.*")
        assert len(matching) == 2

        # Get state for each
        states = []
        for entity_id in matching:
            entity_state = provider._get_entity_state(entity_id)
            if entity_state:
                states.append(entity_state)

        assert len(states) == 2
        assert any(s["entity_id"] == "light.living_room" for s in states)
        assert any(s["entity_id"] == "light.bedroom" for s in states)

    @pytest.mark.asyncio
    async def test_concrete_provider_get_context(self, mock_hass):
        """Test concrete provider's get_context method."""
        provider = ConcreteContextProvider(mock_hass, {})

        result = await provider.get_context("turn on the lights")

        assert result == "Context for: turn on the lights"

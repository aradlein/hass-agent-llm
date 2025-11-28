"""Test that available_services does not include duplicate homeassistant.* services."""

import pytest
from unittest.mock import Mock, patch

from custom_components.home_agent.context_providers.base import (
    get_entity_available_services,
)
from homeassistant.core import HomeAssistant


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = Mock()

    # Mock the service registry - simulates real HA behavior
    # Each domain has its own services, homeassistant domain is separate
    hass.services = Mock()
    hass.services.async_services.return_value = {
        "light": {
            "turn_on": Mock(),
            "turn_off": Mock(),
            "toggle": Mock(),
        },
        "cover": {
            "open_cover": Mock(),
            "close_cover": Mock(),
            "stop_cover": Mock(),
            "set_cover_position": Mock(),
            "toggle": Mock(),
        },
        "homeassistant": {
            "turn_on": Mock(),
            "turn_off": Mock(),
            "toggle": Mock(),
            "update_entity": Mock(),
            "reload_config_entry": Mock(),
        },
    }

    # Mock state for supported_features
    hass.states = Mock()
    hass.states.get.return_value = Mock(
        attributes={"supported_features": 15}  # All cover features
    )

    return hass


def test_no_homeassistant_prefix_in_services(mock_hass):
    """Test that available_services never includes homeassistant.* services."""
    # Test light entity
    light_services = get_entity_available_services(
        mock_hass, "light.living_room"
    )

    assert isinstance(light_services, list)
    assert len(light_services) > 0

    # Verify no service has "homeassistant." prefix
    for service in light_services:
        assert not service.startswith("homeassistant."), (
            f"Found homeassistant.* service: {service}"
        )

    # Verify expected services are present (domain-specific)
    assert "turn_on" in light_services
    assert "turn_off" in light_services
    assert "toggle" in light_services


def test_no_duplicate_services_for_cover(mock_hass):
    """Test that cover entities don't get duplicate homeassistant.* services."""
    cover_services = get_entity_available_services(
        mock_hass, "cover.garage_door"
    )

    assert isinstance(cover_services, list)
    assert len(cover_services) > 0

    # Verify no service has "homeassistant." prefix
    for service in cover_services:
        assert not service.startswith("homeassistant."), (
            f"Found homeassistant.* service: {service}"
        )

    # Verify expected cover services are present
    assert "open_cover" in cover_services
    assert "close_cover" in cover_services
    assert "toggle" in cover_services


def test_only_domain_services_no_update_entity(mock_hass):
    """Test that update_entity and reload_config_entry are not included."""
    light_services = get_entity_available_services(
        mock_hass, "light.bedroom"
    )

    # These homeassistant domain services should NEVER appear
    unwanted_services = [
        "homeassistant.turn_on",
        "homeassistant.turn_off",
        "homeassistant.toggle",
        "homeassistant.update_entity",
        "homeassistant.reload_config_entry",
        "update_entity",  # Even without prefix
        "reload_config_entry",  # Even without prefix
    ]

    for unwanted in unwanted_services:
        assert unwanted not in light_services, (
            f"Found unwanted service: {unwanted}"
        )


def test_fallback_domain_without_mapping(mock_hass):
    """Test fallback behavior for domains without DOMAIN_SERVICE_MAPPINGS entry."""
    # Mock a custom domain not in DOMAIN_SERVICE_MAPPINGS
    mock_hass.services.async_services.return_value["custom_domain"] = {
        "custom_service": Mock(),
        "another_service": Mock(),
    }

    mock_hass.states.get.return_value = Mock(
        attributes={"supported_features": 0}
    )

    custom_services = get_entity_available_services(
        mock_hass, "custom_domain.my_entity"
    )

    # Should get services from the domain registry
    assert isinstance(custom_services, list)
    assert "custom_service" in custom_services
    assert "another_service" in custom_services

    # Should NOT get homeassistant.* services
    for service in custom_services:
        assert not service.startswith("homeassistant."), (
            f"Fallback case leaked homeassistant.* service: {service}"
        )


def test_service_names_are_plain_strings(mock_hass):
    """Test that service names are plain strings without domain prefix."""
    services = get_entity_available_services(
        mock_hass, "light.kitchen"
    )

    # All service names should be plain strings like "turn_on", not "light.turn_on"
    for service in services:
        assert "." not in service, (
            f"Service name should not contain '.': {service}"
        )


@pytest.mark.parametrize(
    "entity_id,expected_services",
    [
        ("light.bedroom", ["turn_on", "turn_off", "toggle"]),
        ("switch.outlet", ["turn_on", "turn_off", "toggle"]),
        ("fan.ceiling", ["turn_on", "turn_off", "toggle"]),
    ],
)
def test_base_services_for_common_domains(mock_hass, entity_id, expected_services):
    """Test that common domains have expected base services (no homeassistant.* prefix)."""
    mock_hass.states.get.return_value = Mock(
        attributes={"supported_features": 0}
    )

    services = get_entity_available_services(mock_hass, entity_id)

    for expected in expected_services:
        assert expected in services, (
            f"Expected service '{expected}' not found for {entity_id}"
        )
        # Verify no prefixed version exists
        assert f"homeassistant.{expected}" not in services


def test_documented_unwanted_services_never_appear(mock_hass):
    """Test that services listed in ENTITY_SERVICES_REFERENCE.md as unwanted never appear.

    According to ENTITY_SERVICES_REFERENCE.md, these should be removed:
    - homeassistant.turn_on (duplicate of domain service)
    - homeassistant.turn_off (duplicate of domain service)
    - homeassistant.toggle (duplicate of domain service)
    - homeassistant.update_entity (not useful)
    - homeassistant.reload_config_entry (not useful)
    """
    # Test multiple entity types
    test_entities = [
        "light.living_room",
        "switch.kitchen",
        "cover.garage",
        "fan.bedroom",
        "climate.thermostat",
    ]

    unwanted_services = [
        "homeassistant.turn_on",
        "homeassistant.turn_off",
        "homeassistant.toggle",
        "homeassistant.update_entity",
        "homeassistant.reload_config_entry",
    ]

    for entity_id in test_entities:
        mock_hass.states.get.return_value = Mock(
            attributes={"supported_features": 15}
        )

        services = get_entity_available_services(mock_hass, entity_id)

        for unwanted in unwanted_services:
            assert unwanted not in services, (
                f"Entity {entity_id} should not have service {unwanted}. "
                f"Found: {services}"
            )


def test_parameter_hints_for_services_with_required_params(mock_hass):
    """Test that services with required parameters include parameter hints."""
    # Cover has set_cover_position which requires 'position' parameter
    cover_services = get_entity_available_services(
        mock_hass, "cover.garage_door"
    )

    # Should have parameter hint for set_cover_position
    assert any("set_cover_position[position]" in service for service in cover_services), (
        f"Expected 'set_cover_position[position]' with parameter hint. Found: {cover_services}"
    )

    # Verify simple services don't have hints
    assert "open_cover" in cover_services
    assert "close_cover" in cover_services


def test_parameter_hints_can_be_disabled(mock_hass):
    """Test that parameter hints can be disabled when needed."""
    # Get services without parameter hints
    cover_services = get_entity_available_services(
        mock_hass, "cover.garage_door", include_parameter_hints=False
    )

    # Should have plain service name without hints
    assert "set_cover_position" in cover_services
    # Should NOT have hints
    assert not any("[" in service for service in cover_services), (
        f"Found parameter hints when they should be disabled: {cover_services}"
    )

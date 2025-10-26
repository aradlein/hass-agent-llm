"""Shared fixtures for tools tests."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock
from homeassistant.core import State
from homeassistant.util import dt as dt_util


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance for tools testing."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.services = MagicMock()
    hass.data = {}

    # Mock async_add_executor_job to execute synchronously
    async def mock_executor_job(func, *args):
        return func(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)

    return hass


@pytest.fixture
def mock_entity_registry():
    """Create a mock entity registry."""
    registry = MagicMock()
    registry.async_get = MagicMock(return_value=MagicMock())
    return registry


@pytest.fixture
def sample_light_state():
    """Create a sample light entity state."""
    return State(
        "light.living_room",
        "on",
        attributes={
            "friendly_name": "Living Room Light",
            "brightness": 128,
            "color_temp": 370,
            "rgb_color": [255, 200, 150],
            "supported_features": 63
        },
        last_changed=dt_util.now() - timedelta(minutes=5),
        last_updated=dt_util.now() - timedelta(minutes=5),
    )


@pytest.fixture
def sample_climate_state():
    """Create a sample climate entity state."""
    return State(
        "climate.thermostat",
        "heat",
        attributes={
            "friendly_name": "Thermostat",
            "temperature": 72,
            "target_temp_high": 75,
            "target_temp_low": 68,
            "current_temperature": 71,
            "hvac_mode": "heat",
            "fan_mode": "auto",
            "supported_features": 91
        },
        last_changed=dt_util.now() - timedelta(hours=2),
        last_updated=dt_util.now() - timedelta(minutes=10),
    )


@pytest.fixture
def sample_sensor_state():
    """Create a sample sensor entity state."""
    return State(
        "sensor.living_room_temperature",
        "72.5",
        attributes={
            "friendly_name": "Living Room Temperature",
            "unit_of_measurement": "°F",
            "device_class": "temperature",
        },
        last_changed=dt_util.now() - timedelta(minutes=15),
        last_updated=dt_util.now() - timedelta(minutes=1),
    )


@pytest.fixture
def sample_switch_state():
    """Create a sample switch entity state."""
    return State(
        "switch.fan",
        "off",
        attributes={
            "friendly_name": "Fan",
        },
        last_changed=dt_util.now() - timedelta(hours=1),
        last_updated=dt_util.now() - timedelta(hours=1),
    )


@pytest.fixture
def exposed_entities():
    """Create a set of exposed entity IDs for testing."""
    return {
        "light.living_room",
        "light.bedroom",
        "climate.thermostat",
        "sensor.living_room_temperature",
        "switch.fan",
        "cover.garage_door",
    }


@pytest.fixture
def mock_history_states():
    """Create mock historical state data."""
    now = dt_util.now()
    states = []

    # Create 24 hourly data points
    for i in range(24):
        state = State(
            "sensor.living_room_temperature",
            str(68 + (i % 10)),  # Temperature varies between 68-77
            attributes={
                "friendly_name": "Living Room Temperature",
                "unit_of_measurement": "°F",
            },
            last_changed=now - timedelta(hours=24-i),
            last_updated=now - timedelta(hours=24-i),
        )
        states.append(state)

    return states


@pytest.fixture
def mock_recorder(mock_hass):
    """Mock the recorder component."""
    with pytest.mock.patch("homeassistant.components.recorder.async_migration_in_progress") as mock_migration, \
         pytest.mock.patch("homeassistant.components.recorder.async_recorder_ready") as mock_ready:
        mock_migration.return_value = False
        mock_ready.return_value = True
        yield mock_migration, mock_ready

"""Shared test fixtures for Home Agent."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant


def _create_mock_hass(*, minimal: bool = False, with_loop: bool = True, tmp_path=None):
    """Create a mock Home Assistant instance with configurable features.

    Args:
        minimal: If True, create a minimal mock suitable for simple unit tests
        with_loop: If True, add event loop reference
        tmp_path: Optional pytest tmp_path for config directory

    Returns:
        Mock HomeAssistant instance
    """
    mock = MagicMock(spec=HomeAssistant)
    mock.data = {}

    # Event loop (optional, for tests that need it)
    if with_loop:
        try:
            mock.loop = asyncio.get_running_loop()
        except RuntimeError:
            mock.loop = asyncio.new_event_loop()

    # Mock states
    mock.states = MagicMock()
    mock.states.get = MagicMock(return_value=None)
    mock.states.async_entity_ids = MagicMock(return_value=[])
    mock.states.async_all = MagicMock(return_value=[])

    # Mock services
    mock.services = MagicMock()
    mock.services.async_call = AsyncMock()

    if not minimal:
        mock.services.async_services = MagicMock(
            return_value={
                "light": {"turn_on": {}, "turn_off": {}, "toggle": {}},
                "switch": {"turn_on": {}, "turn_off": {}, "toggle": {}},
                "climate": {"set_temperature": {}, "set_hvac_mode": {}},
                "homeassistant": {"turn_on": {}, "turn_off": {}, "toggle": {}},
            }
        )

    # Mock config
    mock.config = MagicMock()
    if tmp_path:
        mock.config.config_dir = str(tmp_path)
        mock.config.path = MagicMock(side_effect=lambda *args: "/".join([str(tmp_path)] + list(args)))
    else:
        mock.config.config_dir = "/config"
        mock.config.path = MagicMock(side_effect=lambda *args: "/".join(["/config"] + list(args)))

    if not minimal:
        mock.config.location_name = "Test Home"

    # Mock bus
    mock.bus = MagicMock()
    mock.bus.async_fire = MagicMock(return_value=None)

    if not minimal:
        mock.bus.async_listen = MagicMock(return_value=lambda: None)

    # Mock state attribute for HA lifecycle checks
    if not minimal:
        mock.state = MagicMock()

    # Mock async_add_executor_job - executes the callable immediately
    async def mock_executor_job(func, *args):
        return func(*args) if args else func()

    mock.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)

    # Additional features for integration tests
    if not minimal:
        # Track created tasks for cleanup
        created_tasks = []

        # Mock async_create_task
        def mock_create_task(coro, *args, **kwargs):
            """Create a task that properly awaits the coroutine."""
            task = mock.loop.create_task(coro)
            created_tasks.append(task)
            return task

        mock.async_create_task = MagicMock(side_effect=mock_create_task)
        mock._test_tasks = created_tasks

        # Mock exposed entities data structure (required by VectorDBManager)
        mock_exposed_entities = MagicMock()
        mock_exposed_entities.async_should_expose.return_value = True
        mock.data["homeassistant.exposed_entites"] = mock_exposed_entities

        # Mock registries to prevent AttributeError
        from homeassistant.helpers import entity_registry as er
        from homeassistant.helpers import area_registry as ar
        from homeassistant.helpers import device_registry as dr

        # Entity registry
        mock_entity_registry = MagicMock()
        mock_entity_registry.async_get = MagicMock(return_value=None)
        mock.data[er.DATA_REGISTRY] = mock_entity_registry

        # Area registry
        mock_area_registry = MagicMock()
        mock_area_registry.areas = {}
        mock_area_registry.async_get_area = MagicMock(return_value=None)
        mock.data[ar.DATA_REGISTRY] = mock_area_registry

        # Device registry
        mock_device_registry = MagicMock()
        mock_device_registry.devices = {}
        mock_device_registry._device_data = {}
        mock_device_registry.async_get = MagicMock(return_value=None)
        mock.data[dr.DATA_REGISTRY] = mock_device_registry

    return mock


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance for unit tests.

    This is the AUTHORITATIVE mock_hass fixture. All other mock_hass fixtures
    should be removed in favor of this one or its variants.

    This fixture provides a properly mocked HomeAssistant instance
    with commonly used methods and attributes pre-configured for testing.

    For simple unit tests, this provides all needed functionality.
    For integration tests, use test_hass fixture instead.
    """
    return _create_mock_hass(minimal=True, with_loop=False)


@pytest.fixture
def mock_hass_with_loop():
    """Create a mock Home Assistant instance with event loop for async tests."""
    return _create_mock_hass(minimal=True, with_loop=True)


@pytest.fixture
def mock_hass_full(tmp_path):
    """Create a full-featured mock Home Assistant instance for integration tests.

    Includes:
    - Event loop
    - Task tracking
    - Registry mocks (entity, area, device)
    - Exposed entities
    - Service definitions
    - Temporary config directory (auto-cleanup via tmp_path)
    """
    return _create_mock_hass(minimal=False, with_loop=True, tmp_path=tmp_path)


@pytest.fixture
def mock_llm_client():
    """Mock LLM API client."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response", tool_calls=None))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    return client


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client."""
    with patch("chromadb.Client") as mock:
        collection = MagicMock()
        collection.query.return_value = {
            "ids": [["entity1", "entity2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"entity_id": "light.living_room"}, {"entity_id": "sensor.temp"}]],
        }
        mock.return_value.get_or_create_collection.return_value = collection
        yield mock


@pytest.fixture
def sample_entities():
    """Sample entity data for testing."""
    return [
        {
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {
                "brightness": 128,
                "color_temp": 370,
                "friendly_name": "Living Room Light",
            },
        },
        {
            "entity_id": "sensor.living_room_temperature",
            "state": "72",
            "attributes": {
                "unit_of_measurement": "°F",
                "device_class": "temperature",
                "friendly_name": "Living Room Temperature",
            },
        },
        {
            "entity_id": "climate.thermostat",
            "state": "heat",
            "attributes": {
                "temperature": 72,
                "target_temperature": 70,
                "hvac_mode": "heat",
                "friendly_name": "Thermostat",
            },
        },
    ]


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "name": "Test Home Agent",
        "llm": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-key-123",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 500,
        },
        "context": {
            "mode": "direct",
            "direct": {
                "entities": [
                    {"entity_id": "light.living_room", "attributes": ["state", "brightness"]}
                ],
                "format": "json",
            },
        },
        "history": {"enabled": True, "max_messages": 10, "persist": False},
        "tools": {"enable_native": True, "max_calls_per_turn": 5, "timeout_seconds": 30},
    }


@pytest.fixture
def sample_tool_call():
    """Sample tool call from LLM."""
    return {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "ha_control",
            "arguments": (
                '{"action": "turn_on", "entity_id": "light.living_room", '
                '"parameters": {"brightness_pct": 50}}'
            ),
        },
    }

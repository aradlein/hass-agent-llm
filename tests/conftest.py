"""Shared test fixtures for Home Agent."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component


@pytest.fixture
def hass(event_loop):
    """Create a test Home Assistant instance."""
    hass = MagicMock(spec=HomeAssistant)
    hass.loop = event_loop
    hass.data = {}
    hass.states = MagicMock()
    hass.services = MagicMock()
    return hass


@pytest.fixture
def mock_llm_client():
    """Mock LLM API client."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Test response",
                        tool_calls=None
                    )
                )
            ],
            usage=MagicMock(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
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
            "metadatas": [[{"entity_id": "light.living_room"}, {"entity_id": "sensor.temp"}]]
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
                "friendly_name": "Living Room Light"
            }
        },
        {
            "entity_id": "sensor.living_room_temperature",
            "state": "72",
            "attributes": {
                "unit_of_measurement": "°F",
                "device_class": "temperature",
                "friendly_name": "Living Room Temperature"
            }
        },
        {
            "entity_id": "climate.thermostat",
            "state": "heat",
            "attributes": {
                "temperature": 72,
                "target_temperature": 70,
                "hvac_mode": "heat",
                "friendly_name": "Thermostat"
            }
        }
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
            "max_tokens": 500
        },
        "context": {
            "mode": "direct",
            "direct": {
                "entities": [
                    {
                        "entity_id": "light.living_room",
                        "attributes": ["state", "brightness"]
                    }
                ],
                "format": "json"
            }
        },
        "history": {
            "enabled": True,
            "max_messages": 10,
            "persist": False
        },
        "tools": {
            "enable_native": True,
            "max_calls_per_turn": 5,
            "timeout_seconds": 30
        }
    }


@pytest.fixture
def sample_tool_call():
    """Sample tool call from LLM."""
    return {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "ha_control",
            "arguments": '{"action": "turn_on", "entity_id": "light.living_room", "parameters": {"brightness_pct": 50}}'
        }
    }

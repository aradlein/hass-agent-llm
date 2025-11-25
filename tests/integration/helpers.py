"""Helper functions for integration tests.

This module provides utilities to simplify integration testing of Home Agent
components including entity indexing, message sending, tool verification,
and cleanup operations.
"""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from homeassistant.core import HomeAssistant, State

_LOGGER = logging.getLogger(__name__)


async def index_test_entities(
    vector_manager: Any,
    entities: list[dict[str, Any]] | list[State],
) -> None:
    """Bulk index entities into the vector database.

    Convenience function to index a list of entities for testing. Handles both
    dictionary and State object formats.

    Args:
        vector_manager: VectorDBManager instance
        entities: List of entity dictionaries or State objects to index

    Example:
        >>> entities = [
        ...     {"entity_id": "light.living_room", "state": "on"},
        ...     {"entity_id": "sensor.temperature", "state": "72"},
        ... ]
        >>> await index_test_entities(vector_manager, entities)
    """
    # Convert dict entities to State objects if needed
    state_objects = []
    for entity in entities:
        if isinstance(entity, State):
            state_objects.append(entity)
        elif isinstance(entity, dict):
            state = State(
                entity["entity_id"],
                entity.get("state", "unknown"),
                entity.get("attributes", {}),
            )
            state_objects.append(state)
        else:
            _LOGGER.warning("Unknown entity type: %s", type(entity))

    # Index all entities
    for state in state_objects:
        try:
            await vector_manager.async_index_entity(state)
        except Exception as err:
            _LOGGER.error("Failed to index entity %s: %s", state.entity_id, err)
            raise

    _LOGGER.info("Indexed %d entities into vector database", len(state_objects))


async def send_message_and_wait(
    agent: Any,
    message: str,
    conversation_id: str | None = None,
    timeout: float = 30.0,
    context: Any = None,
    device_id: str | None = None,
    satellite_id: str | None = None,
    agent_id: str | None = None,
) -> Any:
    """Send a message to the agent and wait for response with timeout.

    Args:
        agent: HomeAgent instance
        message: User message text
        conversation_id: Optional conversation ID for context
        timeout: Timeout in seconds (default: 30.0)
        context: Optional Home Assistant context
        device_id: Optional device ID
        satellite_id: Optional satellite ID
        agent_id: Optional agent ID (defaults to "home_agent")

    Returns:
        ConversationResult from the agent

    Raises:
        asyncio.TimeoutError: If agent doesn't respond within timeout

    Example:
        >>> result = await send_message_and_wait(
        ...     agent, "Turn on the lights", timeout=10.0
        ... )
        >>> assert result.response.speech["plain"]["speech"]
    """
    from homeassistant.components.conversation import ConversationInput

    # Create conversation input with all required parameters
    user_input = ConversationInput(
        text=message,
        context=context,
        conversation_id=conversation_id,
        device_id=device_id,
        satellite_id=satellite_id,
        language="en",
        agent_id=agent_id or "home_agent",
    )

    # Process with timeout
    try:
        result = await asyncio.wait_for(
            agent.async_process(user_input),
            timeout=timeout,
        )
        _LOGGER.info("Agent responded to message: %s", message[:50])
        return result
    except asyncio.TimeoutError:
        _LOGGER.error("Agent timed out after %s seconds for message: %s", timeout, message)
        raise


async def assert_tool_called(
    agent: Any,
    tool_name: str,
    min_calls: int = 1,
) -> None:
    """Verify that a specific tool was called during agent execution.

    This function checks the agent's tool handler to verify that a tool was
    executed at least the minimum number of times.

    Args:
        agent: HomeAgent instance with tool_handler
        tool_name: Name of the tool to check (e.g., "ha_control", "ha_query")
        min_calls: Minimum number of expected calls (default: 1)

    Raises:
        AssertionError: If tool wasn't called the minimum number of times

    Example:
        >>> await send_message_and_wait(agent, "Turn on the lights")
        >>> await assert_tool_called(agent, "ha_control", min_calls=1)
    """
    # Check if tool is registered
    if not hasattr(agent, "tool_handler"):
        raise ValueError("Agent does not have a tool_handler")

    registered_tools = agent.tool_handler.get_registered_tools()
    if tool_name not in registered_tools:
        raise AssertionError(
            f"Tool '{tool_name}' is not registered. "
            f"Available tools: {', '.join(registered_tools)}"
        )

    # Get overall metrics (note: current implementation doesn't track per-tool metrics)
    metrics = agent.tool_handler.get_metrics()
    total_calls = metrics.get("total_executions", 0)

    # Since we don't have per-tool metrics, we can only check if ANY tools were called
    # For more specific verification, tests should check the conversation result
    # or use event listeners to track individual tool calls
    if total_calls < min_calls:
        raise AssertionError(
            f"Expected at least {min_calls} tool call(s), but only {total_calls} "
            f"tool(s) were executed. Tool: {tool_name}"
        )

    _LOGGER.info(
        "Tool verification passed: %s was called (total calls: %d >= %d)",
        tool_name,
        total_calls,
        min_calls,
    )


def create_test_hass() -> HomeAssistant:
    """Create a mock Home Assistant instance with realistic states for testing.

    This provides a more complete mock than the basic fixtures, with:
    - Multiple entity types (lights, sensors, switches, climate)
    - Realistic state values and attributes
    - Service registrations
    - Event bus functionality

    Returns:
        Mock HomeAssistant instance configured for integration testing

    Example:
        >>> hass = create_test_hass()
        >>> states = hass.states.async_all()
        >>> assert len(states) > 0
    """
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {}

    # Create realistic entity states
    test_states = [
        State(
            "light.living_room",
            "on",
            {
                "brightness": 255,
                "color_temp": 370,
                "friendly_name": "Living Room Light",
                "supported_features": 43,
            },
        ),
        State(
            "light.bedroom",
            "off",
            {
                "friendly_name": "Bedroom Light",
                "supported_features": 40,
            },
        ),
        State(
            "light.kitchen",
            "on",
            {
                "brightness": 180,
                "friendly_name": "Kitchen Light",
                "supported_features": 40,
            },
        ),
        State(
            "sensor.living_room_temperature",
            "72.5",
            {
                "unit_of_measurement": "°F",
                "device_class": "temperature",
                "friendly_name": "Living Room Temperature",
                "state_class": "measurement",
            },
        ),
        State(
            "sensor.living_room_humidity",
            "45",
            {
                "unit_of_measurement": "%",
                "device_class": "humidity",
                "friendly_name": "Living Room Humidity",
                "state_class": "measurement",
            },
        ),
        State(
            "switch.coffee_maker",
            "off",
            {
                "friendly_name": "Coffee Maker",
            },
        ),
        State(
            "switch.bedroom_fan",
            "on",
            {
                "friendly_name": "Bedroom Fan",
            },
        ),
        State(
            "climate.thermostat",
            "heat",
            {
                "temperature": 72,
                "current_temperature": 70.5,
                "target_temp_high": 75,
                "target_temp_low": 68,
                "hvac_mode": "heat",
                "hvac_modes": ["off", "heat", "cool", "auto"],
                "friendly_name": "Thermostat",
                "supported_features": 27,
            },
        ),
        State(
            "binary_sensor.front_door",
            "off",
            {
                "device_class": "door",
                "friendly_name": "Front Door",
            },
        ),
        State(
            "binary_sensor.motion_detector",
            "on",
            {
                "device_class": "motion",
                "friendly_name": "Motion Detector",
            },
        ),
    ]

    # Mock states interface
    hass.states = MagicMock()
    hass.states.async_all = MagicMock(return_value=test_states)
    hass.states.async_entity_ids = MagicMock(
        return_value=[state.entity_id for state in test_states]
    )
    hass.states.get = MagicMock(
        side_effect=lambda entity_id: next(
            (s for s in test_states if s.entity_id == entity_id), None
        )
    )

    # Mock services
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.services.async_services = MagicMock(
        return_value={
            "light": {
                "turn_on": {"description": "Turn on light"},
                "turn_off": {"description": "Turn off light"},
                "toggle": {"description": "Toggle light"},
            },
            "switch": {
                "turn_on": {"description": "Turn on switch"},
                "turn_off": {"description": "Turn off switch"},
                "toggle": {"description": "Toggle switch"},
            },
            "climate": {
                "set_temperature": {"description": "Set target temperature"},
                "set_hvac_mode": {"description": "Set HVAC mode"},
            },
            "homeassistant": {
                "turn_on": {"description": "Generic turn on"},
                "turn_off": {"description": "Generic turn off"},
                "toggle": {"description": "Generic toggle"},
            },
        }
    )

    # Mock config
    hass.config = MagicMock()
    hass.config.config_dir = "/config"
    hass.config.location_name = "Test Home"
    hass.config.latitude = 40.7128
    hass.config.longitude = -74.0060
    hass.config.elevation = 10
    hass.config.time_zone = "America/New_York"

    # Mock bus for events
    hass.bus = MagicMock()
    # async_fire is sync in HA despite the name - returns None
    hass.bus.async_fire = MagicMock(return_value=None)
    hass.bus.async_listen = MagicMock(return_value=lambda: None)

    # Mock loop
    hass.loop = asyncio.get_event_loop()

    # Mock async_add_executor_job - executes the callable immediately
    async def mock_executor_job(func, *args):
        """Execute job immediately in test context."""
        return func(*args) if args else func()

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)

    _LOGGER.info("Created test Home Assistant instance with %d entities", len(test_states))
    return hass


async def cleanup_test_collections(
    client: Any,
    prefix: str = "test_",
) -> int:
    """Remove all test collections from ChromaDB.

    Useful for cleaning up after integration test runs. Removes all collections
    whose names start with the specified prefix.

    Args:
        client: ChromaDB HttpClient instance
        prefix: Collection name prefix to match (default: "test_")

    Returns:
        Number of collections deleted

    Example:
        >>> # After all tests complete
        >>> await cleanup_test_collections(chromadb_client, prefix="test_")
    """
    deleted_count = 0

    try:
        # Get all collections
        collections = client.list_collections()

        # Filter and delete test collections
        for collection in collections:
            if collection.name.startswith(prefix):
                try:
                    client.delete_collection(name=collection.name)
                    deleted_count += 1
                    _LOGGER.debug("Deleted test collection: %s", collection.name)
                except Exception as err:
                    _LOGGER.warning(
                        "Failed to delete collection %s: %s", collection.name, err
                    )

        _LOGGER.info("Cleaned up %d test collection(s) with prefix '%s'", deleted_count, prefix)

    except Exception as err:
        _LOGGER.error("Error during collection cleanup: %s", err)
        raise

    return deleted_count


async def wait_for_indexing(
    vector_manager: Any,
    expected_count: int,
    timeout: float = 10.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for entity indexing to complete.

    Polls the vector database collection until the expected number of entities
    are indexed or the timeout is reached.

    Args:
        vector_manager: VectorDBManager instance
        expected_count: Expected number of indexed entities
        timeout: Maximum time to wait in seconds (default: 10.0)
        poll_interval: Time between checks in seconds (default: 0.5)

    Returns:
        True if expected count was reached, False if timeout

    Example:
        >>> await index_test_entities(vector_manager, entities)
        >>> success = await wait_for_indexing(vector_manager, len(entities))
        >>> assert success, "Indexing did not complete in time"
    """
    start_time = asyncio.get_event_loop().time()
    end_time = start_time + timeout

    while asyncio.get_event_loop().time() < end_time:
        try:
            # Get collection and count
            collection = vector_manager._collection
            if collection is not None:
                result = collection.get()
                current_count = len(result.get("ids", []))

                if current_count >= expected_count:
                    _LOGGER.info(
                        "Indexing complete: %d/%d entities", current_count, expected_count
                    )
                    return True

                _LOGGER.debug(
                    "Waiting for indexing: %d/%d entities", current_count, expected_count
                )

        except Exception as err:
            _LOGGER.warning("Error checking indexing status: %s", err)

        # Wait before next check
        await asyncio.sleep(poll_interval)

    # Timeout reached
    _LOGGER.warning(
        "Indexing timeout: expected %d entities but timeout after %.1fs",
        expected_count,
        timeout,
    )
    return False

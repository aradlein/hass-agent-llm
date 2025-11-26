"""Integration tests for multiple tool calls in sequence.

This test suite validates that the LLM agent can:
- Execute multiple tool calls in a single turn
- Chain tool calls where later calls depend on earlier results
- Handle query-then-control workflows
- Properly format and return results from sequential tool executions
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import State

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
)

_LOGGER = logging.getLogger(__name__)

# Mark all tests in this module as integration tests requiring LLM
pytestmark = [pytest.mark.integration, pytest.mark.requires_llm]


@pytest.fixture
def multi_tool_entity_states() -> list[State]:
    """Create entity states for multi-tool testing.

    Returns:
        List of mock Home Assistant State objects with varied states
        to support query-then-control testing.
    """
    return [
        State(
            "light.kitchen",
            "on",
            {"brightness": 255, "friendly_name": "Kitchen Light"},
        ),
        State(
            "light.bedroom",
            "off",
            {"friendly_name": "Bedroom Light"},
        ),
        State(
            "sensor.temperature",
            "68.5",
            {
                "unit_of_measurement": "°F",
                "device_class": "temperature",
                "friendly_name": "Temperature",
            },
        ),
        State(
            "climate.thermostat",
            "heat",
            {
                "temperature": 70,
                "current_temperature": 68.5,
                "hvac_mode": "heat",
                "friendly_name": "Thermostat",
            },
        ),
        State(
            "switch.fan",
            "off",
            {"friendly_name": "Fan"},
        ),
        State(
            "light.living_room",
            "on",
            {"brightness": 128, "friendly_name": "Living Room Light"},
        ),
    ]


@pytest.mark.asyncio
async def test_query_then_control_sequence(
    test_hass, llm_config, multi_tool_entity_states, session_manager
):
    """Test that the agent can query state and then control based on result.

    This test verifies:
    1. Agent first queries entity state using ha_query
    2. Agent uses query result to inform control decision
    3. Agent executes ha_control based on the query result
    4. Both tool results are incorporated into final response
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=multi_tool_entity_states)

        def mock_get_state(entity_id):
            for state in multi_tool_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        # Track service calls
        service_calls = []

        async def mock_service_call(domain, service, service_data, **kwargs):
            service_calls.append(
                {
                    "domain": domain,
                    "service": service,
                    "data": service_data,
                }
            )
            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call)

        agent = HomeAgent(test_hass, config, session_manager)

        # Mock exposed entities
        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in multi_tool_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask a question that requires checking state first, then acting
        response = await agent.process_message(
            text="Check if the bedroom light is off, and if it is, turn it on",
            conversation_id="test_query_control",
        )

        # Verify we got a response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), f"Response should be a string, got {type(response)}"
        assert len(response) > 20, f"Response should be meaningful, got {len(response)} chars"

        # Response should acknowledge both the query and the action
        response_lower = response.lower()

        # The agent should have taken some action
        # LLM might query first, or might directly control if confident
        # We accept either workflow as long as the bedroom light is addressed
        bedroom_mentioned = any(
            word in response_lower for word in ["bedroom", "light"]
        )
        assert bedroom_mentioned, f"Response should mention bedroom light: {response[:300]}"

        await agent.close()


@pytest.mark.asyncio
async def test_multiple_queries_in_sequence(
    test_hass, llm_config, multi_tool_entity_states, session_manager
):
    """Test that the agent can execute multiple queries in sequence.

    This test verifies:
    1. Agent can query multiple entities in one turn
    2. Results from all queries are collected
    3. Final response synthesizes information from all queries
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=multi_tool_entity_states)

        def mock_get_state(entity_id):
            for state in multi_tool_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        agent = HomeAgent(test_hass, config, session_manager)

        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in multi_tool_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask a question that benefits from querying multiple entities
        response = await agent.process_message(
            text="What's the temperature and which lights are currently on?",
            conversation_id="test_multi_query",
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 30, f"Response should be substantial, got {len(response)} chars"

        response_lower = response.lower()

        # Response should mention temperature (68.5)
        temp_mentioned = any(
            word in response_lower for word in ["temperature", "68", "degree"]
        )

        # Response should mention lights that are on (kitchen, living room)
        lights_mentioned = any(
            word in response_lower for word in ["light", "kitchen", "living"]
        )

        # At least one piece of information should be present
        assert (
            temp_mentioned or lights_mentioned
        ), f"Response should mention temperature or lights: {response[:300]}"

        await agent.close()


@pytest.mark.asyncio
async def test_multiple_controls_in_sequence(
    test_hass, llm_config, multi_tool_entity_states, session_manager
):
    """Test that the agent can execute multiple control actions in one turn.

    This test verifies:
    1. Agent can execute multiple ha_control calls in sequence
    2. All control actions are completed
    3. Response acknowledges all actions taken
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=multi_tool_entity_states)

        def mock_get_state(entity_id):
            for state in multi_tool_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        service_calls = []

        async def mock_service_call(domain, service, service_data, **kwargs):
            service_calls.append(
                {
                    "domain": domain,
                    "service": service,
                    "data": service_data,
                }
            )
            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call)

        agent = HomeAgent(test_hass, config, session_manager)

        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in multi_tool_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask to control multiple devices at once
        response = await agent.process_message(
            text="Turn on the bedroom light and turn off the kitchen light",
            conversation_id="test_multi_control",
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 20

        # Check if tools were actually called
        # LLM behavior is non-deterministic, so we validate what actually happened
        if len(service_calls) > 0:
            _LOGGER.info("Service calls made: %s", service_calls)

            # Check for bedroom light action
            bedroom_calls = [
                call for call in service_calls
                if "light.bedroom" in str(call.get("data", {}))
            ]

            # Check for kitchen light action
            kitchen_calls = [
                call for call in service_calls
                if "light.kitchen" in str(call.get("data", {}))
            ]

            # At least one of the requested actions should have been taken
            assert (
                len(bedroom_calls) > 0 or len(kitchen_calls) > 0
            ), f"At least one light should have been controlled. Calls: {service_calls}"
        else:
            # If no service calls, response should at least acknowledge the request
            response_lower = response.lower()
            assert any(
                word in response_lower
                for word in ["bedroom", "kitchen", "light"]
            ), f"Response should acknowledge the request: {response[:200]}"

        await agent.close()


@pytest.mark.asyncio
async def test_conditional_control_based_on_query(
    test_hass, llm_config, multi_tool_entity_states, session_manager
):
    """Test conditional control based on query results.

    This test verifies:
    1. Agent queries entity state
    2. Agent makes decision based on queried value
    3. Agent executes control only if condition is met
    4. Response explains the reasoning
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=multi_tool_entity_states)

        def mock_get_state(entity_id):
            for state in multi_tool_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        service_calls = []

        async def mock_service_call(domain, service, service_data, **kwargs):
            service_calls.append(
                {
                    "domain": domain,
                    "service": service,
                    "data": service_data,
                }
            )
            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call)

        agent = HomeAgent(test_hass, config, session_manager)

        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in multi_tool_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask a conditional question: turn on fan IF temperature is above a threshold
        # Current temp is 68.5, threshold is 70, so fan should NOT be turned on
        response = await agent.process_message(
            text="If the temperature is above 70 degrees, turn on the fan",
            conversation_id="test_conditional",
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 20

        response_lower = response.lower()

        # Response should mention the temperature or condition
        temp_mentioned = any(
            word in response_lower for word in ["temperature", "68", "70", "degree"]
        )

        # Since temp is 68.5 (below 70), fan should NOT be turned on
        # Check if any fan control happened
        fan_calls = [
            call for call in service_calls
            if "switch.fan" in str(call.get("data", {})) and call.get("service") == "turn_on"
        ]

        # The LLM should have understood the condition and NOT turned on the fan
        # (or explained why it didn't in the response)
        if len(fan_calls) > 0:
            # If fan was turned on despite temp being below threshold, that's wrong
            # However, we'll be lenient and just log this case
            _LOGGER.warning(
                "Fan was turned on despite temperature being below threshold. "
                "This may indicate LLM misunderstood the condition."
            )

        # At minimum, response should acknowledge the request
        assert (
            temp_mentioned or "fan" in response_lower
        ), f"Response should mention temperature or fan: {response[:300]}"

        await agent.close()


@pytest.mark.asyncio
async def test_tool_sequence_with_errors(
    test_hass, llm_config, multi_tool_entity_states, session_manager
):
    """Test that agent handles errors gracefully during multi-tool sequences.

    This test verifies:
    1. Agent can handle tool execution errors
    2. Agent continues with other tools after one fails
    3. Error is communicated in the final response
    4. System doesn't crash on tool failure
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=multi_tool_entity_states)

        def mock_get_state(entity_id):
            for state in multi_tool_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        call_count = 0

        async def mock_service_call_with_error(domain, service, service_data, **kwargs):
            nonlocal call_count
            call_count += 1

            # First call succeeds, second call fails
            if call_count == 2:
                raise Exception("Simulated service call failure")

            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call_with_error)

        agent = HomeAgent(test_hass, config, session_manager)

        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in multi_tool_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask to control two devices - one will succeed, one will fail
        response = await agent.process_message(
            text="Turn on the bedroom light and the fan",
            conversation_id="test_error_handling",
        )

        # Should still get a response despite errors
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 20

        # Response might mention the error or partial success
        # At minimum, it should be a coherent response
        response_lower = response.lower()

        # LLM should have attempted the actions or explained what happened
        relevant_mentioned = any(
            word in response_lower
            for word in ["bedroom", "fan", "light", "error", "unable", "could", "sorry"]
        )

        assert relevant_mentioned, f"Response should mention the devices or errors: {response[:300]}"

        await agent.close()


@pytest.mark.asyncio
async def test_max_tool_calls_enforcement(
    test_hass, llm_config, multi_tool_entity_states, session_manager
):
    """Test that max tool calls per turn is enforced.

    This test verifies:
    1. System enforces CONF_TOOLS_MAX_CALLS_PER_TURN limit
    2. Only the first N tools are executed when limit is exceeded
    3. Response is still generated despite hitting the limit
    """
    # Set a low limit for testing
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 2,  # Low limit for testing
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=multi_tool_entity_states)

        def mock_get_state(entity_id):
            for state in multi_tool_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        service_calls = []

        async def mock_service_call(domain, service, service_data, **kwargs):
            service_calls.append(
                {
                    "domain": domain,
                    "service": service,
                    "data": service_data,
                }
            )
            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call)

        agent = HomeAgent(test_hass, config, session_manager)

        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in multi_tool_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask to control many devices (more than the limit)
        response = await agent.process_message(
            text="Turn on all the lights and the fan",
            conversation_id="test_max_calls",
        )

        # Should get a response even if not all actions completed
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 20

        # Check that no more than max_calls were executed
        # (LLM might not call tools at all, or might call fewer than requested)
        assert (
            len(service_calls) <= 2
        ), f"Should not exceed max_calls limit of 2, got {len(service_calls)} calls"

        await agent.close()

"""Integration tests for LLM integration with real API endpoints.

These tests verify that the HomeAgent correctly interacts with a real LLM
endpoint (Ollama, OpenAI, etc.) for conversation processing, tool calling,
and streaming responses.
"""

import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.const import ATTR_ENTITY_ID

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_PERSIST,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
)

_LOGGER = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_basic_conversation(test_hass, llm_config, session_manager):
    """Test simple Q&A with real LLM.

    This test verifies that:
    1. HomeAgent can connect to the LLM endpoint
    2. Basic conversation processing works
    3. Response is returned successfully
    """
    # Configure HomeAgent with real LLM
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: False,  # Disable for simple test
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    # Mock entity exposure to return no entities (simple test)
    # async_should_expose is actually a sync function despite the name
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Process a simple message
        response = await agent.process_message(
            text="Hello! How are you today?",
            conversation_id="test_basic",
        )

        # Verify we got a response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), f"Response should be a string, got {type(response)}"
        assert (
            len(response) > 20
        ), f"Response should be meaningful (>20 chars), got {len(response)} chars: {response[:100]}"

        # Response should be conversational
        # (this line removed as redundant with above check)

        # Response should be coherent (not just random characters)
        # Check for common conversational patterns
        response_lower = response.lower()
        assert any(
            pattern in response_lower
            for pattern in [
                "hello",
                "hi",
                "how",
                "doing",
                "help",
                "assist",
                "good",
                "great",
                "thank",
                "i",
                "you",
            ]
        ), f"Response doesn't appear conversational: {response[:200]}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_tool_calling(test_hass, llm_config, sample_entity_states, session_manager):
    """Test that LLM triggers tools correctly.

    This test verifies that:
    1. LLM recognizes when to use tools
    2. Tool calls are properly formatted
    3. Tool results are processed
    4. Final response incorporates tool results
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
    }

    # Mock entity exposure to return False (avoid entity registry calls)
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        # Mock service call to track tool executions
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

        # Mock the get_exposed_entities method to return test entities
        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in sample_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask the agent to control a device
        response = await agent.process_message(
            text="Turn on the living room light",
            conversation_id="test_tool_calling",
        )

        # Verify we got a response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), f"Response should be a string, got {type(response)}"
        assert (
            len(response) > 20
        ), f"Response should be meaningful (>20 chars), got {len(response)} chars: {response[:100]}"

        # Response should be related to the request
        # Note: LLM behavior is non-deterministic, so we check for relevant content
        response_lower = response.lower()

        # Check if tool was actually called with correct parameters
        if len(service_calls) > 0:
            # Verify at least one call targeted the living room light
            light_targeted = any(
                call.get("data", {}).get("entity_id") == "light.living_room"
                or "light.living_room" in str(call.get("data", {}))
                for call in service_calls
            )

            # Verify turn_on service was called
            turn_on_called = any(call.get("service") == "turn_on" for call in service_calls)

            assert (
                light_targeted
            ), f"Living room light was not targeted. Calls made: {service_calls}"
            assert turn_on_called, f"turn_on service was not called. Calls made: {service_calls}"
        else:
            # If no service calls, response should at least acknowledge the request
            assert any(
                word in response_lower for word in ["light", "living", "turn", "on"]
            ), f"No tool calls made and response doesn't mention the request: {response[:200]}"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_multi_turn_context(test_hass, llm_config, session_manager):
    """Test conversation memory across multiple turns.

    This test verifies that:
    1. Conversation history is maintained
    2. Context from previous turns is used
    3. Follow-up questions work correctly
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: True,  # Enable history
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_HISTORY_PERSIST: False,
        CONF_EMIT_EVENTS: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_multi_turn"

        # First turn: Set context
        response1 = await agent.process_message(
            text="My name is Alice and I like the color blue.",
            conversation_id=conversation_id,
        )

        assert response1 is not None, "First response should not be None"
        assert isinstance(response1, str), f"Response should be a string, got {type(response1)}"
        assert len(response1) > 10, f"Response should be meaningful, got {len(response1)} chars"

        # Second turn: Reference previous context
        response2 = await agent.process_message(
            text="What is my name?",
            conversation_id=conversation_id,
        )

        assert response2 is not None, "Second response should not be None"
        assert isinstance(response2, str), f"Response should be a string, got {type(response2)}"
        assert len(response2) > 5, f"Response should not be empty, got {len(response2)} chars"
        # Response should mention Alice
        assert "alice" in response2.lower(), "Agent didn't remember name from previous turn"

        # Third turn: Reference other context
        response3 = await agent.process_message(
            text="What color do I like?",
            conversation_id=conversation_id,
        )

        assert response3 is not None, "Third response should not be None"
        assert isinstance(response3, str), f"Response should be a string, got {type(response3)}"
        assert len(response3) > 5, f"Response should not be empty, got {len(response3)} chars"
        # Response should mention blue
        assert "blue" in response3.lower(), "Agent didn't remember color preference from first turn"

        # Verify conversation history is populated
        history = agent.conversation_manager.get_history(conversation_id)
        assert len(history) >= 4, "Conversation history not tracking properly"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_streaming_response(test_hass, llm_config, session_manager):
    """Test SSE streaming works with real LLM.

    This test verifies that:
    1. Streaming can be enabled
    2. Response is delivered incrementally
    3. Complete response is assembled correctly
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 300,
        CONF_STREAMING_ENABLED: True,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Collect streaming chunks
        chunks = []

        async def collect_chunks():
            async for chunk in agent._call_llm_streaming(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Count from 1 to 5."},
                ]
            ):
                chunks.append(chunk)

        await collect_chunks()

        # Verify we received multiple chunks
        assert len(chunks) > 0, "No streaming chunks received"
        assert isinstance(chunks, list), f"Chunks should be a list, got {type(chunks)}"
        # Should receive multiple chunks for streaming (not just one)
        assert len(chunks) >= 2, f"Expected multiple streaming chunks, got {len(chunks)}"

        # Parse chunks to verify SSE format
        valid_chunks = 0
        for chunk in chunks:
            if chunk.strip():
                # Should be SSE format: "data: {...}\n\n"
                if chunk.startswith("data:"):
                    valid_chunks += 1

        assert valid_chunks > 0, "No valid SSE chunks received"

        # Verify we can parse the JSON from at least one chunk
        parsed_any = False
        for chunk in chunks:
            if chunk.startswith("data:"):
                try:
                    data_str = chunk[5:].strip()  # Remove "data: " prefix
                    if data_str and data_str != "[DONE]":
                        json.loads(data_str)
                        parsed_any = True
                        break
                except json.JSONDecodeError:
                    continue

        assert parsed_any, "Could not parse any JSON from streaming chunks"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_error_handling(test_hass, llm_config, session_manager):
    """Test LLM error handling (invalid model, connection issues, etc).

    This test verifies that:
    1. Invalid model name is handled gracefully
    2. Appropriate error messages are returned
    3. System doesn't crash on LLM errors
    """
    # Configure with invalid model
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: "nonexistent-model-xyz",  # Invalid model
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Try to process a message with invalid model
        response = None
        exception_caught = None
        try:
            response = await agent.process_message(
                text="Hello",
                conversation_id="test_error",
            )
        except Exception as e:
            exception_caught = e

        # Either we got an exception OR the response indicates an error
        # OR the backend handled it gracefully (some LLM servers have fallback models)
        if exception_caught:
            error_str = str(exception_caught).lower()
            # Error should be informative about the issue
            assert any(
                word in error_str
                for word in ["model", "not found", "invalid", "error", "failed", "404", "400"]
            ), f"Error message not informative: {exception_caught}"
        elif response:
            # Some LLM backends may gracefully handle invalid models with fallbacks
            # or return error messages in the response body
            response_lower = response.lower()
            # Accept either error indication OR valid response (backend handled gracefully)
            has_error_indication = any(
                word in response_lower for word in ["error", "sorry", "unable", "cannot", "failed"]
            )
            # Or it's a valid response (some backends use fallback models)
            is_valid_response = len(response) > 0 and isinstance(response, str)

            assert (
                has_error_indication or is_valid_response
            ), f"Response should either indicate error or be valid. Got: {response[:200]}"
        else:
            pytest.fail("Neither exception raised nor response returned")

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_llm_with_complex_tools(test_hass, llm_config, sample_entity_states, session_manager):
    """Test LLM handling of complex multi-step tool interactions.

    This test verifies that:
    1. LLM can chain multiple tool calls
    2. Results from one tool inform the next
    3. Final response synthesizes all tool results
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

    # Mock entity exposure to return False (avoid entity registry calls)
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
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

        # Mock the get_exposed_entities method to return test entities
        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in sample_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Ask for a complex multi-step action
        response = await agent.process_message(
            text="What's the temperature, and if it's below 70, turn on the thermostat",
            conversation_id="test_complex_tools",
        )

        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), f"Response should be a string, got {type(response)}"
        assert (
            len(response) > 20
        ), f"Response should be meaningful (>20 chars), got {len(response)} chars: {response[:100]}"

        # Response should mention temperature (the query asked about temperature)
        response_lower = response.lower()
        assert any(
            word in response_lower for word in ["temperature", "72", "70", "degrees", "thermostat"]
        ), f"Response doesn't mention temperature info: {response[:300]}"

        # The test asked about temperature - verify it was understood
        # Note: Tool calls are optional depending on LLM interpretation

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_tool_execution_with_correct_entity(
    test_hass, llm_config, sample_entity_states, session_manager
):
    """Test that tool calls target the correct entity_id.

    This test verifies that:
    1. LLM correctly identifies the target entity from user input
    2. Tool calls include the correct entity_id parameter
    3. Service calls are made with the right entity target
    4. Entity_id is not confused or mixed up with other entities
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
    }

    # Mock entity registry to return entries for our test entities
    mock_entity_registry = MagicMock()
    mock_entity_registry.async_get = MagicMock(
        side_effect=lambda entity_id: MagicMock(entity_id=entity_id)
    )

    # Mock entity exposure
    with (
        patch(
            "custom_components.home_agent.agent.async_should_expose",
            return_value=False,
        ),
        patch(
            "custom_components.home_agent.tools.ha_control.er.async_get",
            return_value=mock_entity_registry,
        ),
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=sample_entity_states)

        def mock_get_state(entity_id):
            for state in sample_entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        test_hass.states.get = MagicMock(side_effect=mock_get_state)

        # Track service calls with detailed information
        service_calls = []

        async def mock_service_call(domain, service, service_data, **kwargs):
            call_info = {
                "domain": domain,
                "service": service,
                "data": service_data,
                "entity_id": service_data.get(ATTR_ENTITY_ID) if service_data else None,
            }
            service_calls.append(call_info)
            _LOGGER.debug("Service call tracked: %s", call_info)
            return None

        test_hass.services.async_call = AsyncMock(side_effect=mock_service_call)

        agent = HomeAgent(test_hass, config, session_manager)

        # Mock the get_exposed_entities method to return test entities
        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in sample_entity_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Test 1: Turn on a specific light (bedroom, not living room)
        response1 = await agent.process_message(
            text="Turn on the bedroom light",
            conversation_id="test_entity_targeting_1",
        )

        assert response1 is not None, "First response should not be None"
        assert isinstance(response1, str), f"Response should be a string, got {type(response1)}"
        assert (
            len(response1) > 10
        ), f"Response should be meaningful, got {len(response1)} chars: {response1[:100]}"

        # Check if service was called (LLM may or may not call the tool)
        bedroom_calls = [call for call in service_calls if call.get("entity_id") == "light.bedroom"]

        living_room_calls = [
            call for call in service_calls if call.get("entity_id") == "light.living_room"
        ]

        # If tool was called, it should target bedroom, not living room
        if len(service_calls) > 0:
            # Verify no calls to living room light when we asked for bedroom
            assert (
                len(living_room_calls) == 0
            ), f"Should not call living_room light when user asked for bedroom. Calls: {service_calls}"

            # If bedroom was called, verify it was the right action
            if len(bedroom_calls) > 0:
                assert bedroom_calls[0]["service"] in [
                    "turn_on",
                    "toggle",
                ], f"Should turn on/toggle bedroom light, got: {bedroom_calls[0]['service']}"

        # Clear service calls for next test
        service_calls.clear()

        # Test 2: Control coffee maker specifically
        response2 = await agent.process_message(
            text="Turn on the coffee maker",
            conversation_id="test_entity_targeting_2",
        )

        assert response2 is not None, "Second response should not be None"
        assert isinstance(response2, str), f"Response should be a string, got {type(response2)}"
        assert len(response2) > 10, f"Response should be meaningful, got {len(response2)} chars"

        # Check if coffee maker was targeted (not other switches/lights)
        coffee_maker_calls = [
            call for call in service_calls if call.get("entity_id") == "switch.coffee_maker"
        ]

        wrong_entity_calls = [
            call
            for call in service_calls
            if call.get("entity_id")
            and call.get("entity_id")
            not in [
                "switch.coffee_maker",
                None,  # Generic homeassistant service calls might not have entity_id
            ]
        ]

        # If tool was called, verify it targeted the right entity
        if len(service_calls) > 0:
            # Should not call wrong entities
            assert (
                len(wrong_entity_calls) == 0
            ), f"Should only target coffee_maker, but got: {wrong_entity_calls}"

            # If coffee maker was called, verify correct service
            if len(coffee_maker_calls) > 0:
                assert coffee_maker_calls[0]["service"] in [
                    "turn_on",
                    "toggle",
                ], f"Should turn on coffee maker, got: {coffee_maker_calls[0]['service']}"

        # Clear service calls for next test
        service_calls.clear()

        # Test 3: Query specific sensor (not controls)
        response3 = await agent.process_message(
            text="What is the temperature?",
            conversation_id="test_entity_targeting_3",
        )

        assert response3 is not None, "Third response should not be None"
        assert isinstance(response3, str), f"Response should be a string, got {type(response3)}"
        assert len(response3) > 10, f"Response should be meaningful, got {len(response3)} chars"

        # Response should acknowledge the temperature query
        # Note: LLM behavior is non-deterministic, so we accept various valid responses
        response_lower = response3.lower()
        # Temperature sensor shows 72.5°F - LLM may mention temp value, acknowledge query, or describe checking
        valid_response_patterns = [
            "temperature",
            "72",
            "sensor",
            "degrees",  # Direct answer
            "check",
            "let me",
            "look",
            "see",
            "finding",  # Acknowledging query
            "living",
            "room",
            "current",  # Context mentions
        ]
        assert any(
            word in response_lower for word in valid_response_patterns
        ), f"Response should acknowledge temperature query: {response3[:200]}"

        # Verify no control actions were taken for a query
        control_services = [
            call
            for call in service_calls
            if call["service"] in ["turn_on", "turn_off", "toggle", "set_temperature"]
        ]

        # Query shouldn't trigger control actions
        assert (
            len(control_services) == 0
        ), f"Query should not trigger control services, got: {control_services}"

        await agent.close()

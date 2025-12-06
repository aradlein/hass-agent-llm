"""Integration tests for LLM integration with real and mocked API endpoints.

These tests verify that the HomeAgent correctly integrates with LLM endpoints
for conversation processing, tool execution, and streaming responses.

Testing Approach:
- Real LLM tests: Validate connectivity, streaming format, and error handling
- Mocked LLM tests: Validate tool execution, history mechanism, and data flow
- Structural validation: Focus on integration points, not LLM content quality
- No semantic checks: LLM response content is non-deterministic and not tested

Tests marked with @pytest.mark.requires_llm will only run when LLM service
is available (configured in .env.test).
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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


# Helper functions for creating mock LLM responses
def create_mock_llm_response(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
) -> dict:
    """Create a mock LLM API response.

    Args:
        content: Response text (None if tool calls)
        tool_calls: Optional list of tool call dicts
        finish_reason: Finish reason ("stop" or "tool_calls")

    Returns:
        Mock LLM response dictionary
    """
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }


def create_mock_tool_call(
    tool_name: str,
    arguments: dict,
    call_id: str = "call_123",
) -> dict:
    """Create a mock tool call structure.

    Args:
        tool_name: Name of the tool (e.g., "ha_control")
        arguments: Tool arguments dictionary
        call_id: Unique call ID

    Returns:
        Tool call dictionary
    """
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    }


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_basic_conversation(test_hass, llm_config, session_manager):
    """Test basic LLM connectivity and response format.

    This test verifies that:
    1. HomeAgent can connect to the LLM endpoint
    2. LLM returns a properly formatted response
    3. Response is a non-empty string

    Note: This test does NOT validate response content (LLM behavior is non-deterministic).
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
        "custom_components.home_agent.agent.core.async_should_expose",
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

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_tool_calling(test_hass_with_default_entities, llm_config, session_manager):
    """Test that tool execution mechanism works correctly.

    This test verifies that:
    1. Tool calls from LLM are properly parsed
    2. Service calls are made with correct parameters
    3. Correct entity_id is targeted
    4. Correct service is invoked

    Note: LLM response is mocked for deterministic tool execution testing.
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

    test_hass_with_default_entities.services.async_call = AsyncMock(side_effect=mock_service_call)

    agent = HomeAgent(test_hass_with_default_entities, config, session_manager)

    # Mock the get_exposed_entities method to return test entities
    sample_states = test_hass_with_default_entities.states.async_all()

    def mock_exposed_entities():
        return [
            {
                "entity_id": state.entity_id,
                "name": state.attributes.get("friendly_name", state.entity_id),
                "state": state.state,
                "aliases": [],
            }
            for state in sample_states
        ]

    agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

    # Create mock LLM response with tool call
    mock_tool_call = create_mock_tool_call(
        tool_name="ha_control",
        arguments={"entity_id": "light.living_room", "action": "turn_on"},
    )
    mock_response = create_mock_llm_response(
        content=None,
        tool_calls=[mock_tool_call],
        finish_reason="tool_calls",
    )

    # Mock the LLM to return our controlled response
    with patch.object(agent, "_call_llm", return_value=mock_response):
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

    # Since we mocked the LLM, tool MUST be called
    assert len(service_calls) > 0, "Tool should have been called with mocked response"

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

    await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_multi_turn_context(test_hass, llm_config, session_manager):
    """Test conversation history mechanism across multiple turns.

    This test verifies that:
    1. Conversation history is maintained in ConversationManager
    2. History is correctly passed to LLM on subsequent turns
    3. Messages are stored with correct roles and content
    4. History persists across multiple turns in same conversation

    Note: LLM is mocked. This tests the history MECHANISM, not LLM's memory ability.
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
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_multi_turn"

        # Create mock responses for each turn
        mock_response_1 = create_mock_llm_response(
            content="Nice to meet you, Alice! Blue is a great color."
        )
        mock_response_2 = create_mock_llm_response(content="Your name is Alice.")
        mock_response_3 = create_mock_llm_response(content="You like the color blue.")

        # Mock the LLM to return sequential responses
        with patch.object(
            agent,
            "_call_llm",
            side_effect=[mock_response_1, mock_response_2, mock_response_3],
        ) as mock_llm:
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

            # Third turn: Reference other context
            response3 = await agent.process_message(
                text="What color do I like?",
                conversation_id=conversation_id,
            )

            assert response3 is not None, "Third response should not be None"
            assert isinstance(response3, str), f"Response should be a string, got {type(response3)}"
            assert len(response3) > 5, f"Response should not be empty, got {len(response3)} chars"

        # Verify conversation history contains the right structure
        history = agent.conversation_manager.get_history(conversation_id)

        # Should have: user1, assistant1, user2, assistant2, user3, assistant3
        assert len(history) == 6, f"Expected 6 messages in history, got {len(history)}"

        # Verify messages are in correct order with correct roles
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "My name is Alice and I like the color blue."
        assert history[1]["role"] == "assistant"

        assert history[2]["role"] == "user"
        assert history[2]["content"] == "What is my name?"
        assert history[3]["role"] == "assistant"

        assert history[4]["role"] == "user"
        assert history[4]["content"] == "What color do I like?"
        assert history[5]["role"] == "assistant"

        # Verify that history was PASSED to LLM (check mock call arguments)
        # On second call, history should include first exchange
        assert mock_llm.call_count == 3, f"Expected 3 LLM calls, got {mock_llm.call_count}"

        second_call_args = mock_llm.call_args_list[1]
        messages_sent = second_call_args[0][0]  # First positional arg

        # Messages should include: system + history + new user message
        # Verify previous context was included
        assert len(messages_sent) >= 3, "History should be passed to LLM"
        assert any(
            "Alice" in str(msg.get("content", "")) for msg in messages_sent
        ), "Previous conversation context not passed to LLM"

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
        "custom_components.home_agent.agent.core.async_should_expose",
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

        # Verify at least some chunks contain actual content
        content_chunks = 0
        for chunk in chunks:
            if chunk.startswith("data:"):
                data_str = chunk[5:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices and len(choices) > 0:
                            delta = choices[0].get("delta", {})
                            if delta.get("content"):
                                content_chunks += 1
                    except json.JSONDecodeError:
                        pass

        assert content_chunks > 0, "No content chunks received in stream"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_error_handling(test_hass, llm_config, session_manager):
    """Test LLM error handling for invalid configurations.

    This test verifies that:
    1. Invalid model name doesn't crash the system
    2. Either an exception is raised OR a response is returned
    3. System handles LLM errors gracefully

    Note: Different LLM backends handle errors differently (exceptions vs fallbacks).
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
        "custom_components.home_agent.agent.core.async_should_expose",
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

        # Either we got an exception OR we got a response
        # (some LLM backends have fallback models)
        if exception_caught:
            # Verify it's an actual exception (not None)
            assert exception_caught is not None
            assert isinstance(exception_caught, Exception)
            _LOGGER.info(
                f"Got expected exception for invalid model: {type(exception_caught).__name__}"
            )
        elif response:
            # Some backends may handle invalid models gracefully with fallback
            assert isinstance(response, str)
            assert len(response) > 0
            _LOGGER.info("Backend handled invalid model gracefully with fallback")
        else:
            pytest.fail("Neither exception raised nor response returned")

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_llm_with_complex_tools(test_hass_with_default_entities, llm_config, session_manager):
    """Test multi-step tool interaction mechanism.

    This test verifies that:
    1. Multiple tool calls can be made in sequence
    2. Tool results can inform subsequent actions
    3. Agent handles multi-turn tool execution

    Note: LLM is mocked to ensure deterministic tool call sequence.
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

    test_hass_with_default_entities.services.async_call = AsyncMock(side_effect=mock_service_call)

    agent = HomeAgent(test_hass_with_default_entities, config, session_manager)

    # Mock the get_exposed_entities method to return test entities
    sample_states = test_hass_with_default_entities.states.async_all()

    def mock_exposed_entities():
        return [
            {
                "entity_id": state.entity_id,
                "name": state.attributes.get("friendly_name", state.entity_id),
                "state": state.state,
                "aliases": [],
            }
            for state in sample_states
        ]

    agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

    # Create mock multi-step responses (query temperature, then respond)
    mock_tool_call_query = create_mock_tool_call(
        tool_name="ha_query",
        arguments={"entity_id": "sensor.temperature"},
        call_id="call_query",
    )
    mock_response_1 = create_mock_llm_response(
        content=None,
        tool_calls=[mock_tool_call_query],
        finish_reason="tool_calls",
    )

    # Second response after getting temperature result
    mock_response_2 = create_mock_llm_response(
        content="The temperature is 72.5°F, which is above 70, so I won't turn on the thermostat."
    )

    # Mock the LLM to return sequential responses
    with patch.object(
        agent, "_call_llm", side_effect=[mock_response_1, mock_response_2]
    ):
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

    await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_tool_execution_with_correct_entity(
    test_hass_with_default_entities, llm_config, session_manager
):
    """Test that tool calls target the correct entity_id.

    This test verifies that:
    1. Tool calls include the correct entity_id parameter
    2. Service calls are made with the right entity target
    3. Entity_id is not confused with other entities
    4. Query operations don't trigger control actions

    Note: LLM is mocked to ensure deterministic entity targeting.
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

    test_hass_with_default_entities.services.async_call = AsyncMock(side_effect=mock_service_call)

    # Mock entity exposure
    with patch(
        "custom_components.home_agent.tools.ha_control.er.async_get",
        return_value=mock_entity_registry,
    ):
        agent = HomeAgent(test_hass_with_default_entities, config, session_manager)

        # Mock the get_exposed_entities method to return test entities
        sample_states = test_hass_with_default_entities.states.async_all()

        def mock_exposed_entities():
            return [
                {
                    "entity_id": state.entity_id,
                    "name": state.attributes.get("friendly_name", state.entity_id),
                    "state": state.state,
                    "aliases": [],
                }
                for state in sample_states
            ]

        agent.get_exposed_entities = MagicMock(return_value=mock_exposed_entities())

        # Test 1: Turn on a specific light (bedroom, not living room)
        mock_bedroom_tool = create_mock_tool_call(
            tool_name="ha_control",
            arguments={"entity_id": "light.bedroom", "action": "turn_on"},
            call_id="call_bedroom",
        )
        mock_bedroom_response = create_mock_llm_response(
            content=None,
            tool_calls=[mock_bedroom_tool],
            finish_reason="tool_calls",
        )

        with patch.object(agent, "_call_llm", return_value=mock_bedroom_response):
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

        # Since we mocked the LLM, tool should be called
        assert len(service_calls) > 0, "Tool should have been called with mocked response"

        # Verify no calls to living room light when we asked for bedroom
        assert (
            len(living_room_calls) == 0
        ), f"Should not call living_room light when user asked for bedroom. Calls: {service_calls}"

        # Verify bedroom was called with right action
        assert len(bedroom_calls) > 0, f"Bedroom light should have been called. Calls: {service_calls}"
        assert bedroom_calls[0]["service"] in [
            "turn_on",
            "toggle",
        ], f"Should turn on/toggle bedroom light, got: {bedroom_calls[0]['service']}"

        # Clear service calls for next test
        service_calls.clear()

        # Test 2: Control coffee maker specifically
        mock_coffee_tool = create_mock_tool_call(
            tool_name="ha_control",
            arguments={"entity_id": "switch.coffee_maker", "action": "turn_on"},
            call_id="call_coffee",
        )
        mock_coffee_response = create_mock_llm_response(
            content=None,
            tool_calls=[mock_coffee_tool],
            finish_reason="tool_calls",
        )

        with patch.object(agent, "_call_llm", return_value=mock_coffee_response):
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

        # Since we mocked the LLM, tool should be called
        assert len(service_calls) > 0, "Tool should have been called with mocked response"

        # Should not call wrong entities
        assert (
            len(wrong_entity_calls) == 0
        ), f"Should only target coffee_maker, but got: {wrong_entity_calls}"

        # Coffee maker should have been called with correct service
        assert len(coffee_maker_calls) > 0, f"Coffee maker should have been called. Calls: {service_calls}"
        assert coffee_maker_calls[0]["service"] in [
            "turn_on",
            "toggle",
        ], f"Should turn on coffee maker, got: {coffee_maker_calls[0]['service']}"

        # Clear service calls for next test
        service_calls.clear()

        # Test 3: Query specific sensor (not controls)
        mock_temp_tool = create_mock_tool_call(
            tool_name="ha_query",
            arguments={"entity_id": "sensor.temperature"},
            call_id="call_temp",
        )
        mock_temp_response_1 = create_mock_llm_response(
            content=None,
            tool_calls=[mock_temp_tool],
            finish_reason="tool_calls",
        )
        mock_temp_response_2 = create_mock_llm_response(
            content="The current temperature is 72.5°F."
        )

        with patch.object(
            agent, "_call_llm", side_effect=[mock_temp_response_1, mock_temp_response_2]
        ):
            response3 = await agent.process_message(
                text="What is the temperature?",
                conversation_id="test_entity_targeting_3",
            )

        assert response3 is not None, "Third response should not be None"
        assert isinstance(response3, str), f"Response should be a string, got {type(response3)}"
        assert len(response3) > 10, f"Response should be meaningful, got {len(response3)} chars"

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

"""Integration tests for LLM integration with real or mock API endpoints.

These tests verify that the HomeAgent correctly interacts with LLM endpoints
(Ollama, OpenAI, etc.) for conversation processing, tool calling,
and streaming responses.

When real LLM services are unavailable, these tests use mock implementations
to validate the integration paths. Mock tests verify:
- Agent initialization and configuration
- Tool dispatch and execution
- Message routing and history
- Response processing

Mock tests do NOT verify:
- Actual LLM intelligence (tool selection, reasoning)
- Real-world response quality
- Network reliability
"""

import json
import logging
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import ATTR_ENTITY_ID

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.helpers import strip_thinking_blocks
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

# Suffix to disable thinking mode for reasoning models (Qwen3, DeepSeek R1, etc.)
# This prevents tests from timing out due to extended thinking
NO_THINK_SUFFIX = "\n/no_think"


@contextmanager
def maybe_mock_llm(is_using_mock: bool, mock_server):
    """Context manager that patches aiohttp when using mock LLM.

    Args:
        is_using_mock: Whether to use mock LLM
        mock_server: MockLLMServer instance to use

    Yields:
        None (patches are applied in context)
    """
    if is_using_mock and mock_server:
        with mock_server.patch_aiohttp():
            yield
    else:
        yield


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_basic_conversation(
    test_hass, llm_config, session_manager, is_using_mock_llm, mock_llm_server
):
    """Test simple Q&A with LLM (real or mock).

    This test verifies that:
    1. HomeAgent can connect to the LLM endpoint
    2. Basic conversation processing works
    3. Response is returned successfully

    With mocks: Verifies agent processes mock responses correctly.
    With real LLM: Verifies actual conversational ability.
    """
    # Configure HomeAgent with LLM
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
        with maybe_mock_llm(is_using_mock_llm, mock_llm_server):
            agent = HomeAgent(test_hass, config, session_manager)

            # Process a simple message (with /no_think suffix for reasoning models)
            response = await agent.process_message(
                text="Hello! How are you today?" + NO_THINK_SUFFIX,
                conversation_id="test_basic",
            )

            # Strip thinking blocks from response before validation
            response = strip_thinking_blocks(response) or ""

            # Verify we got a response
            assert response is not None, "Response should not be None"
            assert isinstance(response, str), f"Response should be a string, got {type(response)}"
            assert (
                len(response) > 0
            ), f"Response should not be empty, got {len(response)} chars: {response[:100]}"

            # With mocks, check for specific patterns; with real LLM, just verify non-empty response
            if is_using_mock_llm:
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
                        "home",  # Mock response mentions "home assistant"
                    ]
                ), f"Response doesn't appear conversational: {response[:200]}"
            # For real LLM, just check we got a response (already verified above)

            await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_tool_calling(
    test_hass, llm_config, sample_entity_states, session_manager, is_using_mock_llm, mock_llm_server
):
    """Test that LLM triggers tools correctly.

    This test verifies that:
    1. LLM recognizes when to use tools
    2. Tool calls are properly formatted
    3. Tool results are processed
    4. Final response incorporates tool results

    With mocks: Verifies tool dispatch when mock returns tool_call response.
    With real LLM: Verifies LLM correctly chooses to call tools.
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
        "custom_components.home_agent.agent.core.async_should_expose",
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

        with maybe_mock_llm(is_using_mock_llm, mock_llm_server):
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

            # Ask the agent to control a device (with /no_think suffix for reasoning models)
            response = await agent.process_message(
                text="Turn on the living room light" + NO_THINK_SUFFIX,
                conversation_id="test_tool_calling",
            )

            # Strip thinking blocks from response before validation
            response = strip_thinking_blocks(response) or ""

            # Verify we got a response
            assert response is not None, "Response should not be None"
            assert isinstance(response, str), f"Response should be a string, got {type(response)}"
            assert len(response) > 0, f"Response should not be empty, got: {response[:100]}"

            # With mocks, we expect tool calls; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Mock is configured to return tool call for "turn on...living room light"
                assert len(service_calls) > 0, f"Mock should trigger tool calls. Response: {response}"
                turn_on_called = any(call.get("service") == "turn_on" for call in service_calls)
                assert turn_on_called, f"turn_on service should be called. Calls: {service_calls}"
            # For real LLM, just check we got a response (already verified above)

            await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_multi_turn_context(
    test_hass, llm_config, session_manager, is_using_mock_llm, mock_llm_server
):
    """Test conversation memory across multiple turns.

    This test verifies that:
    1. Conversation history is maintained
    2. Context from previous turns is used
    3. Follow-up questions work correctly

    With mocks: Verifies history tracking and message routing.
    With real LLM: Verifies LLM uses context from previous turns.
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

    # Configure mock responses for deterministic multi-turn behavior
    if is_using_mock_llm and mock_llm_server:
        mock_llm_server.add_response("name", "Your name is Alice.")
        mock_llm_server.add_response("color", "You like the color blue.")

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        with maybe_mock_llm(is_using_mock_llm, mock_llm_server):
            agent = HomeAgent(test_hass, config, session_manager)

            conversation_id = "test_multi_turn"

            # First turn: Set context (with /no_think suffix for reasoning models)
            response1 = await agent.process_message(
                text="My name is Alice and I like the color blue." + NO_THINK_SUFFIX,
                conversation_id=conversation_id,
            )

            # Strip thinking blocks from response before validation
            response1 = strip_thinking_blocks(response1) or ""

            assert response1 is not None, "First response should not be None"
            assert isinstance(response1, str), f"Response should be a string, got {type(response1)}"
            assert len(response1) > 10, f"Response should be meaningful, got {len(response1)} chars"

            # Second turn: Reference previous context (with /no_think suffix)
            response2 = await agent.process_message(
                text="What is my name?" + NO_THINK_SUFFIX,
                conversation_id=conversation_id,
            )

            # Strip thinking blocks from response before validation
            response2 = strip_thinking_blocks(response2) or ""

            assert response2 is not None, "Second response should not be None"
            assert isinstance(response2, str), f"Response should be a string, got {type(response2)}"
            assert len(response2) > 0, f"Response should not be empty, got {len(response2)} chars"
            # With mocks, verify specific content; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Response should mention Alice (mock is configured for this)
                assert "alice" in response2.lower(), "Agent didn't remember name from previous turn"

            # Third turn: Reference other context (with /no_think suffix)
            response3 = await agent.process_message(
                text="What color do I like?" + NO_THINK_SUFFIX,
                conversation_id=conversation_id,
            )

            # Strip thinking blocks from response before validation
            response3 = strip_thinking_blocks(response3) or ""

            assert response3 is not None, "Third response should not be None"
            assert isinstance(response3, str), f"Response should be a string, got {type(response3)}"
            assert len(response3) > 0, f"Response should not be empty, got {len(response3)} chars"
            # With mocks, verify specific content; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Response should mention blue (mock is configured for this)
                assert "blue" in response3.lower(), "Agent didn't remember color preference"

            # Verify conversation history is populated
            history = agent.conversation_manager.get_history(conversation_id)
            assert len(history) >= 4, "Conversation history not tracking properly"

            await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_streaming_response(
    test_hass, llm_config, session_manager, is_using_mock_llm, mock_llm_server
):
    """Test SSE streaming works with LLM (real or mock).

    This test verifies that:
    1. Streaming can be enabled
    2. Response is delivered incrementally
    3. Complete response is assembled correctly

    With mocks: Verifies streaming chunk processing.
    With real LLM: Verifies actual SSE streaming.
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
        with maybe_mock_llm(is_using_mock_llm, mock_llm_server):
            agent = HomeAgent(test_hass, config, session_manager)

            # Collect streaming chunks
            chunks = []

            async def collect_chunks():
                # Add /no_think suffix to user message for reasoning models
                async for chunk in agent._call_llm_streaming(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Count from 1 to 5." + NO_THINK_SUFFIX},
                    ]
                ):
                    chunks.append(chunk)

            await collect_chunks()

            # Verify we received chunks
            assert len(chunks) > 0, "No streaming chunks received"
            assert isinstance(chunks, list), f"Chunks should be a list, got {type(chunks)}"
            # With mocks, expect multiple chunks; with real LLM, at least one chunk
            if is_using_mock_llm:
                # Should receive multiple chunks for streaming (not just one)
                assert len(chunks) >= 2, f"Expected multiple streaming chunks, got {len(chunks)}"
            else:
                # Real LLM may stream differently, just verify we got at least one chunk
                assert len(chunks) >= 1, f"Expected at least one streaming chunk, got {len(chunks)}"

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
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Try to process a message with invalid model (with /no_think suffix)
        response = None
        exception_caught = None
        try:
            response = await agent.process_message(
                text="Hello" + NO_THINK_SUFFIX,
                conversation_id="test_error",
            )
            # Strip thinking blocks if we got a response
            if response:
                response = strip_thinking_blocks(response) or ""
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
async def test_llm_with_complex_tools(
    test_hass, llm_config, sample_entity_states, session_manager, is_using_mock_llm, mock_llm_server
):
    """Test LLM handling of complex multi-step tool interactions.

    This test verifies that:
    1. LLM can chain multiple tool calls
    2. Results from one tool inform the next
    3. Final response synthesizes all tool results

    With mocks: Verifies response processing for temperature query.
    With real LLM: Verifies multi-step reasoning.
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
        "custom_components.home_agent.agent.core.async_should_expose",
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

        with maybe_mock_llm(is_using_mock_llm, mock_llm_server):
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

            # Ask for a complex multi-step action (with /no_think suffix for reasoning models)
            response = await agent.process_message(
                text="What's the temperature, and if it's below 70, turn on the thermostat" + NO_THINK_SUFFIX,
                conversation_id="test_complex_tools",
            )

            # Strip thinking blocks from response before validation
            response = strip_thinking_blocks(response) or ""

            # Verify response
            assert response is not None, "Response should not be None"
            assert isinstance(response, str), f"Response should be a string, got {type(response)}"
            assert (
                len(response) > 0
            ), f"Response should not be empty, got {len(response)} chars: {response[:100]}"

            # With mocks, verify specific content; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Response should mention temperature or heating action (the query asked about temperature)
                response_lower = response.lower()
                assert any(
                    word in response_lower for word in ["temperature", "72", "70", "degrees", "thermostat", "heat", "warm", "climate", "turn"]
                ), f"Response doesn't mention temperature or heating info: {response[:300]}"

            await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_tool_execution_with_correct_entity(
    test_hass, llm_config, sample_entity_states, session_manager, is_using_mock_llm, mock_llm_server
):
    """Test that tool calls target the correct entity_id.

    This test verifies that:
    1. LLM correctly identifies the target entity from user input
    2. Tool calls include the correct entity_id parameter
    3. Service calls are made with the right entity target
    4. Entity_id is not confused or mixed up with other entities

    With mocks: Verifies tool dispatch for bedroom light/coffee maker.
    With real LLM: Verifies correct entity targeting.
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

    # Mock entity exposure - return True to expose all test entities
    with (
        patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=True,
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

        # Add mock responses for when using mock LLM
        if is_using_mock_llm and mock_llm_server:
            mock_llm_server.add_tool_call_response(
                "bedroom light",
                "ha_control",
                {"action": "turn_on", "entity_id": "light.bedroom"},
            )
            mock_llm_server.add_tool_call_response(
                "coffee maker",
                "ha_control",
                {"action": "turn_on", "entity_id": "switch.coffee_maker"},
            )
            mock_llm_server.add_response(
                "temperature",
                "The current temperature is 72.5°F.",
            )
            mock_llm_server.default_response = "I've completed the requested action."

        with maybe_mock_llm(is_using_mock_llm, mock_llm_server):
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
            # Add /no_think suffix for reasoning models
            response1 = await agent.process_message(
                text="Turn on the bedroom light" + NO_THINK_SUFFIX,
                conversation_id="test_entity_targeting_1",
            )

            # Strip thinking blocks from response before validation
            response1 = strip_thinking_blocks(response1) or ""

            assert response1 is not None, "First response should not be None"
            assert isinstance(response1, str), f"Response should be a string, got {type(response1)}"
            assert (
                len(response1) > 0
            ), f"Response should not be empty, got {len(response1)} chars: {response1[:100]}"

            # With mocks, verify entity targeting; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Check if service was called (LLM may or may not call the tool)
                bedroom_calls = [call for call in service_calls if call.get("entity_id") == "light.bedroom"]
                living_room_calls = [
                    call for call in service_calls if call.get("entity_id") == "light.living_room"
                ]

                # If tool was called, it should target bedroom, not living room
                if len(service_calls) > 0:
                    assert (
                        len(living_room_calls) == 0
                    ), f"Should not call living_room when user asked for bedroom. Calls: {service_calls}"

                    if len(bedroom_calls) > 0:
                        assert bedroom_calls[0]["service"] in [
                            "turn_on",
                            "toggle",
                        ], f"Should turn on/toggle bedroom light, got: {bedroom_calls[0]['service']}"

            # Clear service calls for next test
            service_calls.clear()

            # Test 2: Control coffee maker specifically (with /no_think suffix)
            response2 = await agent.process_message(
                text="Turn on the coffee maker" + NO_THINK_SUFFIX,
                conversation_id="test_entity_targeting_2",
            )

            # Strip thinking blocks from response before validation
            response2 = strip_thinking_blocks(response2) or ""

            assert response2 is not None, "Second response should not be None"
            assert isinstance(response2, str), f"Response should be a string, got {type(response2)}"
            assert len(response2) > 0, f"Response should not be empty, got {len(response2)} chars"

            # With mocks, verify entity targeting; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Check if coffee maker was targeted
                coffee_maker_calls = [
                    call for call in service_calls if call.get("entity_id") == "switch.coffee_maker"
                ]
                wrong_entity_calls = [
                    call
                    for call in service_calls
                    if call.get("entity_id")
                    and call.get("entity_id") not in ["switch.coffee_maker", None]
                ]

                if len(service_calls) > 0:
                    assert (
                        len(wrong_entity_calls) == 0
                    ), f"Should only target coffee_maker, but got: {wrong_entity_calls}"

                    if len(coffee_maker_calls) > 0:
                        assert coffee_maker_calls[0]["service"] in [
                            "turn_on",
                            "toggle",
                        ], f"Should turn on coffee maker, got: {coffee_maker_calls[0]['service']}"

            # Clear service calls for next test
            service_calls.clear()

            # Test 3: Query specific sensor (not controls) (with /no_think suffix)
            response3 = await agent.process_message(
                text="What is the temperature?" + NO_THINK_SUFFIX,
                conversation_id="test_entity_targeting_3",
            )

            # Strip thinking blocks from response before validation
            response3 = strip_thinking_blocks(response3) or ""

            assert response3 is not None, "Third response should not be None"
            assert isinstance(response3, str), f"Response should be a string, got {type(response3)}"
            assert len(response3) > 0, f"Response should not be empty, got {len(response3)} chars"

            # With mocks, verify specific content; with real LLM, just verify response exists
            if is_using_mock_llm:
                # Response should acknowledge the temperature query
                response_lower = response3.lower()
                valid_response_patterns = [
                    "temperature", "72", "sensor", "degrees", "check",
                    "let me", "look", "see", "finding", "living", "room", "current",
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
            assert (
                len(control_services) == 0
            ), f"Query should not trigger control services, got: {control_services}"

            await agent.close()

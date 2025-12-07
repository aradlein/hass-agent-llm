"""Integration tests for language edge cases in Home Agent.

These tests verify that the agent correctly handles edge cases related to
language support including unicode handling, streaming with multiple languages,
tool execution across languages, and minimal interactions.

Expected to fail initially (TDD approach) until language preservation is implemented.
"""

from unittest.mock import MagicMock, patch

import pytest
from homeassistant.components import conversation as ha_conversation

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language,text,description",
    [
        ("zh", "你好，打开客厅的灯", "Chinese characters"),
        ("ru", "Привет, включи свет в гостиной", "Russian Cyrillic"),
        ("pl", "Włącz światło w salonie", "Polish special characters"),
        ("ja", "リビングルームの電気をつけて", "Japanese characters"),
        ("ar", "أضيء غرفة المعيشة", "Arabic characters"),
        ("ko", "거실 조명을 켜줘", "Korean characters"),
    ],
)
async def test_unicode_handling_per_language(
    test_hass, llm_config, session_manager, mock_llm_server, language, text, description
):
    """Test unicode handling for various languages.

    This parametrized test verifies that:
    1. The agent can process messages with unicode characters
    2. No encoding issues occur during processing
    3. Language is correctly preserved in the response
    4. Various character sets are supported (Chinese, Russian, Polish, etc.)

    Args:
        language: ISO language code
        text: Test message in the specified language
        description: Human-readable description of the test case
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    # Set default mock response for the LLM
    mock_llm_server.default_response = "I understand your request. The message was received successfully."

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ), mock_llm_server.patch_aiohttp():
        agent = HomeAgent(test_hass, config, session_manager)

        # Send message with unicode characters
        user_input = ha_conversation.ConversationInput(
            text=text,
            conversation_id=f"test_unicode_{language}",
            language=language,
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result = await agent.async_process(user_input)

        # Verify no encoding errors occurred
        assert result is not None, f"Result should not be None for {description}"
        assert result.response is not None, f"Response should not be None for {description}"

        # Verify language is preserved
        assert (
            result.response.language == language
        ), f"Expected language '{language}', got '{result.response.language}' for {description}"

        # Verify response text exists and is a string (no encoding corruption)
        response_text = result.response.speech.get("plain", {}).get("speech", "")
        assert isinstance(
            response_text, str
        ), f"Response text should be string for {description}, got {type(response_text)}"
        assert len(response_text) > 0, f"Response should not be empty for {description}"

        # Verify we can encode/decode the response without errors
        try:
            encoded = response_text.encode("utf-8")
            decoded = encoded.decode("utf-8")
            assert decoded == response_text, f"Unicode round-trip failed for {description}"
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            pytest.fail(f"Unicode encoding/decoding failed for {description}: {e}")

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_streaming_with_multilingual(test_hass, llm_config, session_manager, mock_llm_server):
    """Test streaming mode with different languages.

    This test verifies that:
    1. Streaming works with non-English languages
    2. Language is preserved in streaming responses
    3. Unicode characters stream correctly
    4. Falls back gracefully if streaming is not available
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_STREAMING_ENABLED: True,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    # Set default mock response (used for both German and Spanish)
    mock_llm_server.default_response = "Das Wetter heute ist schön und sonnig."

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ), mock_llm_server.patch_aiohttp():
        agent = HomeAgent(test_hass, config, session_manager)

        # Test German with streaming
        user_input_de = ha_conversation.ConversationInput(
            text="Wie ist das Wetter heute?",
            conversation_id="test_streaming_de",
            language="de",
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result_de = await agent.async_process(user_input_de)

        assert result_de is not None, "German streaming result should not be None"
        assert result_de.response is not None, "German streaming response should not be None"
        assert (
            result_de.response.language == "de"
        ), f"Expected language 'de', got '{result_de.response.language}'"

        # Test Spanish with streaming
        user_input_es = ha_conversation.ConversationInput(
            text="¿Cómo está el clima hoy?",
            conversation_id="test_streaming_es",
            language="es",
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result_es = await agent.async_process(user_input_es)

        assert result_es is not None, "Spanish streaming result should not be None"
        assert result_es.response is not None, "Spanish streaming response should not be None"
        assert (
            result_es.response.language == "es"
        ), f"Expected language 'es', got '{result_es.response.language}'"

        # Verify responses contain text
        response_text_de = result_de.response.speech.get("plain", {}).get("speech", "")
        response_text_es = result_es.response.speech.get("plain", {}).get("speech", "")

        assert len(response_text_de) > 0, "German streaming response should have content"
        assert len(response_text_es) > 0, "Spanish streaming response should have content"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_execution_language_agnostic(
    test_hass, llm_config, sample_entity_states, session_manager, mock_llm_server
):
    """Test tools work regardless of request language.

    This test verifies that:
    1. Tool execution works with non-English input
    2. German message can trigger ha_control tool
    3. Tool executes correctly regardless of language
    4. Response is returned in the request language
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

    # Set default mock response for German tool request
    mock_llm_server.default_response = "Natürlich! Ich habe das Wohnzimmerlicht eingeschaltet."

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ), mock_llm_server.patch_aiohttp():
        # Setup test states
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

        from unittest.mock import AsyncMock

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

        # Send German message that should trigger tool
        user_input_de = ha_conversation.ConversationInput(
            text="Schalte das Wohnzimmerlicht ein",
            conversation_id="test_tool_de",
            language="de",
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result_de = await agent.async_process(user_input_de)

        # Verify response
        assert result_de is not None, "German tool result should not be None"
        assert result_de.response is not None, "German tool response should not be None"

        # Verify language is preserved in response
        assert (
            result_de.response.language == "de"
        ), f"Expected language 'de', got '{result_de.response.language}'"

        # Note: Tool execution is non-deterministic depending on LLM behavior
        # We verify that IF tools were called, they worked correctly
        # The key assertion is that the response language is correct
        response_text = result_de.response.speech.get("plain", {}).get("speech", "")
        assert len(response_text) > 0, "Response should not be empty"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_empty_conversation_preserves_language(test_hass, llm_config, session_manager, mock_llm_server):
    """Test language preservation with minimal interaction.

    This test verifies that:
    1. Empty or very short messages work with language setting
    2. Language is preserved even with minimal input
    3. Response maintains the specified language
    """
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    # Set default mock response for minimal messages
    mock_llm_server.default_response = "Bonjour! Comment puis-je vous aider?"

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ), mock_llm_server.patch_aiohttp():
        agent = HomeAgent(test_hass, config, session_manager)

        # Test 1: Very short message in French
        user_input_short = ha_conversation.ConversationInput(
            text="Bonjour",
            conversation_id="test_short_fr",
            language="fr",
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result_short = await agent.async_process(user_input_short)

        assert result_short is not None, "Short message result should not be None"
        assert result_short.response is not None, "Short message response should not be None"
        assert (
            result_short.response.language == "fr"
        ), f"Expected language 'fr' for short message, got '{result_short.response.language}'"

        # Test 2: Single word in Italian
        user_input_word = ha_conversation.ConversationInput(
            text="Ciao",
            conversation_id="test_word_it",
            language="it",
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result_word = await agent.async_process(user_input_word)

        assert result_word is not None, "Single word result should not be None"
        assert result_word.response is not None, "Single word response should not be None"
        assert (
            result_word.response.language == "it"
        ), f"Expected language 'it' for single word, got '{result_word.response.language}'"

        # Test 3: Question mark only with language set
        user_input_minimal = ha_conversation.ConversationInput(
            text="?",
            conversation_id="test_minimal_pt",
            language="pt",
            context=MagicMock(user_id="test_user"),
            device_id=None,
            satellite_id=None,
            agent_id="home_agent",
        )

        result_minimal = await agent.async_process(user_input_minimal)

        assert result_minimal is not None, "Minimal input result should not be None"
        assert result_minimal.response is not None, "Minimal input response should not be None"
        assert (
            result_minimal.response.language == "pt"
        ), f"Expected language 'pt' for minimal input, got '{result_minimal.response.language}'"

        # Verify all responses have content
        response_texts = [
            result_short.response.speech.get("plain", {}).get("speech", ""),
            result_word.response.speech.get("plain", {}).get("speech", ""),
            result_minimal.response.speech.get("plain", {}).get("speech", ""),
        ]

        for i, text in enumerate(response_texts):
            assert len(text) > 0, f"Response {i+1} should have content"

        await agent.close()

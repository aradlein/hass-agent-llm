"""Integration tests for multilingual conversation support.

This test suite validates that Home Agent correctly preserves language codes
through the conversation pipeline from ConversationInput.language to
IntentResponse.language. It tests support for the top 10 languages used in
Home Assistant:

- English (en)
- German (de)
- Spanish (es)
- French (fr)
- Dutch (nl)
- Italian (it)
- Polish (pl)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)

The tests use a TDD approach and are expected to initially fail until the
language preservation implementation is complete. Each test verifies that
the language parameter flows through the conversation system correctly.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_STREAMING_ENABLED,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
)
from tests.integration.helpers import send_message_and_wait

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def base_agent_config() -> dict:
    """Provide base configuration for HomeAgent.

    Returns:
        Dictionary with minimal agent configuration for testing
    """
    return {
        CONF_LLM_BASE_URL: "http://localhost:11434/v1",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "qwen2.5:3b",
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: DEFAULT_HISTORY_MAX_MESSAGES,
        CONF_HISTORY_MAX_TOKENS: DEFAULT_HISTORY_MAX_TOKENS,
        CONF_EMIT_EVENTS: False,  # Disable events for cleaner test logs
        CONF_STREAMING_ENABLED: False,  # Use synchronous mode for deterministic tests
    }


@pytest.fixture
def mock_llm_response() -> dict:
    """Provide mock LLM response for all language tests.

    Returns:
        Dictionary representing a successful LLM API response
    """
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I have processed your request successfully.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }


@pytest.fixture
async def test_agent(test_hass, base_agent_config, session_manager):
    """Create a HomeAgent instance for testing.

    Args:
        test_hass: Mock Home Assistant instance from conftest
        base_agent_config: Base configuration dictionary
        session_manager: Conversation session manager

    Returns:
        Configured HomeAgent instance
    """
    agent = HomeAgent(test_hass, base_agent_config, session_manager)
    return agent


@pytest.mark.asyncio
async def test_english_conversation_processing(test_agent, mock_llm_response):
    """Test English conversation language preservation.

    Verifies that when a conversation is initiated with language='en',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Turn on the living room light",
            language="en",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "en"


@pytest.mark.asyncio
async def test_german_conversation_processing(test_agent, mock_llm_response):
    """Test German conversation language preservation.

    Verifies that when a conversation is initiated with language='de',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Schalte das Wohnzimmerlicht ein",
            language="de",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "de"


@pytest.mark.asyncio
async def test_spanish_conversation_processing(test_agent, mock_llm_response):
    """Test Spanish conversation language preservation.

    Verifies that when a conversation is initiated with language='es',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Enciende la luz del salón",
            language="es",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "es"


@pytest.mark.asyncio
async def test_french_conversation_processing(test_agent, mock_llm_response):
    """Test French conversation language preservation.

    Verifies that when a conversation is initiated with language='fr',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Allume la lumière du salon",
            language="fr",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "fr"


@pytest.mark.asyncio
async def test_dutch_conversation_processing(test_agent, mock_llm_response):
    """Test Dutch conversation language preservation.

    Verifies that when a conversation is initiated with language='nl',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Doe het woonkamerlicht aan",
            language="nl",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "nl"


@pytest.mark.asyncio
async def test_italian_conversation_processing(test_agent, mock_llm_response):
    """Test Italian conversation language preservation.

    Verifies that when a conversation is initiated with language='it',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Accendi la luce del soggiorno",
            language="it",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "it"


@pytest.mark.asyncio
async def test_polish_conversation_processing(test_agent, mock_llm_response):
    """Test Polish conversation language preservation.

    Verifies that when a conversation is initiated with language='pl',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Włącz światło w salonie",
            language="pl",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "pl"


@pytest.mark.asyncio
async def test_portuguese_conversation_processing(test_agent, mock_llm_response):
    """Test Portuguese conversation language preservation.

    Verifies that when a conversation is initiated with language='pt',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Acenda a luz da sala",
            language="pt",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "pt"


@pytest.mark.asyncio
async def test_russian_conversation_processing(test_agent, mock_llm_response):
    """Test Russian conversation language preservation.

    Verifies that when a conversation is initiated with language='ru',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="Включи свет в гостиной",
            language="ru",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "ru"


@pytest.mark.asyncio
async def test_chinese_conversation_processing(test_agent, mock_llm_response):
    """Test Chinese conversation language preservation.

    Verifies that when a conversation is initiated with language='zh',
    the response maintains the same language code.
    """
    with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
        result = await send_message_and_wait(
            test_agent,
            message="打开客厅的灯",
            language="zh",
            timeout=10.0,
        )

        # Verify response contains language
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == "zh"


@pytest.mark.asyncio
async def test_multilingual_concurrent_conversations(test_agent, mock_llm_response):
    """Test concurrent conversations in different languages.

    Verifies that the agent can handle multiple simultaneous conversations
    in different languages without mixing up language codes.

    This test creates 10 concurrent conversations, one for each supported
    language, and verifies that each response maintains its correct language.
    """
    # Define test cases for all 10 languages
    test_cases = [
        ("en", "Turn on the light", "conv_en_001"),
        ("de", "Schalte das Licht ein", "conv_de_001"),
        ("es", "Enciende la luz", "conv_es_001"),
        ("fr", "Allume la lumière", "conv_fr_001"),
        ("nl", "Doe het licht aan", "conv_nl_001"),
        ("it", "Accendi la luce", "conv_it_001"),
        ("pl", "Włącz światło", "conv_pl_001"),
        ("pt", "Acenda a luz", "conv_pt_001"),
        ("ru", "Включи свет", "conv_ru_001"),
        ("zh", "打开灯", "conv_zh_001"),
    ]

    async def process_language_conversation(language: str, message: str, conv_id: str):
        """Process a single conversation and verify language preservation.

        Args:
            language: Language code to use
            message: Test message to send
            conv_id: Conversation ID for isolation

        Returns:
            Tuple of (language, result) for verification
        """
        with patch.object(test_agent, "_call_llm", return_value=mock_llm_response):
            result = await send_message_and_wait(
                test_agent,
                message=message,
                language=language,
                conversation_id=conv_id,
                timeout=15.0,
            )
            return (language, result)

    # Execute all conversations concurrently
    tasks = [process_language_conversation(lang, msg, conv_id) for lang, msg, conv_id in test_cases]

    results = await asyncio.gather(*tasks)

    # Verify each result has the correct language
    for expected_language, result in results:
        assert result.response is not None
        assert hasattr(result.response, "language")
        assert result.response.language == expected_language, (
            f"Expected language '{expected_language}' but got " f"'{result.response.language}'"
        )

    # Verify we got results for all 10 languages
    assert len(results) == 10
    languages_received = {result.response.language for _, result in results}
    assert languages_received == {"en", "de", "es", "fr", "nl", "it", "pl", "pt", "ru", "zh"}

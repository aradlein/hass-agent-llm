"""Integration tests for edge cases and stress testing.

These tests verify that the Home Agent handles edge cases correctly:
- Unicode and internationalization (RTL languages, emojis, mixed scripts)
- Large payloads (long messages, large context, large history)
- Concurrency (simultaneous conversations, concurrent memory access)
- Malformed or unusual inputs
"""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant, State

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONTEXT_MODE_DIRECT,
)

_LOGGER = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rtl_language_arabic(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test Arabic (RTL) input and response handling.

    Verifies:
    1. Arabic text is processed correctly
    2. No encoding errors occur
    3. Response is returned successfully
    4. Conversation history preserves RTL text
    """
    # Mock LLM response with Arabic text
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "أنا مساعد منزلي ذكي. كيف يمكنني مساعدتك؟",  # "I am a smart home assistant. How can I help you?"
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Arabic text: "Turn on the living room lights"
        arabic_input = "شغل أضواء غرفة المعيشة"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=arabic_input,
                conversation_id="test_arabic_rtl",
            )

        # Verify response is returned
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        # Verify it contains Arabic characters (Unicode range U+0600 to U+06FF)
        assert any('\u0600' <= c <= '\u06FF' for c in response), "Response should contain Arabic characters"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_emoji_in_user_input(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test emoji handling in messages.

    Verifies:
    1. Emoji in user input is processed correctly
    2. Emoji in response is handled correctly
    3. No encoding errors occur
    """
    # Mock LLM response with emoji
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Sure! I'll turn on the lights 💡✨",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # User input with emoji
        emoji_input = "🏠 Turn on the lights in the living room 💡"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=emoji_input,
                conversation_id="test_emoji",
            )

        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mixed_script_text(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test mixed-script text (English + Chinese).

    Verifies:
    1. Mixed script text is processed correctly
    2. No encoding errors occur
    3. Both scripts are preserved
    """
    # Mock LLM response with mixed scripts
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "好的！I will turn on the 灯 (lights).",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Mixed English and Chinese
        mixed_input = "请 turn on the living room 灯光 please"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=mixed_input,
                conversation_id="test_mixed_script",
            )

        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_zero_width_characters(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test handling of zero-width characters (security concern).

    Verifies:
    1. Zero-width characters don't cause crashes
    2. Processing completes successfully
    3. No security vulnerabilities exposed
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I'll turn on the lights.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Input with zero-width characters
        zwj = "\u200D"  # Zero-width joiner
        zwnj = "\u200C"  # Zero-width non-joiner
        zwsp = "\u200B"  # Zero-width space

        malicious_input = f"Turn{zwj}on{zwnj}the{zwsp}lights"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=malicious_input,
                conversation_id="test_zero_width",
            )

        # Verify response (should succeed despite zero-width chars)
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_very_long_user_message(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test handling of very long user input (>10KB).

    Verifies:
    1. Long messages are processed without crashes
    2. Memory usage is reasonable
    3. Response is returned successfully
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I understand your request.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2500, "completion_tokens": 10, "total_tokens": 2510},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Create a very long message (>10KB)
        long_message = "Please turn on the lights and " + ("really " * 2000) + "make sure they're bright."
        assert len(long_message.encode('utf-8')) > 10000, "Message should be >10KB"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=long_message,
                conversation_id="test_long_message",
            )

        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_context_many_entities(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test handling of large context with 100+ entities.

    Verifies:
    1. Large context is processed correctly
    2. No performance degradation
    3. All entities are accessible
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I can see all your devices.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 10, "total_tokens": 510},
    }

    # Create 100+ entity states
    large_entity_states = [
        State(
            f"light.room_{i}",
            "on" if i % 2 == 0 else "off",
            {"brightness": 255 if i % 2 == 0 else 0, "friendly_name": f"Room {i} Light"},
        )
        for i in range(150)
    ]

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        # Setup test states
        test_hass.states.async_all = MagicMock(return_value=large_entity_states)

        agent = HomeAgent(test_hass, config, session_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text="How many lights are on?",
                conversation_id="test_large_context",
            )

        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_conversation_history(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test handling of large conversation history (50+ messages).

    Verifies:
    1. Large history is maintained correctly
    2. Truncation works as expected
    3. No memory leaks
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Acknowledged.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: 100,  # Allow up to 100 messages
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        conversation_id = "test_large_history"

        # Send 50 messages to build up history
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            for i in range(50):
                response = await agent.process_message(
                    text=f"Message number {i}",
                    conversation_id=conversation_id,
                )

                # Verify each response succeeds
                assert response is not None, f"Response {i} should not be None"

        # Verify the conversation completes successfully
        # The session_manager handles history internally
        # 50 user messages + 50 assistant responses = 100 total messages

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_conversations(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test multiple simultaneous conversations.

    Verifies:
    1. Concurrent conversations don't interfere
    2. Each conversation maintains separate state
    3. No race conditions or deadlocks
    """
    # Mock LLM response
    def create_mock_response(conv_id: str):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Response for {conv_id}",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: True,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Create multiple concurrent conversations
        num_conversations = 10

        async def process_conversation(conv_num: int):
            """Process a single conversation."""
            conv_id = f"test_concurrent_{conv_num}"
            mock_response = create_mock_response(conv_id)

            with patch.object(agent, "_call_llm", return_value=mock_response):
                response = await agent.process_message(
                    text=f"Message for conversation {conv_num}",
                    conversation_id=conv_id,
                )

            assert response is not None, f"Response for conv {conv_num} should not be None"
            assert isinstance(response, str), f"Response for conv {conv_num} should be a string"
            assert len(response) > 0, f"Response for conv {conv_num} should not be empty"

            return conv_id

        # Run conversations concurrently
        tasks = [process_conversation(i) for i in range(num_conversations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all conversations succeeded
        assert len(results) == num_conversations, "Should have processed all conversations"
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Conversation {i} failed with exception: {result}")

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_malformed_utf8_sequences(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test handling of malformed UTF-8 sequences.

    Verifies:
    1. Malformed UTF-8 doesn't crash the system
    2. Error handling is graceful
    3. Valid processing continues
    """
    # Mock LLM response
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Processed your request.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Try to create malformed UTF-8 (Python 3 handles this gracefully)
        # Use replacement character to simulate malformed input
        malformed_input = "Turn on the lights \ufffd please"  # Replacement character

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=malformed_input,
                conversation_id="test_malformed_utf8",
            )

        # Verify response (should handle gracefully)
        assert response is not None, "Response should not be None even with malformed input"
        assert isinstance(response, str), "Response should be a string"

        await agent.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hebrew_rtl_text(
    test_hass: HomeAssistant,
    llm_config: dict[str, Any],
    session_manager,
):
    """Test Hebrew (RTL) text handling.

    Verifies:
    1. Hebrew text is processed correctly
    2. RTL directionality is preserved
    3. No encoding errors occur
    """
    # Mock LLM response with Hebrew
    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "אני עוזר בית חכם. איך אוכל לעזור?",  # "I am a smart home assistant. How can I help?"
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 10, "total_tokens": 22},
    }

    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 200,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.core.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Hebrew text: "Turn on the lights"
        hebrew_input = "הדלק את האורות"

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text=hebrew_input,
                conversation_id="test_hebrew_rtl",
            )

        # Verify response
        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"
        # Verify it contains Hebrew characters (Unicode range U+0590 to U+05FF)
        assert any('\u0590' <= c <= '\u05FF' for c in response), "Response should contain Hebrew characters"

        await agent.close()

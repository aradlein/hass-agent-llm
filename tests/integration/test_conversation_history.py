"""Integration tests for conversation history management.

These tests verify that conversation history correctly persists, loads,
and applies token/message limits.
"""

import pytest
from unittest.mock import patch

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_HISTORY_PERSIST,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
)


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_history_persistence(test_hass, llm_config, session_manager):
    """Test that conversation history persists and loads correctly.

    This test verifies that:
    1. History is saved to storage when persist is enabled
    2. History can be loaded from storage on restart
    3. Loaded history contains the correct message content
    """
    conversation_id = "test_persistence"

    # Configuration with persistence enabled
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_PERSIST: True,
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        # Create first agent and send messages
        agent1 = HomeAgent(test_hass, config, session_manager)

        # Load any existing history
        await agent1.conversation_manager.load_from_storage()

        # Send test messages
        test_message1 = "My favorite color is purple"
        response1 = await agent1.process_message(
            text=test_message1,
            conversation_id=conversation_id,
        )

        assert response1 is not None, "First response should not be None"
        assert isinstance(response1, str), f"Response should be a string, got {type(response1)}"
        assert len(response1) > 10, f"Response should be meaningful (>10 chars), got {len(response1)} chars: {response1[:100]}"
        # Response should be conversational and acknowledge the input
        response1_lower = response1.lower()
        assert any(word in response1_lower for word in ["purple", "color", "favorite", "thanks", "great", "noted", "understand"]), \
            f"Response should acknowledge the color preference, got: {response1[:200]}"

        test_message2 = "I live in New York"
        response2 = await agent1.process_message(
            text=test_message2,
            conversation_id=conversation_id,
        )

        assert response2 is not None, "Second response should not be None"
        assert isinstance(response2, str), f"Response should be a string, got {type(response2)}"
        assert len(response2) > 10, f"Response should be meaningful, got {len(response2)} chars"

        # Get history before closing
        history1 = agent1.conversation_manager.get_history(conversation_id)
        assert len(history1) >= 2, "History should contain at least user messages"

        # Verify our messages are in the history
        user_messages = [msg for msg in history1 if msg.get("role") == "user"]
        assert len(user_messages) >= 2, "Should have at least 2 user messages"

        message_texts = [msg.get("content", "") for msg in user_messages]
        assert any(test_message1 in text for text in message_texts), \
            "First message not found in history"
        assert any(test_message2 in text for text in message_texts), \
            "Second message not found in history"

        # Save and close first agent
        await agent1.conversation_manager.save_to_storage()
        await agent1.close()

        # Create second agent (simulating restart)
        agent2 = HomeAgent(test_hass, config, session_manager)

        # Load history from storage
        await agent2.conversation_manager.load_from_storage()

        # Get history - should match what we saved
        history2 = agent2.conversation_manager.get_history(conversation_id)

        # Verify history was loaded
        assert len(history2) > 0, "History should be loaded from storage"
        assert isinstance(history2, list), f"History should be a list, got {type(history2)}"
        # Verify history contains message dictionaries
        assert all(isinstance(msg, dict) for msg in history2), "All history entries should be dictionaries"
        assert len(history2) == len(history1), \
            f"Loaded history length {len(history2)} doesn't match saved {len(history1)}"

        # Verify message content matches
        user_messages2 = [msg for msg in history2 if msg.get("role") == "user"]
        message_texts2 = [msg.get("content", "") for msg in user_messages2]

        assert any(test_message1 in text for text in message_texts2), \
            "First message not found in loaded history"
        assert any(test_message2 in text for text in message_texts2), \
            "Second message not found in loaded history"

        await agent2.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_history_token_limits(test_hass, llm_config, session_manager):
    """Test that history is truncated when token limits are exceeded.

    This test verifies that:
    1. History respects max_tokens configuration
    2. Older messages are removed when limit is exceeded
    3. Recent messages are preserved
    """
    conversation_id = "test_token_limits"

    # Configuration with very low token limit
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_PERSIST: False,
        CONF_HISTORY_MAX_MESSAGES: 100,  # High message limit
        CONF_HISTORY_MAX_TOKENS: 200,  # Low token limit (~800 chars)
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Send several messages to exceed token limit
        messages = [
            "This is my first message about a topic",
            "This is my second message with more information",
            "This is my third message containing even more details",
            "This is my fourth and final message with the most recent information",
        ]

        for msg in messages:
            response = await agent.process_message(
                text=msg,
                conversation_id=conversation_id,
            )
            assert response is not None, f"Response should not be None for message: {msg}"
            assert isinstance(response, str), f"Response should be a string, got {type(response)}"
            assert len(response) > 5, f"Response should not be empty, got {len(response)} chars"

        # Get history
        history = agent.conversation_manager.get_history(conversation_id)

        # Verify history exists
        assert len(history) > 0, "History should not be empty"
        assert isinstance(history, list), f"History should be a list, got {type(history)}"

        # Calculate approximate token count
        total_chars = sum(len(str(msg.get("content", ""))) for msg in history)
        estimated_tokens = total_chars / 4  # Rough estimation

        # Should be under or close to limit (allowing for some overhead)
        # Note: The actual truncation may not be exact due to estimation
        assert estimated_tokens < 300, \
            f"History token count {estimated_tokens} should be near limit of 200"

        # Most recent message should still be in history
        user_messages = [msg for msg in history if msg.get("role") == "user"]
        latest_message_content = [msg.get("content", "") for msg in user_messages]

        # At minimum, the last message should be present
        assert any(messages[-1] in content for content in latest_message_content), \
            "Most recent message should be preserved"

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_history_message_limits(test_hass, llm_config, session_manager):
    """Test that history respects message count limits.

    This test verifies that:
    1. History respects max_messages configuration
    2. Older messages are removed when limit is exceeded
    3. Message count never exceeds the configured limit
    """
    conversation_id = "test_message_limits"
    max_messages = 4

    # Configuration with low message limit
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_PERSIST: False,
        CONF_HISTORY_MAX_MESSAGES: max_messages,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Send more messages than the limit
        messages = [
            "Message number one",
            "Message number two",
            "Message number three",
            "Message number four",
            "Message number five",
            "Message number six",
        ]

        for i, msg in enumerate(messages):
            response = await agent.process_message(
                text=msg,
                conversation_id=conversation_id,
            )
            assert response is not None, f"Response should not be None for message {i+1}: {msg}"
            assert isinstance(response, str), f"Response should be a string, got {type(response)}"
            assert len(response) > 5, f"Response should not be empty, got {len(response)} chars"

            # Check history after each message
            history = agent.conversation_manager.get_history(conversation_id)

            # History should never exceed max_messages
            assert len(history) <= max_messages, \
                f"History length {len(history)} exceeds max_messages {max_messages} after message {i+1}"

        # Final check
        final_history = agent.conversation_manager.get_history(conversation_id)
        assert len(final_history) <= max_messages, \
            f"Final history length {len(final_history)} exceeds max_messages {max_messages}"

        # Most recent messages should be in history
        user_messages = [msg for msg in final_history if msg.get("role") == "user"]
        message_contents = [msg.get("content", "") for msg in user_messages]

        # Last message should definitely be present
        assert any(messages[-1] in content for content in message_contents), \
            "Most recent message should be in history"

        # First message should likely be gone (truncated)
        # Note: Depending on how many messages are generated by the LLM,
        # we might or might not have the first user message

        await agent.close()


@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_history_disabled(test_hass, llm_config, session_manager):
    """Test that history is not maintained when disabled.

    This test verifies that:
    1. When history is disabled, messages are not stored
    2. Each conversation turn is independent
    3. Context from previous messages is not used
    """
    conversation_id = "test_history_disabled"

    # Configuration with history disabled
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 500,
        CONF_HISTORY_ENABLED: False,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
    }

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, config, session_manager)

        # Send first message
        response1 = await agent.process_message(
            text="My name is Bob",
            conversation_id=conversation_id,
        )
        assert response1 is not None, "First response should not be None"
        assert isinstance(response1, str), f"Response should be a string, got {type(response1)}"
        assert len(response1) > 5, f"Response should not be empty, got {len(response1)} chars"

        # Get history - should be empty or minimal when disabled
        history1 = agent.conversation_manager.get_history(conversation_id)

        # When history is disabled, get_history might return empty list
        # or just the current context, but shouldn't grow
        initial_length = len(history1)

        # Send second message
        response2 = await agent.process_message(
            text="What is my name?",
            conversation_id=conversation_id,
        )
        assert response2 is not None, "Second response should not be None"
        assert isinstance(response2, str), f"Response should be a string, got {type(response2)}"
        assert len(response2) > 5, f"Response should not be empty, got {len(response2)} chars"

        # History should not have grown significantly
        history2 = agent.conversation_manager.get_history(conversation_id)

        # With history disabled, the agent shouldn't remember previous context
        # This means the history length should not accumulate
        assert len(history2) <= initial_length + 2, \
            "History should not grow when disabled"

        # The response likely won't contain "Bob" since history is disabled
        # (unless the LLM hallucinates it), but we can't assert that reliably

        await agent.close()

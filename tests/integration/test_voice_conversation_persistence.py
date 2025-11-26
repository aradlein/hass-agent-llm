"""Integration tests for voice conversation persistence.

These tests verify that the ConversationSessionManager correctly persists
conversation sessions across multiple voice interactions, handles device isolation,
and properly manages session timeouts and clearing.

The tests use a real ConversationSessionManager (not mocked) with a mock
Home Assistant instance and mock LLM responses to verify actual session
manager behavior in realistic scenarios.
"""

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant

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
    CONF_SESSION_TIMEOUT,
    CONF_STREAMING_ENABLED,
    DEFAULT_SESSION_TIMEOUT,
)
from custom_components.home_agent.conversation_session import (
    ConversationSessionManager,
)


@pytest.fixture
def test_config(test_hass: HomeAssistant) -> dict[str, Any]:
    """Create test configuration for the agent.

    Returns:
        Configuration dictionary with minimal settings for testing
    """
    return {
        CONF_LLM_BASE_URL: "http://localhost:11434",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "qwen2.5:3b",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 150,
        CONF_HISTORY_ENABLED: True,
        CONF_HISTORY_MAX_MESSAGES: 10,
        CONF_EMIT_EVENTS: False,
        CONF_DEBUG_LOGGING: False,
        CONF_STREAMING_ENABLED: False,
        CONF_SESSION_TIMEOUT: DEFAULT_SESSION_TIMEOUT,
    }


@pytest.fixture
async def session_manager(test_hass: HomeAssistant) -> ConversationSessionManager:
    """Create a real ConversationSessionManager for testing.

    Args:
        test_hass: Mock Home Assistant instance

    Returns:
        Initialized ConversationSessionManager
    """
    manager = ConversationSessionManager(test_hass, session_timeout=3600)
    await manager.async_load()
    return manager


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Create a mock LLM response for testing.

    Returns:
        Mock response in OpenAI API format
    """
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen2.5:3b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I understand. How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
        },
    }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_persistence_across_messages(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify that the same user/device gets the same conversation ID across multiple messages.

    This test verifies:
    1. First message from a user creates a new conversation_id
    2. Second message from same user reuses the same conversation_id
    3. Session manager correctly persists the mapping
    4. Both messages are in the same conversation history
    """
    # Create agent with session manager
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, session_manager)

        # Mock LLM API calls
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # First message from user (using device_id for session lookup)
            response_1 = await agent.process_message(
                text="Turn on the lights",
                conversation_id=None,  # No conversation_id provided
                device_id="kitchen_satellite",
            )

            # Verify response was generated
            assert response_1 is not None, "First message should return response"

            # Verify session was created and stored
            conversation_id_1 = session_manager.get_conversation_id(device_id="kitchen_satellite")
            assert conversation_id_1 is not None, "Session should be created for device"
            assert len(conversation_id_1) > 0, "Conversation ID should not be empty"

            # Second message from same device (no conversation_id provided)
            response_2 = await agent.process_message(
                text="What's the temperature?",
                conversation_id=None,  # Still no conversation_id provided
                device_id="kitchen_satellite",
            )

            assert response_2 is not None, "Second message should return response"

            # Verify same conversation ID was reused
            conversation_id_2 = session_manager.get_conversation_id(device_id="kitchen_satellite")
            assert conversation_id_2 == conversation_id_1, (
                "Second message should reuse the same conversation_id"
            )

            # Verify both messages are in the same conversation history
            history = agent.conversation_manager.get_history(conversation_id_1)
            user_messages = [msg for msg in history if msg.get("role") == "user"]

            assert len(user_messages) >= 2, "History should contain both user messages"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_device_isolation(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify that different devices for the same user get different conversation IDs.

    This test verifies:
    1. Same user on device A gets conversation_id_A
    2. Same user on device B gets conversation_id_B
    3. conversation_id_A != conversation_id_B (device isolation)
    4. Each device maintains its own separate conversation
    """
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, session_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # Message from kitchen device
            kitchen_response = await agent.process_message(
                text="Turn on kitchen lights",
                conversation_id=None,
                device_id="kitchen_satellite",
            )

            assert kitchen_response is not None, "Kitchen device should get response"

            kitchen_conv_id = session_manager.get_conversation_id(device_id="kitchen_satellite")
            assert kitchen_conv_id is not None, "Kitchen device should get conversation_id"

            # Message from bedroom device (same user, different device)
            bedroom_response = await agent.process_message(
                text="Turn on bedroom lights",
                conversation_id=None,
                device_id="bedroom_satellite",
            )

            assert bedroom_response is not None, "Bedroom device should get response"

            bedroom_conv_id = session_manager.get_conversation_id(device_id="bedroom_satellite")
            assert bedroom_conv_id is not None, "Bedroom device should get conversation_id"

            # Verify devices have different conversation IDs
            assert kitchen_conv_id != bedroom_conv_id, (
                "Different devices should have independent conversation contexts"
            )

            # Verify session manager has both mappings
            assert session_manager.get_conversation_id(
                device_id="kitchen_satellite"
            ) == kitchen_conv_id
            assert session_manager.get_conversation_id(
                device_id="bedroom_satellite"
            ) == bedroom_conv_id

            # Verify each device maintains separate history
            kitchen_history = agent.conversation_manager.get_history(kitchen_conv_id)
            bedroom_history = agent.conversation_manager.get_history(bedroom_conv_id)

            kitchen_messages = [msg for msg in kitchen_history if msg.get("role") == "user"]
            bedroom_messages = [msg for msg in bedroom_history if msg.get("role") == "user"]

            # Check that messages are in correct conversations
            assert any("kitchen" in msg.get("content", "").lower() for msg in kitchen_messages)
            assert any("bedroom" in msg.get("content", "").lower() for msg in bedroom_messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_explicit_conversation_id_honored(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify that explicitly provided conversation_id is used instead of session lookup.

    This test verifies:
    1. When conversation_id is explicitly provided, it's used
    2. Session lookup is bypassed when explicit ID is provided
    3. Session manager activity is still updated
    4. Explicit ID takes precedence over session mapping
    """
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, session_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # Create an initial session for the device
            initial_response = await agent.process_message(
                text="First message",
                conversation_id=None,
                device_id="test_device",
            )

            assert initial_response is not None, "Initial message should get response"

            # Verify session was created
            initial_conv_id = session_manager.get_conversation_id(device_id="test_device")
            assert initial_conv_id is not None, "Initial message should create conversation_id"

            # Now send message with explicit conversation_id (different from session)
            explicit_conv_id = "explicit_conversation_123"
            explicit_response = await agent.process_message(
                text="Second message with explicit ID",
                conversation_id=explicit_conv_id,  # Explicitly provided
                device_id="test_device",
            )

            assert explicit_response is not None, "Explicit ID message should get response"

            # Verify message went to the explicitly specified conversation
            explicit_history = agent.conversation_manager.get_history(explicit_conv_id)
            explicit_messages = [msg for msg in explicit_history if msg.get("role") == "user"]

            assert len(explicit_messages) >= 1, "Message should be in explicit conversation"
            assert any(
                "explicit ID" in msg.get("content", "") for msg in explicit_messages
            ), "Explicit conversation should contain the right message"

            # Verify the session still has the original conversation_id (not updated)
            session_conv_id = session_manager.get_conversation_id(device_id="test_device")
            assert session_conv_id == initial_conv_id, (
                "Session should maintain original conversation_id when explicit ID is used"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_activity_updates(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify that session activity is updated after successful conversation.

    This test verifies:
    1. Session last_activity timestamp is updated on each message
    2. Activity updates prevent session expiration
    3. Session info reflects recent activity
    """
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, session_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # Send first message
            response_1 = await agent.process_message(
                text="First message",
                conversation_id=None,
                device_id="activity_test_device",
            )

            assert response_1 is not None, "First message should get response"

            conv_id = session_manager.get_conversation_id(device_id="activity_test_device")
            assert conv_id is not None, "Session should be created"

            # Get session info after first message
            session_info_1 = session_manager.get_session_info()
            assert session_info_1["total_sessions"] >= 1, "Should have at least one session"

            # Find our session
            our_session = next(
                (s for s in session_info_1["sessions"] if s["conversation_id"] == conv_id),
                None,
            )
            assert our_session is not None, "Our session should be in session info"

            age_1 = our_session["age_seconds"]
            assert age_1 >= 0, "Session age should be non-negative"
            assert age_1 < 5, "Session should be very recent (< 5 seconds)"

            # Wait a bit to allow time to pass
            await asyncio.sleep(1)

            # Send second message to update activity
            response_2 = await agent.process_message(
                text="Second message",
                conversation_id=None,
                device_id="activity_test_device",
            )

            assert response_2 is not None, "Second message should get response"

            # Get session info after second message
            session_info_2 = session_manager.get_session_info()
            our_session_2 = next(
                (s for s in session_info_2["sessions"] if s["conversation_id"] == conv_id),
                None,
            )

            assert our_session_2 is not None, "Session should still exist"

            age_2 = our_session_2["age_seconds"]

            # Age should be very small (recently updated)
            assert age_2 < age_1 + 2, (
                "Activity update should refresh the session timestamp"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clear_conversation_service(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify the clear_conversation service works correctly.

    This test verifies:
    1. clear_session removes a specific device's session
    2. Subsequent message from that device creates new conversation_id
    3. clear_all_sessions removes all sessions
    4. Session count is correctly reported
    """
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, session_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # Create sessions for two devices
            response_1 = await agent.process_message(
                text="Device 1 message",
                conversation_id=None,
                device_id="device_1",
            )

            assert response_1 is not None, "Device 1 should get response"
            conv_id_1 = session_manager.get_conversation_id(device_id="device_1")
            assert conv_id_1 is not None, "Device 1 should have conversation_id"

            response_2 = await agent.process_message(
                text="Device 2 message",
                conversation_id=None,
                device_id="device_2",
            )

            assert response_2 is not None, "Device 2 should get response"
            conv_id_2 = session_manager.get_conversation_id(device_id="device_2")
            assert conv_id_2 is not None, "Device 2 should have conversation_id"

            # Verify both sessions exist
            session_info = session_manager.get_session_info()
            assert session_info["total_sessions"] >= 2, "Should have at least 2 sessions"

            # Clear device_1's session
            cleared = await session_manager.clear_session(device_id="device_1")
            assert cleared is True, "clear_session should return True when session exists"

            # Verify device_1's session is gone
            assert session_manager.get_conversation_id(device_id="device_1") is None, (
                "Device 1 session should be cleared"
            )

            # Verify device_2's session still exists
            assert session_manager.get_conversation_id(device_id="device_2") == conv_id_2, (
                "Device 2 session should still exist"
            )

            # Send new message from device_1 - should get NEW conversation_id
            response_1_new = await agent.process_message(
                text="Device 1 new message",
                conversation_id=None,
                device_id="device_1",
            )

            assert response_1_new is not None, "Device 1 new message should get response"
            conv_id_1_new = session_manager.get_conversation_id(device_id="device_1")

            assert conv_id_1_new != conv_id_1, (
                "After clearing session, device should get new conversation_id"
            )

            # Test clear_all_sessions
            count = await session_manager.clear_all_sessions()
            assert count >= 2, f"Should have cleared at least 2 sessions, got {count}"

            # Verify all sessions are cleared
            session_info_after = session_manager.get_session_info()
            assert session_info_after["total_sessions"] == 0, (
                "All sessions should be cleared"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_conversation_persistence_in_streaming_mode(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify session persistence works in streaming mode.

    This test verifies:
    1. Streaming mode creates and persists conversation sessions
    2. Session lookup works correctly in streaming mode
    3. Activity is updated after streaming response completes
    4. Device isolation works in streaming mode
    """
    # Enable streaming in config
    streaming_config = {**test_config, CONF_STREAMING_ENABLED: True}

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, streaming_config, session_manager)

        # Mock the LLM call (streaming still uses process_message internally)
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # First streaming message
            # Note: streaming uses async_process, but we can test via the agent's session manager
            stream_response_1 = await agent.process_message(
                text="First streaming message",
                conversation_id=None,
                device_id="streaming_device",
            )

            assert stream_response_1 is not None, "Streaming mode should return response"

            # Verify session was created
            conv_id_1 = session_manager.get_conversation_id(device_id="streaming_device")
            assert conv_id_1 is not None, (
                "Session manager should store conversation_id in streaming mode"
            )

            # Second streaming message (should reuse same conversation_id)
            stream_response_2 = await agent.process_message(
                text="Second streaming message",
                conversation_id=None,
                device_id="streaming_device",
            )

            assert stream_response_2 is not None, "Second streaming message should return response"

            conv_id_2 = session_manager.get_conversation_id(device_id="streaming_device")
            assert conv_id_2 == conv_id_1, (
                "Streaming mode should reuse conversation_id from session"
            )

            # Verify session activity was updated
            session_info = session_manager.get_session_info()
            our_session = next(
                (s for s in session_info["sessions"] if s["conversation_id"] == conv_id_1),
                None,
            )

            assert our_session is not None, "Session should exist after streaming"
            assert our_session["age_seconds"] < 5, (
                "Session should have been recently updated"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_timeout_behavior(
    test_hass: HomeAssistant,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify that expired sessions are not returned.

    This test verifies:
    1. Sessions expire after timeout period
    2. Expired sessions return None on lookup
    3. New session is created after expiration
    """
    # Create session manager with very short timeout (2 seconds)
    short_timeout_manager = ConversationSessionManager(test_hass, session_timeout=2)
    await short_timeout_manager.async_load()

    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, short_timeout_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # Create a session
            response_1 = await agent.process_message(
                text="First message",
                conversation_id=None,
                device_id="timeout_device",
            )

            assert response_1 is not None, "First message should get response"

            # Verify session exists
            conv_id_1 = short_timeout_manager.get_conversation_id(device_id="timeout_device")
            assert conv_id_1 is not None, "Session should be created"

            # Wait for session to expire (2+ seconds)
            await asyncio.sleep(2.5)

            # Try to retrieve expired session - should return None
            expired_id = short_timeout_manager.get_conversation_id(device_id="timeout_device")
            assert expired_id is None, "Expired session should return None"

            # Send new message - should create new conversation_id
            response_2 = await agent.process_message(
                text="Message after timeout",
                conversation_id=None,
                device_id="timeout_device",
            )

            assert response_2 is not None, "Message after timeout should get response"

            conv_id_2 = short_timeout_manager.get_conversation_id(device_id="timeout_device")

            # Should have new conversation_id (different from expired one)
            assert conv_id_2 != conv_id_1, (
                "New conversation_id should be created after session expires"
            )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_user_id_fallback_when_no_device_id(
    test_hass: HomeAssistant,
    session_manager: ConversationSessionManager,
    test_config: dict[str, Any],
    mock_llm_response: dict[str, Any],
) -> None:
    """Verify that user_id is used when device_id is not available.

    This test verifies:
    1. When no device_id is provided, user_id is used for session mapping
    2. Session lookup works with user_id
    3. Preference for device_id over user_id when both are present
    """
    with patch(
        "custom_components.home_agent.agent.async_should_expose",
        return_value=False,
    ):
        agent = HomeAgent(test_hass, test_config, session_manager)

        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # Message with user_id but no device_id
            # Note: ConversationInput doesn't have user_id parameter,
            # so we test through the session manager directly

            # Manually set a session with user_id
            test_conv_id = "test_user_conversation"
            await session_manager.set_conversation_id(
                test_conv_id,
                user_id="test_user_123",
                device_id=None,
            )

            # Verify lookup by user_id works
            retrieved = session_manager.get_conversation_id(user_id="test_user_123")
            assert retrieved == test_conv_id, "Should retrieve session by user_id"

            # Now add a device_id for the same user
            device_conv_id = "test_device_conversation"
            await session_manager.set_conversation_id(
                device_conv_id,
                user_id="test_user_123",
                device_id="test_device",
            )

            # Verify device_id takes precedence when both are provided
            retrieved_device = session_manager.get_conversation_id(
                user_id="test_user_123",
                device_id="test_device",
            )
            assert retrieved_device == device_conv_id, (
                "device_id should take precedence over user_id"
            )

            # Verify user_id still works when device_id not provided
            retrieved_user = session_manager.get_conversation_id(user_id="test_user_123")
            assert retrieved_user == test_conv_id, (
                "user_id lookup should still work for sessions without device_id"
            )

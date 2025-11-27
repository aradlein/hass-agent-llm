"""Test conversation session manager."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.conversation_session import (
    DEFAULT_SESSION_TIMEOUT,
    ConversationSessionManager,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {}
    return hass


@pytest.fixture
async def session_manager(mock_hass):
    """Create a conversation session manager."""
    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        # Mock the Store's async_load and async_save methods
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        manager = ConversationSessionManager(mock_hass)
        await manager.async_load()
        return manager


@pytest.mark.asyncio
async def test_get_conversation_id_not_found(session_manager):
    """Test getting conversation ID when not found."""
    result = session_manager.get_conversation_id(user_id="user_123")
    assert result is None


@pytest.mark.asyncio
async def test_set_and_get_conversation_id(session_manager):
    """Test setting and getting conversation ID."""
    await session_manager.set_conversation_id(
        "conv_456",
        user_id="user_123",
    )

    result = session_manager.get_conversation_id(user_id="user_123")
    assert result == "conv_456"


@pytest.mark.asyncio
async def test_device_id_priority(session_manager):
    """Test that device_id takes priority over user_id."""
    # Set with both user_id and device_id
    await session_manager.set_conversation_id(
        "conv_device",
        user_id="user_123",
        device_id="device_456",
    )

    # Should use device_id as key
    result = session_manager.get_conversation_id(
        user_id="user_123",
        device_id="device_456",
    )
    assert result == "conv_device"

    # Different device should not find it
    result = session_manager.get_conversation_id(
        user_id="user_123",
        device_id="device_789",
    )
    assert result is None


@pytest.mark.asyncio
async def test_session_expiration(mock_hass):
    """Test that sessions expire after timeout."""
    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        # Create session manager with 1 second timeout
        manager = ConversationSessionManager(mock_hass, session_timeout=1)
        await manager.async_load()

        await manager.set_conversation_id("conv_123", user_id="user_123")

        # Should be found immediately
        result = manager.get_conversation_id(user_id="user_123")
        assert result == "conv_123"

        # Wait for expiration
        time.sleep(1.1)

        # Should not be found after expiration
        result = manager.get_conversation_id(user_id="user_123")
        assert result is None


@pytest.mark.asyncio
async def test_clear_session(session_manager):
    """Test clearing a session."""
    await session_manager.set_conversation_id("conv_123", user_id="user_123")

    success = await session_manager.clear_session(user_id="user_123")
    assert success is True

    result = session_manager.get_conversation_id(user_id="user_123")
    assert result is None


@pytest.mark.asyncio
async def test_clear_nonexistent_session(session_manager):
    """Test clearing a session that doesn't exist."""
    success = await session_manager.clear_session(user_id="user_999")
    assert success is False


@pytest.mark.asyncio
async def test_clear_all_sessions(session_manager):
    """Test clearing all sessions."""
    await session_manager.set_conversation_id("conv_1", user_id="user_1")
    await session_manager.set_conversation_id("conv_2", user_id="user_2")
    await session_manager.set_conversation_id("conv_3", device_id="device_1")

    count = await session_manager.clear_all_sessions()
    assert count == 3

    # All should be gone
    assert session_manager.get_conversation_id(user_id="user_1") is None
    assert session_manager.get_conversation_id(user_id="user_2") is None
    assert session_manager.get_conversation_id(device_id="device_1") is None


@pytest.mark.asyncio
async def test_update_activity(session_manager):
    """Test updating session activity."""
    await session_manager.set_conversation_id("conv_123", user_id="user_123")

    # Get initial timestamp
    session = session_manager._sessions["user_123"]
    initial_time = session["last_activity"]

    # Wait a bit and update
    time.sleep(0.1)
    await session_manager.update_activity(user_id="user_123")

    # Should have newer timestamp
    updated_time = session_manager._sessions["user_123"]["last_activity"]
    assert updated_time > initial_time


@pytest.mark.asyncio
async def test_session_info(session_manager):
    """Test getting session information."""
    await session_manager.set_conversation_id("conv_1", user_id="user_1")
    await session_manager.set_conversation_id("conv_2", device_id="device_1")

    info = session_manager.get_session_info()

    assert info["total_sessions"] == 2
    assert info["timeout_seconds"] == DEFAULT_SESSION_TIMEOUT
    assert len(info["sessions"]) == 2


@pytest.mark.asyncio
async def test_no_user_or_device_id(session_manager):
    """Test behavior when neither user_id nor device_id is provided."""
    result = session_manager.get_conversation_id()
    assert result is None

    # Should log warning but not crash
    await session_manager.set_conversation_id("conv_123")
    # Verify nothing was stored
    assert len(session_manager._sessions) == 0


@pytest.mark.asyncio
async def test_session_persistence_on_load(mock_hass):
    """Test that sessions are loaded from storage on startup."""
    existing_sessions = {
        "user_1": {
            "conversation_id": "conv_1",
            "last_activity": time.time(),
            "user_id": "user_1",
            "device_id": None,
        },
        "device_1": {
            "conversation_id": "conv_2",
            "last_activity": time.time(),
            "user_id": None,
            "device_id": "device_1",
        },
    }

    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value={"sessions": existing_sessions})
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        manager = ConversationSessionManager(mock_hass)
        await manager.async_load()

        # Sessions should be loaded
        assert manager.get_conversation_id(user_id="user_1") == "conv_1"
        assert manager.get_conversation_id(device_id="device_1") == "conv_2"


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_on_load(mock_hass):
    """Test that expired sessions are cleaned up on load."""
    # Create sessions with one expired and one active
    current_time = time.time()
    existing_sessions = {
        "user_1": {
            "conversation_id": "conv_1",
            "last_activity": current_time - 7200,  # 2 hours ago (expired)
            "user_id": "user_1",
            "device_id": None,
        },
        "user_2": {
            "conversation_id": "conv_2",
            "last_activity": current_time - 10,  # 10 seconds ago (active)
            "user_id": "user_2",
            "device_id": None,
        },
    }

    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value={"sessions": existing_sessions})
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        manager = ConversationSessionManager(mock_hass)
        await manager.async_load()

        # Expired session should be gone
        assert manager.get_conversation_id(user_id="user_1") is None
        # Active session should remain
        assert manager.get_conversation_id(user_id="user_2") == "conv_2"


@pytest.mark.asyncio
async def test_session_metadata_stored_correctly(session_manager):
    """Test that session metadata is stored correctly."""
    await session_manager.set_conversation_id(
        "conv_123",
        user_id="user_123",
        device_id="kitchen_satellite",
    )

    # Get session directly from internal storage
    session = session_manager._sessions["kitchen_satellite"]

    assert session["conversation_id"] == "conv_123"
    assert session["user_id"] == "user_123"
    assert session["device_id"] == "kitchen_satellite"
    assert isinstance(session["last_activity"], float)
    assert session["last_activity"] <= time.time()


@pytest.mark.asyncio
async def test_multiple_users_independent_sessions(session_manager):
    """Test that multiple users have independent sessions."""
    await session_manager.set_conversation_id("conv_1", user_id="user_1")
    await session_manager.set_conversation_id("conv_2", user_id="user_2")
    await session_manager.set_conversation_id("conv_3", device_id="device_1")

    # Each should have their own conversation ID
    assert session_manager.get_conversation_id(user_id="user_1") == "conv_1"
    assert session_manager.get_conversation_id(user_id="user_2") == "conv_2"
    assert session_manager.get_conversation_id(device_id="device_1") == "conv_3"

    # Clearing one should not affect others
    await session_manager.clear_session(user_id="user_1")

    assert session_manager.get_conversation_id(user_id="user_1") is None
    assert session_manager.get_conversation_id(user_id="user_2") == "conv_2"
    assert session_manager.get_conversation_id(device_id="device_1") == "conv_3"


@pytest.mark.asyncio
async def test_session_info_age_calculation(session_manager):
    """Test that session age is calculated correctly in info."""
    await session_manager.set_conversation_id("conv_123", user_id="user_123")

    # Wait a bit so age is measurable
    time.sleep(0.2)

    info = session_manager.get_session_info()
    session_info = info["sessions"][0]

    # Age should be > 0 and < 1 second
    assert session_info["age_seconds"] >= 0
    assert session_info["age_seconds"] < 1


@pytest.mark.asyncio
async def test_update_activity_nonexistent_session(session_manager):
    """Test that updating activity of nonexistent session doesn't crash."""
    # Should not raise an exception
    await session_manager.update_activity(user_id="nonexistent_user")

    # Verify nothing was created
    assert len(session_manager._sessions) == 0


@pytest.mark.asyncio
async def test_session_persistence_disabled_returns_none(mock_hass):
    """Test that when session_timeout=0, get_conversation_id always returns None even if there's a session stored."""
    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        # Create session manager with persistence disabled (timeout=0)
        manager = ConversationSessionManager(mock_hass, session_timeout=0)
        await manager.async_load()

        # Set a conversation ID
        await manager.set_conversation_id("conv_123", user_id="user_123")

        # Should return None because persistence is disabled
        result = manager.get_conversation_id(user_id="user_123")
        assert result is None


@pytest.mark.asyncio
async def test_session_persistence_disabled_negative_timeout(mock_hass):
    """Test that negative timeout values also disable persistence (returns None)."""
    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        # Create session manager with negative timeout
        manager = ConversationSessionManager(mock_hass, session_timeout=-10)
        await manager.async_load()

        # Set a conversation ID
        await manager.set_conversation_id("conv_456", user_id="user_456")

        # Should return None because negative timeout disables persistence
        result = manager.get_conversation_id(user_id="user_456")
        assert result is None


@pytest.mark.asyncio
async def test_session_persistence_enabled_with_custom_timeout(mock_hass):
    """Test that a custom timeout (e.g., 300 seconds / 5 minutes) works correctly."""
    with patch("custom_components.home_agent.conversation_session.Store") as mock_store:
        store_instance = MagicMock()
        store_instance.async_load = AsyncMock(return_value=None)
        store_instance.async_save = AsyncMock()
        mock_store.return_value = store_instance

        # Create session manager with custom 300 second (5 minute) timeout
        manager = ConversationSessionManager(mock_hass, session_timeout=300)
        await manager.async_load()

        # Set a conversation ID
        await manager.set_conversation_id("conv_789", user_id="user_789")

        # Should be found immediately
        result = manager.get_conversation_id(user_id="user_789")
        assert result == "conv_789"

        # Verify the session info shows the custom timeout
        info = manager.get_session_info()
        assert info["timeout_seconds"] == 300
        assert info["total_sessions"] == 1

        # Session should still be valid after a short delay (well within 300 seconds)
        time.sleep(0.5)
        result = manager.get_conversation_id(user_id="user_789")
        assert result == "conv_789"

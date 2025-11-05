"""Conversation session manager for persistent voice conversations."""

from __future__ import annotations

import logging
import time
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.conversation_sessions"
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour in seconds


class ConversationSessionManager:
    """Manage persistent conversation sessions for users/devices.

    This class maps user_id or device_id to persistent conversation_id values,
    enabling multi-turn voice conversations by reusing the same conversation ID
    across multiple interactions from the same user or device.

    The manager automatically handles:
    - Session expiration based on configurable timeout
    - Persistent storage across Home Assistant restarts
    - Automatic cleanup of expired sessions
    - Priority handling (device_id over user_id for better multi-device support)
    """

    def __init__(
        self,
        hass: HomeAssistant,
        session_timeout: int = DEFAULT_SESSION_TIMEOUT,
    ) -> None:
        """Initialize conversation session manager.

        Args:
            hass: Home Assistant instance
            session_timeout: Time in seconds before sessions expire (default: 1 hour)
        """
        self._hass = hass
        self._session_timeout = session_timeout
        self._sessions: dict[str, dict[str, Any]] = {}
        self._store = Store(hass, STORAGE_VERSION, STORAGE_KEY)

    async def async_load(self) -> None:
        """Load sessions from persistent storage."""
        try:
            data = await self._store.async_load()
            if data and isinstance(data, dict):
                self._sessions = data.get("sessions", {})
                _LOGGER.info("Loaded %d conversation session(s)", len(self._sessions))

                # Clean up expired sessions on load
                self._cleanup_expired_sessions()
        except Exception as err:
            _LOGGER.error("Failed to load conversation sessions: %s", err)
            self._sessions = {}

    async def async_save(self) -> None:
        """Save sessions to persistent storage."""
        try:
            await self._store.async_save({"sessions": self._sessions})
            _LOGGER.debug("Saved %d conversation session(s)", len(self._sessions))
        except Exception as err:
            _LOGGER.error("Failed to save conversation sessions: %s", err)

    def get_conversation_id(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> str | None:
        """Get conversation ID for a user or device.

        Prefers device_id over user_id for better multi-device support. This allows
        different devices for the same user to maintain independent conversation contexts.

        Args:
            user_id: User ID from conversation context
            device_id: Device ID from conversation input

        Returns:
            Conversation ID if found and not expired, None otherwise

        Example:
            >>> manager.get_conversation_id(user_id="user_123", device_id="kitchen")
            "01HWXYZ123..."  # Returns existing conversation ID
        """
        # Prefer device_id for better multi-device support
        key = device_id if device_id else user_id

        if not key:
            _LOGGER.warning("No user_id or device_id provided")
            return None

        session = self._sessions.get(key)
        if not session:
            return None

        # Check if session has expired
        last_activity = session.get("last_activity", 0)
        if time.time() - last_activity > self._session_timeout:
            _LOGGER.debug("Session expired for %s", key)
            del self._sessions[key]
            return None

        return session.get("conversation_id")

    async def set_conversation_id(
        self,
        conversation_id: str,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> None:
        """Set conversation ID for a user or device.

        Creates or updates a session mapping, storing the conversation ID
        along with metadata for tracking and cleanup.

        Args:
            conversation_id: Conversation ID to store (typically a ULID)
            user_id: User ID from conversation context
            device_id: Device ID from conversation input

        Example:
            >>> await manager.set_conversation_id(
            ...     "01HWXYZ123...",
            ...     user_id="user_123",
            ...     device_id="kitchen_satellite"
            ... )
        """
        key = device_id if device_id else user_id

        if not key:
            _LOGGER.warning("No user_id or device_id provided")
            return

        self._sessions[key] = {
            "conversation_id": conversation_id,
            "last_activity": time.time(),
            "user_id": user_id,
            "device_id": device_id,
        }

        _LOGGER.debug("Set conversation %s for %s", conversation_id, key)
        await self.async_save()

    async def update_activity(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> None:
        """Update last activity time for a session.

        This prevents active sessions from expiring by updating the
        last_activity timestamp on each interaction.

        Args:
            user_id: User ID from conversation context
            device_id: Device ID from conversation input
        """
        key = device_id if device_id else user_id

        if not key or key not in self._sessions:
            return

        self._sessions[key]["last_activity"] = time.time()
        await self.async_save()

    async def clear_session(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> bool:
        """Clear conversation session for a user or device.

        This allows users to manually reset their conversation context
        and start fresh.

        Args:
            user_id: User ID from conversation context
            device_id: Device ID from conversation input

        Returns:
            True if session was cleared, False if not found
        """
        key = device_id if device_id else user_id

        if not key or key not in self._sessions:
            return False

        conversation_id = self._sessions[key]["conversation_id"]
        del self._sessions[key]

        _LOGGER.info("Cleared session %s for %s", conversation_id, key)
        await self.async_save()
        return True

    async def clear_all_sessions(self) -> int:
        """Clear all conversation sessions.

        Returns:
            Number of sessions cleared
        """
        count = len(self._sessions)
        self._sessions = {}
        await self.async_save()
        _LOGGER.info("Cleared all %d conversation session(s)", count)
        return count

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions.

        This is called automatically on load and when getting session info
        to prevent unbounded growth of the sessions dictionary.
        """
        current_time = time.time()
        expired_keys = [
            key
            for key, session in self._sessions.items()
            if current_time - session.get("last_activity", 0) > self._session_timeout
        ]

        for key in expired_keys:
            del self._sessions[key]

        if expired_keys:
            _LOGGER.info("Cleaned up %d expired session(s)", len(expired_keys))

    def get_session_info(self) -> dict[str, Any]:
        """Get information about active sessions.

        Performs cleanup before returning statistics, ensuring accuracy.

        Returns:
            Dictionary with session statistics including:
            - total_sessions: Number of active sessions
            - timeout_seconds: Configured timeout value
            - sessions: List of session details with age information
        """
        self._cleanup_expired_sessions()

        return {
            "total_sessions": len(self._sessions),
            "timeout_seconds": self._session_timeout,
            "sessions": [
                {
                    "key": key,
                    "conversation_id": session["conversation_id"],
                    "user_id": session.get("user_id"),
                    "device_id": session.get("device_id"),
                    "age_seconds": int(time.time() - session["last_activity"]),
                }
                for key, session in self._sessions.items()
            ],
        }

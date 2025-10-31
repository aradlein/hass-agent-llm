# Execution Plan: Persistent Voice Conversations

## Problem Statement

Currently, each voice interaction with Home Assistant creates a new `conversation_id`, resulting in lost context between interactions. Users cannot have multi-turn conversations via voice assistants.

**Root Cause:** The agent doesn't maintain a mapping between users/devices and their conversation IDs, so each interaction starts fresh.

## Solution Overview

Implement a conversation ID mapping system that:
1. Maps `user_id` or `device_id` to persistent `conversation_id`
2. Reuses the same `conversation_id` for the same user/device
3. Allows natural timeouts and cleanup
4. Provides user control to reset conversations

## Implementation Plan

### Phase 1: Add Conversation Mapping Storage

**Objective:** Create a system to store and retrieve user/device → conversation_id mappings

#### Step 1.1: Create ConversationSessionManager Class

**File:** `/workspaces/home-agent/custom_components/home_agent/conversation_session.py` (NEW FILE)

**Implementation:**
```python
"""Conversation session manager for persistent voice conversations."""
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
    """Manage persistent conversation sessions for users/devices."""

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
        """Load sessions from storage."""
        try:
            data = await self._store.async_load()
            if data and isinstance(data, dict):
                self._sessions = data.get("sessions", {})
                _LOGGER.info("Loaded %d conversation sessions", len(self._sessions))

                # Clean up expired sessions on load
                self._cleanup_expired_sessions()
        except Exception as err:
            _LOGGER.error("Failed to load conversation sessions: %s", err)
            self._sessions = {}

    async def async_save(self) -> None:
        """Save sessions to storage."""
        try:
            await self._store.async_save({"sessions": self._sessions})
            _LOGGER.debug("Saved %d conversation sessions", len(self._sessions))
        except Exception as err:
            _LOGGER.error("Failed to save conversation sessions: %s", err)

    def get_conversation_id(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> str | None:
        """Get conversation ID for a user or device.

        Args:
            user_id: User ID from conversation context
            device_id: Device ID from conversation input

        Returns:
            Conversation ID if found and not expired, None otherwise
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

        Args:
            conversation_id: Conversation ID to store
            user_id: User ID from conversation context
            device_id: Device ID from conversation input
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
        _LOGGER.info("Cleared all %d conversation sessions", count)
        return count

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        current_time = time.time()
        expired_keys = [
            key
            for key, session in self._sessions.items()
            if current_time - session.get("last_activity", 0) > self._session_timeout
        ]

        for key in expired_keys:
            del self._sessions[key]

        if expired_keys:
            _LOGGER.info("Cleaned up %d expired sessions", len(expired_keys))

    def get_session_info(self) -> dict[str, Any]:
        """Get information about active sessions.

        Returns:
            Dictionary with session statistics
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
```

**Key Design Decisions:**
- Use `device_id` as primary key (better for multi-device scenarios)
- Fall back to `user_id` if device_id not available
- 1-hour default timeout (configurable)
- Persistent storage survives restarts
- Auto-cleanup on load and info queries

#### Step 1.2: Add Configuration Constants

**File:** `/workspaces/home-agent/custom_components/home_agent/const.py`

**Add these constants:**
```python
# Conversation session configuration
CONF_SESSION_TIMEOUT = "session_timeout"
DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour
```

#### Step 1.3: Update manifest.json

**File:** `/workspaces/home-agent/custom_components/home_agent/manifest.json`

**Ensure storage is available (should already be present):**
```json
{
  "domain": "home_agent",
  "name": "Home Agent",
  "codeowners": ["@yourusername"],
  "config_flow": true,
  "dependencies": ["conversation"],
  "documentation": "https://github.com/yourusername/home-agent",
  "iot_class": "calculated",
  "requirements": [],
  "version": "0.0.1"
}
```

---

### Phase 2: Integrate Session Manager into Agent

**Objective:** Modify the agent to use persistent conversation IDs

#### Step 2.1: Initialize Session Manager in __init__.py

**File:** `/workspaces/home-agent/custom_components/home_agent/__init__.py`

**Modifications:**

1. Import the new manager:
```python
from .conversation_session import ConversationSessionManager
```

2. Initialize in `async_setup_entry`:
```python
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Home Agent from a config entry."""

    # ... existing code ...

    # Initialize conversation session manager
    session_timeout = entry.data.get(CONF_SESSION_TIMEOUT, DEFAULT_SESSION_TIMEOUT)
    session_manager = ConversationSessionManager(hass, session_timeout)
    await session_manager.async_load()

    # Store in hass.data
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    hass.data[DOMAIN][entry.entry_id] = {
        "session_manager": session_manager,
        # ... other existing data ...
    }

    # ... rest of existing code ...
```

#### Step 2.2: Update Agent to Use Session Manager

**File:** `/workspaces/home-agent/custom_components/home_agent/agent.py`

**Modifications:**

1. Add session_manager to __init__:
```python
def __init__(
    self,
    hass: HomeAssistant,
    config: dict[str, Any],
    session_manager: ConversationSessionManager,
) -> None:
    """Initialize the agent."""
    self.hass = hass
    self.config = config
    self.session_manager = session_manager  # NEW
    # ... rest of existing code ...
```

2. Modify `async_process` to use persistent conversation IDs:

**Location:** Around line 540 in `async_process` method

**Replace this section:**
```python
async def async_process(
    self,
    user_input: str,
    conversation_id: str | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
) -> str:
```

**With:**
```python
async def async_process(
    self,
    user_input: str,
    conversation_id: str | None = None,
    user_id: str | None = None,
    device_id: str | None = None,
) -> str:
    """Process user input and return response.

    Args:
        user_input: User's message
        conversation_id: Optional conversation ID (if None, will look up or create)
        user_id: User ID from conversation context
        device_id: Device ID from conversation input

    Returns:
        Agent's response text
    """
    metrics: dict[str, Any] = {
        "tool_calls": 0,
        "performance": {},
    }

    # NEW: Get or create persistent conversation ID
    if conversation_id is None:
        # Try to get existing conversation for this user/device
        conversation_id = self.session_manager.get_conversation_id(
            user_id=user_id,
            device_id=device_id,
        )

        if conversation_id:
            _LOGGER.debug(
                "Reusing conversation %s for user=%s device=%s",
                conversation_id,
                user_id,
                device_id,
            )
        else:
            # Generate new conversation ID using Home Assistant's ULID format
            from homeassistant.util.ulid import ulid_now
            conversation_id = ulid_now()

            _LOGGER.info(
                "Created new conversation %s for user=%s device=%s",
                conversation_id,
                user_id,
                device_id,
            )

            # Store the mapping
            await self.session_manager.set_conversation_id(
                conversation_id,
                user_id=user_id,
                device_id=device_id,
            )
    else:
        # Update activity for explicitly provided conversation_id
        await self.session_manager.update_activity(
            user_id=user_id,
            device_id=device_id,
        )

    # ... rest of existing code continues unchanged ...
```

3. Update activity on successful completion:

**Location:** Around line 650, after saving conversation history

**Add:**
```python
                # Save to conversation history
                if self.config.get(CONF_HISTORY_ENABLED, True):
                    self.conversation_manager.add_message(
                        conversation_id, "user", user_message
                    )
                    self.conversation_manager.add_message(
                        conversation_id, "assistant", final_content
                    )

                    # NEW: Update session activity
                    await self.session_manager.update_activity(
                        user_id=user_id,
                        device_id=device_id,
                    )

                return final_content
```

#### Step 2.3: Update ConversationEntity Integration

**File:** `/workspaces/home-agent/custom_components/home_agent/conversation.py` (the entity file, not the history manager)

**Locate the `async_process` method in the ConversationEntity class:**

**Modification:** Pass user_id and device_id to agent:

```python
async def async_process(
    self, user_input: ConversationInput
) -> ConversationResult:
    """Process a sentence."""

    # Extract user and device IDs from input
    user_id = user_input.context.user_id if user_input.context else None
    device_id = user_input.device_id

    response = await self._agent.async_process(
        user_input.text,
        conversation_id=user_input.conversation_id,
        user_id=user_id,  # NEW
        device_id=device_id,  # NEW
    )

    # ... rest of existing code ...
```

---

### Phase 3: Add User-Facing Controls

**Objective:** Allow users to reset their conversation context

#### Step 3.1: Add Service to Clear Conversation

**File:** `/workspaces/home-agent/custom_components/home_agent/__init__.py`

**Add service registration in `async_setup_entry`:**

```python
async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Home Agent from a config entry."""

    # ... existing setup code ...

    # Register services
    async def handle_clear_conversation(call: ServiceCall) -> None:
        """Handle clear conversation service call."""
        user_id = call.data.get("user_id")
        device_id = call.data.get("device_id")

        session_manager = hass.data[DOMAIN][entry.entry_id]["session_manager"]

        if user_id or device_id:
            # Clear specific session
            success = await session_manager.clear_session(
                user_id=user_id,
                device_id=device_id,
            )
            if success:
                _LOGGER.info("Cleared conversation for user_id=%s device_id=%s", user_id, device_id)
            else:
                _LOGGER.warning("No active conversation found for user_id=%s device_id=%s", user_id, device_id)
        else:
            # Clear all sessions
            count = await session_manager.clear_all_sessions()
            _LOGGER.info("Cleared all %d conversation sessions", count)

    hass.services.async_register(
        DOMAIN,
        "clear_conversation",
        handle_clear_conversation,
        schema=vol.Schema({
            vol.Optional("user_id"): cv.string,
            vol.Optional("device_id"): cv.string,
        }),
    )

    # ... rest of existing code ...
```

#### Step 3.2: Create Services YAML

**File:** `/workspaces/home-agent/custom_components/home_agent/services.yaml` (NEW FILE)

```yaml
clear_conversation:
  name: Clear conversation
  description: Clear conversation history and start a new conversation session
  fields:
    user_id:
      name: User ID
      description: Clear conversation for specific user (leave empty to clear all)
      required: false
      example: "a1b2c3d4e5f6"
      selector:
        text:
    device_id:
      name: Device ID
      description: Clear conversation for specific device (leave empty to clear all)
      required: false
      example: "satellite_kitchen"
      selector:
        text:
```

#### Step 3.3: Update Strings for User-Facing Text

**File:** `/workspaces/home-agent/custom_components/home_agent/strings.json`

**Add service translations:**
```json
{
  "config": {
    "step": {
      "user": {
        "title": "Configure Home Agent",
        "description": "Configure your Home Agent settings",
        "data": {
          "session_timeout": "Conversation session timeout (seconds)"
        }
      }
    }
  },
  "services": {
    "clear_conversation": {
      "name": "Clear conversation",
      "description": "Clear conversation history and start fresh. If no user_id or device_id provided, clears all conversations."
    }
  }
}
```

---

### Phase 4: Testing Strategy

#### Step 4.1: Unit Tests

**File:** `/workspaces/home-agent/tests/unit/test_conversation_session.py` (NEW FILE)

```python
"""Test conversation session manager."""
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.conversation_session import (
    ConversationSessionManager,
    DEFAULT_SESSION_TIMEOUT,
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
    with patch("custom_components.home_agent.conversation_session.Store"):
        manager = ConversationSessionManager(mock_hass)
        return manager


async def test_get_conversation_id_not_found(session_manager):
    """Test getting conversation ID when not found."""
    result = session_manager.get_conversation_id(user_id="user_123")
    assert result is None


async def test_set_and_get_conversation_id(session_manager):
    """Test setting and getting conversation ID."""
    await session_manager.set_conversation_id(
        "conv_456",
        user_id="user_123",
    )

    result = session_manager.get_conversation_id(user_id="user_123")
    assert result == "conv_456"


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


async def test_session_expiration(session_manager):
    """Test that sessions expire after timeout."""
    # Create session manager with 1 second timeout
    manager = ConversationSessionManager(session_manager._hass, session_timeout=1)

    await manager.set_conversation_id("conv_123", user_id="user_123")

    # Should be found immediately
    result = manager.get_conversation_id(user_id="user_123")
    assert result == "conv_123"

    # Wait for expiration
    time.sleep(1.1)

    # Should not be found after expiration
    result = manager.get_conversation_id(user_id="user_123")
    assert result is None


async def test_clear_session(session_manager):
    """Test clearing a session."""
    await session_manager.set_conversation_id("conv_123", user_id="user_123")

    success = await session_manager.clear_session(user_id="user_123")
    assert success is True

    result = session_manager.get_conversation_id(user_id="user_123")
    assert result is None


async def test_clear_nonexistent_session(session_manager):
    """Test clearing a session that doesn't exist."""
    success = await session_manager.clear_session(user_id="user_999")
    assert success is False


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


async def test_session_info(session_manager):
    """Test getting session information."""
    await session_manager.set_conversation_id("conv_1", user_id="user_1")
    await session_manager.set_conversation_id("conv_2", device_id="device_1")

    info = session_manager.get_session_info()

    assert info["total_sessions"] == 2
    assert info["timeout_seconds"] == DEFAULT_SESSION_TIMEOUT
    assert len(info["sessions"]) == 2
```

#### Step 4.2: Integration Tests

**File:** `/workspaces/home-agent/tests/integration/test_voice_persistence.py` (NEW FILE)

```python
"""Test voice conversation persistence."""
from unittest.mock import AsyncMock, patch

import pytest

from homeassistant.components.conversation import ConversationInput
from homeassistant.core import Context


async def test_voice_conversation_persistence(hass, config_entry, mock_agent):
    """Test that voice conversations persist across multiple interactions."""

    # First interaction
    input1 = ConversationInput(
        text="Turn on the living room lights",
        context=Context(user_id="user_123"),
        device_id="kitchen_satellite",
        conversation_id=None,  # No conversation ID provided (voice scenario)
    )

    with patch.object(mock_agent, "async_process", return_value="Lights turned on") as mock_process:
        result1 = await mock_agent.async_process(
            input1.text,
            conversation_id=input1.conversation_id,
            user_id=input1.context.user_id,
            device_id=input1.device_id,
        )

        # Get the conversation_id that was used
        call_kwargs = mock_process.call_args.kwargs
        first_conversation_id = call_kwargs.get("conversation_id")

    # Second interaction (simulating new voice command)
    input2 = ConversationInput(
        text="Now turn them off",
        context=Context(user_id="user_123"),
        device_id="kitchen_satellite",
        conversation_id=None,  # Still no conversation ID (voice scenario)
    )

    with patch.object(mock_agent, "async_process", return_value="Lights turned off") as mock_process:
        result2 = await mock_agent.async_process(
            input2.text,
            conversation_id=input2.conversation_id,
            user_id=input2.context.user_id,
            device_id=input2.device_id,
        )

        # Should reuse the same conversation_id
        call_kwargs = mock_process.call_args.kwargs
        second_conversation_id = call_kwargs.get("conversation_id")

    # Verify same conversation ID was reused
    assert first_conversation_id == second_conversation_id


async def test_different_devices_different_conversations(hass, config_entry, mock_agent):
    """Test that different devices have different conversations."""

    # User on kitchen device
    input1 = ConversationInput(
        text="Turn on the lights",
        context=Context(user_id="user_123"),
        device_id="kitchen_satellite",
        conversation_id=None,
    )

    # User on bedroom device
    input2 = ConversationInput(
        text="Turn on the lights",
        context=Context(user_id="user_123"),
        device_id="bedroom_satellite",
        conversation_id=None,
    )

    # Both should have different conversation IDs
    # (because they're on different devices)

    # Test implementation...
```

---

### Phase 5: Configuration UI Updates

#### Step 5.1: Add Session Timeout to Config Flow

**File:** `/workspaces/home-agent/custom_components/home_agent/config_flow.py`

**Add to user step schema:**
```python
from .const import CONF_SESSION_TIMEOUT, DEFAULT_SESSION_TIMEOUT

DATA_SCHEMA = vol.Schema({
    # ... existing fields ...
    vol.Optional(
        CONF_SESSION_TIMEOUT,
        default=DEFAULT_SESSION_TIMEOUT,
    ): vol.All(vol.Coerce(int), vol.Range(min=60, max=86400)),
})
```

**Add validation description:**
- Minimum: 60 seconds (1 minute)
- Maximum: 86400 seconds (24 hours)
- Default: 3600 seconds (1 hour)

---

### Phase 6: Documentation

#### Step 6.1: Update README

**File:** `/workspaces/home-agent/README.md`

**Add section:**
```markdown
## Voice Conversation Persistence

Home Agent automatically maintains conversation context across multiple voice interactions. This means you can have natural, multi-turn conversations with your voice assistant.

### How It Works

- Each user/device combination has a persistent conversation session
- Sessions automatically expire after 1 hour of inactivity (configurable)
- Conversation history is maintained within each session
- You can manually reset conversations using the `home_agent.clear_conversation` service

### Configuration

You can configure the session timeout in the integration settings:
- Minimum: 1 minute
- Maximum: 24 hours
- Default: 1 hour

### Clearing Conversations

To start a fresh conversation:

```yaml
# Clear conversation for current user
service: home_agent.clear_conversation

# Clear conversation for specific device
service: home_agent.clear_conversation
data:
  device_id: "kitchen_satellite"

# Clear all conversations
service: home_agent.clear_conversation
```

### Multi-Device Behavior

Each device maintains its own conversation context. This means:
- Kitchen satellite remembers kitchen-related conversations
- Bedroom satellite has its own independent context
- Same user on different devices = different conversations
```

#### Step 6.2: Add Code Comments

Ensure all new code has comprehensive docstrings explaining:
- Purpose of each method
- Parameters and return values
- Examples where helpful
- Design decisions (e.g., why device_id takes priority)

---

## Migration Strategy

### Existing Users

No migration needed - feature is opt-in by design:
- If `conversation_id` is explicitly provided, use it (old behavior)
- If `conversation_id` is None, use persistent sessions (new behavior)
- Existing integrations continue to work unchanged

### Rollback Plan

If issues arise:
1. Feature can be disabled by always passing explicit `conversation_id` values
2. Storage file can be deleted to clear all sessions
3. Service to clear all sessions provides manual reset

---

## Testing Checklist

- [ ] Unit tests for ConversationSessionManager
- [ ] Unit tests for session expiration
- [ ] Unit tests for device_id vs user_id priority
- [ ] Integration test for voice persistence
- [ ] Integration test for multi-device scenarios
- [ ] Manual testing with real voice assistant
- [ ] Manual testing with clear_conversation service
- [ ] Manual testing with different timeout values
- [ ] Verify storage persistence across restarts
- [ ] Verify performance with many sessions

---

## Performance Considerations

### Memory Usage

- Each session: ~200 bytes
- 1000 sessions: ~200KB
- Auto-cleanup prevents unbounded growth

### Storage I/O

- Debounced saves (avoid excessive writes)
- Load once on startup
- Cleanup on load removes expired sessions

### Lookup Performance

- O(1) dictionary lookup by key
- No scanning or iteration for normal operations
- Cleanup is O(n) but infrequent

---

## Success Criteria

1. ✅ Voice users can have multi-turn conversations
2. ✅ Context preserved across interactions (same device)
3. ✅ Different devices have independent contexts
4. ✅ Sessions expire after timeout
5. ✅ Users can manually reset conversations
6. ✅ No breaking changes to existing behavior
7. ✅ Unit test coverage >95%
8. ✅ Integration tests pass
9. ✅ Documentation complete

---

## Timeline Estimate

- **Phase 1** (Storage): 2-3 hours
- **Phase 2** (Integration): 2-3 hours
- **Phase 3** (Services): 1-2 hours
- **Phase 4** (Testing): 3-4 hours
- **Phase 5** (Config UI): 1 hour
- **Phase 6** (Documentation): 1-2 hours

**Total: 10-15 hours**

---

## Notes for Implementation

1. **Import ulid_now:** Use `from homeassistant.util.ulid import ulid_now` for conversation ID generation
2. **Thread Safety:** All async operations use Home Assistant's event loop
3. **Error Handling:** All storage operations wrapped in try/except
4. **Logging:** Use appropriate log levels (debug for frequent, info for important events)
5. **Type Hints:** Maintain full type hint coverage
6. **Code Style:** Follow Home Assistant conventions (run ruff and pylint)

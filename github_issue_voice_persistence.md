# Feature: Persistent Voice Conversations

## Problem

Currently, each voice interaction with Home Assistant creates a new `conversation_id`, resulting in lost context between interactions. Users cannot have multi-turn conversations via voice assistants.

**Example of the problem:**
```
User: "Turn on the living room lights"
Agent: "I've turned on the living room lights"

[30 seconds later]

User: "Now turn them off"
Agent: "I'm sorry, I don't know what you're referring to"
```

The agent has no memory of the previous interaction because each voice command starts a fresh conversation with a new `conversation_id`.

## Root Cause

The integration doesn't maintain a mapping between users/devices and their conversation IDs. Home Assistant's voice assistant generates a new ULID for each interaction, and our agent treats each as a brand new conversation.

**Key code location:** `/custom_components/home_agent/agent.py` - The `async_process` method receives `conversation_id=None` from voice interactions.

## Proposed Solution

Implement a conversation session manager that:
1. Maps `user_id` or `device_id` to persistent `conversation_id` values
2. Reuses the same `conversation_id` for the same user/device
3. Allows natural timeouts (default: 1 hour) and cleanup
4. Provides user control to reset conversations via service call

## Implementation Details

See the complete execution plan: [voice_persistence_execution_plan.md](../voice_persistence_execution_plan.md)

### High-Level Architecture

```
Voice Input → ConversationInput (user_id, device_id, conversation_id=None)
                    ↓
         ConversationSessionManager
                    ↓
    Get/Create persistent conversation_id
                    ↓
         Agent.async_process(conversation_id, user_id, device_id)
                    ↓
       ConversationHistoryManager (existing)
                    ↓
              LLM with full context
```

### Key Components

1. **ConversationSessionManager** (NEW)
   - File: `/custom_components/home_agent/conversation_session.py`
   - Manages user/device → conversation_id mappings
   - Handles session expiration and cleanup
   - Provides persistent storage via Home Assistant's Store

2. **Agent Modifications** (EXISTING - MODIFY)
   - File: `/custom_components/home_agent/agent.py`
   - Update `async_process` to check for existing conversation IDs
   - Create new conversation IDs only when needed
   - Update session activity after each interaction

3. **Service Registration** (EXISTING - MODIFY)
   - File: `/custom_components/home_agent/__init__.py`
   - Add `home_agent.clear_conversation` service
   - Allow users to reset their conversation context

4. **Configuration** (EXISTING - MODIFY)
   - Add `session_timeout` configuration option
   - Default: 3600 seconds (1 hour)
   - Range: 60-86400 seconds

## Implementation Phases

### Phase 1: Core Session Management ⚙️
- [ ] Create `ConversationSessionManager` class
- [ ] Add storage integration
- [ ] Implement get/set/clear methods
- [ ] Add session expiration logic
- [ ] Add configuration constants

**Files to create:**
- `/custom_components/home_agent/conversation_session.py`

**Files to modify:**
- `/custom_components/home_agent/const.py`

**Estimated time:** 2-3 hours

---

### Phase 2: Agent Integration 🔌
- [ ] Initialize session manager in `__init__.py`
- [ ] Pass session manager to Agent
- [ ] Modify `async_process` to use persistent conversation IDs
- [ ] Update ConversationEntity to pass user_id/device_id
- [ ] Add activity tracking

**Files to modify:**
- `/custom_components/home_agent/__init__.py`
- `/custom_components/home_agent/agent.py`
- `/custom_components/home_agent/conversation.py` (entity file)

**Estimated time:** 2-3 hours

---

### Phase 3: User Controls 🎛️
- [ ] Register `clear_conversation` service
- [ ] Create `services.yaml`
- [ ] Update `strings.json` with service descriptions
- [ ] Add service call handler

**Files to modify:**
- `/custom_components/home_agent/__init__.py`

**Files to create:**
- `/custom_components/home_agent/services.yaml`

**Files to modify:**
- `/custom_components/home_agent/strings.json`

**Estimated time:** 1-2 hours

---

### Phase 4: Testing 🧪
- [ ] Unit tests for `ConversationSessionManager`
- [ ] Test session creation and retrieval
- [ ] Test session expiration
- [ ] Test device_id vs user_id priority
- [ ] Integration tests for voice persistence
- [ ] Integration tests for multi-device scenarios
- [ ] Manual testing with real voice assistant

**Files to create:**
- `/tests/unit/test_conversation_session.py`
- `/tests/integration/test_voice_persistence.py`

**Estimated time:** 3-4 hours

---

### Phase 5: Configuration UI 🖥️
- [ ] Add session_timeout to config flow
- [ ] Add validation (60-86400 seconds)
- [ ] Update config flow strings

**Files to modify:**
- `/custom_components/home_agent/config_flow.py`
- `/custom_components/home_agent/strings.json`

**Estimated time:** 1 hour

---

### Phase 6: Documentation 📚
- [ ] Update README with voice persistence section
- [ ] Document configuration options
- [ ] Document clear_conversation service
- [ ] Add code comments and docstrings
- [ ] Update CHANGELOG

**Files to modify:**
- `/README.md`
- `/CHANGELOG.md`

**Estimated time:** 1-2 hours

---

## Code Examples

### 1. Getting Persistent Conversation ID

```python
# In agent.py async_process method
if conversation_id is None:
    # Try to get existing conversation for this user/device
    conversation_id = self.session_manager.get_conversation_id(
        user_id=user_id,
        device_id=device_id,
    )

    if conversation_id:
        _LOGGER.debug("Reusing conversation %s", conversation_id)
    else:
        # Generate new conversation ID
        from homeassistant.util.ulid import ulid_now
        conversation_id = ulid_now()

        # Store the mapping
        await self.session_manager.set_conversation_id(
            conversation_id,
            user_id=user_id,
            device_id=device_id,
        )
```

### 2. Using the Clear Service

```yaml
# Clear conversation for current user (in automation)
service: home_agent.clear_conversation

# Clear conversation for specific device
service: home_agent.clear_conversation
data:
  device_id: "kitchen_satellite"

# Clear all conversations (admin)
service: home_agent.clear_conversation
```

### 3. Session Manager API

```python
# Get existing conversation ID
conversation_id = manager.get_conversation_id(
    user_id="user_123",
    device_id="kitchen_satellite"
)

# Set new conversation ID mapping
await manager.set_conversation_id(
    "conv_abc123",
    user_id="user_123",
    device_id="kitchen_satellite"
)

# Update last activity time
await manager.update_activity(
    user_id="user_123",
    device_id="kitchen_satellite"
)

# Clear specific session
await manager.clear_session(device_id="kitchen_satellite")

# Clear all sessions
await manager.clear_all_sessions()

# Get session info
info = manager.get_session_info()
```

## Testing Strategy

### Unit Tests
- Session creation, retrieval, and expiration
- Device ID priority over user ID
- Clear operations (single and all)
- Activity updates
- Storage persistence

### Integration Tests
- Full voice conversation flow
- Multi-device scenarios
- Service calls
- Config flow updates

### Manual Testing
- Real voice assistant interactions
- Multiple devices
- Session timeouts
- Service calls from UI

## Success Criteria

- [x] Voice users can have multi-turn conversations
- [x] Context is preserved across interactions on the same device
- [x] Different devices maintain independent contexts
- [x] Sessions expire after configured timeout
- [x] Users can manually reset conversations
- [x] No breaking changes to existing behavior
- [x] Unit test coverage >95%
- [x] Integration tests pass
- [x] Documentation complete

## User Experience

### Before (Current Behavior)
```
User: "What's the temperature in the living room?"
Agent: "The living room is 72°F"

[Later]
User: "What about the bedroom?"
Agent: "I don't have context about what you're asking about"
```

### After (With Persistent Conversations)
```
User: "What's the temperature in the living room?"
Agent: "The living room is 72°F"

[Later]
User: "What about the bedroom?"
Agent: "The bedroom temperature is 68°F"
```

## Configuration Example

```yaml
# configuration.yaml or via UI
home_agent:
  # ... other settings ...
  session_timeout: 3600  # 1 hour (default)
```

## Migration Notes

- **Backward Compatible:** Existing integrations work unchanged
- **Opt-in:** Only activates when `conversation_id=None` (voice scenario)
- **No Data Migration:** Feature starts fresh on installation
- **Rollback:** Can disable by clearing all sessions or uninstalling

## Performance Impact

- **Memory:** ~200 bytes per session (~200KB for 1000 sessions)
- **Storage I/O:** Debounced saves, one load on startup
- **Lookup:** O(1) dictionary lookup
- **Cleanup:** Automatic expiration prevents growth

## Related Issues

- Issue #XXX: Voice assistant context loss
- Issue #XXX: Multi-turn conversations not working

## Additional Context

This feature aligns with Home Assistant's vision of natural, conversational interactions. It makes the voice assistant feel more intelligent and responsive by remembering context within reasonable time windows.

The implementation follows Home Assistant's patterns:
- Uses `Store` for persistence
- Uses `ulid_now()` for conversation IDs
- Follows async/await patterns
- Includes comprehensive testing
- Provides user control via services

## Questions?

See the full execution plan for detailed implementation instructions: [voice_persistence_execution_plan.md](../voice_persistence_execution_plan.md)

---

**Labels:** `enhancement`, `voice`, `conversation`, `good first issue`
**Milestone:** v0.2.0
**Assignee:** TBD

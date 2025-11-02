# API Reference

Complete reference for Home Agent services, events, and tools.

## Table of Contents

- [Services Reference](#services-reference)
- [Events Reference](#events-reference)
- [Tools Reference](#tools-reference)

---

## Services Reference

### home_agent.process

Process a conversation message through the Home Agent.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `text` | string | Yes | The user's message text to process | "Turn on the living room lights" |
| `conversation_id` | string | No | Optional conversation ID for history tracking | "living_room_conversation" |
| `user_id` | string | No | Optional user ID for the conversation | "user123" |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- Response text from the agent

**Example YAML:**
```yaml
service: home_agent.process
data:
  text: "What's the temperature in the living room?"
  conversation_id: "main_conversation"
```

**Example Automation:**
```yaml
automation:
  - alias: "Voice Command Shortcut"
    trigger:
      - platform: event
        event_type: custom_voice_command
    action:
      - service: home_agent.process
        data:
          text: "{{ trigger.event.data.command }}"
          conversation_id: "automation_{{ trigger.event.data.user }}"
```

---

### home_agent.clear_history

Clear conversation history for a specific conversation or all conversations.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `conversation_id` | string | No | Specific conversation to clear (omit to clear all) | "living_room_conversation" |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- None

**Example YAML:**
```yaml
# Clear specific conversation
service: home_agent.clear_history
data:
  conversation_id: "living_room_conversation"

# Clear all conversations
service: home_agent.clear_history
```

**Example Automation:**
```yaml
automation:
  - alias: "Clear History Daily"
    trigger:
      - platform: time
        at: "00:00:00"
    action:
      - service: home_agent.clear_history
```

---

### home_agent.reload_context

Reload entity context (useful after entity changes).

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- None

**Example YAML:**
```yaml
service: home_agent.reload_context
```

**Use Cases:**
- After adding/removing entities
- After changing entity exposure settings
- After updating entity attributes
- Manual context refresh

---

### home_agent.execute_tool

Manually execute a tool for testing and debugging purposes.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `tool_name` | string | Yes | The name of the tool to execute | "ha_query" |
| `parameters` | object | Yes | Tool parameters as a JSON object | `{"entity_id": "light.living_room"}` |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- Tool execution result with success status

**Example YAML:**
```yaml
# Query entity state
service: home_agent.execute_tool
data:
  tool_name: ha_query
  parameters:
    entity_id: light.living_room

# Control entity
service: home_agent.execute_tool
data:
  tool_name: ha_control
  parameters:
    action: turn_on
    entity_id: light.living_room
    parameters:
      brightness: 128
```

---

### home_agent.reindex_entities

Force a full reindex of all Home Assistant entities into the vector database.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- None

**Example YAML:**
```yaml
service: home_agent.reindex_entities
```

**Use Cases:**
- After major entity changes
- Vector DB corruption recovery
- Initial setup of vector DB mode
- Performance optimization

**Notes:**
- Only required when using vector DB context mode
- Can take several minutes with many entities
- Check logs for indexing progress

---

### home_agent.index_entity

Index a specific entity into the vector database.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `entity_id` | string | Yes | The entity ID to index | "sensor.temperature" |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- None

**Example YAML:**
```yaml
service: home_agent.index_entity
data:
  entity_id: sensor.temperature
```

**Use Cases:**
- Add newly created entity to vector DB
- Update entity after significant attribute changes
- Targeted indexing for specific entities

---

### home_agent.list_memories

List all stored memories with optional filtering by type and limit.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `memory_type` | select | No | Filter by memory type | "preference" |
| `limit` | number (1-1000) | No | Maximum number of memories to return | 50 |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Valid memory types:**
- `fact`
- `preference`
- `context`
- `event`

**Returns:**
- List of memory objects

**Example YAML:**
```yaml
# List all preferences
service: home_agent.list_memories
data:
  memory_type: preference
  limit: 50

# List all memories
service: home_agent.list_memories
```

---

### home_agent.search_memories

Search memories using semantic similarity to find relevant information.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `query` | string | Yes | What to search for in stored memories | "temperature preferences" |
| `limit` | number (1-100) | No | Maximum number of results to return | 10 |
| `min_importance` | number (0.0-1.0) | No | Filter results by minimum importance score | 0.5 |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- List of matching memories with relevance scores

**Example YAML:**
```yaml
service: home_agent.search_memories
data:
  query: "bedroom temperature preferences"
  limit: 10
  min_importance: 0.5
```

**Example Automation:**
```yaml
automation:
  - alias: "Search User Preferences"
    trigger:
      - platform: state
        entity_id: input_button.search_preferences
    action:
      - service: home_agent.search_memories
        data:
          query: "{{ states('input_text.search_query') }}"
          limit: 20
        response_variable: search_results
      - service: notify.persistent_notification
        data:
          message: "Found {{ search_results.memories | length }} memories"
```

---

### home_agent.add_memory

Manually add a memory to the system.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `content` | text | Yes | The memory content to store | "User prefers bedroom temperature at 68°F for sleeping" |
| `type` | select | No | Memory type | "preference" |
| `importance` | number (0.0-1.0) | No | Importance score from 0.0 (trivial) to 1.0 (critical) | 0.7 |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Valid types:**
- `fact` - Concrete information
- `preference` - User preferences
- `context` - Background information
- `event` - Significant events

**Returns:**
- Memory ID of created memory

**Example YAML:**
```yaml
service: home_agent.add_memory
data:
  content: "User prefers bedroom temperature at 68°F for sleeping"
  type: preference
  importance: 0.8
```

---

### home_agent.delete_memory

Delete a specific memory by its ID.

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `memory_id` | string | Yes | The ID of the memory to delete | "mem_abc123xyz" |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- None

**Example YAML:**
```yaml
service: home_agent.delete_memory
data:
  memory_id: "mem_abc123xyz"
```

---

### home_agent.clear_memories

Delete all stored memories (requires confirmation - this cannot be undone!).

**Parameters:**

| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `confirm` | boolean | Yes | Must be set to true to confirm deletion of all memories | true |
| `entry_id` | string | No (Advanced) | Optional config entry ID to use (defaults to first entry) | - |

**Returns:**
- None

**Example YAML:**
```yaml
service: home_agent.clear_memories
data:
  confirm: true
```

**Warning:** This permanently deletes all memories. There is no undo.

---

## Events Reference

Home Agent fires various events that can be used in automations to monitor and respond to agent activity.

### home_agent.conversation.started

Triggered when a new conversation begins.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `conversation_id` | string | Unique conversation identifier | "living_room_conv_123" |
| `user_id` | string | Home Assistant user ID | "abc123" |
| `device_id` | string | Device that triggered conversation | "living_room_speaker" |
| `timestamp` | float | Unix timestamp | 1699564800.123 |
| `context_mode` | string | Context injection mode | "direct" or "vector_db" |

**Example Event Data:**
```json
{
  "conversation_id": "main_conversation",
  "user_id": "user123",
  "device_id": "living_room_assistant",
  "timestamp": 1699564800.123,
  "context_mode": "vector_db"
}
```

**Example Automation:**
```yaml
automation:
  - alias: "Log Conversation Start"
    trigger:
      - platform: event
        event_type: home_agent.conversation.started
    action:
      - service: logbook.log
        data:
          name: "Home Agent"
          message: "Conversation started: {{ trigger.event.data.conversation_id }}"
```

---

### home_agent.conversation.finished

Triggered when conversation completes.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `conversation_id` | string | Conversation identifier | "main_conversation" |
| `user_id` | string | Home Assistant user ID | "abc123" |
| `tool_calls` | integer | Number of tools executed | 3 |
| `tool_breakdown` | object | Dict of tool names and call counts | `{"ha_query": 2, "ha_control": 1}` |
| `tokens` | object | Token usage statistics | See below |
| `duration_ms` | integer | Total processing time in milliseconds | 2345 |
| `performance` | object | Performance metrics | See below |
| `context` | object | Context-related metrics | See below |

**Tokens Object:**
```json
{
  "prompt": 150,
  "completion": 75,
  "total": 225
}
```

**Performance Object:**
```json
{
  "llm_latency_ms": 1200,
  "tool_latency_ms": 450,
  "context_latency_ms": 150
}
```

**Complete Example Event Data:**
```json
{
  "conversation_id": "main_conversation",
  "user_id": "user123",
  "tool_calls": 3,
  "tool_breakdown": {
    "ha_query": 2,
    "ha_control": 1
  },
  "tokens": {
    "prompt": 150,
    "completion": 75,
    "total": 225
  },
  "duration_ms": 2345,
  "performance": {
    "llm_latency_ms": 1200,
    "tool_latency_ms": 450,
    "context_latency_ms": 150
  }
}
```

**Example Automation:**
```yaml
automation:
  - alias: "Track Slow Conversations"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    condition:
      - condition: template
        value_template: "{{ trigger.event.data.duration_ms > 5000 }}"
    action:
      - service: notify.admin
        data:
          message: "Slow conversation detected: {{ trigger.event.data.duration_ms }}ms"
```

---

### home_agent.tool.executed

Triggered after each tool execution.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tool_name` | string | Tool that was executed | "ha_control" |
| `parameters` | object | Tool parameters | `{"action": "turn_on", "entity_id": "light.living_room"}` |
| `result` | any | Execution result (truncated if large) | `{"success": true, "state": "on"}` |
| `success` | boolean | Whether execution succeeded | true |
| `duration_ms` | float | Execution time in milliseconds | 45.2 |
| `conversation_id` | string | Associated conversation | "main_conversation" |

**Example Event Data:**
```json
{
  "tool_name": "ha_control",
  "parameters": {
    "action": "turn_on",
    "entity_id": "light.living_room",
    "parameters": {
      "brightness": 128
    }
  },
  "result": {
    "success": true,
    "entity_id": "light.living_room",
    "new_state": "on"
  },
  "success": true,
  "duration_ms": 45.2,
  "conversation_id": "main_conversation"
}
```

**Example Automation:**
```yaml
automation:
  - alias: "Log Tool Executions"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
    action:
      - service: system_log.write
        data:
          message: "Tool {{ trigger.event.data.tool_name }} executed in {{ trigger.event.data.duration_ms }}ms"
```

---

### home_agent.tool.progress

Triggered during tool execution to indicate progress status.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `tool_name` | string | Tool being executed | "ha_control" |
| `tool_call_id` | string | Tool call identifier | "call_abc123" |
| `status` | string | Current status | "started", "completed", or "failed" |
| `timestamp` | float | Unix timestamp | 1699564800.123 |
| `success` | boolean | Whether execution succeeded (completed/failed only) | true |
| `error` | string | Error message (failed only) | "Entity not found" |
| `error_type` | string | Error type (failed only) | "ToolExecutionError" |

**Example Event Data (Started):**
```json
{
  "tool_name": "ha_query",
  "tool_call_id": "call_abc123",
  "status": "started",
  "timestamp": 1699564800.123
}
```

**Example Event Data (Completed):**
```json
{
  "tool_name": "ha_query",
  "tool_call_id": "call_abc123",
  "status": "completed",
  "timestamp": 1699564801.456,
  "success": true
}
```

**Example Event Data (Failed):**
```json
{
  "tool_name": "ha_control",
  "tool_call_id": "call_xyz789",
  "status": "failed",
  "error": "Entity not found: light.nonexistent",
  "error_type": "ToolExecutionError",
  "timestamp": 1699564802.789,
  "success": false
}
```

**Use Cases:**
- Real-time tool execution monitoring
- Progress indicators for long-running tools
- Error handling and retry logic

---

### home_agent.context.injected

Triggered when context is added to LLM call.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `conversation_id` | string | Conversation identifier | "main_conversation" |
| `mode` | string | Context injection mode | "direct" or "vector_db" |
| `entities_included` | list | List of entity IDs | ["light.living_room", "sensor.temperature"] |
| `token_count` | integer | Approximate tokens used | 250 |
| `vector_db_query` | string | Query used (if vector DB mode) | "living room temperature" |

**Example Event Data (Direct Mode):**
```json
{
  "conversation_id": "main_conversation",
  "mode": "direct",
  "entities_included": [
    "light.living_room",
    "sensor.living_room_temperature",
    "climate.thermostat"
  ],
  "token_count": 250
}
```

**Example Event Data (Vector DB Mode):**
```json
{
  "conversation_id": "main_conversation",
  "mode": "vector_db",
  "entities_included": [
    "sensor.living_room_temperature",
    "climate.thermostat"
  ],
  "token_count": 180,
  "vector_db_query": "living room temperature"
}
```

---

### home_agent.context.optimized

Triggered when context is optimized/compressed.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `conversation_id` | string | Conversation identifier | "main_conversation" |
| `original_tokens` | integer | Token count before optimization | 5000 |
| `optimized_tokens` | integer | Token count after optimization | 3000 |
| `compression_level` | string | Level of compression applied | "medium" |
| `optimization_type` | string | Type of optimization performed | "summarization" |

**Example Event Data:**
```json
{
  "conversation_id": "main_conversation",
  "original_tokens": 5000,
  "optimized_tokens": 3000,
  "compression_level": "medium",
  "optimization_type": "summarization"
}
```

---

### home_agent.history.saved

Triggered when conversation history is saved.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `conversation_id` | string | Conversation identifier | "main_conversation" |
| `message_count` | integer | Number of messages saved | 12 |
| `storage_type` | string | Where history is stored | "persistent" or "memory" |

**Example Event Data:**
```json
{
  "conversation_id": "main_conversation",
  "message_count": 12,
  "storage_type": "persistent"
}
```

---

### home_agent.vector_db.queried

Triggered when vector database is queried.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `query` | string | Search query used | "bedroom temperature" |
| `results_count` | integer | Number of results returned | 5 |
| `similarity_scores` | list | Similarity scores for results | [0.95, 0.87, 0.82, 0.78, 0.71] |
| `duration_ms` | float | Query duration in milliseconds | 125.3 |
| `collection` | string | Collection name queried | "home_entities" |

**Example Event Data:**
```json
{
  "query": "bedroom temperature sensors",
  "results_count": 5,
  "similarity_scores": [0.95, 0.87, 0.82, 0.78, 0.71],
  "duration_ms": 125.3,
  "collection": "home_entities"
}
```

---

### home_agent.memory.extracted

Triggered when memories are extracted from a conversation.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `conversation_id` | string | Conversation identifier | "main_conversation" |
| `memories_extracted` | integer | Number of memories extracted | 3 |
| `extraction_llm` | string | LLM used for extraction | "external" or "local" |
| `timestamp` | string | ISO timestamp | "2024-01-15T10:30:00" |

**Example Event Data:**
```json
{
  "conversation_id": "main_conversation",
  "memories_extracted": 3,
  "extraction_llm": "external",
  "timestamp": "2024-01-15T10:30:00"
}
```

**Example Automation:**
```yaml
automation:
  - alias: "Notify Memory Extraction"
    trigger:
      - platform: event
        event_type: home_agent.memory.extracted
    condition:
      - condition: template
        value_template: "{{ trigger.event.data.memories_extracted > 0 }}"
    action:
      - service: notify.admin
        data:
          message: "Extracted {{ trigger.event.data.memories_extracted }} new memories"
```

---

### home_agent.error

Triggered on errors.

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `error_type` | string | Exception class name | "ToolExecutionError" |
| `error_message` | string | Error description | "Tool 'ha_control' failed: Entity not found" |
| `conversation_id` | string | Associated conversation (if applicable) | "main_conversation" |
| `component` | string | Component where error occurred | "agent" |
| `context` | object | Additional debug information | `{"text_length": 150}` |

**Example Event Data:**
```json
{
  "error_type": "ToolExecutionError",
  "error_message": "Tool 'ha_control' failed: Entity light.nonexistent not found",
  "conversation_id": "main_conversation",
  "component": "tool_handler",
  "context": {
    "tool_name": "ha_control",
    "entity_id": "light.nonexistent"
  }
}
```

**Example Automation:**
```yaml
automation:
  - alias: "Alert on Home Agent Errors"
    trigger:
      - platform: event
        event_type: home_agent.error
    action:
      - service: notify.admin
        data:
          title: "Home Agent Error"
          message: >
            Type: {{ trigger.event.data.error_type }}
            Message: {{ trigger.event.data.error_message }}
```

---

### home_agent.streaming.error

Triggered on streaming failures (with automatic fallback to synchronous mode).

**Event Data:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `error` | string | Error description | "SSE parsing failed" |
| `error_type` | string | Error type | "StreamingError" |
| `fallback` | boolean | Whether fallback to synchronous occurred | true |

**Example Event Data:**
```json
{
  "error": "SSE connection lost during streaming",
  "error_type": "ConnectionError",
  "fallback": true
}
```

**Use Cases:**
- Monitor streaming reliability
- Alert on streaming issues
- Track fallback frequency

---

## Tools Reference

Home Agent provides several built-in tools that the LLM can call during conversations.

### ha_control

Control Home Assistant devices and services.

**Tool Definition:**
```json
{
  "name": "ha_control",
  "description": "Control Home Assistant devices and services. Use this to turn on/off lights, adjust thermostats, lock doors, etc.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform",
        "enum": ["turn_on", "turn_off", "toggle", "set_value"]
      },
      "entity_id": {
        "type": "string",
        "description": "The entity ID to control (e.g., 'light.living_room')"
      },
      "parameters": {
        "type": "object",
        "description": "Additional parameters for the action (e.g., brightness, temperature)"
      }
    },
    "required": ["action", "entity_id"]
  }
}
```

**Actions:**
- `turn_on` - Turn entity on
- `turn_off` - Turn entity off
- `toggle` - Toggle entity state
- `set_value` - Set specific value

**Common Parameters:**
- **Lights:**
  - `brightness`: 0-255
  - `rgb_color`: [R, G, B] (0-255 each)
  - `color_temp`: Mireds
- **Climate:**
  - `temperature`: Target temperature
  - `hvac_mode`: heat, cool, auto, off
- **Covers:**
  - `position`: 0-100 (0 = closed, 100 = open)
- **Media Players:**
  - `volume_level`: 0.0-1.0

**Example Tool Calls:**

```json
// Turn on light
{
  "action": "turn_on",
  "entity_id": "light.living_room"
}

// Turn on light with brightness
{
  "action": "turn_on",
  "entity_id": "light.living_room",
  "parameters": {
    "brightness": 128
  }
}

// Set thermostat temperature
{
  "action": "set_value",
  "entity_id": "climate.thermostat",
  "parameters": {
    "temperature": 72
  }
}

// Toggle switch
{
  "action": "toggle",
  "entity_id": "switch.fan"
}
```

**Response Format:**
```json
{
  "success": true,
  "entity_id": "light.living_room",
  "action": "turn_on",
  "new_state": "on",
  "parameters": {
    "brightness": 128
  }
}
```

---

### ha_query

Get current state and attributes of Home Assistant entities.

**Tool Definition:**
```json
{
  "name": "ha_query",
  "description": "Get current state and attributes of Home Assistant entities. Use this to check if lights are on, get sensor values, etc.",
  "parameters": {
    "type": "object",
    "properties": {
      "entity_id": {
        "type": "string",
        "description": "The entity ID to query (e.g., 'sensor.temperature'). Can use wildcards (e.g., 'light.*')"
      },
      "attributes": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Specific attributes to retrieve (optional, returns all if not specified)"
      },
      "history": {
        "type": "object",
        "description": "Optional: retrieve historical data",
        "properties": {
          "duration": {
            "type": "string",
            "description": "Time range (e.g., '1h', '24h', '7d')"
          },
          "aggregate": {
            "type": "string",
            "enum": ["avg", "min", "max", "sum", "count"]
          }
        }
      }
    },
    "required": ["entity_id"]
  }
}
```

**Example Tool Calls:**

```json
// Query single entity
{
  "entity_id": "sensor.living_room_temperature"
}

// Query specific attributes
{
  "entity_id": "light.living_room",
  "attributes": ["state", "brightness"]
}

// Query with wildcard
{
  "entity_id": "light.*"
}

// Query with history
{
  "entity_id": "sensor.temperature",
  "history": {
    "duration": "24h",
    "aggregate": "avg"
  }
}
```

**Response Format (Single Entity):**
```json
{
  "success": true,
  "entity_id": "sensor.living_room_temperature",
  "state": "23.5",
  "attributes": {
    "unit_of_measurement": "°C",
    "friendly_name": "Living Room Temperature",
    "device_class": "temperature"
  }
}
```

**Response Format (Multiple Entities):**
```json
{
  "success": true,
  "entities": [
    {
      "entity_id": "light.living_room",
      "state": "on",
      "attributes": {"brightness": 128}
    },
    {
      "entity_id": "light.bedroom",
      "state": "off"
    }
  ]
}
```

**Response Format (With History):**
```json
{
  "success": true,
  "entity_id": "sensor.temperature",
  "current_state": "23.5",
  "history": {
    "duration": "24h",
    "aggregate": "avg",
    "value": 22.8
  }
}
```

---

### query_external_llm

Query a more capable external LLM for complex analysis (optional tool).

**Availability:** Only available when `CONF_EXTERNAL_LLM_ENABLED = true`

**Tool Definition:**
```json
{
  "name": "query_external_llm",
  "description": "Query a more capable external LLM for complex analysis, detailed explanations, or comprehensive answers. Use this when you need help with complex reasoning or detailed analysis.",
  "parameters": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "The question or prompt to send to the external LLM"
      },
      "context": {
        "type": "object",
        "description": "Additional explicit context to provide (e.g., sensor data, previous tool results). Note: Full conversation history is NOT automatically included."
      }
    },
    "required": ["prompt"]
  }
}
```

**Example Tool Calls:**

```json
// Simple query
{
  "prompt": "Should I adjust the thermostat based on current conditions?"
}

// Query with context
{
  "prompt": "Analyze this energy usage data and suggest optimizations",
  "context": {
    "current_temperature": "68°F",
    "energy_usage_kwh": 45.2,
    "time_period": "last 24 hours"
  }
}
```

**Response Format (Success):**
```json
{
  "success": true,
  "result": "Based on the current temperature of 68°F, the thermostat setting is optimal for comfort and energy efficiency...",
  "error": null
}
```

**Response Format (Error):**
```json
{
  "success": false,
  "result": null,
  "error": "Failed to query external LLM: Connection timeout after 30s"
}
```

**Notes:**
- Errors are returned transparently to primary LLM
- Only explicit prompt and context are passed (conversation history NOT included)
- Counts toward `max_calls_per_turn` limit

---

### store_memory

Store a memory for future recall (memory tool).

**Availability:** Only available when `CONF_MEMORY_ENABLED = true`

**Tool Definition:**
```json
{
  "name": "store_memory",
  "description": "Store important information as a memory for future conversations. Use this to remember user preferences, facts about the home, or important context.",
  "parameters": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The information to remember"
      },
      "memory_type": {
        "type": "string",
        "enum": ["fact", "preference", "context", "event"],
        "description": "Type of memory"
      },
      "importance": {
        "type": "number",
        "description": "Importance score from 0.0 to 1.0"
      }
    },
    "required": ["content"]
  }
}
```

**Example Tool Call:**
```json
{
  "content": "User prefers bedroom temperature at 68°F for sleeping",
  "memory_type": "preference",
  "importance": 0.8
}
```

**Response Format:**
```json
{
  "success": true,
  "memory_id": "mem_abc123xyz",
  "message": "Memory stored successfully"
}
```

---

### recall_memory

Search and retrieve relevant memories (memory tool).

**Availability:** Only available when `CONF_MEMORY_ENABLED = true`

**Tool Definition:**
```json
{
  "name": "recall_memory",
  "description": "Search stored memories to recall relevant information from past conversations.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "What to search for in memories"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of memories to retrieve"
      }
    },
    "required": ["query"]
  }
}
```

**Example Tool Call:**
```json
{
  "query": "temperature preferences",
  "limit": 5
}
```

**Response Format:**
```json
{
  "success": true,
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User prefers bedroom temperature at 68°F for sleeping",
      "type": "preference",
      "importance": 0.8,
      "created_at": "2024-01-15T10:30:00",
      "relevance_score": 0.95
    },
    {
      "id": "mem_xyz789",
      "content": "Living room thermostat typically set to 72°F during day",
      "type": "fact",
      "importance": 0.6,
      "created_at": "2024-01-14T15:20:00",
      "relevance_score": 0.87
    }
  ]
}
```

---

## Tool Response Format

All tools return standardized responses:

**Success Response:**
```json
{
  "success": true,
  "result": <tool_specific_data>,
  "error": null
}
```

**Error Response:**
```json
{
  "success": false,
  "result": null,
  "error": "Error message describing what went wrong"
}
```

This standardized format ensures consistent error handling and makes it easy for the LLM to understand tool execution results.

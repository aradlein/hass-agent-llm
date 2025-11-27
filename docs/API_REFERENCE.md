# API Reference

Quick reference for Home Agent services, events, and tools.

## Core Services

| Service | Description | Key Parameters |
|---------|-------------|----------------|
| **home_agent.process** | Process a conversation message | `text` (required), `conversation_id` (optional) |
| **home_agent.clear_history** | Clear conversation history | `conversation_id` (optional - omit to clear all) |
| **home_agent.execute_tool** | Manually execute a tool for testing | `tool_name`, `parameters` |
| **home_agent.add_memory** | Manually add a memory | `content`, `type`, `importance` |
| **home_agent.search_memories** | Search memories semantically | `query`, `limit`, `min_importance` |

## Key Events

| Event | Triggered When | Key Data Fields |
|-------|----------------|-----------------|
| **home_agent.conversation.finished** | Conversation completes | `tool_calls`, `tokens`, `duration_ms`, `performance` |
| **home_agent.tool.executed** | Tool execution completes | `tool_name`, `parameters`, `result`, `success`, `duration_ms` |
| **home_agent.memory.extracted** | Memories extracted from conversation | `conversation_id`, `memories_extracted`, `extraction_llm` |
| **home_agent.error** | Error occurs | `error_type`, `error_message`, `component` |
| **home_agent.context.injected** | Context added to LLM call | `mode`, `entities_included`, `token_count` |

## Built-in Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| **ha_control** | Control HA devices/services | `action` (turn_on/off/toggle/set_value), `entity_id`, `parameters` (optional: brightness, temperature, etc.) |
| **ha_query** | Get entity state and attributes | `entity_id`, `attributes` (optional), `history` (optional: duration, aggregate) |
| **query_external_llm** | Query external LLM for complex analysis | `prompt`, `context` (optional) |

*Note: `query_external_llm` only available when external LLM is enabled.*

## Quick Examples

### Service Call Example

Process a conversation:

```yaml
service: home_agent.process
data:
  text: "Turn on the living room lights and set them to 50% brightness"
  conversation_id: "main_conversation"
```

Search memories:

```yaml
service: home_agent.search_memories
data:
  query: "bedroom temperature preferences"
  limit: 10
  min_importance: 0.5
```

### Event Automation Example

Track token usage:

```yaml
automation:
  - alias: "Track Token Usage"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: input_number.set_value
        target:
          entity_id: input_number.total_tokens_used
        data:
          value: >
            {{ states('input_number.total_tokens_used') | int +
               trigger.event.data.tokens.total | int }}
```

Alert on errors:

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

### Tool Usage Example

The LLM automatically calls these tools, but you can test them manually:

**Control a device:**
```yaml
service: home_agent.execute_tool
data:
  tool_name: ha_control
  parameters:
    action: turn_on
    entity_id: light.living_room
    parameters:
      brightness: 128
```

**Query entity state:**
```yaml
service: home_agent.execute_tool
data:
  tool_name: ha_query
  parameters:
    entity_id: sensor.living_room_temperature
```

**Query with history:**
```yaml
service: home_agent.execute_tool
data:
  tool_name: ha_query
  parameters:
    entity_id: sensor.temperature
    history:
      duration: "24h"
      aggregate: "avg"
```

## Service Details

### home_agent.process

Process a conversation message.

**Parameters:**
- `text` (string, required): The message to process
- `conversation_id` (string, optional): ID for history tracking
- `user_id` (string, optional): User ID for the conversation

**Example:**
```yaml
service: home_agent.process
data:
  text: "What's the temperature in the living room?"
  conversation_id: "main_conversation"
```

### home_agent.add_memory

Manually store a memory.

**Parameters:**
- `content` (text, required): The memory content
- `type` (select, optional): Memory type (fact, preference, context, event)
- `importance` (number 0.0-1.0, optional): Importance score

**Example:**
```yaml
service: home_agent.add_memory
data:
  content: "User prefers bedroom temperature at 68°F for sleeping"
  type: preference
  importance: 0.8
```

### home_agent.clear_history

Clear conversation history.

**Parameters:**
- `conversation_id` (string, optional): Specific conversation to clear (omit to clear all)

**Examples:**
```yaml
# Clear specific conversation
service: home_agent.clear_history
data:
  conversation_id: "living_room_conversation"

# Clear all conversations
service: home_agent.clear_history
```

## Event Details

### home_agent.conversation.finished

Triggered when a conversation completes.

**Key Data Fields:**
```json
{
  "conversation_id": "main_conversation",
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

### home_agent.tool.executed

Triggered after each tool execution.

**Key Data Fields:**
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

### home_agent.memory.extracted

Triggered when memories are extracted.

**Key Data Fields:**
```json
{
  "conversation_id": "main_conversation",
  "memories_extracted": 3,
  "extraction_llm": "external",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Tool Details

### ha_control

Control Home Assistant devices.

**Actions:**
- `turn_on` - Turn entity on
- `turn_off` - Turn entity off
- `toggle` - Toggle entity state
- `set_value` - Set specific value

**Common Parameters:**
- **Lights:** `brightness` (0-255), `rgb_color` ([R,G,B])
- **Climate:** `temperature`, `hvac_mode` (heat/cool/auto/off)
- **Covers:** `position` (0-100)
- **Media:** `volume_level` (0.0-1.0)

**Examples:**
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

// Set thermostat
{
  "action": "set_value",
  "entity_id": "climate.thermostat",
  "parameters": {
    "temperature": 72
  }
}
```

### ha_query

Get entity state and attributes.

**Parameters:**
- `entity_id` (required): Entity to query (supports wildcards like `light.*`)
- `attributes` (optional): Specific attributes to retrieve
- `history` (optional): Retrieve historical data
  - `duration`: Time range (e.g., "1h", "24h", "7d")
  - `aggregate`: Aggregation method (avg, min, max, sum, count)

**Examples:**
```json
// Query single entity
{
  "entity_id": "sensor.living_room_temperature"
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

### query_external_llm

Query external LLM for complex analysis (only available when external LLM is enabled).

**Parameters:**
- `prompt` (required): Question or prompt to send
- `context` (optional): Additional context (sensor data, tool results, etc.)

**Examples:**
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

## Need More Details?

See the [Complete Reference](reference/API_REFERENCE.md) for comprehensive API documentation with all services, events, and tools.

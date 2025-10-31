# Home Agent

A highly customizable Home Assistant custom component that provides intelligent conversational AI capabilities with advanced tool calling, context injection, and conversation history management.

## Overview

Home Agent extends Home Assistant's native conversation platform to provide:

- **OpenAI-Compatible LLM Integration** - Works with any OpenAI-compatible API (OpenAI, Ollama, LM Studio, LocalAI, etc.)
- **Smart Context Injection** - Automatically provides relevant entity states to the LLM
- **Conversation History** - Maintains conversation context across multiple interactions
- **Tool Calling** - Native Home Assistant control and query capabilities
- **Flexible Configuration** - Extensive UI-based configuration options

## Features

### Phase 1 (Current - MVP) ✅

- ✅ OpenAI-compatible LLM API integration
- ✅ Direct entity context injection
- ✅ Conversation history management
- ✅ Core tools: `ha_control` and `ha_query`
- ✅ UI-based configuration flow
- ✅ Service endpoints for automation
- ✅ Event system for monitoring
- ✅ Comprehensive unit tests (376+ passing tests)

### Phase 2 (Planned)

- Vector DB (ChromaDB) integration for semantic context retrieval
- Enhanced history management with persistence
- Context optimization and compression
- Advanced event system

### Phase 3 (Planned)

- External LLM tool (delegate complex queries to more powerful models)
- Custom tool definition framework
- Streaming response support

## Installation

### HACS (Recommended - Coming Soon)

1. Open HACS in Home Assistant
2. Search for "Home Agent"
3. Install the integration
4. Restart Home Assistant

### Manual Installation

1. Copy the `custom_components/home_agent` directory to your Home Assistant `custom_components` folder
2. Restart Home Assistant
3. Go to Settings > Devices & Services > Add Integration
4. Search for "Home Agent" and follow the setup wizard

## Configuration

### Initial Setup

1. **Add Integration**: Go to Settings > Devices & Services > Add Integration
2. **Configure LLM**:
   - **Name**: Friendly name for this instance (default: "Home Agent")
   - **LLM Base URL**: Your OpenAI-compatible endpoint (e.g., `https://api.openai.com/v1` or `http://localhost:11434/v1` for Ollama)
   - **API Key**: Your API key (if required)
   - **Model**: Model name (e.g., `gpt-4o-mini`, `llama2`, etc.)
   - **Temperature**: Creativity level, 0.0-2.0 (default: 0.7)
   - **Max Tokens**: Maximum response length (default: 500)

### Advanced Configuration (Options)

Access via: Settings > Devices & Services > Home Agent > Configure

Home Agent provides a menu-based options flow with the following categories:

#### LLM Settings
Edit the primary LLM connection and parameters:
- **LLM Base URL**: OpenAI-compatible API endpoint
- **API Key**: Authentication key for the LLM service
- **Model**: Model name to use
- **Temperature**: Creativity level, 0.0-2.0 (default: 0.7)
- **Max Tokens**: Maximum response length (default: 500)

#### Context Settings
Configure how entity context is provided to the LLM:
- **Context Mode**: Direct (Phase 1) or Vector DB (Phase 2)
- **Context Format**: JSON, Natural Language, or Hybrid
- **Entities to Include**: Comma-separated list of entity IDs or patterns

#### Conversation History
Manage conversation history:
- **Enable History**: Track conversation context across turns
- **Max Messages**: Maximum conversation turns to retain (default: 10)
- **Max Tokens**: Token-based limit for history (default: 4000)

#### System Prompt
Customize the agent's behavior:
- **Use Default Prompt**: Use Home Agent's built-in system prompt
- **Custom Additions**: Additional instructions to append

#### Tool Configuration
Control tool execution limits:
- **Max Tool Calls Per Turn**: Maximum executions per message (default: 5)
- **Tool Timeout**: Timeout in seconds for each tool call (default: 30)

#### External LLM (Phase 3)
Configure an optional external LLM for complex queries:
- **Enable External LLM**: Expose query_external_llm tool
- **Base URL, API Key, Model**: External LLM connection details
- **Tool Description**: When to use the external LLM
- **Auto-include Context**: Pass conversation history automatically

#### Debug Settings
Enable detailed logging:
- **Debug Logging**: Enable verbose logging for troubleshooting

## Usage

### As a Service

Call the `home_agent.process` service to interact with the agent:

```yaml
service: home_agent.process
data:
  text: "Turn on the living room lights to 50%"
  conversation_id: "living_room_chat"  # Optional, for history tracking
  user_id: "user123"  # Optional
```

### In Automations

```yaml
automation:
  - alias: "Voice Command Handler"
    trigger:
      - platform: voice_assistant
        event_type: intent
    action:
      - service: home_agent.process
        data:
          text: "{{ trigger.event.data.text }}"
          conversation_id: "{{ trigger.event.data.conversation_id }}"
```

### Services

#### `home_agent.process`
Process a conversation message through the agent.

**Parameters:**
- `text` (required): The user's message
- `conversation_id` (optional): ID for history tracking
- `user_id` (optional): User identifier

#### `home_agent.clear_history`
Clear conversation history.

**Parameters:**
- `conversation_id` (optional): Specific conversation to clear (omit for all)

#### `home_agent.reload_context`
Reload entity context (useful after entity changes).

#### `home_agent.execute_tool`
Manually execute a tool for testing/debugging.

**Parameters:**
- `tool_name` (required): Tool to execute (e.g., "ha_control", "ha_query")
- `parameters` (required): Tool parameters as JSON

## Tool System

### ha_control

Control Home Assistant devices and services.

**Actions:**
- `turn_on` - Turn on entities
- `turn_off` - Turn off entities
- `toggle` - Toggle state
- `set_value` - Set specific values (brightness, temperature, etc.)

**Example:**
```json
{
  "action": "turn_on",
  "entity_id": "light.living_room",
  "parameters": {
    "brightness_pct": 50
  }
}
```

### ha_query

Query Home Assistant entity states and history.

**Features:**
- Current state queries
- Wildcard matching (`light.*`, `*.bedroom`)
- Attribute filtering
- Historical data with aggregation (avg, min, max, sum, count)

**Example:**
```json
{
  "entity_id": "sensor.temperature",
  "history": {
    "duration": "24h",
    "aggregate": "avg"
  }
}
```

## Context Injection

### Direct Mode (Phase 1)

Specify entities to always include in LLM context:

```yaml
context_entities:
  - entity_id: sensor.living_room_temperature
    attributes: [state, unit_of_measurement]
  - entity_id: light.living_room
    attributes: [state, brightness]
  - entity_id: climate.*  # Wildcard support
```

## Events

Home Agent fires events for monitoring and automation:

### `home_agent.conversation.started`
Fired when a conversation begins.

**Data:**
- `conversation_id`
- `user_id`
- `timestamp`
- `context_mode`

### `home_agent.conversation.finished`
Fired when a conversation completes.

**Data:**
- `conversation_id`
- `user_id`
- `duration_ms`
- `tool_calls`
- `tokens_used`

### `home_agent.tool.executed`
Fired after each tool execution.

**Data:**
- `tool_name`
- `parameters`
- `result`
- `success`
- `duration_ms`

### `home_agent.context.injected`
Fired when context is injected.

**Data:**
- `conversation_id`
- `mode`
- `entities_included`
- `token_count`

### `home_agent.error`
Fired on errors.

**Data:**
- `error_type`
- `error_message`
- `conversation_id`
- `component`

## Examples

### Basic Light Control

```yaml
service: home_agent.process
data:
  text: "Turn on the bedroom lights"
```

### Complex Query with History

```yaml
service: home_agent.process
data:
  text: "What was the average temperature yesterday?"
  conversation_id: "climate_analysis"
```

### Automation Trigger

```yaml
automation:
  - alias: "Morning Routine Assistant"
    trigger:
      - platform: time
        at: "07:00:00"
    action:
      - service: home_agent.process
        data:
          text: "Good morning! Please prepare for the day."
```

## Development

### Running Tests

```bash
# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Run tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=custom_components.home_agent --cov-report=html
```

### Project Structure

```
custom_components/home_agent/
├── __init__.py              # Component initialization
├── agent.py                 # Main conversation agent
├── config_flow.py           # UI configuration
├── const.py                 # Constants
├── context_manager.py       # Context injection orchestration
├── conversation.py          # History management
├── exceptions.py            # Custom exceptions
├── helpers.py               # Utility functions
├── manifest.json            # Component metadata
├── services.yaml            # Service definitions
├── strings.json             # UI text
├── tool_handler.py          # Tool execution orchestration
├── context_providers/       # Context injection strategies
│   ├── base.py             # Base provider interface
│   └── direct.py           # Direct entity injection
├── tools/                   # Tool implementations
│   ├── registry.py         # Tool registry
│   ├── ha_control.py       # Home Assistant control
│   └── ha_query.py         # Home Assistant queries
└── translations/            # Localization
    └── en.json
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Follow the development standards in `docs/DEVELOPMENT.md`
6. Submit a pull request

## Testing

This project maintains >80% code coverage with comprehensive unit and integration tests.

**Current Test Status:**
- ✅ 376+ passing unit tests
- ✅ Comprehensive coverage of core functionality
- ✅ Mock-based testing for fast execution

## Documentation

- [Project Specification](docs/PROJECT_SPEC.md) - Complete feature specifications
- [Development Standards](docs/DEVELOPMENT.md) - Coding standards and test requirements

## Troubleshooting

### LLM Connection Issues

- Verify your API endpoint is accessible
- Check API key is correct
- Ensure model name is valid for your endpoint
- Check Home Assistant logs for detailed error messages

### Tool Execution Failures

- Verify entities are exposed to the agent
- Check entity IDs are correct
- Ensure required integrations (like recorder for history) are available

### Debug Mode

Enable debug logging in the integration options to see detailed execution logs.

## License

[Add your license here]

## Credits

Built with inspiration from the extended_openai_conversation integration.

## Support

- [GitHub Issues](https://github.com/yourusername/home-agent/issues)
- [GitHub Discussions](https://github.com/yourusername/home-agent/discussions)

## Changelog

### 0.1.0 (Phase 1 MVP)

- Initial release
- OpenAI-compatible LLM integration
- Direct context injection
- Conversation history
- Core tools (ha_control, ha_query)
- UI configuration
- Service endpoints
- Event system

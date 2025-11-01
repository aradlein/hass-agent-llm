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

### Phase 1 (MVP) ✅

- ✅ OpenAI-compatible LLM API integration
- ✅ Direct entity context injection
- ✅ Conversation history management
- ✅ Core tools: `ha_control` and `ha_query`
- ✅ UI-based configuration flow
- ✅ Service endpoints for automation
- ✅ Event system for monitoring
- ✅ Comprehensive unit tests (376+ passing tests)

### Phase 2 (Current) ✅

- ✅ **Vector DB (ChromaDB) Integration** - Semantic entity search using embeddings
- ✅ **History Persistence** - Conversations saved across HA restarts
- ✅ **Context Optimization** - Intelligent compression to stay within token limits
- ✅ **Enhanced Events** - Detailed metrics including tokens, latency, and compression ratios
- ✅ **Smart Truncation** - Preserves important information when context is large
- ✅ **Entity Prioritization** - Ranks entities by relevance to user query

### Phase 3 ✅

- ✅ **External LLM Tool** - Delegate complex queries to more powerful models via `query_external_llm`
- ✅ **Custom Tool Framework** - Define custom tools in `configuration.yaml` (REST + Service handlers)
- ✅ Tool execution with standardized response format and error handling

### Phase 4 (Planned)

- **Streaming Response Support** - Stream LLM responses for better user experience
- Progress indicators during tool execution

### Phase 5 (Planned)

- **MCP Server Integration** - Connect to external Model Context Protocol servers for data collection
- Extended custom tool handlers with OAuth support

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

- **Enable External LLM**: Expose `query_external_llm` tool to primary LLM
- **Base URL, API Key, Model**: External LLM connection details
- **Tool Description**: Customize when primary LLM should delegate to external LLM
- **Context Handling**: Only explicit `prompt` and `context` parameters are passed (not full conversation history)

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
  conversation_id: "living_room_chat" # Optional, for history tracking
  user_id: "user123" # Optional
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

### query_external_llm (Phase 3) 🆕

Delegate complex queries to a more powerful external LLM.

**Use Cases:**

- Complex analysis requiring advanced reasoning
- Detailed explanations or recommendations
- Tasks requiring larger context windows
- Specialized tasks better suited for specific models

**Configuration:**

Enable via Integration Options > External LLM:

- **Enable External LLM**: Expose the tool to primary LLM
- **Base URL**: External LLM endpoint (e.g., `https://api.openai.com/v1`)
- **API Key**: Authentication key
- **Model**: Model name (e.g., `gpt-4o`, `claude-3-5-sonnet`)
- **Tool Description**: Customize when primary LLM should delegate

**Example:**

```json
{
  "prompt": "Analyze the energy consumption patterns and provide optimization recommendations",
  "context": "Living room lights used 45kWh this month, bedroom 32kWh, kitchen 28kWh"
}
```

**Parameters:**

- `prompt` (required): The query to send to external LLM
- `context` (optional): Additional context to provide

**Note:** Only explicit parameters are passed to external LLM - full conversation history is NOT automatically included for efficiency.

## Context Injection

### Direct Mode (Phase 1)

Specify entities to always include in LLM context:

```yaml
context_entities:
  - entity_id: sensor.living_room_temperature
    attributes: [state, unit_of_measurement]
  - entity_id: light.living_room
    attributes: [state, brightness]
  - entity_id: climate.* # Wildcard support
```

### Vector DB Mode (Phase 2) 🆕

Use semantic search to dynamically find relevant entities based on user query:

**Setup:**

1. Install and run ChromaDB server
2. Configure Vector DB settings in integration options
3. Index entities: Call `home_agent.index_entities` service (coming soon)

**Configuration:**

```yaml
context_mode: vector_db
vector_db:
  host: localhost
  port: 8000
  collection: home_entities
  embedding_model: text-embedding-3-small
  top_k: 5 # Number of entities to retrieve
  similarity_threshold: 0.7
```

**How it works:**

- User query is embedded using the configured model
- ChromaDB finds semantically similar entities
- Only relevant entities are included in context
- More efficient token usage compared to direct mode

## Custom Tools (Phase 3) 🆕

Define custom tools in your `configuration.yaml` to extend the LLM's capabilities with REST APIs and Home Assistant services.

### REST Tools

Call external HTTP APIs with full template support:

```yaml
home_agent:
  tools_custom:
    - name: check_weather
      description: "Get weather forecast for a location"
      parameters:
        type: object
        properties:
          location:
            type: string
            description: "City name or coordinates"
        required:
          - location
      handler:
        type: rest
        url: "https://api.weather.com/v1/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ secrets.weather_api_key }}"
        query_params:
          location: "{{ location }}"
          format: "json"
```

**REST Handler Options:**

- `url` (required): API endpoint (supports Jinja2 templates)
- `method` (required): HTTP method (GET, POST, PUT, DELETE)
- `headers` (optional): Request headers with template support
- `query_params` (optional): URL query parameters with template support
- `body` (optional): JSON request body for POST/PUT requests
- `timeout` (optional): Request timeout in seconds

**Template Variables:**

- Tool parameters are available as template variables
- Access secrets via `{{ secrets.key_name }}`
- Standard Home Assistant Jinja2 template syntax

### Service Tools

Trigger Home Assistant services, automations, scripts, and scenes:

```yaml
home_agent:
  tools_custom:
    # Simple automation trigger
    - name: trigger_morning_routine
      description: "Trigger the morning routine automation"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine

    # Script with parameters
    - name: notify_family
      description: "Send a notification to the family with a custom message"
      parameters:
        type: object
        properties:
          message:
            type: string
            description: "The message to send"
        required:
          - message
      handler:
        type: service
        service: script.notify_family
        data:
          message: "{{ message }}"

    # Scene activation with target
    - name: set_movie_scene
      description: "Activate the movie watching scene"
      handler:
        type: service
        service: scene.turn_on
        target:
          entity_id: scene.movie_time
```

**Service Handler Options:**

- `service` (required): Service to call (format: `domain.service_name`)
- `data` (optional): Service data with template support
- `target` (optional): Target entities, devices, or areas
  - `entity_id`: Single entity, list, or templated
  - `device_id`: Device identifier
  - `area_id`: Area identifier

**Common Use Cases:**

- **Trigger Automations**: `automation.trigger` with `entity_id`
- **Run Scripts**: `script.my_script` with templated parameters
- **Control Scenes**: `scene.turn_on` with target
- **Custom Notifications**: `notify.mobile_app` with message templates
- **Climate Control**: `climate.set_temperature` with templated values

### Response Format

All custom tools return a standardized response:

```json
{
  "success": true,
  "result": { /* API response or success message */ },
  "error": null
}
```

On error:

```json
{
  "success": false,
  "result": null,
  "error": "Error message description"
}
```

### Advanced Examples

#### POST Request with Body

```yaml
- name: create_task
  description: "Create a new task in external system"
  parameters:
    type: object
    properties:
      title:
        type: string
      priority:
        type: string
        enum: [low, medium, high]
  handler:
    type: rest
    url: "https://api.tasks.com/v1/tasks"
    method: POST
    headers:
      Content-Type: "application/json"
      Authorization: "Bearer {{ secrets.tasks_api_key }}"
    body:
      title: "{{ title }}"
      priority: "{{ priority }}"
      created_by: "home_assistant"
```

#### Dynamic Service Targeting

```yaml
- name: control_room_lights
  description: "Turn on/off lights in a specific room"
  parameters:
    type: object
    properties:
      room:
        type: string
      action:
        type: string
        enum: [turn_on, turn_off]
  handler:
    type: service
    service: "light.{{ action }}"
    target:
      area_id: "{{ room }}"
```

#### Climate Control with Templates

```yaml
- name: set_room_temperature
  description: "Set temperature for a specific room"
  parameters:
    type: object
    properties:
      room:
        type: string
      temperature:
        type: number
  handler:
    type: service
    service: climate.set_temperature
    data:
      temperature: "{{ temperature }}"
    target:
      area_id: "{{ room }}"
```

### Configuration Tips

1. **Parameter Schema**: Use JSON Schema to define tool parameters for proper LLM understanding
2. **Descriptions**: Write clear descriptions to help the LLM know when to use each tool
3. **Templates**: Leverage Jinja2 templates for dynamic values
4. **Error Handling**: Tools automatically handle errors and return structured responses
5. **Validation**: Service tools validate that services exist at startup (warns if not found)

### Complete Configuration Example

Here's a complete example showing both REST and service tools in your `configuration.yaml`:

```yaml
home_agent:
  tools_custom:
    # Example REST tool - calls external weather API
    - name: check_weather
      description: "Get weather forecast"
      handler:
        type: rest
        url: "https://api.open-meteo.com/v1/forecast"
        method: GET
        query_params:
          latitude: "47.6788491"
          longitude: "-122.3971093"
          forecast_days: 3
          precipitation_unit: "inch"
          current: "temperature_2m,precipitation"
          hourly: "temperature_2m,precipitation,showers"

    # Example service tool - triggers Home Assistant automation
    - name: trigger_morning_routine
      description: "Trigger the morning routine automation"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine

    # Example service tool with parameters - runs script with template variables
    - name: notify_family
      description: "Send a notification to the family with a custom message"
      parameters:
        type: object
        properties:
          message:
            type: string
            description: "The message to send"
        required:
          - message
      handler:
        type: service
        service: script.notify_family
        data:
          message: "{{ message }}"

    # Example service tool with target - controls scene
    - name: set_movie_scene
      description: "Activate the movie watching scene"
      handler:
        type: service
        service: scene.turn_on
        target:
          entity_id: scene.movie_time
```

This configuration provides:
- **External API integration** with weather data
- **Automation control** for morning routines
- **Parameterized scripts** for family notifications
- **Scene management** for movie time

### Context Optimization (Phase 2) 🆕

Automatically compresses context when approaching token limits:

- **Smart Truncation** - Preserves entity IDs and critical information
- **Entity Prioritization** - Ranks by relevance to user query
- **Whitespace Removal** - Removes redundant formatting
- **Compression Levels** - Low, medium, high
- **Metrics Tracking** - Monitor compression ratios via events

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

**Data (Enhanced in Phase 2):**

- `conversation_id`
- `user_id`
- `duration_ms`
- `tool_calls`
- `tokens` - Breakdown: `prompt`, `completion`, `total` 🆕
- `performance` - Latency: `llm_latency_ms`, `tool_latency_ms`, `context_latency_ms` 🆕
- `context` - Optimization metrics: `original_tokens`, `optimized_tokens`, `compression_ratio` 🆕

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

### `home_agent.context.optimized` 🆕

Fired when context is compressed (Phase 2).

**Data:**

- `original_tokens`
- `optimized_tokens`
- `compression_ratio`
- `was_truncated`

### `home_agent.history.saved` 🆕

Fired when history is persisted (Phase 2).

**Data:**

- `conversation_count`
- `message_count`
- `size_bytes`
- `timestamp`

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

### Code Quality

Home Agent follows Home Assistant's code quality standards using `ruff` for linting and formatting.

```bash
# Quick format (auto-fix + format)
./scripts/format.sh

# Full linting (includes pylint)
./scripts/lint.sh

# Or run tools individually:
python3 -m ruff check --fix custom_components/home_agent/
python3 -m ruff format custom_components/home_agent/
python3 -m pylint custom_components/home_agent/
```

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

- ✅ 400+ passing unit tests
- ✅ 16+ integration tests
- ✅ Comprehensive coverage of core functionality including:
  - Phase 1: LLM integration, context injection, history, core tools
  - Phase 2: Vector DB, persistence, optimization
  - Phase 3: Custom tools (REST + Service handlers), external LLM
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

### 0.3.0 (Phase 3)

- Custom tool framework with REST and service handlers
- Define custom tools in `configuration.yaml`
- REST API integration with template support
- Home Assistant service tool integration
- External LLM delegation via `query_external_llm` tool
- Comprehensive error handling and validation
- Standardized tool response format

### 0.2.0 (Phase 2)

- Vector DB (ChromaDB) integration for semantic entity search
- Conversation history persistence across restarts
- Context optimization with smart compression
- Enhanced event system with performance metrics
- Entity prioritization and relevance ranking
- Token usage tracking and optimization

### 0.1.0 (Phase 1 MVP)

- Initial release
- OpenAI-compatible LLM integration
- Direct context injection
- Conversation history
- Core tools (ha_control, ha_query)
- UI configuration
- Service endpoints
- Event system

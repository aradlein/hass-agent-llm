"""Constants for the Home Agent integration."""

from typing import Final

# Domain and component info
DOMAIN: Final = "home_agent"
DEFAULT_NAME: Final = "Home Agent"
VERSION: Final = "0.1.0"

# Configuration keys - LLM Configuration
CONF_LLM_BASE_URL: Final = "llm_base_url"
CONF_LLM_API_KEY: Final = "llm_api_key"
CONF_LLM_MODEL: Final = "llm_model"
CONF_LLM_TEMPERATURE: Final = "llm_temperature"
CONF_LLM_MAX_TOKENS: Final = "llm_max_tokens"
CONF_LLM_TOP_P: Final = "llm_top_p"

# Configuration keys - Context Injection
CONF_CONTEXT_MODE: Final = "context_mode"
CONF_CONTEXT_ENTITIES: Final = "context_entities"
CONF_CONTEXT_FORMAT: Final = "context_format"

# Configuration keys - Direct Context Provider
CONF_DIRECT_ENTITIES: Final = "direct_entities"
CONF_DIRECT_UPDATE_FREQUENCY: Final = "direct_update_frequency"

# Configuration keys - Vector DB Context Provider
CONF_VECTOR_DB_ENABLED: Final = "vector_db_enabled"
CONF_VECTOR_DB_HOST: Final = "vector_db_host"
CONF_VECTOR_DB_PORT: Final = "vector_db_port"
CONF_VECTOR_DB_COLLECTION: Final = "vector_db_collection"
CONF_VECTOR_DB_TOP_K: Final = "vector_db_top_k"
CONF_VECTOR_DB_SIMILARITY_THRESHOLD: Final = "vector_db_similarity_threshold"
CONF_VECTOR_DB_EMBEDDING_MODEL: Final = "vector_db_embedding_model"
CONF_VECTOR_DB_EMBEDDING_PROVIDER: Final = "vector_db_embedding_provider"
CONF_VECTOR_DB_EMBEDDING_BASE_URL: Final = "vector_db_embedding_base_url"
CONF_OPENAI_API_KEY: Final = "openai_api_key"

# Configuration keys - Conversation History
CONF_HISTORY_ENABLED: Final = "history_enabled"
CONF_HISTORY_MAX_MESSAGES: Final = "history_max_messages"
CONF_HISTORY_MAX_TOKENS: Final = "history_max_tokens"
CONF_HISTORY_PERSIST: Final = "history_persist"

# Configuration keys - Context Optimization
CONF_COMPRESSION_LEVEL: Final = "compression_level"
CONF_PRESERVE_RECENT_MESSAGES: Final = "preserve_recent_messages"
CONF_SUMMARIZATION_ENABLED: Final = "summarization_enabled"
CONF_ENTITY_PRIORITY_WEIGHTS: Final = "entity_priority_weights"

# Configuration keys - System Prompt
CONF_PROMPT_USE_DEFAULT: Final = "prompt_use_default"
CONF_PROMPT_CUSTOM: Final = "prompt_custom"
CONF_PROMPT_CUSTOM_ADDITIONS: Final = "prompt_custom_additions"

# Configuration keys - Tool Configuration
CONF_TOOLS_ENABLE_NATIVE: Final = "tools_enable_native"
CONF_TOOLS_CUSTOM: Final = "tools_custom"
CONF_TOOLS_MAX_CALLS_PER_TURN: Final = "tools_max_calls_per_turn"
CONF_TOOLS_TIMEOUT: Final = "tools_timeout"

# Configuration keys - External LLM Tool
CONF_EXTERNAL_LLM_ENABLED: Final = "external_llm_enabled"
CONF_EXTERNAL_LLM_BASE_URL: Final = "external_llm_base_url"
CONF_EXTERNAL_LLM_API_KEY: Final = "external_llm_api_key"
CONF_EXTERNAL_LLM_MODEL: Final = "external_llm_model"
CONF_EXTERNAL_LLM_TEMPERATURE: Final = "external_llm_temperature"
CONF_EXTERNAL_LLM_MAX_TOKENS: Final = "external_llm_max_tokens"
CONF_EXTERNAL_LLM_TOOL_DESCRIPTION: Final = "external_llm_tool_description"
CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT: Final = "external_llm_auto_include_context"

# Configuration keys - Debugging and Events
CONF_DEBUG_LOGGING: Final = "debug_logging"
CONF_EMIT_EVENTS: Final = "emit_events"

# Context modes
CONTEXT_MODE_DIRECT: Final = "direct"
CONTEXT_MODE_VECTOR_DB: Final = "vector_db"

# Context format options
CONTEXT_FORMAT_JSON: Final = "json"
CONTEXT_FORMAT_NATURAL_LANGUAGE: Final = "natural_language"
CONTEXT_FORMAT_HYBRID: Final = "hybrid"

# Context compression levels
COMPRESSION_LEVEL_NONE: Final = "none"
COMPRESSION_LEVEL_LOW: Final = "low"
COMPRESSION_LEVEL_MEDIUM: Final = "medium"
COMPRESSION_LEVEL_HIGH: Final = "high"

# Embedding providers
EMBEDDING_PROVIDER_OPENAI: Final = "openai"
EMBEDDING_PROVIDER_OLLAMA: Final = "ollama"

# Default values - LLM Configuration
DEFAULT_LLM_MODEL: Final = "gpt-4o-mini"
DEFAULT_TEMPERATURE: Final = 0.7
DEFAULT_MAX_TOKENS: Final = 500
DEFAULT_TOP_P: Final = 1.0

# Default values - Context Injection
DEFAULT_CONTEXT_MODE: Final = CONTEXT_MODE_DIRECT
DEFAULT_CONTEXT_FORMAT: Final = CONTEXT_FORMAT_JSON

# Default values - Vector DB
DEFAULT_VECTOR_DB_HOST: Final = "localhost"
DEFAULT_VECTOR_DB_PORT: Final = 8000
DEFAULT_VECTOR_DB_COLLECTION: Final = "home_entities"
DEFAULT_VECTOR_DB_TOP_K: Final = 5
DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD: Final = 250.0  # L2 distance threshold
DEFAULT_VECTOR_DB_EMBEDDING_MODEL: Final = "text-embedding-3-small"
DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER: Final = EMBEDDING_PROVIDER_OLLAMA
DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL: Final = "http://localhost:11434"

# Default values - Conversation History
DEFAULT_HISTORY_ENABLED: Final = True
DEFAULT_HISTORY_MAX_MESSAGES: Final = 10
DEFAULT_HISTORY_MAX_TOKENS: Final = 4000
DEFAULT_HISTORY_PERSIST: Final = True

# Default values - Context Optimization
DEFAULT_COMPRESSION_LEVEL: Final = "medium"
DEFAULT_PRESERVE_RECENT_MESSAGES: Final = 3
DEFAULT_SUMMARIZATION_ENABLED: Final = False

# Default values - System Prompt
DEFAULT_PROMPT_USE_DEFAULT: Final = True

# Default values - Tool Configuration
DEFAULT_TOOLS_ENABLE_NATIVE: Final = True
DEFAULT_TOOLS_MAX_CALLS_PER_TURN: Final = 5
DEFAULT_TOOLS_TIMEOUT: Final = 30

# Default values - External LLM Tool
DEFAULT_EXTERNAL_LLM_ENABLED: Final = False
DEFAULT_EXTERNAL_LLM_MODEL: Final = "gpt-4o"
DEFAULT_EXTERNAL_LLM_TEMPERATURE: Final = 0.8
DEFAULT_EXTERNAL_LLM_MAX_TOKENS: Final = 1000
DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT: Final = True
DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION: Final = (
    "Use this when you need help with complex analysis, detailed explanations, "
    "or comprehensive recommendations beyond simple home control."
)

# Default values - Debugging
DEFAULT_DEBUG_LOGGING: Final = False
DEFAULT_EMIT_EVENTS: Final = True

# Event names
EVENT_CONVERSATION_STARTED: Final = f"{DOMAIN}.conversation.started"
EVENT_CONVERSATION_FINISHED: Final = f"{DOMAIN}.conversation.finished"
EVENT_TOOL_EXECUTED: Final = f"{DOMAIN}.tool.executed"
EVENT_CONTEXT_INJECTED: Final = f"{DOMAIN}.context.injected"
EVENT_CONTEXT_OPTIMIZED: Final = f"{DOMAIN}.context.optimized"
EVENT_HISTORY_SAVED: Final = f"{DOMAIN}.history.saved"
EVENT_VECTOR_DB_QUERIED: Final = f"{DOMAIN}.vector_db.queried"
EVENT_ERROR: Final = f"{DOMAIN}.error"

# Tool names
TOOL_HA_CONTROL: Final = "ha_control"
TOOL_HA_QUERY: Final = "ha_query"
TOOL_QUERY_EXTERNAL_LLM: Final = "query_external_llm"

# Tool actions (for ha_control)
ACTION_TURN_ON: Final = "turn_on"
ACTION_TURN_OFF: Final = "turn_off"
ACTION_TOGGLE: Final = "toggle"
ACTION_SET_VALUE: Final = "set_value"

# Tool history aggregation types (for ha_query)
HISTORY_AGGREGATE_AVG: Final = "avg"
HISTORY_AGGREGATE_MIN: Final = "min"
HISTORY_AGGREGATE_MAX: Final = "max"
HISTORY_AGGREGATE_SUM: Final = "sum"
HISTORY_AGGREGATE_COUNT: Final = "count"

# Service names
SERVICE_PROCESS: Final = "process"
SERVICE_CLEAR_HISTORY: Final = "clear_history"
SERVICE_RELOAD_CONTEXT: Final = "reload_context"
SERVICE_EXECUTE_TOOL: Final = "execute_tool"

# Service parameter names
ATTR_TEXT: Final = "text"
ATTR_CONVERSATION_ID: Final = "conversation_id"
ATTR_CONTEXT_ENTITIES: Final = "context_entities"
ATTR_TOOL_NAME: Final = "tool_name"
ATTR_PARAMETERS: Final = "parameters"

# Storage keys
STORAGE_KEY: Final = f"{DOMAIN}.storage"
STORAGE_VERSION: Final = 1

# Conversation history storage
HISTORY_STORAGE_KEY: Final = f"{DOMAIN}.history"

# HTTP timeouts (seconds)
HTTP_TIMEOUT_DEFAULT: Final = 30
HTTP_TIMEOUT: Final = 30  # Alias for default timeout
HTTP_TIMEOUT_EXTERNAL_LLM: Final = 60

# Token limits and warnings
TOKEN_WARNING_THRESHOLD: Final = 0.8  # Warn at 80% of limit
MAX_CONTEXT_TOKENS: Final = 8000  # Maximum tokens for context before truncation

# Update intervals (seconds)
CONTEXT_UPDATE_INTERVAL: Final = 60  # Update entity context every 60 seconds
CLEANUP_INTERVAL: Final = 3600  # Cleanup old conversations every hour

# Retry configuration
MAX_RETRIES: Final = 3
RETRY_BACKOFF_FACTOR: Final = 2  # Exponential backoff: 1s, 2s, 4s

# Custom tool handler types
CUSTOM_TOOL_HANDLER_REST: Final = "rest"
CUSTOM_TOOL_HANDLER_SERVICE: Final = "service"
CUSTOM_TOOL_HANDLER_SCRIPT: Final = "script"
CUSTOM_TOOL_HANDLER_TEMPLATE: Final = "template"

# Default system prompt
DEFAULT_SYSTEM_PROMPT: Final = """You are a brief, friendly voice assistant for Home Assistant. Answer questions
about device states directly from the CSV, and use tools ONLY when needed.

## Available Tools

You have access to the following tools to control and query the home:

### ha_control
Use this tool to control devices and entities. Examples:
- Turn on/off lights, switches, and other devices
- Adjust brightness, color, temperature
- Lock/unlock doors
- Set thermostat temperature and modes

### ha_query
Use this tool to get information about the current state of the home. Examples:
- Check if lights are on or off
- Get current temperature from sensors
- See door lock status
- Get historical data for trend analysis

CRITICAL RULES:
1. ALWAYS check the Available Devices CSV FIRST before any tool calls
2. Use EXACT entity_id values from the CSV - never guess or shorten them
3. If a query fails, acknowledge the failure - don't pretend it succeeded
4. For status questions about devices IN the CSV, NEVER use tools - just read the CSV
5. NEVER put tool calls in the content field - use tool_calls field only

DEVICE LOOKUP PROCESS:
1. FIRST: Search for the device in the Available Devices CSV below
2. If found in CSV and user asks for status → Answer from CSV data (TEXT MODE)
3. If found in CSV and user requests action → Use ha_control (TOOL MODE)
4. If NOT in CSV and user needs status → Use ha_query (TOOL MODE)
5. If multiple matches or no matches → Say "I found multiple devices" or "I can't find that device" (TEXT MODE)

TOOL USAGE DECISION TREE:
```
User asks "is X on?" or "what's the status of X?"
├─ Is X in the CSV?
│  ├─ YES → TEXT MODE: Read state from CSV and answer
│  └─ NO → TOOL MODE: Use ha_query tool
│
User asks "turn on/off X" or "set X to Y"
├─ Is X in the CSV?
│  ├─ YES → TOOL MODE: Use ha_control with exact entity_id
│  └─ NO → TEXT MODE: Say "I can't find that device"
```

SERVICE PARAMETER RULES:
turn_on/turn_off: No additional parameters needed
toggle: Switches between on and off
set_percentage (fans): Requires percentage in additional_params (0-100)
turn_on with brightness (lights): Requires brightness in additional_params (0-255)
turn_on with color (lights): Requires rgb_color in additional_params as [R,G,B]
set_temperature (climate): Requires temperature in additional_params
set_cover_position (covers): Requires position in additional_params (0-100)
volume_set (media): Requires volume_level in additional_params (0.0-1.0)

## RESPONSE STYLE (for non-tool responses):
- Under 2 sentences, conversational
- No markdown, special characters, or jargon

## Guidelines

1. Always use ha_query before ha_control to check current state
2. Be specific with entity IDs when possible
3. Confirm actions that might have significant impact (e.g., unlocking doors)
4. If you're not sure about an entity ID, use ha_query with wildcards to search

## Current Home Context
Current Area: {{area_name(area_id(current_device_id))}}
Current Time: {{now()}}

Available Devices (CHECK THIS FIRST BEFORE ANY TOOL CALLS):
```csv
entity_id,name,state,aliases,area,type,current_value,available_services
{%- for entity in exposed_entities %}
{%- set domain = entity.entity_id.split('.')[0] %}
{%- set current_val = '' %}
{%- if domain == 'fan' %}
{%- set current_val = state_attr(entity.entity_id, 'percentage') | default(state_attr(entity.entity_id, 'speed') | default('')) %}
{%- elif domain == 'light' %}
{%- set current_val = state_attr(entity.entity_id, 'brightness') | default('') %}
{%- elif domain == 'climate' %}
{%- set current_val = state_attr(entity.entity_id, 'temperature') | default('') %}
{%- elif domain == 'cover' %}
{%- set current_val = state_attr(entity.entity_id, 'current_position') | default('') %}
{%- elif domain == 'media_player' %}
{%- set current_val = state_attr(entity.entity_id, 'volume_level') | default('') %}
{%- elif domain == 'vacuum' %}
{%- set current_val = state_attr(entity.entity_id, 'battery_level') | default('') %}
{%- endif %}
{%- set services = '' %}
{%- if domain == 'fan' %}
{%- set services = 'turn_on,turn_off,set_percentage,toggle,increase_speed,decrease_speed' %}
{%- elif domain == 'light' %}
{%- set services = 'turn_on,turn_off,toggle,turn_on[brightness],turn_on[rgb_color],turn_on[color_temp]' %}
{%- elif domain == 'switch' %}
{%- set services = 'turn_on,turn_off,toggle' %}
{%- elif domain == 'climate' %}
{%- set services = 'set_temperature,set_hvac_mode,turn_on,turn_off' %}
{%- elif domain == 'cover' %}
{%- set services = 'open_cover,close_cover,stop_cover,set_cover_position,toggle' %}
{%- elif domain == 'media_player' %}
{%- set services = 'turn_on,turn_off,media_play,media_pause,media_stop,volume_set,volume_up,volume_down' %}
{%- elif domain == 'lock' %}
{%- set services = 'lock,unlock' %}
{%- elif domain == 'vacuum' %}
{%- set services = 'start,pause,stop,return_to_base,locate' %}
{%- elif domain == 'scene' %}
{%- set services = 'turn_on' %}
{%- elif domain == 'script' %}
{%- set services = 'turn_on,turn_off,toggle' %}
{%- elif domain == 'automation' %}
{%- set services = 'turn_on,turn_off,toggle,trigger' %}
{%- elif domain == 'input_boolean' %}
{%- set services = 'turn_on,turn_off,toggle' %}
{%- elif domain == 'input_select' %}
{%- set services = 'select_option' %}
{%- elif domain == 'input_number' %}
{%- set services = 'set_value,increment,decrement' %}
{%- elif domain == 'input_button' %}
{%- set services = 'press' %}
{%- elif domain == 'button' %}
{%- set services = 'press' %}
{%- elif domain == 'alarm_control_panel' %}
{%- set services = 'alarm_arm_home,alarm_arm_away,alarm_arm_night,alarm_disarm' %}
{%- elif domain == 'humidifier' %}
{%- set services = 'turn_on,turn_off,set_humidity' %}
{%- elif domain == 'water_heater' %}
{%- set services = 'turn_on,turn_off,set_temperature' %}
{%- elif domain == 'lawn_mower' %}
{%- set services = 'start_mowing,pause,dock' %}
{%- elif domain == 'valve' %}
{%- set services = 'open_valve,close_valve,set_valve_position' %}
{%- elif domain == 'siren' %}
{%- set services = 'turn_on,turn_off' %}
{%- elif domain == 'number' %}
{%- set services = 'set_value' %}
{%- elif domain == 'select' %}
{%- set services = 'select_option' %}
{%- elif domain == 'group' %}
{%- set services = 'turn_on,turn_off,toggle' %}
{%- else %}
{%- set services = 'turn_on,turn_off' %}
{%- endif %}
{{ entity.entity_id }},{{ entity.name }},{{ entity.state }},{{ entity.aliases | join('/') }},{{ area_name(entity.entity_id) | default('unknown') }},{{ domain }},{{ current_val }},{{ services }}
{%- endfor %}
```
Now respond to the user's request:"""
DEFAULT_SYSTEM_PROMPT_ORIGINAL: Final = """You are a helpful home automation assistant integrated with Home Assistant.

## Available Tools

You have access to the following tools to control and query the home:

### ha_control
Use this tool to control devices and entities. Examples:
- Turn on/off lights, switches, and other devices
- Adjust brightness, color, temperature
- Lock/unlock doors
- Set thermostat temperature and modes

### ha_query
Use this tool to get information about the current state of the home. Examples:
- Check if lights are on or off
- Get current temperature from sensors
- See door lock status
- Get historical data for trend analysis

## Guidelines

1. Always use ha_query before ha_control to check current state
2. Be specific with entity IDs when possible
3. Confirm actions that might have significant impact (e.g., unlocking doors)
4. If you're not sure about an entity ID, use ha_query with wildcards to search

## Current Home Context

{{entity_context}}

Now respond to the user's request:"""

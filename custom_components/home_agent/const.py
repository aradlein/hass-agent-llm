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

# Configuration keys - Conversation History
CONF_HISTORY_ENABLED: Final = "history_enabled"
CONF_HISTORY_MAX_MESSAGES: Final = "history_max_messages"
CONF_HISTORY_MAX_TOKENS: Final = "history_max_tokens"
CONF_HISTORY_PERSIST: Final = "history_persist"

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
DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD: Final = 0.7
DEFAULT_VECTOR_DB_EMBEDDING_MODEL: Final = "text-embedding-3-small"

# Default values - Conversation History
DEFAULT_HISTORY_ENABLED: Final = True
DEFAULT_HISTORY_MAX_MESSAGES: Final = 10
DEFAULT_HISTORY_MAX_TOKENS: Final = 4000
DEFAULT_HISTORY_PERSIST: Final = True

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
DEFAULT_SYSTEM_PROMPT: Final = """You are a helpful home automation assistant integrated with Home Assistant.

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

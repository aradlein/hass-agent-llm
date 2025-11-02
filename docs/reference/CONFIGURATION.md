# Configuration Reference

Complete reference for all Home Agent configuration options.

## Table of Contents

- [Overview](#overview)
- [Primary LLM Settings](#primary-llm-settings)
- [Context Settings](#context-settings)
- [History Settings](#history-settings)
- [Tool Settings](#tool-settings)
- [External LLM Settings](#external-llm-settings)
- [Memory System Settings](#memory-system-settings)
- [Streaming Settings](#streaming-settings)
- [Debug Settings](#debug-settings)
- [Configuration Examples](#configuration-examples)

---

## Overview

Home Agent configuration is managed through Home Assistant's configuration UI. Access it via:

**Settings → Devices & Services → Home Agent → Configure**

Configuration is organized into logical sections using a menu-based interface. All settings have sensible defaults, so you can start with minimal configuration and add features as needed.

### Configuration Storage

- Settings are stored in Home Assistant's config entries
- Custom tools are defined in `configuration.yaml`
- Conversation history persists in `.storage/home_agent.history` (if enabled)
- Memories persist in `.storage/home_agent.memories` (if memory system enabled)

---

## Primary LLM Settings

Settings for the main LLM that handles conversations and tool execution.

### CONF_LLM_BASE_URL

**Type:** String (URL)
**Required:** Yes
**Description:** Base URL for the OpenAI-compatible LLM API endpoint

**Valid Values:**
- OpenAI: `https://api.openai.com/v1`
- Ollama: `http://localhost:11434/v1`
- LocalAI: `http://localhost:8080/v1`
- LM Studio: `http://localhost:1234/v1`
- Any OpenAI-compatible endpoint

**Example:**
```yaml
llm_base_url: "https://api.openai.com/v1"
```

---

### CONF_LLM_API_KEY

**Type:** String
**Required:** Yes (for most providers)
**Description:** API key for authenticating with the LLM provider

**Notes:**
- Keep this secret and secure
- OpenAI keys start with `sk-`
- Local models (Ollama) may not require a key
- Can be left empty for unauthenticated endpoints

**Example:**
```yaml
llm_api_key: "sk-proj-abc123..."
```

---

### CONF_LLM_MODEL

**Type:** String
**Required:** Yes
**Default:** `gpt-4o-mini`

**Description:** Name of the LLM model to use

**Common Values:**
- **OpenAI:**
  - `gpt-4o` - Most capable, slower, more expensive
  - `gpt-4o-mini` - Fast, cost-effective, recommended
  - `gpt-3.5-turbo` - Fastest, cheapest, basic capabilities
- **Ollama:**
  - `llama2` - General purpose
  - `mistral` - Fast and capable
  - `codellama` - Code-focused
  - Run `ollama list` to see installed models
- **Local Models:** Check your provider's documentation

**Example:**
```yaml
llm_model: "gpt-4o-mini"
```

---

### CONF_LLM_TEMPERATURE

**Type:** Float
**Required:** No
**Default:** `0.7`
**Range:** `0.0 - 2.0`

**Description:** Controls randomness in LLM responses

**Guidelines:**
- `0.0` - Deterministic, consistent (good for tool calling)
- `0.3-0.5` - Mostly focused, slight variation
- `0.7` - Balanced (default, recommended)
- `1.0` - Creative, varied responses
- `1.5-2.0` - Very random (not recommended for home automation)

**Example:**
```yaml
llm_temperature: 0.7
```

---

### CONF_LLM_MAX_TOKENS

**Type:** Integer
**Required:** No
**Default:** `500`
**Range:** `1 - model_max`

**Description:** Maximum tokens in LLM response

**Guidelines:**
- `150-300` - Short, concise responses (recommended for voice)
- `500` - Balanced (default)
- `1000+` - Detailed explanations
- Higher values = slower responses + higher costs

**Notes:**
- Does not include prompt tokens
- Model limits vary (GPT-4: 8192+, GPT-3.5: 4096)
- Affects response latency and cost

**Example:**
```yaml
llm_max_tokens: 500
```

---

### CONF_LLM_TOP_P

**Type:** Float
**Required:** No
**Default:** `1.0`
**Range:** `0.0 - 1.0`

**Description:** Nucleus sampling parameter (alternative to temperature)

**Guidelines:**
- `1.0` - Consider all tokens (default, recommended)
- `0.9` - Slightly more focused
- `0.5` - Very focused, limited vocabulary

**Notes:**
- Generally recommended to leave at default
- Use temperature for controlling randomness instead

**Example:**
```yaml
llm_top_p: 1.0
```

---

## Context Settings

Settings for how entity context is injected into conversations.

### CONF_CONTEXT_MODE

**Type:** Select
**Required:** No
**Default:** `"direct"`

**Description:** Strategy for entity context injection

**Options:**
- `direct` - Include configured entities directly in every prompt
- `vector_db` - Use ChromaDB to retrieve relevant entities based on query

**Comparison:**

| Feature | Direct Mode | Vector DB Mode |
|---------|-------------|----------------|
| Setup complexity | Simple | Requires ChromaDB |
| Token efficiency | Lower | Higher |
| Relevance | All configured entities | Only relevant entities |
| Performance | Fast | Slightly slower (query overhead) |
| Best for | Small entity lists | Large entity lists |

**Example:**
```yaml
context_mode: "direct"
```

---

### CONF_CONTEXT_FORMAT

**Type:** Select
**Required:** No
**Default:** `"json"`

**Description:** Format for entity context in prompt

**Options:**
- `json` - Structured JSON format (precise, machine-readable)
- `natural_language` - Human-readable descriptions
- `hybrid` - JSON with natural language annotations

**Example:**
```yaml
context_format: "json"
```

**Format Examples:**

**JSON:**
```json
{
  "entity_id": "light.living_room",
  "state": "on",
  "attributes": {"brightness": 128}
}
```

**Natural Language:**
```
The living room light is on with brightness at 50%.
```

---

### CONF_CONTEXT_ENTITIES

**Type:** List of entity configurations
**Required:** No (when using direct mode)
**Default:** `[]`

**Description:** List of entities to include in direct context mode

**Format:**
```yaml
context_entities:
  - entity_id: sensor.living_room_temperature
    attributes:
      - temperature
      - unit_of_measurement
  - entity_id: light.living_room
    attributes:
      - state
      - brightness
  - entity_id: climate.*  # Wildcards supported
```

**Notes:**
- Only used in direct mode
- Supports wildcards (`*`)
- Empty attributes list includes all attributes
- More entities = more tokens = slower/costlier responses

---

### Vector DB Settings (CONF_VECTOR_DB_*)

Only applicable when `context_mode = "vector_db"`

#### CONF_VECTOR_DB_ENABLED

**Type:** Boolean
**Default:** `false`

#### CONF_VECTOR_DB_HOST

**Type:** String
**Default:** `"localhost"`

**Description:** ChromaDB server host

#### CONF_VECTOR_DB_PORT

**Type:** Integer
**Default:** `8000`

**Description:** ChromaDB server port

#### CONF_VECTOR_DB_COLLECTION

**Type:** String
**Default:** `"home_entities"`

**Description:** ChromaDB collection name for entities

#### CONF_VECTOR_DB_TOP_K

**Type:** Integer
**Default:** `5`
**Range:** `1-50`

**Description:** Number of most relevant entities to retrieve

#### CONF_VECTOR_DB_SIMILARITY_THRESHOLD

**Type:** Float
**Default:** `250.0`

**Description:** L2 distance threshold for relevance (lower = more similar)

#### CONF_VECTOR_DB_EMBEDDING_MODEL

**Type:** String
**Default:** `"text-embedding-3-small"`

**Description:** Embedding model for vectorization

**Common Values:**
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
- Ollama: Model-specific

#### CONF_VECTOR_DB_EMBEDDING_PROVIDER

**Type:** Select
**Default:** `"ollama"`

**Options:**
- `openai` - Use OpenAI embeddings
- `ollama` - Use local Ollama embeddings

#### CONF_VECTOR_DB_EMBEDDING_BASE_URL

**Type:** String
**Default:** `"http://localhost:11434"`

**Description:** Base URL for embedding provider (Ollama)

---

## History Settings

Settings for conversation history management.

### CONF_HISTORY_ENABLED

**Type:** Boolean
**Default:** `true`

**Description:** Enable conversation history tracking

**Effects:**
- When enabled: Previous messages included in context
- When disabled: Each message is independent (no memory of conversation)

---

### CONF_HISTORY_MAX_MESSAGES

**Type:** Integer
**Default:** `10`
**Range:** `1-100`

**Description:** Maximum conversation turns to retain

**Guidelines:**
- `5-10` - Short-term context (recommended)
- `10-20` - Medium-term context
- `20+` - Long conversations (increases token usage)

**Notes:**
- One turn = user message + assistant response
- Higher values = better context but more tokens

---

### CONF_HISTORY_MAX_TOKENS

**Type:** Integer
**Default:** `4000`
**Range:** `100-32000`

**Description:** Maximum tokens for conversation history (alternative limit to max_messages)

**Notes:**
- Acts as a ceiling when max_messages would exceed this
- Prevents context window overflow
- Adjust based on model's context window

---

### CONF_HISTORY_PERSIST

**Type:** Boolean
**Default:** `true`

**Description:** Save conversation history across Home Assistant restarts

**Effects:**
- When enabled: History survives restarts (stored in `.storage/`)
- When disabled: History cleared on restart (memory-only)

---

### Context Optimization Settings

#### CONF_COMPRESSION_LEVEL

**Type:** Select
**Default:** `"medium"`

**Options:**
- `none` - No compression
- `low` - Minimal compression
- `medium` - Balanced compression
- `high` - Aggressive compression

**Description:** Level of context compression to apply when approaching token limits

---

#### CONF_PRESERVE_RECENT_MESSAGES

**Type:** Integer
**Default:** `3`

**Description:** Number of recent messages to never compress/summarize

---

#### CONF_SUMMARIZATION_ENABLED

**Type:** Boolean
**Default:** `false`

**Description:** Use LLM to summarize old conversation history

**Notes:**
- Requires additional LLM calls (costs tokens)
- Helps maintain context in long conversations
- Experimental feature

---

## Tool Settings

Settings for tool execution and custom tools.

### CONF_TOOLS_ENABLE_NATIVE

**Type:** Boolean
**Default:** `true`

**Description:** Enable built-in tools (ha_control, ha_query)

**Notes:**
- Disabling removes core functionality
- Only disable for testing or specific use cases

---

### CONF_TOOLS_MAX_CALLS_PER_TURN

**Type:** Integer
**Default:** `5`
**Range:** `1-20`

**Description:** Maximum tool calls allowed per conversation turn

**Guidelines:**
- `3-5` - Normal usage (recommended)
- `5-10` - Complex automations
- `10+` - Advanced scenarios (risk of loops)

**Purpose:**
- Prevents infinite loops
- Controls costs and latency
- Enforces reasonable complexity

---

### CONF_TOOLS_TIMEOUT

**Type:** Integer (seconds)
**Default:** `30`
**Range:** `5-300`

**Description:** Timeout for individual tool execution

**Guidelines:**
- `10-30` - Normal tools (default)
- `60+` - Slow external APIs
- `5-10` - Fast-only mode

---

### CONF_TOOLS_CUSTOM

**Type:** List
**Default:** `[]`
**Location:** `configuration.yaml`

**Description:** Custom tool definitions

**Format:**
```yaml
home_agent:
  custom_tools:
    - name: check_weather
      description: "Get weather forecast for location"
      parameters:
        type: object
        properties:
          location:
            type: string
            description: "City name"
      handler:
        type: rest
        url: "https://api.weather.com/v1/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ weather_api_key }}"

    - name: trigger_automation
      description: "Trigger a Home Assistant automation"
      parameters:
        type: object
        properties:
          automation_id:
            type: string
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: "{{ automation_id }}"
```

**Supported Handler Types:**
- `rest` - HTTP API calls
- `service` - Home Assistant service calls

**Notes:**
- Defined in `configuration.yaml` under `home_agent:` key
- Validated on integration setup
- Errors prevent integration from loading

---

## External LLM Settings

Settings for optional external LLM tool (dual-LLM strategy).

### CONF_EXTERNAL_LLM_ENABLED

**Type:** Boolean
**Default:** `false`

**Description:** Enable external LLM as a tool for the primary LLM

**Use Case:**
- Primary LLM (fast/local) handles tool execution
- Primary can delegate complex queries to external LLM (GPT-4, Claude)
- Best of both worlds: speed + capability

---

### CONF_EXTERNAL_LLM_BASE_URL

**Type:** String (URL)
**Required:** If external LLM enabled

**Description:** Base URL for external LLM API

**Example:**
```yaml
external_llm_base_url: "https://api.openai.com/v1"
```

---

### CONF_EXTERNAL_LLM_API_KEY

**Type:** String
**Required:** If external LLM enabled

**Description:** API key for external LLM

**Notes:**
- Can be different from primary LLM key
- Allows using different providers

---

### CONF_EXTERNAL_LLM_MODEL

**Type:** String
**Default:** `"gpt-4o"`

**Description:** Model name for external LLM

**Recommended:**
- `gpt-4o` - OpenAI's most capable model
- `claude-3-opus` - Anthropic (if using compatible endpoint)
- Use most capable model available

---

### CONF_EXTERNAL_LLM_TEMPERATURE

**Type:** Float
**Default:** `0.8`
**Range:** `0.0-2.0`

**Description:** Temperature for external LLM

**Notes:**
- Default is higher (0.8) for more creative responses
- External LLM typically used for analysis/explanation

---

### CONF_EXTERNAL_LLM_MAX_TOKENS

**Type:** Integer
**Default:** `1000`

**Description:** Maximum tokens for external LLM response

**Notes:**
- Higher than primary LLM default (500)
- External LLM used for detailed responses

---

### CONF_EXTERNAL_LLM_TOOL_DESCRIPTION

**Type:** String (multiline)
**Default:** See below

**Description:** Description shown to primary LLM about when to use external LLM

**Default Value:**
```
Use this when you need help with complex analysis, detailed explanations,
or comprehensive recommendations beyond simple home control.
```

**Customization:**
```yaml
external_llm_tool_description: |
  Use this tool when:
  - User asks for detailed analysis
  - Complex reasoning required
  - Need comprehensive recommendations
  - Unsure how to help user
```

---

### CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT

**Type:** Boolean
**Default:** `true`

**Description:** Automatically include entity context when calling external LLM

**Notes:**
- When true: Entity context passed to external LLM
- When false: Only explicit prompt and context parameter passed

---

### HTTP_TIMEOUT_EXTERNAL_LLM

**Type:** Integer (seconds)
**Default:** `90`

**Description:** Timeout for external LLM calls

**Notes:**
- Higher than primary LLM timeout (60s)
- Accounts for potentially slower external APIs

---

## Memory System Settings

Settings for long-term memory and learning.

### CONF_MEMORY_ENABLED

**Type:** Boolean
**Default:** `true`

**Description:** Enable the memory system

**Effects:**
- When enabled: Memories extracted and recalled
- When disabled: No memory persistence (privacy mode)

**Privacy Note:** User-controlled on/off for entire memory system

---

### CONF_MEMORY_EXTRACTION_ENABLED

**Type:** Boolean
**Default:** `true`

**Description:** Enable automatic memory extraction from conversations

**Notes:**
- Requires CONF_MEMORY_ENABLED = true
- Extraction happens after each conversation
- Uses configured extraction LLM

---

### CONF_MEMORY_EXTRACTION_LLM

**Type:** Select
**Default:** `"external"`

**Options:**
- `external` - Use external LLM for extraction (more capable)
- `local` - Use primary/local LLM for extraction (faster, free)

**Notes:**
- External requires CONF_EXTERNAL_LLM_ENABLED = true
- External typically produces better quality extractions

---

### CONF_MEMORY_MAX_MEMORIES

**Type:** Integer
**Default:** `100`
**Range:** `10-10000`

**Description:** Maximum number of memories to store

**Guidelines:**
- `50-100` - Personal home (default)
- `100-500` - Large home or detailed tracking
- `500+` - Enterprise or extensive history

**Notes:**
- Oldest/least important memories purged when limit reached
- ChromaDB can handle thousands efficiently

---

### CONF_MEMORY_MIN_IMPORTANCE

**Type:** Float
**Default:** `0.3`
**Range:** `0.0-1.0`

**Description:** Minimum importance score for memory recall

**Guidelines:**
- `0.0` - Include all memories
- `0.3` - Filter trivial memories (default, recommended)
- `0.5` - Only moderately important memories
- `0.7+` - Only critical memories

---

### CONF_MEMORY_COLLECTION_NAME

**Type:** String
**Default:** `"home_agent_memories"`

**Description:** ChromaDB collection name for memories

**Notes:**
- Change if running multiple instances
- Must be valid ChromaDB collection name

---

### CONF_MEMORY_IMPORTANCE_DECAY

**Type:** Float
**Default:** `0.0`
**Range:** `0.0-1.0`

**Description:** Optional importance decay rate over time

**Examples:**
- `0.0` - No decay (default)
- `0.01` - Slow decay
- `0.1` - Memories fade quickly

**Notes:**
- Experimental feature
- Decay applied periodically based on age

---

### CONF_MEMORY_DEDUP_THRESHOLD

**Type:** Float
**Default:** `0.95`
**Range:** `0.0-1.0`

**Description:** Similarity threshold for deduplication

**Guidelines:**
- `0.95` - Very similar memories deduplicated (default)
- `0.85` - More aggressive dedup
- `1.0` - Only exact duplicates

---

### CONF_MEMORY_CONTEXT_TOP_K

**Type:** Integer
**Default:** `5`
**Range:** `1-20`

**Description:** Number of relevant memories to inject into context

**Guidelines:**
- `3-5` - Balanced (default)
- `5-10` - More context
- `1-3` - Minimal token usage

---

### Memory TTL Settings

Control how long different memory types persist.

#### CONF_MEMORY_EVENT_TTL

**Type:** Integer (seconds) or None
**Default:** `300` (5 minutes)

**Description:** Time-to-live for event memories

**Notes:**
- Events are transient by nature
- Short TTL prevents clutter

#### CONF_MEMORY_FACT_TTL

**Type:** Integer (seconds) or None
**Default:** `None` (no expiration)

**Description:** Time-to-live for fact memories

**Notes:**
- Facts should persist indefinitely

#### CONF_MEMORY_PREFERENCE_TTL

**Type:** Integer (seconds) or None
**Default:** `7776000` (90 days)

**Description:** Time-to-live for preference memories

**Notes:**
- Preferences may change over time
- 90 days allows for seasonal/habit changes

---

### CONF_MEMORY_CLEANUP_INTERVAL

**Type:** Integer (seconds)
**Default:** `300` (5 minutes)

**Description:** How often to run memory cleanup task

**Notes:**
- Removes expired memories
- Updates importance scores
- Runs in background

---

## Streaming Settings

Settings for real-time response streaming.

### CONF_STREAMING_ENABLED

**Type:** Boolean
**Default:** `false`

**Description:** Enable streaming responses via Home Assistant's assist pipeline

**Requirements:**
- Wyoming TTS integration
- Voice assistant pipeline configured
- ChatLog support in conversation platform

**Effects:**
- When enabled: Audio starts playing before complete response
- When disabled: Traditional synchronous responses

**Performance:**
- First audio chunk: ~500ms (vs 5+ seconds synchronous)
- Better user experience for voice interactions

**Notes:**
- Automatically falls back to synchronous on errors
- Only works with voice assistant, not service calls
- Experimental feature

---

## Debug Settings

Settings for debugging and monitoring.

### CONF_DEBUG_LOGGING

**Type:** Boolean
**Default:** `false`

**Description:** Enable verbose debug logging

**Log Output:**
- LLM request/response details
- Tool execution parameters and results
- Context injection details
- Memory extraction process
- Token usage statistics
- Performance metrics

**Notes:**
- Generates significant log volume
- May expose sensitive data in logs
- Use temporarily for troubleshooting

---

### CONF_EMIT_EVENTS

**Type:** Boolean
**Default:** `true`

**Description:** Fire Home Assistant events for monitoring and automation

**Events Fired:**
- `home_agent.conversation.started`
- `home_agent.conversation.finished`
- `home_agent.tool.executed`
- `home_agent.tool.progress`
- `home_agent.context.injected`
- `home_agent.memory.extracted`
- `home_agent.error`
- And more...

**Use Cases:**
- Performance monitoring
- Error tracking
- Usage analytics
- Automation triggers

---

## Configuration Examples

### Minimal Configuration

Basic setup for getting started quickly.

```yaml
# Configuration UI
llm_base_url: "https://api.openai.com/v1"
llm_api_key: "sk-..."
llm_model: "gpt-4o-mini"

# Defaults for everything else
```

**Features:**
- Direct context mode (no vector DB)
- Basic conversation history
- Native tools only
- No external LLM
- Memory system enabled

---

### Recommended Configuration

Balanced setup for most users.

```yaml
# Primary LLM
llm_base_url: "https://api.openai.com/v1"
llm_api_key: "sk-..."
llm_model: "gpt-4o-mini"
llm_temperature: 0.7
llm_max_tokens: 500

# Context
context_mode: "direct"
context_format: "json"

# History
history_enabled: true
history_max_messages: 10
history_persist: true

# Tools
tools_enable_native: true
tools_max_calls_per_turn: 5
tools_timeout: 30

# Memory
memory_enabled: true
memory_extraction_enabled: true
memory_extraction_llm: "local"  # Use primary LLM for free extraction
memory_max_memories: 100
memory_min_importance: 0.3

# Debug
debug_logging: false
emit_events: true
```

**Best For:**
- Home users
- Cost-conscious setups
- Simple to moderate complexity

---

### Advanced Configuration (All Features)

Full-featured setup with all capabilities enabled.

```yaml
# Primary LLM (Local/Fast)
llm_base_url: "http://localhost:11434/v1"
llm_api_key: ""  # Ollama doesn't need key
llm_model: "mistral"
llm_temperature: 0.5  # Lower for more consistent tool calling
llm_max_tokens: 300  # Shorter for faster responses

# External LLM (Cloud/Capable)
external_llm_enabled: true
external_llm_base_url: "https://api.openai.com/v1"
external_llm_api_key: "sk-..."
external_llm_model: "gpt-4o"
external_llm_temperature: 0.8
external_llm_max_tokens: 1000

# Vector DB Context
context_mode: "vector_db"
vector_db_enabled: true
vector_db_host: "localhost"
vector_db_port: 8000
vector_db_collection: "home_entities"
vector_db_top_k: 5
vector_db_embedding_provider: "ollama"
vector_db_embedding_base_url: "http://localhost:11434"

# History with Optimization
history_enabled: true
history_max_messages: 20
history_max_tokens: 6000
history_persist: true
compression_level: "medium"
preserve_recent_messages: 3
summarization_enabled: true

# Memory System (Full)
memory_enabled: true
memory_extraction_enabled: true
memory_extraction_llm: "external"  # Use GPT-4 for quality
memory_max_memories: 500
memory_min_importance: 0.2
memory_context_top_k: 10
memory_importance_decay: 0.01
memory_dedup_threshold: 0.95

# Streaming
streaming_enabled: true

# Debug
debug_logging: true
emit_events: true
```

**Custom Tools (configuration.yaml):**
```yaml
home_agent:
  custom_tools:
    - name: get_weather
      description: "Get weather forecast"
      parameters:
        type: object
        properties:
          location:
            type: string
      handler:
        type: rest
        url: "https://api.openweathermap.org/data/2.5/weather"
        method: GET
        params:
          q: "{{ location }}"
          appid: "{{ weather_api_key }}"
```

**Best For:**
- Power users
- Complex automations
- Large homes with many entities
- Users wanting maximum capability

---

### Performance-Optimized Configuration

Optimized for speed and low latency.

```yaml
# Fast Local LLM
llm_base_url: "http://localhost:11434/v1"
llm_api_key: ""
llm_model: "mistral"  # Fast local model
llm_temperature: 0.3  # Lower for faster, more deterministic
llm_max_tokens: 150  # Short responses only

# Minimal Context
context_mode: "direct"
context_format: "json"
# Keep entity list small

# Reduced History
history_enabled: true
history_max_messages: 5  # Minimal history
history_max_tokens: 2000
history_persist: false  # Memory-only for speed

# Fast Tools
tools_max_calls_per_turn: 3  # Limit iterations
tools_timeout: 15  # Fast timeout

# Memory Disabled (Optional)
memory_enabled: false  # Disable for maximum speed

# No External LLM
external_llm_enabled: false

# Streaming for Responsiveness
streaming_enabled: true

# Minimal Events
emit_events: false  # Skip event overhead
debug_logging: false
```

**Performance Targets:**
- Simple queries: < 2 seconds
- Tool execution: < 5 seconds
- Minimal token costs

**Best For:**
- Voice assistant use cases
- Latency-sensitive applications
- Resource-constrained systems

---

### Privacy-Focused Configuration

Maximum privacy and data control.

```yaml
# Local LLM Only
llm_base_url: "http://localhost:11434/v1"
llm_api_key: ""
llm_model: "llama2"
llm_temperature: 0.7
llm_max_tokens: 500

# No External Services
external_llm_enabled: false  # No cloud LLMs
vector_db_enabled: false  # Local-only
context_mode: "direct"

# Local History Only
history_enabled: true
history_max_messages: 10
history_persist: true  # Stored locally in HA

# Memory - User Controlled
memory_enabled: true  # Can be disabled by user
memory_extraction_enabled: true
memory_extraction_llm: "local"  # Use local LLM
memory_max_memories: 100

# No Custom Tools (Optional)
# Only use built-in tools to avoid external API calls

# Events for Transparency
emit_events: true  # User can monitor all activity
debug_logging: false
```

**Privacy Features:**
- No data leaves local network
- All processing on local hardware
- User control over memory system
- Full transparency via events
- No third-party API calls

**Best For:**
- Privacy-conscious users
- Offline/air-gapped systems
- GDPR compliance requirements
- Complete data sovereignty

---

## Configuration Validation

Home Agent validates configuration on setup and updates. Common validation rules:

**URLs:**
- Must be valid HTTP/HTTPS URLs
- Should be accessible from Home Assistant

**Numeric Ranges:**
- Temperature: 0.0 - 2.0
- Top P: 0.0 - 1.0
- Max tokens: > 0
- History max messages: > 0

**Required Fields:**
- `llm_base_url` - Always required
- `llm_model` - Always required
- `llm_api_key` - Required for most providers
- External LLM fields - Required if external LLM enabled

**Dependencies:**
- Vector DB settings require `context_mode = "vector_db"`
- Memory extraction requires `memory_enabled = true`
- External LLM tool requires `external_llm_enabled = true`
- Streaming requires Wyoming TTS integration

---

## Troubleshooting Configuration

**Configuration Not Saving:**
1. Check for validation errors in logs
2. Verify all required fields are present
3. Ensure numeric values are in valid ranges
4. Check YAML syntax in `configuration.yaml`

**Integration Won't Load:**
1. Check custom tools are valid YAML
2. Verify handler types are supported
3. Review logs for specific validation errors
4. Try minimal configuration first

**Settings Not Taking Effect:**
1. Restart Home Assistant after changes
2. Clear conversation history after config changes
3. Check active config entry is being used
4. Verify changes saved successfully

**Getting Help:**
- See [TROUBLESHOOTING.md](/workspaces/home-agent/docs/TROUBLESHOOTING.md) for detailed debugging
- Check logs with debug logging enabled
- Review event data for insights
- Test with minimal configuration first

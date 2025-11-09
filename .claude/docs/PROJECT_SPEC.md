# Home Agent - Comprehensive Project Specification

## Project Overview

**Project Name:** Home Agent
**Base Reference:** Extended OpenAI Conversation (jekalmin/extended_openai_conversation)
**Purpose:** A highly customizable Home Assistant custom component that extends conversational AI capabilities with advanced function execution, external integrations, and intelligent automation management.

---

## Development Progress & Roadmap

**Last Updated:** 2025-11-05
**Current Version:** v0.4.7-beta
**Current Phase:** Phase 7 (Documentation & Production Readiness)

### Completed Phases ✅

#### Phase 1: Foundation (MVP) ✅ **COMPLETE**
**Status:** Production-ready | **Test Coverage:** >80% | **Tests:** 376+

- ✅ Core architecture and component setup
- ✅ Home Assistant ConversationEntity implementation
- ✅ OpenAI-compatible API client integration
- ✅ Direct entity context injection
- ✅ Basic conversation history management
- ✅ Core tools: `ha_control` and `ha_query`
- ✅ Default system prompt template
- ✅ Configuration flow (UI-based setup)
- ✅ Basic error handling and exception system
- ✅ Comprehensive test coverage

**Key Files:** `agent.py`, `tools/ha_control.py`, `tools/ha_query.py`, `config_flow.py`, `context_manager.py`, `conversation.py`, `tool_handler.py`, `exceptions.py`

---

#### Phase 2: Enhanced Context & History ✅ **COMPLETE**
**Status:** Production-ready | **Test Coverage:** >80%

- ✅ Vector DB (ChromaDB) integration for semantic search
- ✅ Configurable context strategies (direct vs vector DB)
- ✅ Advanced history management (token limits, persistence)
- ✅ Context optimization and compression
- ✅ Event system with detailed metrics
- ✅ Enhanced configuration UI with menu-based options
- ✅ Additional ChromaDB collections support (v0.4.7-beta)
  - Two-tier ranking system (entity priority + supplementary context)
  - Multi-collection querying with intelligent merging
  - Separate thresholds and top K for additional collections
  - Graceful error handling for non-existent collections

**Key Files:** `vector_db_manager.py`, `context_optimizer.py`, `context_providers/vector_db.py`
**Services:** `reindex_entities`, `index_entity`
**Events:** `EVENT_CONTEXT_OPTIMIZED`, `EVENT_HISTORY_SAVED`, `EVENT_VECTOR_DB_QUERIED`

---

#### Phase 3: External LLM & Custom Tools ✅ **COMPLETE**
**Status:** Production-ready | **Test Coverage:** 95.58%

- ✅ External LLM tool (`query_external_llm`) for delegation
- ✅ Custom tool framework (REST + Service handlers)
- ✅ Configuration via `configuration.yaml`
- ✅ YAML schema validation on integration setup
- ✅ Standardized response format for all tools
- ✅ Comprehensive error handling

**Key Files:** `tools/external_llm.py`, `tools/custom_tools.py`
**Documentation:** `docs/CUSTOM_TOOLS.md`, `docs/EXTERNAL_LLM.md`
**Tests:** 76 unit + 23 integration tests

---

#### Phase 3.5: Long-Term Memory System ✅ **COMPLETE**
**Status:** Production-ready | **Test Coverage:** >80%

- ✅ Core memory manager with persistent storage
- ✅ ChromaDB indexing for semantic search
- ✅ Automatic memory extraction from conversations
- ✅ **Memory quality validation** - Multi-layer filtering to reject low-value content
- ✅ Manual memory tools (`store_memory`, `recall_memory`)
- ✅ Memory context provider for enhanced responses
- ✅ Memory management UI and services
- ✅ Configurable LLM selection for extraction

**Key Files:** `memory_manager.py`, `context_providers/memory.py`, `tools/memory_tools.py`
**Services:** `list_memories`, `delete_memory`, `clear_memories`, `search_memories`, `add_memory`
**Storage:** `.storage/home_agent.memories`
**Tests:** 7 memory test files (includes quality validation tests)

---

#### Phase 4: Streaming Response Support ✅ **COMPLETE**
**Status:** Production-ready | **Test Coverage:** >80%

- ✅ Migration from `async_process()` to `_async_handle_message()` API
- ✅ LLM response streaming using aiohttp SSE
- ✅ `chat_log.async_add_delta_content_stream()` integration
- ✅ Tool progress indicators (`EVENT_TOOL_PROGRESS`)
- ✅ Streaming + tool call integration (pause/resume)
- ✅ Configuration toggle (default: disabled)
- ✅ Comprehensive testing (unit + integration)
- ✅ Fallback to synchronous on errors
- ✅ Error event emission (`EVENT_STREAMING_ERROR`)

**Key Files:** `streaming.py`, `agent.py`, `tool_handler.py`
**Performance:** First audio chunk ~500ms (vs 5+ seconds synchronous)
**Tests:** 16 unit + 10 integration tests
**Documentation:** `docs/MANUAL_TESTING_ISSUE10.md`

---

### Current Phase: Phase 7 - Documentation & Production Ready 🚧

**Status:** In Progress | **Priority:** High
**Target:** Complete user-facing documentation for public release

#### Completed ✅
- ✅ 9 comprehensive documentation files created
  - `docs/INSTALLATION.md` - Installation and setup guide
  - `docs/VECTOR_DB_SETUP.md` - ChromaDB configuration
  - `docs/MEMORY_SYSTEM.md` - Memory system guide
  - `docs/TROUBLESHOOTING.md` - Problem-solving guide
  - `docs/API_REFERENCE.md` - Services and events reference
  - `docs/CONFIGURATION.md` - Configuration options
  - `docs/EXAMPLES.md` - Automation examples
  - `docs/MIGRATION.md` - Migration from extended_openai_conversation
  - `docs/FAQ.md` - Frequently asked questions
- ✅ README.md restructured (user-focused, removed roadmap tracking)
- ✅ PROJECT_SPEC.md enhanced with progress tracking (this document)

#### In Progress 🚧
- [ ] Review and validate all generated documentation
- [ ] Test all code examples in documentation
- [ ] Add screenshots/diagrams where helpful
- [ ] Spellcheck and proofread all documents
- [ ] Cross-reference validation between docs

#### Remaining Tasks 📋
- [ ] Security audit (code review, dependency check, privacy review)
- [ ] HACS integration preparation
  - [ ] Repository structure validation
  - [ ] manifest.json compliance check
  - [ ] Version tagging strategy
  - [ ] Changelog maintenance
- [ ] Beta testing program
  - [ ] Recruit beta testers
  - [ ] Gather feedback
  - [ ] Identify edge cases
- [ ] Community engagement
  - [ ] Home Assistant forums thread
  - [ ] Release announcements
- [ ] Initial release (v1.0.0) preparation
  - [ ] Final review of all documentation
  - [ ] Release notes creation
  - [ ] Known issues documentation

---

### Planned Phases (Not Started)

#### Phase 5: MCP Server Integration 📅
**Status:** Planned | **Priority:** Medium

- [ ] MCP (Model Context Protocol) server support
- [ ] MCP server handler type for custom tools
- [ ] Configuration for MCP server endpoints
- [ ] Authentication and authorization for MCP servers
- [ ] Read-only data collection (no local execution)
- [ ] Security validation and rate limiting
- [ ] MCP integration tests and documentation

**Target:** Connect to external MCP servers for enhanced data collection

---

#### Phase 6: Reliability & Resource Management 📅
**Status:** Planned | **Priority:** Medium | **Target:** v1.0.0

**Estimated Effort:** 11-15 hours (revised from 28-36 hours)

- [ ] **Resource Management** (3-4 hours)
  - [ ] Conversation history cleanup
  - [ ] Memory-efficient storage
  - [ ] Connection pooling
  - [ ] Proper async patterns
- [ ] **Graceful Degradation** (4-5 hours)
  - [ ] ChromaDB fallback to direct mode
  - [ ] Memory system failure handling
  - [ ] Ollama timeout handling
  - [ ] Partial feature degradation
- [ ] **Connection Reliability** (2-3 hours)
  - [ ] Simple retry logic (2 attempts, 1s delay)
  - [ ] Timeout handling
  - [ ] Health checks
- [ ] **User-Friendly Errors** (2-3 hours)
  - [ ] Clear error messages
  - [ ] Helpful suggestions
  - [ ] Proper error propagation

**Target:** Reliable, resource-efficient operation on Home Assistant hardware

**Note:** Enterprise-scale features (caching, rate limiting quotas, performance metrics, circuit breakers) explicitly excluded as inappropriate for single-instance Home Assistant deployments.

---

### Success Metrics

#### Current Status (as of 2025-11-02)

- **Code Quality:** ✅ **400+ tests passing**, >80% coverage
  - Phase 1-2: 376+ unit tests
  - Phase 3: 95.58% coverage (76 unit + 23 integration)
  - Phase 3.5: 6+ memory test files
  - Phase 4: 26 streaming tests
- **Functionality:** ✅ Phases 1-4 + 3.5 complete and tested
- **Performance:** ✅ **Good**
  - Streaming: ~500ms first audio (✅ Phase 4)
  - Typical queries: 1-2s with local LLM
  - Performance adequate for household usage
- **Documentation:** 🚧 **In Progress** (Phase 7)
  - 9 comprehensive docs created
  - README restructured
  - PROJECT_SPEC enhanced
  - Validation pending
- **Reliability:** ⏳ **Needs Work**
  - Basic error handling ✅
  - Graceful degradation ⏳ (Phase 6)
  - Resource management ⏳ (Phase 6)
  - Connection reliability ⏳ (Phase 6)
- **Security:** ⏳ **In Progress**
  - Entity access control ✅
  - API key handling ✅
  - Security audit ⏳ (Phase 7)
- **Release Readiness:** 🚧 **In Progress**
  - HACS integration ⏳ (Phase 7)
  - Beta testing ⏳ (Phase 7)

#### Target Metrics (v1.0.0 Release)

- **Functionality:** All core features working reliably
- **Performance:**
  - < 2s response for simple queries
  - < 5s response for complex queries
  - > 50% cache hit rate (Phase 6)
- **Reliability:** > 99% uptime
- **User Satisfaction:** Positive community feedback
- **Adoption:**
  - HACS installation available
  - 100+ installations within 3 months
  - Active community engagement
- **Code Quality:** Maintain > 80% test coverage
- **Security:** No critical vulnerabilities

---

## 1. Executive Summary

This project creates a Home Assistant custom component that integrates with Home Assistant's native LLM API to provide intelligent, context-aware home automation through natural language. The component acts as a bridge between HA's conversation platform and any OpenAI-compatible LLM, providing smart context injection, conversation history management, and flexible tool calling capabilities.

### Key Differentiators
- **Native HA LLM API Integration** - Uses Home Assistant's built-in LLM conversation platform
- **Flexible Context Injection** - Toggle between direct entity inclusion or vector DB (Chroma) retrieval
- **Smart Conversation Memory** - Configurable chat history automatically added as context
- **Generic Tool Calling** - Simple, LLM-friendly tool definitions for HA operations
- **Dual LLM Strategy** - Primary LLM for tool execution + optional external LLM for comprehensive responses
- **OpenAI-Compatible** - Works with any OpenAI-compatible endpoint (OpenAI, local models, etc.)

---

## 2. Core Architecture

### 2.1 Component Structure

```
custom_components/home_agent/
├── __init__.py              # Component initialization and setup
├── manifest.json            # Component metadata and dependencies
├── config_flow.py           # Configuration UI implementation
├── const.py                 # Constants and default configurations
├── exceptions.py            # Custom exception definitions
├── agent.py                 # Main conversation agent (HA LLM API integration)
├── conversation.py          # Conversation history management
├── context_manager.py       # Context injection and formatting
├── tool_handler.py          # Tool call execution and registration
├── services.py              # Service registration and handlers
├── services.yaml            # Service schema definitions
├── helpers.py               # Utility functions
├── strings.json             # UI text and localization
├── context_providers/       # Context injection strategies
│   ├── __init__.py
│   ├── base.py             # Base context provider interface
│   ├── direct.py           # Direct entity/attribute injection
│   └── vector_db.py        # Vector DB (Chroma) integration
├── tools/                   # Tool definitions and executors
│   ├── __init__.py
│   ├── registry.py         # Tool registration and management
│   ├── ha_control.py       # Home Assistant control tool
│   ├── ha_query.py         # Home Assistant state query tool
│   └── custom.py           # User-defined custom tools
└── translations/            # Multi-language support
    └── en.json
```

### 2.2 Key Components

#### Conversation Agent (`agent.py`)
- Implements Home Assistant's `ConversationEntity` interface
- Integrates with HA's native LLM API (`llm` platform)
- Orchestrates context injection, history, and tool calling
- Communicates with OpenAI-compatible LLM endpoints
- Manages dual LLM strategy (primary + optional comprehensive)
- Fires events for automation triggers and monitoring

#### Context Manager (`context_manager.py`)
- Handles context injection strategies (direct vs vector DB)
- Formats entity and attribute data for LLM consumption
- Manages conversation history storage and retrieval
- Implements configurable history limits
- Optimizes context size for token efficiency

#### Tool Handler (`tool_handler.py`)
- Registers and manages available tools
- Translates HA capabilities into LLM-friendly tool definitions
- Executes tool calls from LLM responses
- Validates permissions and entity access
- Formats tool results for LLM consumption

#### Context Providers (`context_providers/`)
- **Direct Provider**: Injects configured entities/attributes directly
- **Vector DB Provider**: Queries Chroma for relevant context based on user input
- Extensible interface for additional strategies

---

## 3. Feature Specifications

### 3.1 Home Assistant LLM API Integration

This component integrates with Home Assistant's native LLM conversation platform, making it compatible with any OpenAI-compatible API endpoint.

#### Primary LLM Configuration
- **Purpose:** Main conversation handling and tool execution
- **API Compatibility:** OpenAI-compatible endpoints (OpenAI, Ollama, LM Studio, LocalAI, etc.)
- **Configuration:**
  - Base URL (e.g., `https://api.openai.com/v1` or `http://localhost:11434/v1`)
  - API key (if required)
  - Model name
  - Temperature (0.0 - 2.0)
  - Max tokens per response
  - Top P (0.0 - 1.0)
  - Tool calling support (enabled by default)
  - Keep Alive: Duration to keep model loaded in memory (sent with all API requests)
    - Default: `5m` (5 minutes)
    - Options: Duration string (e.g., `10m`, `1h`) or `-1` for indefinite
    - Primarily used by Ollama to control model unloading
    - Other providers (OpenAI, etc.) will ignore this parameter
    - Not used for embedding models (see Vector DB configuration)

#### External LLM Tool (Optional)
- **Purpose:** Exposed as a tool that primary LLM can call when it needs help
- **Use Case:** Local/fast model handles tool execution, can delegate to powerful external LLM when needed
- **Configuration:**
  - Enable/disable toggle
  - Separate base URL and API key
  - Model name
  - Temperature and generation parameters
  - Tool description for primary LLM
  - Keep Alive: Duration to keep external model loaded in memory
    - Default: `5m` (5 minutes)
    - Same behavior as primary LLM keep_alive setting

**Example Workflow:**
1. User asks: "What's the temperature and should I adjust the thermostat?"
2. Primary LLM (local Ollama) uses `ha_query` tool to get temperature
3. Primary LLM decides it needs comprehensive analysis, calls `query_external_llm` tool
4. External LLM (GPT-4) receives context and provides detailed recommendation
5. Primary LLM returns the external LLM's response to user

### 3.2 Native Tool Calling

The component exposes tools to the LLM using Home Assistant's native tool call format. Tools are designed to be generic and easy for LLMs to interpret.

#### Core Tools

**1. `ha_control` - Control Home Assistant Entities**
```json
{
  "name": "ha_control",
  "description": "Control Home Assistant devices and services. Use this to turn on/off lights, adjust thermostats, lock doors, etc.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "description": "The action to perform (e.g., 'turn_on', 'turn_off', 'set_temperature')",
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

**2. `ha_query` - Query Home Assistant State**
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
          "duration": {"type": "string", "description": "Time range (e.g., '1h', '24h', '7d')"},
          "aggregate": {"type": "string", "enum": ["avg", "min", "max", "sum", "count"]}
        }
      }
    },
    "required": ["entity_id"]
  }
}
```

**3. `query_external_llm` - Query External LLM (Optional Tool)**
```json
{
  "name": "query_external_llm",
  "description": "Query a more capable external LLM for complex analysis, detailed explanations, or comprehensive answers. Use this when you need help with complex reasoning, detailed explanations, or when the user asks for analysis beyond simple home control.",
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

**Response Format:**
All tools (including external LLM and custom tools) return standardized responses:
```json
{
  "success": true,
  "result": "External LLM's response or tool result",
  "error": null
}
```

**Error Handling:**
If external LLM call fails (timeout, API error, rate limit):
```json
{
  "success": false,
  "result": null,
  "error": "Failed to query external LLM: Connection timeout after 30s"
}
```
Error messages are returned transparently to the primary LLM, which then communicates the issue to the user.

#### Custom Tool Definition (Phase 3)

Users can define additional custom tools in `configuration.yaml`:

```yaml
# configuration.yaml
home_agent:
  custom_tools:
    - name: check_weather
      description: Get weather forecast for a location
      parameters:
        type: object
        properties:
          location:
            type: string
            description: City name or coordinates
      handler:
        type: rest  # Supported: rest, service (no script execution)
        url: "https://api.weather.com/v1/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ weather_api_key }}"

    - name: trigger_automation
      description: Trigger a Home Assistant automation
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

**Supported Handler Types (Phase 3):**
- `rest` - HTTP API calls with configurable headers, method, body
- `service` - Call Home Assistant services
- **Script execution is NOT supported** for security reasons

**All custom tools return standardized format:**
```json
{
  "success": true,
  "result": {...},
  "error": null
}
```

#### Tool Execution Flow
1. Primary LLM receives tool definitions in system prompt (including `query_external_llm` if configured)
2. User query processed by primary LLM
3. Primary LLM responds with tool call(s) - could be `ha_control`, `ha_query`, `query_external_llm`, or custom tools
4. Component executes tool(s):
   - `ha_control`/`ha_query` → executed against Home Assistant
   - `query_external_llm` → forwarded to external LLM API
   - Custom tools → executed per configuration
5. Tool results returned to primary LLM
6. Primary LLM formulates natural language response (or makes additional tool calls)
7. Response returned to user

**Example Multi-Tool Flow:**
```
User: "What's the living room temperature and should I adjust it?"
→ Primary LLM calls ha_query(entity_id="sensor.living_room_temperature")
→ Result: 68°F
→ Primary LLM calls query_external_llm(prompt="Living room is 68°F. Should I adjust?", context={...})
→ External LLM: "68°F is comfortable for most. Only adjust if you prefer warmer..."
→ Primary LLM returns external LLM's answer to user
```

### 3.3 Context Injection Strategies

The component provides flexible context injection to give the LLM relevant information about the home state.

#### Direct Entity Injection (Default)
- **Configuration:** Specify entities and attributes to always include
- **Format:** Structured JSON or natural language description
- **Example:**
  ```yaml
  context_entities:
    - entity_id: sensor.living_room_temperature
      attributes: [temperature, unit_of_measurement]
    - entity_id: light.living_room
      attributes: [state, brightness]
    - entity_id: climate.thermostat
      attributes: [temperature, target_temperature, hvac_mode]
  ```
- **Automatic Appending:** Injected into every LLM call before user message
- **Toggle:** Can be disabled entirely if not needed

#### Vector Database Injection (Advanced)
- **Purpose:** Dynamically retrieve relevant entities based on user query
- **Integration:** ChromaDB vector store
- **Configuration:**
  - ChromaDB connection details (host, port, collection name)
  - Embedding model for query vectorization
  - Top K results to retrieve
  - Similarity threshold
  - Embedding Keep Alive: Duration to keep embedding model loaded in memory
    - Default: `5m` (5 minutes)
    - Options: Duration string (e.g., `10m`, `1h`) or `-1` for indefinite
    - Separate from main LLM keep_alive setting
    - Only applies when using Ollama for embeddings
    - Ignored by OpenAI and other providers
- **Workflow:**
  1. User query embedded using configured model
  2. Vector DB queried for semantically similar entity contexts
  3. Top results injected into LLM context
  4. More efficient token usage - only relevant entities included
- **Example:** User asks "Is it cold in the bedroom?" → retrieves bedroom temperature sensor context

#### Additional Collections Support ✅ **IMPLEMENTED**
- **Purpose:** Query additional ChromaDB collections alongside entity embeddings for supplementary context
- **Use Cases:**
  - Product manuals and documentation
  - FAQs and knowledge bases
  - Custom procedures and notes
  - Domain-specific information
- **Two-Tier Ranking System:**
  - **Entity Collection (Priority):** Always queries main entity collection first
    - Returns top K entities based on `CONF_TOP_K`
    - Filtered by `L2_DISTANCE_THRESHOLD`
    - Results always appear first in context
  - **Additional Collections (Supplementary):** Queries all specified additional collections
    - Results from ALL additional collections merged together
    - Sorted by distance and filtered by `CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD`
    - Returns top K from merged pool based on `CONF_ADDITIONAL_TOP_K`
    - Results appear after entity context with clear header
- **Configuration:**
  - `CONF_ADDITIONAL_COLLECTIONS`: Comma-separated collection names (default: empty)
  - `CONF_ADDITIONAL_TOP_K`: Number of results from additional collections (default: 5)
  - `CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD`: Separate threshold for additional collections (default: 250.0)
- **Context Format:**
  ```
  [Entity embeddings results - CSV/JSON format]

  ### RELEVANT ADDITIONAL CONTEXT FOR ANSWERING QUESTIONS, NOT CONTROL ###
  [Additional collections results - JSON with metadata]
  ```
- **Error Handling:** Non-existent collections skipped silently with warning logged
- **Documentation:** See `docs/reference/VECTOR_DB_SETUP.md` (lines 597-676)
- **Status:** ✅ Implemented in v0.4.7-beta - PR #19
- **Key Files:** `context_providers/vector_db.py`, `const.py`, `config_flow.py`, `vector_db_manager.py`
- **Tests:** 8 comprehensive test cases in `tests/unit/test_context_providers/test_vector_db.py::TestAdditionalCollections`

#### Context Format Options
- **JSON Format:** Structured data for precise parsing
- **Natural Language:** Human-readable entity descriptions
- **Hybrid:** JSON with natural language annotations

### 3.4 Conversation History Management

#### Automatic History Injection
- **Configurable Message Limit:** Keep last N messages (default: 10)
- **Storage:** Per-conversation ID in component state
- **Format:** Standard OpenAI message format (role + content)
- **Automatic Appending:** History added to LLM context before user message

#### History Configuration
- **Max Messages:** Maximum conversation turns to retain (user + assistant pairs)
- **Max Tokens:** Token-based limit (alternative to message count)
- **Clear on Restart:** Option to persist across HA restarts or clear
- **Manual Clear:** Service call to clear specific conversation

#### Context Window Management
When context (history + entities + system prompt + user message) exceeds limits:
- **Truncate History:** Remove oldest messages first
- **Compress Entities:** Reduce entity attribute verbosity
- **Summarize History:** Use LLM to create summary of old messages (optional)

### 3.5 Default System Prompt

The component provides a comprehensive default prompt that explains tool usage:

```
You are a helpful home automation assistant integrated with Home Assistant.

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

### query_external_llm (if available)
Use this tool when you need help with:
- Complex analysis or reasoning
- Detailed explanations
- Comprehensive recommendations
- Situations where you're unsure how to best help the user
Pass relevant context (like sensor readings, user preferences) to get better answers.

## Guidelines

1. Always use ha_query before ha_control to check current state
2. Be specific with entity IDs when possible
3. Confirm actions that might have significant impact (e.g., unlocking doors)
4. For complex questions or analysis, consider using query_external_llm
5. If you're not sure about an entity ID, use ha_query with wildcards to search

## Current Home Context

{{entity_context}}

## Conversation History

{{conversation_history}}

Now respond to the user's request:
```

**Customization:** Users can fully customize or extend this prompt via configuration.

### 3.6 Configuration Options

#### Initial Setup
- **Integration Name:** Display name for this instance
- **Primary LLM Configuration:**
  - Base URL (OpenAI-compatible endpoint)
  - API Key
  - Model name
  - Temperature, Top P, Max Tokens

#### Context Injection
- **Strategy Selection:** Direct or Vector DB
- **Direct Mode:**
  - Entity list configuration
  - Attribute selection
  - Update frequency
  - Format (JSON/natural language)
- **Vector DB Mode:**
  - ChromaDB host/port
  - Collection name
  - Embedding model
  - Top K results
  - Similarity threshold

#### Conversation History
- **Enable History:** Toggle on/off
- **Max Messages:** Number of conversation turns to keep
- **Max Tokens:** Alternative token-based limit
- **Persist Across Restarts:** Save to storage or memory-only

#### System Prompt
- **Use Default:** Toggle for built-in prompt
- **Custom Prompt:** Full template editor
- **Template Variables:** Access to entity_context, conversation_history, etc.

#### Tool Configuration
- **Enable Native Tools:** ha_control and ha_query
- **Custom Tools:** YAML definitions for additional tools
- **Max Tool Calls:** Limit per conversation turn
- **Tool Timeout:** Execution timeout per tool

#### External LLM Tool (Optional - Phase 3)
- **Enable:** Toggle to expose `query_external_llm` tool to primary LLM
- **Base URL:** External LLM endpoint (OpenAI-compatible)
- **API Key:** Separate credentials
- **Model:** Model name (e.g., "gpt-4o", "claude-3-opus")
- **Temperature:** Generation temperature
- **Max Tokens:** Response limit
- **Keep Alive:** Duration to keep external model loaded (default: `5m`, range: duration string or `-1`)
- **Tool Description:** Customize when primary LLM should use this tool
- **Context Handling:** Only explicit `prompt` and `context` parameters are passed (conversation history is NOT automatically included)
- **Error Behavior:** Errors returned transparently to primary LLM
- **Tool Call Limit:** External LLM calls count toward `max_calls_per_turn`

### 3.7 Services

#### `home_agent.process` (Main Service)
- Process a conversation message through the LLM
- **Parameters:**
  - `text`: User message/query
  - `conversation_id`: Optional conversation ID for history tracking
  - `context_entities`: Optional override of entity context

#### `home_agent.clear_history`
- Clear conversation history
- **Parameters:**
  - `conversation_id`: Specific conversation to clear (optional, clears all if omitted)

#### `home_agent.reload_context`
- Reload entity context (useful after entity changes)
- **Parameters:** None

#### `home_agent.execute_tool` (Debug/Testing)
- Manually execute a tool for testing
- **Parameters:**
  - `tool_name`: Tool to execute
  - `parameters`: Tool parameters (JSON)

---

## 4. Events

### 4.1 Fired Events

#### `home_agent.conversation.started`
- Triggered when a new conversation begins
- **Data:**
  - `conversation_id`: Unique conversation identifier
  - `user_id`: HA user ID
  - `timestamp`: ISO timestamp
  - `context_mode`: "direct" or "vector_db"

#### `home_agent.conversation.finished`
- Triggered when conversation completes
- **Data:**
  - `conversation_id`: Conversation identifier
  - `user_id`: HA user ID
  - `tool_calls`: Number of tools executed
  - `tool_breakdown`: Dict of tool names and call counts
  - `tokens_used`: Token consumption (primary + external LLM)
  - `duration_ms`: Total processing time
  - `used_external_llm`: Boolean (if query_external_llm was called)

#### `home_agent.tool.executed`
- Triggered after each tool execution
- **Data:**
  - `tool_name`: Tool that was executed
  - `parameters`: Tool parameters
  - `result`: Execution result (truncated if large)
  - `success`: Boolean
  - `duration_ms`: Execution time
  - `conversation_id`: Associated conversation

#### `home_agent.context.injected`
- Triggered when context is added to LLM call
- **Data:**
  - `conversation_id`: Conversation identifier
  - `mode`: "direct" or "vector_db"
  - `entities_included`: List of entity IDs
  - `token_count`: Approximate tokens used
  - `vector_db_query`: Query used (if vector DB mode)

#### `home_agent.error`
- Triggered on errors
- **Data:**
  - `error_type`: Exception class name
  - `error_message`: Error description
  - `conversation_id`: Associated conversation (if applicable)
  - `component`: Component where error occurred
  - `context`: Additional debug information

---

## 5. Security & Permissions

### 5.1 Entity Access Control
- Entity exposure via voice assistant configuration
- Explicit entity whitelisting per config entry
- Domain-level restrictions
- Service execution permissions

### 5.2 Function Permissions
- Per-function enable/disable
- Service call restrictions
- External API domain whitelisting
- SQL query validation

### 5.3 Rate Limiting (New)
- Per-user conversation limits
- Function execution throttling
- API call rate limiting
- Token usage quotas

### 5.4 Audit Logging (New)
- Function execution logs
- Service call tracking
- User activity monitoring
- Security event logging

---

## 6. Error Handling

### 6.1 Exception Types
- `FunctionNotFound` - Requested function doesn't exist
- `FunctionExecutionError` - Function failed during execution
- `AuthenticationError` - LLM provider auth failure
- `TokenLimitExceeded` - Context too large
- `RateLimitExceeded` - API rate limit hit
- `PermissionDenied` - Insufficient permissions
- `ValidationError` - Invalid configuration/parameters

### 6.2 Error Recovery
- Automatic retry with exponential backoff
- Fallback to alternative providers
- Graceful degradation (disable functions on repeated failures)
- User notification system

---

## 7. Performance Optimization

### 7.1 Caching
- Template rendering results
- Entity state snapshots
- Function execution results (configurable TTL)
- LLM response caching (for identical prompts)

### 7.2 Async Operations
- Non-blocking function execution
- Parallel function calls where possible
- Streaming response handling
- Background task processing

### 7.3 Resource Management
- Connection pooling for REST clients
- Token usage monitoring
- Memory-efficient conversation storage
- Cleanup of old conversation histories

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Function executor validation
- Template rendering
- Configuration validation
- Provider abstraction layer

### 8.2 Integration Tests
- End-to-end conversation flows
- Function execution chains
- Multi-provider scenarios
- Error handling paths

### 8.3 Mock Environments
- LLM provider mocking
- Home Assistant state simulation
- External API mocking

---

## 9. Documentation Requirements

### 9.1 User Documentation
- Installation guide (HACS + manual)
- Configuration walkthrough
- Function reference guide
- Examples and use cases
- Troubleshooting guide

### 9.2 Developer Documentation
- Architecture overview
- Adding custom function types
- Provider implementation guide
- API reference
- Contribution guidelines

---

## 10. Migration & Compatibility

### 10.1 Migration from Extended OpenAI Conversation
- Configuration import tool
- Function definition converter
- Backward compatibility mode
- Migration guide documentation

### 10.2 Home Assistant Compatibility
- Minimum version: 2024.1.0
- Core dependency tracking
- Breaking change management

---

## 11. Future Enhancements

### 11.1 Planned Features
- Multi-agent conversations
- Agent memory/long-term storage
- Voice assistant integration improvements
- Mobile app support
- Conversation analytics dashboard
- Function marketplace/sharing

### 11.2 Research Areas
- On-device model execution
- Proactive automation suggestions
- Learning from user feedback
- Natural language automation builder UI

---

## 12. Development Phases

### Phase 1: Foundation (MVP) ✅ COMPLETE
- [x] Core architecture setup
- [x] Home Assistant ConversationEntity implementation
- [x] OpenAI-compatible API client
- [x] Direct entity context injection
- [x] Basic conversation history management
- [x] Core tools: ha_control and ha_query
- [x] Default system prompt
- [x] Configuration flow (basic)
- [x] Basic error handling and exceptions system
- [x] Comprehensive test coverage (376+ tests passing)

**Implementation Files:**
- `agent.py` - Main agent class (ConversationEntity)
- `tools/ha_control.py` - Control tool
- `tools/ha_query.py` - Query tool
- `config_flow.py` - Configuration UI
- `context_manager.py` - Context injection
- `conversation.py` - History management
- `tool_handler.py` - Tool execution
- `exceptions.py` - Custom exception definitions

### Phase 2: Enhanced Context & History ✅ COMPLETE
- [x] Vector DB (Chroma) integration
- [x] Configurable context strategies (direct vs vector DB)
- [x] Advanced history management (token limits, persistence)
- [x] Context optimization and compression
- [x] Event system implementation
- [x] Enhanced configuration UI with menu-based options

**Implementation Files:**
- `vector_db_manager.py` - Vector DB integration
- `context_optimizer.py` - Context compression
- `context_providers/vector_db.py` - Vector DB provider
- Services: `reindex_entities`, `index_entity`
- Events: `EVENT_CONTEXT_OPTIMIZED`, `EVENT_HISTORY_SAVED`, `EVENT_VECTOR_DB_QUERIED`

### Phase 3: External LLM Tool & Custom Tools ✅ COMPLETE
- [x] **External LLM Tool (`query_external_llm`)**
  - Expose as tool for primary LLM to delegate complex queries
  - Context passthrough: prompt + explicit context parameter only
  - Error handling: Return error message transparently to user
  - Counts toward `max_calls_per_turn` limit
  - Standardized response format: `{success, result, error}`
- [x] **Custom Tool Framework**
  - Configuration via `configuration.yaml` under `home_agent:` section
  - REST handler: HTTP API calls with headers, query params, body
  - Service handler: Call Home Assistant services
  - **No script execution** (security restriction)
  - YAML schema validation on integration setup
  - Standardized response format for all custom tools
- [x] **Integration & Testing**
  - Unit tests for external LLM tool and custom tool handlers (76 tests)
  - Integration tests for dual-LLM workflow (23 tests)
  - Error handling tests
  - Documentation with examples (`docs/CUSTOM_TOOLS.md`, `docs/EXTERNAL_LLM.md`)
  - **Test Coverage**: 95.58% for Phase 3 code (exceeds 80% requirement)

**Configuration Location:** `configuration.yaml`
```yaml
home_agent:
  custom_tools:
    - name: check_weather
      description: "Get weather forecast"
      parameters:
        type: object
        properties:
          location:
            type: string
      handler:
        type: rest
        url: "https://api.weather.com/v1/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ weather_api_key }}"
```

### Phase 3.5: Long-Term Memory System ✅ COMPLETE
- [x] **Core Memory Manager**
  - Persistent storage using Home Assistant Store (`.storage/home_agent.memories`)
  - ChromaDB indexing for semantic search (collection: configurable via `CONF_MEMORY_COLLECTION_NAME`)
  - Memory data structure with metadata:
    - `id`: Unique memory identifier (UUID)
    - `type`: fact, preference, context, event
    - `content`: The actual memory text
    - `source_conversation_id`: Origin conversation
    - `extracted_at`: Timestamp
    - `last_accessed`: Last retrieval time
    - `importance`: Score 0.0-1.0 with access boost (0.05)
    - `metadata`: Additional context (entities, topics, etc.)
    - `expires_at`: TTL support for transient memories
    - `is_transient`: Flag for temporary memories
  - Deduplication logic with configurable similarity threshold (`CONF_MEMORY_DEDUP_THRESHOLD`)
  - Importance scoring with access boost and optional decay (`CONF_MEMORY_IMPORTANCE_DECAY`)
  - Memory retention policies with per-type TTL settings
  - Periodic cleanup task for expired memories (`CONF_MEMORY_CLEANUP_INTERVAL`)
  - **Enhanced quality detection** (`_is_transient_state()` method):
    - Detects transient device states: "is on", "is off", "temperature is", etc.
    - Detects low-value conversational patterns: "conversation occurred", "we discussed", "there is no", etc.
    - 12+ low-value patterns for filtering meta-information and negative existence statements
  - **Implementation**: `memory_manager.py` (887 lines), quality detection at lines 595-655
- [x] **Automatic Memory Extraction**
  - Post-conversation hook triggered by conversation completion in `agent.py`
  - **Configurable LLM selection**: Use external LLM OR local LLM for extraction (`CONF_MEMORY_EXTRACTION_LLM`)
  - Extraction prompt builder (`_build_extraction_prompt()`) with conversation context
  - **Enhanced extraction prompt** with explicit quality guidance:
    - DO NOT extract: meta-information, timestamps, negative existence statements, questions without answers
    - DO extract: actionable facts, specific preferences, important context, patterns/routines
    - Minimum information density requirement (10+ meaningful words)
  - Parse structured JSON response from LLM with markdown code block handling
  - **Multi-layer quality validation** before storing memories:
    1. **Word count validation**: Reject memories with < 10 meaningful words (>2 chars)
    2. **Low-value prefix detection**: Reject content starting with phrases like "there is no", "we discussed"
    3. **Importance threshold**: Reject memories with importance < 0.4
    4. **Pattern-based filtering**: Detect and reject transient states and conversational meta-information
  - Store extracted memories in MemoryManager + ChromaDB indexing
  - Error handling for extraction failures with event firing
  - Event: `EVENT_MEMORY_EXTRACTED` with extraction details
  - Debug logging for all memory rejections with specific reasons
  - **Implementation**: `agent.py` lines 1226-1587, 1469-1520 (validation)
- [x] **Manual Memory Tools**
  - `store_memory` tool: Explicitly save a fact/preference during conversation
  - `recall_memory` tool: Search and retrieve relevant memories semantically
  - Standardized response format: `{success, result, error}`
  - Tool definitions exposed to primary LLM in tool registry
  - **Implementation**: `tools/memory_tools.py`
- [x] **Memory Context Integration**
  - `MemoryContextProvider` for semantic search via ChromaDB
  - Integrate with `ContextManager` to inject relevant memories into system prompt
  - Query ChromaDB based on user input for relevant recall
  - Format memories for system prompt injection with metadata
  - Track access patterns (updates `last_accessed` timestamp)
  - Configurable top-K retrieval (`CONF_MEMORY_CONTEXT_TOP_K`)
  - Importance threshold filtering (`CONF_MEMORY_MIN_IMPORTANCE`)
  - **Implementation**: `context_providers/memory.py`
- [x] **Memory Management UI & Services**
  - Configuration options in `const.py`:
    - `CONF_MEMORY_ENABLED` - Enable/disable memory system (privacy toggle)
    - `CONF_MEMORY_EXTRACTION_ENABLED` - Enable/disable automatic extraction
    - `CONF_MEMORY_EXTRACTION_LLM` - Choose extraction LLM ("external" or "local")
    - `CONF_MEMORY_MAX_MEMORIES` - Max memories limit
    - `CONF_MEMORY_MIN_IMPORTANCE` - Minimum importance threshold
    - `CONF_MEMORY_COLLECTION_NAME` - ChromaDB collection name
    - `CONF_MEMORY_IMPORTANCE_DECAY` - Optional decay rate
    - `CONF_MEMORY_DEDUP_THRESHOLD` - Deduplication similarity threshold
    - `CONF_MEMORY_CONTEXT_TOP_K` - Number of memories to inject
    - `CONF_MEMORY_EVENT_TTL`, `CONF_MEMORY_FACT_TTL`, etc. - Per-type TTL settings
    - `CONF_MEMORY_CLEANUP_INTERVAL` - Cleanup task frequency
  - Home Assistant services (in `services.yaml`):
    - `home_agent.list_memories` - List all stored memories
    - `home_agent.delete_memory` - Delete specific memory by ID
    - `home_agent.clear_memories` - Clear all memories (with confirmation)
    - `home_agent.search_memories` - Manually search memories
    - `home_agent.add_memory` - Manually add a memory
  - Configuration UI integration in `config_flow.py` (Memory System menu)
- [x] **Configuration & Testing**
  - Memory-related constants added to `const.py`
  - Default values for all memory settings
  - Unit tests: `test_memory_manager.py`, `test_memory_extraction.py`, `test_memory_tools.py`, `test_memory_quality_validation.py`
  - Quality validation tests: 10+ test cases covering positive/negative/edge cases for memory filtering
  - Service tests: `test_memory_services.py`
  - Context provider tests: `test_memory_context_provider.py`
  - Integration tests for memory + conversation flows
  - **Test Coverage**: >80% for memory system code

**Memory Scope:** Global (shared across all users and conversations)

**Privacy:** User-controlled on/off toggle for the entire memory system

**Architecture:**
```
User Conversation
    ↓
Primary LLM Response
    ↓
EVENT_CONVERSATION_FINISHED
    ↓
Memory Extraction (External OR Local LLM)
    ↓
MemoryManager.add_memory()
    ↓
├─→ Home Assistant Store (.storage/home_agent.memories)
└─→ ChromaDB (semantic indexing)

Next Conversation:
    User Input
       ↓
    MemoryContextProvider.search_memories()
       ↓
    Relevant memories injected into context
       ↓
    Primary LLM (with entity context + memory context)
```

**Configuration Example:**
```yaml
# In Home Assistant configuration UI
memory:
  enabled: true                           # Privacy toggle
  automatic_extraction: true              # Enable post-conversation extraction
  extraction_llm: "external"              # "external" or "local"
  max_memories: 100                       # Maximum stored memories
  min_importance: 0.3                     # Filter low-importance memories
  collection_name: "home_agent_memories"  # ChromaDB collection
  context_top_k: 5                        # Number of memories to inject
```

### Phase 4: Streaming Response Support ✅ (Completed)

#### Completed Items:
- [x] Migrate from `async_process()` to `_async_handle_message()` API
- [x] Implement LLM response streaming using aiohttp
- [x] Add `chat_log.async_add_delta_content_stream()` integration
- [x] Implement tool progress indicators (`EVENT_TOOL_PROGRESS`)
- [x] Add streaming + tool call integration (pause/resume)
- [x] Configuration toggle (default: disabled)
- [x] Comprehensive testing (unit + integration)
- [x] Fallback to synchronous on errors
- [x] Error event emission (`EVENT_STREAMING_ERROR`)

#### Implementation Details:
- **Streaming Handler**: OpenAI-compatible SSE parser (`streaming.py`)
- **Agent Integration**: ChatLog-based streaming in `agent.py`
- **Tool Progress**: Started/completed/failed events in `tool_handler.py`
- **UI Configuration**: Debug Settings menu toggle
- **Tests**: 16 unit tests + 10 integration tests (all passing)
- **Coverage**: >80% maintained

#### Performance:
- First audio chunk: ~500ms (vs 5+ seconds synchronous)
- Streaming overhead: <10%
- Automatic fallback on any streaming error

#### Event Schema:
- `EVENT_TOOL_PROGRESS`: Fired during tool execution (status: started/completed/failed)
- `EVENT_STREAMING_ERROR`: Fired on streaming failures with fallback indication

#### Configuration:
- Enable via: Settings → Devices & Services → Home Agent → Configure → Debug Settings
- Default: Disabled for backward compatibility
- Requires: Voice Assistant pipeline with Wyoming TTS integration

#### Testing:
See [Manual Testing Guide](docs/MANUAL_TESTING_ISSUE10.md) for comprehensive validation instructions.

### Phase 5: MCP Server Integration
- [ ] **MCP (Model Context Protocol) Server Support**
  - Integration with external MCP servers for data collection
  - MCP server handler type for custom tools
  - Configuration for MCP server endpoints
  - Authentication and authorization for MCP servers
  - **Read-only data collection** (no local execution from MCP servers)
- [ ] **Custom Tool Handler Types** (Extended)
  - `mcp`: Query external MCP servers for context/data
  - Template handler: Jinja2 template rendering
  - Enhanced REST handler with OAuth support
- [ ] **Security & Validation**
  - MCP server authentication
  - Response validation and sanitization
  - Rate limiting for external calls
  - Domain whitelisting options
- [ ] **Testing & Documentation**
  - MCP integration tests
  - Security audit
  - User guides for MCP setup
  - Example MCP configurations

**Example MCP Tool Configuration:**
```yaml
home_agent:
  custom_tools:
    - name: query_knowledge_base
      description: "Query external knowledge base via MCP"
      parameters:
        type: object
        properties:
          query:
            type: string
      handler:
        type: mcp
        server_url: "https://mcp.example.com"
        method: "knowledge/query"
        auth:
          type: bearer
          token: "{{ secrets.mcp_token }}"
```

### Phase 6: Reliability & Resource Management
**Status:** Planned | **Priority:** Medium | **Target:** v1.0.0 completion
**Context:** Home Assistant single-instance deployment (not cloud-scale)

#### Design Philosophy
This phase focuses on making the integration reliable and resource-efficient for Home Assistant deployments running on modest hardware (Raspberry Pi, NUC, home servers). The integration serves 1-10 household users, not millions. Enterprise-scale features like extensive caching, rate limiting quotas, and load testing are **explicitly excluded** as inappropriate for this context.

#### Core Features for v1.0.0

- [ ] **Resource Management** (~3-4 hours)
  - Conversation history cleanup (auto-delete conversations older than 7 days)
  - Memory-efficient conversation storage (limit messages per conversation)
  - Connection pooling for external services (ChromaDB, Ollama, external LLM)
  - Proper async/await patterns (don't block Home Assistant event loop)
  - Background task management for cleanup operations
  - Prevent memory leaks in long-running sessions

- [ ] **Graceful Degradation** (~4-5 hours)
  - Fallback to direct mode if ChromaDB unavailable
  - Continue without memories if memory system fails
  - Handle Ollama/embedding service timeouts gracefully
  - External LLM failure handling (return error to primary LLM)
  - Partial feature degradation instead of complete failure
  - User-friendly fallback messages

- [ ] **Connection Reliability** (~2-3 hours)
  - Simple retry logic for external services (2 attempts, 1s delay between)
  - Timeout handling for long-running operations (ChromaDB, Ollama, external LLM)
  - Connection health checks on integration startup
  - Proper error handling for network failures

- [ ] **User-Friendly Error Messages** (~2-3 hours)
  - Replace stack traces with clear, actionable error messages
  - Helpful suggestions for common errors (entity not found, service unavailable)
  - Proper error propagation to voice assistant interface
  - Logging with appropriate levels (debug, info, warning, error)
  - Context-rich error messages for debugging

**Total Estimated Effort:** 11-15 hours

#### Explicitly Excluded (Not Applicable for Home Assistant Context)

The following features were removed as they are designed for enterprise cloud-scale systems and add unnecessary complexity for a single-instance Home Assistant integration:

- ❌ **LLM Response Caching** - Local LLM usage means no API costs; dynamic home automation context results in <5% cache hit rate
- ❌ **Entity State Caching** - Vector DB mode already limits to ~10 entities; overhead exceeds benefit
- ❌ **ChromaDB Query Caching** - Low hit rate due to varied queries; complex invalidation logic not justified
- ❌ **Per-User Rate Limiting/Quotas** - Household usage (1-10 users) doesn't require per-user limits; tool call limits already implemented
- ❌ **Token Usage Quotas** - No cost with local LLM (Ollama); external LLM usage is rare and optional
- ❌ **Enterprise Performance Metrics** - P50/P95/P99 latency percentiles, load testing, stress testing not relevant for single-instance
- ❌ **Circuit Breaker Patterns** - Overkill for home automation; simple retry logic sufficient
- ❌ **Provider Failover** - Complex configuration for minimal benefit; users can manually switch providers
- ❌ **Automated Performance Testing** - CI/CD complexity not justified; Home Assistant logs provide sufficient visibility

#### Optional Enhancements (Post-v1.0.0)

- Simple conversation-level throttling to prevent runaway automation loops
- Template compilation caching (minor optimization, 10ms savings)
- Enhanced debug logging and telemetry
- Connection pool statistics and monitoring

### Phase 7: Polish & Production Ready
- [ ] **Complete Documentation**
  - Comprehensive user documentation
    - Installation guide (HACS + manual installation)
    - Initial setup and configuration walkthrough
    - LLM provider setup guides (OpenAI, Ollama, LocalAI, etc.)
    - Feature guide for each major feature
    - Configuration reference (all options explained)
  - Memory system documentation
    - User guide for memory features
    - How memory extraction works
    - Configuring memory settings
    - Managing memories via services
    - Privacy and data retention information
  - Vector DB setup guide
    - ChromaDB installation and configuration
    - Embedding model selection
    - Indexing entities for vector search
    - Troubleshooting vector DB issues
  - Troubleshooting guide
    - Common issues and solutions
    - LLM connection problems
    - Tool execution errors
    - Memory/vector DB issues
    - Debugging techniques
  - API reference documentation
    - Service call documentation
    - Event schema reference
    - Configuration schema reference
    - Tool development guide
  - Example automations and use cases
    - Voice assistant integration examples
    - Automation triggers using events
    - Multi-LLM workflow examples
    - Custom tool examples
  - FAQ section
    - Common questions about features
    - Performance and cost considerations
    - Comparison with extended_openai_conversation
    - Privacy and security FAQs
- [ ] **Migration Tools and Guides**
  - Migration from extended_openai_conversation
    - Configuration import tool
    - Function definition converter (REST/service)
    - Step-by-step migration guide
    - Breaking changes documentation
    - Compatibility mode (if needed)
  - Version migration tools
    - Database schema migrations
    - Configuration format updates
    - Deprecation warnings
- [ ] **Security Audit**
  - Code security review
    - Input validation audit
    - API key handling verification
    - Entity access control review
    - SQL injection prevention (if applicable)
  - Dependency security audit
    - Check for known vulnerabilities
    - Pin dependency versions
    - Regular security updates
  - Privacy review
    - Data retention policies
    - Memory storage security
    - Conversation history encryption (optional)
    - GDPR compliance considerations
  - Penetration testing
    - API endpoint security testing
    - Authentication bypass attempts
    - Rate limiting effectiveness
    - Tool execution sandbox verification
- [ ] **HACS Integration**
  - HACS compliance checklist
    - Repository structure validation
    - manifest.json compliance
    - Version tagging strategy
    - Changelog maintenance
  - HACS submission
    - Submit to HACS default repository
    - Category assignment (integration)
    - Documentation URL configuration
    - Issue tracker setup
  - Release workflow
    - GitHub Actions for releases
    - Automated version bumping
    - Release notes generation
    - Asset packaging
- [ ] **Community Feedback and Iteration**
  - Beta testing program
    - Recruit beta testers from community
    - Gather feedback on usability
    - Identify edge cases and bugs
    - Performance testing on diverse setups
  - Issue tracking and management
    - GitHub issue templates
    - Bug triage process
    - Feature request management
    - Community contribution guidelines
  - Community engagement
    - Home Assistant forums thread
    - Discord/Reddit presence
    - Release announcements
    - User testimonials and case studies
- [ ] **Release Preparation**
  - Pre-release checklist
    - All tests passing (>80% coverage)
    - Documentation complete and reviewed
    - Security audit passed
    - Performance benchmarks met
    - HACS compliance verified
  - Release strategy
    - Semantic versioning policy
    - Breaking change communication
    - Deprecation timeline
    - Support policy (versions supported)
  - Initial release (v1.0.0)
    - Release notes highlighting features
    - Upgrade instructions
    - Known issues documentation
    - Community announcement
  - Post-release monitoring
    - Issue response SLA
    - Performance monitoring
    - User feedback collection
    - Hotfix process

---

## 13. Success Metrics

### Current Status (as of 2025-11-02)
- **Code Quality:** ✅ **400+ tests passing**, >80% test coverage achieved
  - Phase 1-2: 376+ unit tests
  - Phase 3: 95.58% coverage (76 unit + 23 integration tests)
  - Phase 3.5: 6+ memory test files
  - Phase 4: 26 streaming tests (16 unit + 10 integration)
- **Functionality:** ✅ Phases 1-4 + 3.5 complete and tested
- **Performance:** ⏳ In Progress
  - Streaming: First audio chunk ~500ms (Phase 4)
  - Target: < 2s response time for simple queries
  - Target: < 5s response time for complex queries
- **Reliability:** ⏳ In Progress
  - Basic error handling implemented
  - Target: > 99% uptime for core features
  - Missing: Rate limiting, caching, automatic retry
- **User Satisfaction:** ⏳ Not yet measured
  - Pre-release phase
  - Beta testing not started
  - Community feedback pending
- **Adoption:** ⏳ Not yet released
  - HACS integration pending (Phase 7)
  - Installation metrics pending release
- **Security:** ⏳ In Progress
  - Entity access control implemented
  - API key handling secure
  - Security audit pending (Phase 7)

### Target Metrics (v1.0.0 Release)
- **Functionality:** All core features working reliably (Phases 1-4, 3.5)
- **Performance:**
  - < 2s response time for typical queries with local LLM
  - Streaming responses provide perceived latency ~500ms (Phase 4)
  - Graceful handling of slow external services
- **Reliability:**
  - Graceful degradation when dependencies fail
  - Clear error messages for common issues
  - Resource-efficient for Home Assistant hardware
- **User Satisfaction:** Positive community feedback and testimonials
- **Adoption:**
  - HACS installation available (public repository)
  - 50+ active installations within 3 months
  - Active community engagement (issues, PRs, discussions)
- **Code Quality:**
  - Maintain > 80% test coverage
  - All critical paths covered
  - No critical security vulnerabilities

---

## 14. Dependencies

### Required Python Packages
- `aiohttp >= 3.9.0` (HTTP client for LLM API calls)
- `chromadb >= 0.4.0` (optional, for vector DB context injection)
- `openai >= 1.3.8` (optional, for embeddings if using vector DB)
- `jinja2` (included with Home Assistant)

### Home Assistant Dependencies
- `conversation` (core conversation platform)
- `llm` (native LLM integration)
- `homeassistant.helpers.intent` (for intent handling)

### Optional Dependencies (for advanced features)
- `numpy` (for vector operations)
- `tiktoken` (for accurate token counting)

---

## 15. Configuration Schema Reference

### Example Configuration Entry
```yaml
# Primary LLM Configuration
name: "Home Agent"
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-..."
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 500
  keep_alive: "5m"  # Keep model loaded for 5 minutes (Ollama-specific, ignored by others)

# Context Injection Strategy
context:
  mode: "direct"  # or "vector_db"

  # Direct mode configuration
  direct:
    entities:
      - entity_id: "sensor.living_room_temperature"
        attributes: ["state", "unit_of_measurement"]
      - entity_id: "light.living_room"
        attributes: ["state", "brightness"]
      - entity_id: "climate.*"  # wildcard support
        attributes: ["temperature", "target_temperature", "hvac_mode"]
    format: "json"  # or "natural_language"

  # Vector DB mode configuration (alternative to direct)
  vector_db:
    enabled: false
    host: "localhost"
    port: 8000
    collection: "home_entities"
    top_k: 5
    similarity_threshold: 0.7
    embedding_model: "text-embedding-3-small"
    embedding_keep_alive: "5m"  # Keep embedding model loaded (separate from main LLM)

# Conversation History
history:
  enabled: true
  max_messages: 10
  max_tokens: 4000
  persist: true  # save across restarts

# System Prompt
prompt:
  use_default: true
  custom_additions: |
    Additional context about my home:
    - The thermostat prefers temperatures between 68-72°F
    - The front door should remain locked after 10 PM

# Tool Configuration
tools:
  enable_native: true  # ha_control and ha_query
  max_calls_per_turn: 5
  timeout_seconds: 30

  # Custom tools
  custom:
    - name: "get_weather"
      description: "Get current weather for configured location"
      parameters:
        type: object
        properties:
          location:
            type: string
            description: "City name"
      handler:
        type: "rest"
        url: "https://api.weather.com/v1/current"
        method: "GET"
        headers:
          Authorization: "Bearer {{ secrets.weather_api_key }}"

# External LLM Tool (Optional - Phase 3)
external_llm:
  enabled: false
  base_url: "https://api.openai.com/v1"
  api_key: "sk-different-key..."
  model: "gpt-4o"
  temperature: 0.8
  max_tokens: 1000
  keep_alive: "5m"  # Keep external model loaded (if using Ollama)
  tool_description: |
    Use this when you need help with complex analysis, detailed explanations,
    or comprehensive recommendations beyond simple home control.
  # Note: Only explicit prompt + context parameters are passed to external LLM
  # Conversation history is NOT automatically included

# Debugging
debug_logging: false
emit_events: true
```

### Example Tool Call Flows

#### Example 1: Simple Control (No External LLM Needed)

**User:** "Turn on the living room lights to 50% brightness"

**Primary LLM Response:**
```json
{
  "tool_calls": [
    {
      "name": "ha_control",
      "parameters": {
        "action": "turn_on",
        "entity_id": "light.living_room",
        "parameters": {
          "brightness_pct": 50
        }
      }
    }
  ]
}
```

**Tool Execution Result:**
```json
{
  "success": true,
  "entity_id": "light.living_room",
  "new_state": "on",
  "brightness": 128
}
```

**Final Response (from Primary LLM):**
"I've turned on the living room lights to 50% brightness."

---

#### Example 2: Complex Query (Primary LLM Delegates to External LLM)

**User:** "Analyze my energy usage this week and suggest optimizations"

**Primary LLM Tool Call 1:**
```json
{
  "name": "ha_query",
  "parameters": {
    "entity_id": "sensor.energy_*",
    "history": {
      "duration": "7d",
      "aggregate": "sum"
    }
  }
}
```

**Result:** Returns energy data for the week

**Primary LLM Tool Call 2:**
```json
{
  "name": "query_external_llm",
  "parameters": {
    "prompt": "Analyze this home energy usage data and provide optimization suggestions",
    "context": {
      "energy_data": "...",
      "user_question": "Analyze my energy usage this week and suggest optimizations"
    }
  }
}
```

**External LLM Response:**
"Based on your energy data, I notice peak usage between 6-9 PM. Here are three optimizations: 1) Shift dishwasher/laundry to off-peak hours... 2) Your HVAC accounts for 45% of usage... 3) Consider automating lights..."

**Final Response (from Primary LLM):**
[Returns the external LLM's detailed analysis]

---

## 16. Notes for Development

### Code Style
- Follow Home Assistant development guidelines
- Use type hints throughout
- Async/await for all I/O operations
- Comprehensive docstrings
- Meaningful variable names

### Git Workflow
- Feature branches from main
- PR reviews required
- Conventional commits
- Semantic versioning

### Release Process
- CHANGELOG.md maintenance
- Version bumping
- GitHub releases
- HACS compatibility validation

---

**Document Version:** 2.0
**Last Updated:** 2025-11-02
**Status:** Active Development - Phases 1-4 + 3.5 Complete

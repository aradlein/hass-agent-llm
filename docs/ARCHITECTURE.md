# Architecture Overview

This document provides a comprehensive overview of Home Agent's architecture, including component relationships, data flow, and module structure.

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [Conversation Flow](#conversation-flow)
- [Module Structure](#module-structure)
- [Component Details](#component-details)
- [Data Flow Patterns](#data-flow-patterns)

---

## High-Level Architecture

Home Agent is built as a modular Home Assistant custom component that integrates with the native conversation platform. The architecture follows a layered design with clear separation of concerns.

```mermaid
graph TB
    subgraph "Home Assistant"
        HA[Home Assistant Core]
        ConvPlatform[Conversation Platform]
        Entities[Entity Registry]
        Services[Services]
    end

    subgraph "Home Agent Core"
        Agent[HomeAgent<br/>Main Orchestrator]
        ContextMgr[ContextManager<br/>Context Injection]
        ConvHistory[ConversationHistoryManager<br/>History Tracking]
        ToolHandler[ToolHandler<br/>Tool Execution]
        SessionMgr[ConversationSessionManager<br/>Persistent Sessions]
    end

    subgraph "Context Providers"
        DirectProvider[DirectContextProvider<br/>Entity Filtering]
        VectorDBProvider[VectorDBContextProvider<br/>Semantic Search]
        MemoryProvider[MemoryContextProvider<br/>Long-term Memory]
    end

    subgraph "Tools"
        HAControl[HomeAssistantControlTool<br/>Device Control]
        HAQuery[HomeAssistantQueryTool<br/>State Queries]
        ExternalLLM[ExternalLLMTool<br/>Delegate to Other LLM]
        CustomTools[CustomToolHandler<br/>REST/Service Tools]
        MemoryTools[Memory Tools<br/>Store/Recall]
    end

    subgraph "Storage & External"
        MemoryMgr[MemoryManager<br/>Long-term Storage]
        VectorDB[(ChromaDB<br/>Vector Storage)]
        LLM[LLM Provider<br/>OpenAI/Ollama/etc]
        HAStore[(Home Assistant<br/>Storage)]
    end

    ConvPlatform -->|process input| Agent
    Agent --> ContextMgr
    Agent --> ConvHistory
    Agent --> ToolHandler
    Agent --> SessionMgr

    ContextMgr --> DirectProvider
    ContextMgr --> VectorDBProvider
    ContextMgr --> MemoryProvider

    DirectProvider --> Entities
    VectorDBProvider --> VectorDB
    MemoryProvider --> MemoryMgr

    ToolHandler --> HAControl
    ToolHandler --> HAQuery
    ToolHandler --> ExternalLLM
    ToolHandler --> CustomTools
    ToolHandler --> MemoryTools

    HAControl --> Services
    HAQuery --> Entities
    CustomTools --> Services
    MemoryTools --> MemoryMgr

    Agent -->|API calls| LLM
    ConvHistory --> HAStore
    MemoryMgr --> HAStore
    MemoryMgr --> VectorDB
    SessionMgr --> HAStore

    style Agent fill:#4CAF50
    style ContextMgr fill:#2196F3
    style ToolHandler fill:#FF9800
    style MemoryMgr fill:#9C27B0
```

### Key Components

1. **HomeAgent**: Central orchestrator that coordinates all operations
2. **ContextManager**: Manages entity context injection strategies
3. **ConversationHistoryManager**: Tracks conversation history across turns
4. **ToolHandler**: Registers and executes tools called by the LLM
5. **MemoryManager**: Handles long-term memory storage and retrieval
6. **ConversationSessionManager**: Manages persistent voice conversation sessions

---

## Conversation Flow

This diagram shows the complete flow of a user conversation through the system, including context injection, tool calling, and memory extraction.

```mermaid
sequenceDiagram
    participant User
    participant HA as Home Assistant
    participant Agent as HomeAgent
    participant Context as ContextManager
    participant Memory as MemoryManager
    participant LLM as LLM Provider
    participant Tools as ToolHandler
    participant History as ConversationHistory
    participant Session as SessionManager

    User->>HA: Voice/Text Input
    HA->>Agent: async_process(user_input)

    Note over Agent: Ensure tools registered
    Agent->>Session: get_conversation_id(user, device)
    Session-->>Agent: conversation_id

    Note over Agent,Memory: Context Assembly Phase
    par Parallel Context Retrieval
        Agent->>Context: get_formatted_context(user_input)
        Context->>Context: get_context (entities)
        Context->>Memory: get_context (memories)
    end
    Context-->>Agent: formatted_context

    Agent->>History: get_history(conversation_id)
    History-->>Agent: previous_messages

    Note over Agent: Build System Prompt
    Agent->>Agent: _build_system_prompt(context)

    Note over Agent,Tools: LLM Interaction Loop (max 5 iterations)
    loop Tool Calling Loop
        Agent->>LLM: call_llm(messages, tools)
        LLM-->>Agent: response (content + tool_calls)

        alt Has Tool Calls
            Agent->>Tools: execute_tool(name, params)
            Tools->>Tools: validate_tool_call

            alt Tool Type: ha_control
                Tools->>HA: call_service(entity_id, action)
                HA-->>Tools: result
            else Tool Type: ha_query
                Tools->>HA: get_state(entity_id)
                HA-->>Tools: state
            else Tool Type: custom
                Tools->>HA: REST/Service call
                HA-->>Tools: result
            end

            Tools-->>Agent: tool_result
            Agent->>Agent: Add tool result to messages
        else No Tool Calls
            Note over Agent: Final response ready
        end
    end

    Note over Agent,History: Save & Extract
    Agent->>History: add_message(user + assistant)
    History->>History: persist to storage

    par Async Memory Extraction
        Agent->>Agent: _extract_and_store_memories
        Agent->>LLM: Extract memories from conversation
        LLM-->>Agent: extracted_memories
        Agent->>Memory: add_memory(content, type, importance)
        Memory->>Memory: Check duplicates, store
    end

    Agent->>Session: update_activity(user, device)
    Agent-->>HA: ConversationResult
    HA-->>User: Response (voice/text)

    Note over Agent: Emit Events
    Agent->>HA: Fire conversation_finished event
```

### Flow Stages

1. **Input Processing**: User input received via Home Assistant conversation platform
2. **Session Management**: Retrieve or create persistent conversation session
3. **Context Assembly**: Parallel retrieval of entity context and memory context
4. **History Integration**: Load previous conversation messages
5. **LLM Interaction**: Iterative loop supporting multiple tool calls per turn
6. **Tool Execution**: Execute tools based on LLM decisions
7. **Response Generation**: Final response from LLM after all tool calls
8. **Persistence**: Save conversation history and extract memories
9. **Event Emission**: Fire Home Assistant events for observability

---

## Module Structure

This diagram shows the directory layout and key classes within the Home Agent codebase.

```mermaid
graph TD
    subgraph "custom_components/home_agent/"
        Init["__init__.py<br/>• async_setup<br/>• async_setup_entry<br/>• Service registration"]

        subgraph "agent/"
            AgentCore["core.py<br/>HomeAgent<br/>• async_process<br/>• process_message<br/>• _process_conversation"]
            AgentLLM["llm.py<br/>LLMMixin<br/>• _call_llm<br/>• _call_llm_streaming"]
            AgentStream["streaming.py<br/>StreamingMixin<br/>• _async_process_streaming<br/>• _can_stream"]
            AgentMemExt["memory_extraction.py<br/>MemoryExtractionMixin<br/>• _extract_and_store_memories"]
        end

        subgraph "context_providers/"
            ProviderBase["base.py<br/>ContextProvider<br/>• Abstract interface"]
            ProviderDirect["direct.py<br/>DirectContextProvider<br/>• Entity filtering"]
            ProviderVectorDB["vector_db.py<br/>VectorDBContextProvider<br/>• Semantic search"]
            ProviderMemory["memory.py<br/>MemoryContextProvider<br/>• Memory injection"]
        end

        subgraph "tools/"
            ToolRegistry["registry.py<br/>ToolRegistry<br/>• Tool registration"]
            ToolHAControl["ha_control.py<br/>HomeAssistantControlTool<br/>• turn_on/off/toggle"]
            ToolHAQuery["ha_query.py<br/>HomeAssistantQueryTool<br/>• get_state"]
            ToolExtLLM["external_llm.py<br/>ExternalLLMTool<br/>• Delegate queries"]
            ToolCustom["custom.py<br/>CustomToolHandler<br/>• REST/Service tools"]
            ToolMemory["memory_tools.py<br/>• StoreMemoryTool<br/>• RecallMemoryTool"]
        end

        subgraph "config/"
            ConfigFlow["flow.py<br/>• Configuration UI flows"]
            ConfigSchemas["schemas.py<br/>• Config validation schemas"]
            ConfigValidators["validators.py<br/>• Field validators"]
        end

        ContextManager["context_manager.py<br/>ContextManager<br/>• get_formatted_context<br/>• Provider orchestration"]

        Conversation["conversation.py<br/>ConversationHistoryManager<br/>• add_message<br/>• get_history<br/>• Persistence"]

        ConvSession["conversation_session.py<br/>ConversationSessionManager<br/>• Session tracking<br/>• Auto-expiration"]

        ToolHandlerMain["tool_handler.py<br/>ToolHandler<br/>• execute_tool<br/>• validate_tool_call<br/>• Metrics"]

        MemoryManagerMain["memory_manager.py<br/>MemoryManager<br/>• add_memory<br/>• search_memories<br/>• Dual storage"]

        VectorDBManager["vector_db_manager.py<br/>VectorDBManager<br/>• ChromaDB interface<br/>• Embeddings"]

        Streaming["streaming.py<br/>OpenAIStreamingHandler<br/>• Stream transformation<br/>• Token usage tracking"]

        ConfigFlowMain["config_flow.py<br/>HomeAgentConfigFlow<br/>• UI configuration<br/>• Options flow"]

        Const["const.py<br/>• Constants<br/>• Defaults<br/>• Event names"]

        Exceptions["exceptions.py<br/>• Custom exceptions"]

        Helpers["helpers.py<br/>• Utility functions"]
    end

    Init --> AgentCore
    Init --> MemoryManagerMain
    Init --> VectorDBManager
    Init --> ConvSession

    AgentCore --> AgentLLM
    AgentCore --> AgentStream
    AgentCore --> AgentMemExt
    AgentCore --> ContextManager
    AgentCore --> Conversation
    AgentCore --> ToolHandlerMain
    AgentCore --> ConvSession

    ContextManager --> ProviderBase
    ProviderBase --> ProviderDirect
    ProviderBase --> ProviderVectorDB
    ProviderBase --> ProviderMemory

    ProviderVectorDB --> VectorDBManager
    ProviderMemory --> MemoryManagerMain

    ToolHandlerMain --> ToolHAControl
    ToolHandlerMain --> ToolHAQuery
    ToolHandlerMain --> ToolExtLLM
    ToolHandlerMain --> ToolCustom
    ToolHandlerMain --> ToolMemory

    ToolMemory --> MemoryManagerMain

    AgentStream --> Streaming

    ConfigFlowMain --> ConfigFlow
    ConfigFlowMain --> ConfigSchemas
    ConfigFlowMain --> ConfigValidators

    style AgentCore fill:#4CAF50
    style ContextManager fill:#2196F3
    style ToolHandlerMain fill:#FF9800
    style MemoryManagerMain fill:#9C27B0
```

### Directory Organization

```
custom_components/home_agent/
├── agent/                      # Main agent implementation (mixin-based)
│   ├── core.py                # HomeAgent orchestrator class
│   ├── llm.py                 # LLM API communication
│   ├── streaming.py           # Streaming response support
│   └── memory_extraction.py   # Memory extraction logic
├── context_providers/         # Context injection strategies
│   ├── base.py               # Abstract provider interface
│   ├── direct.py             # Direct entity filtering
│   ├── vector_db.py          # ChromaDB semantic search
│   └── memory.py             # Memory-based context
├── tools/                     # LLM-callable tools
│   ├── ha_control.py         # Home Assistant control
│   ├── ha_query.py           # Home Assistant queries
│   ├── external_llm.py       # External LLM delegation
│   ├── custom.py             # Custom REST/service tools
│   └── memory_tools.py       # Memory operations
├── config/                    # Configuration management
│   ├── flow.py               # Config flow steps
│   ├── schemas.py            # Validation schemas
│   └── validators.py         # Field validators
├── context_manager.py         # Context orchestration
├── conversation.py            # History management
├── conversation_session.py    # Session persistence
├── tool_handler.py            # Tool execution
├── memory_manager.py          # Long-term memory
├── vector_db_manager.py       # ChromaDB interface
├── streaming.py               # Streaming utilities
├── config_flow.py             # UI configuration
└── const.py                   # Constants & defaults
```

---

## Component Details

### HomeAgent (Core)

**Purpose**: Central orchestrator for all conversation-related functionality

**Responsibilities**:
- Process user inputs through Home Assistant's conversation platform
- Build and manage conversation context (system prompts, entity states)
- Execute multi-turn conversations with tool calling support
- Coordinate between LLM, tools, and Home Assistant services
- Track conversation history and metrics
- Support both streaming and synchronous response modes

**Key Methods**:
- `async_process()`: Main entry point from Home Assistant
- `process_message()`: Direct message processing
- `_process_conversation()`: Tool calling loop implementation
- `_build_system_prompt()`: Construct system prompt with context

**Architecture**: Uses mixin-based design inheriting from:
- `LLMMixin`: LLM API communication
- `StreamingMixin`: Real-time streaming responses
- `MemoryExtractionMixin`: Automatic memory extraction
- `AbstractConversationAgent`: Home Assistant integration

---

### ContextManager

**Purpose**: Manages context injection strategies for LLM conversations

**Responsibilities**:
- Orchestrate different context providers (direct, vector DB, memory)
- Optimize context size to stay within token limits
- Cache context when appropriate
- Fire events for observability

**Context Modes**:
1. **Direct Mode**: Static entity list, always includes configured entities
2. **Vector DB Mode**: Dynamic semantic search based on user query
3. **Memory Mode**: Inject relevant long-term memories

**Key Methods**:
- `get_formatted_context()`: Main entry point, returns optimized context
- `get_context()`: Retrieve raw context from provider(s)
- `_optimize_context_size()`: Compress and truncate if needed
- `set_provider()`: Switch context strategy
- `set_memory_provider()`: Enable memory context

---

### ConversationHistoryManager

**Purpose**: Maintain conversation history across multiple turns

**Features**:
- Per-conversation history tracking
- Message and token limits
- Persistent storage across Home Assistant restarts
- Debounced saves to reduce I/O
- Token estimation for context management

**Storage Format**:
```json
{
  "version": 1,
  "conversations": {
    "conversation_id": [
      {"role": "user", "content": "...", "timestamp": 1234567890},
      {"role": "assistant", "content": "..."}
    ]
  }
}
```

**Key Methods**:
- `add_message()`: Add message to history
- `get_history()`: Retrieve recent messages with limits
- `clear_history()`: Clear specific conversation
- `estimate_tokens()`: Estimate token usage

---

### ToolHandler

**Purpose**: Manage tool registration, validation, and execution

**Features**:
- Tool registration and validation
- Timeout enforcement
- Parallel tool execution support
- Execution metrics tracking
- Tool progress events

**Tool Interface**: Each tool must implement:
- `name`: Unique tool identifier
- `execute(**params)`: Async execution method
- `get_definition()`: Return tool schema
- `to_openai_format()`: Format for LLM consumption

**Key Methods**:
- `register_tool()`: Add tool to registry
- `execute_tool()`: Execute with timeout and metrics
- `get_tool_definitions()`: Format all tools for LLM
- `validate_tool_call()`: Pre-execution validation

---

### MemoryManager

**Purpose**: Long-term memory storage and retrieval system

**Features**:
- Dual storage: Home Assistant Store + ChromaDB
- Memory types: facts, preferences, context, events
- Importance scoring with decay
- Deduplication via semantic similarity
- TTL-based expiration
- Periodic cleanup

**Memory Lifecycle**:
1. **Extraction**: Async extraction from conversations
2. **Validation**: Filter out transient/low-quality content
3. **Deduplication**: Check semantic similarity to existing memories
4. **Storage**: Store in both HA Store and ChromaDB
5. **Retrieval**: Semantic search based on user queries
6. **Decay**: Importance score decreases over time
7. **Expiration**: Remove based on TTL and importance

**Key Methods**:
- `add_memory()`: Store new memory with deduplication
- `search_memories()`: Semantic similarity search
- `apply_importance_decay()`: Reduce importance over time
- `_cleanup_expired_memories()`: Remove expired memories

---

### ConversationSessionManager

**Purpose**: Maintain persistent conversation sessions for voice interactions

**Features**:
- User/device-based session mapping
- Automatic expiration (configurable timeout)
- Persistent storage across restarts
- Session activity tracking

**Use Case**: Enables natural multi-turn voice conversations where follow-up questions maintain context without explicitly providing conversation IDs.

**Example**:
```
User: "What's the temperature in the living room?"
Agent: "The living room is 72°F"
[Later, same device]
User: "What about the bedroom?"
Agent: "The bedroom temperature is 68°F"  # Maintains context
```

---

## Data Flow Patterns

### Context Injection Pattern

```mermaid
flowchart LR
    Input[User Input] --> ContextMgr

    subgraph Parallel Context Retrieval
        ContextMgr --> EntityContext[Entity Context<br/>Direct or VectorDB]
        ContextMgr --> MemoryContext[Memory Context<br/>If enabled]
    end

    EntityContext --> Combine
    MemoryContext --> Combine
    Combine[Combine Contexts] --> Optimize[Optimize Size<br/>Token limits]
    Optimize --> SystemPrompt[System Prompt]
```

### Tool Execution Pattern

```mermaid
flowchart TD
    LLMResponse[LLM Response] --> CheckTools{Has Tool Calls?}

    CheckTools -->|No| FinalResponse[Return Response]
    CheckTools -->|Yes| ValidateTools[Validate Tool Calls]

    ValidateTools --> ExecuteParallel[Execute Tools<br/>in Parallel]
    ExecuteParallel --> Timeout[Apply Timeout]
    Timeout --> Results[Collect Results]
    Results --> AddToMessages[Add Results to Messages]
    AddToMessages --> NextIteration[Next LLM Call]
    NextIteration --> CheckTools

    CheckTools -->|Max iterations<br/>reached| MaxError[Return Error]
```

### Memory Storage Pattern

```mermaid
flowchart TD
    ConvEnd[Conversation Ends] --> Extract[Extract Memories<br/>via LLM]
    Extract --> Validate{Valid Memory?}

    Validate -->|No - Transient| Skip[Skip]
    Validate -->|Yes| CheckDup{Check Duplicates<br/>Semantic Similarity}

    CheckDup -->|Duplicate Found| Merge[Merge with Existing<br/>Boost importance]
    CheckDup -->|No Duplicate| CreateNew[Create New Memory]

    Merge --> StoreHA[Store in HA Store]
    CreateNew --> StoreHA

    StoreHA --> StoreChroma[Store in ChromaDB<br/>with embedding]
    StoreChroma --> CheckLimit{Over limit?}

    CheckLimit -->|Yes| Prune[Prune low-importance<br/>memories]
    CheckLimit -->|No| Done[Done]
    Prune --> Done
```

### Session Persistence Pattern

```mermaid
flowchart TD
    Input[User Input] --> CheckSession{Session Exists<br/>for user/device?}

    CheckSession -->|Yes| GetConvID[Get Conversation ID]
    CheckSession -->|No| CreateConvID[Create New<br/>Conversation ID]

    GetConvID --> CheckTimeout{Session<br/>Expired?}
    CheckTimeout -->|Yes| CreateConvID
    CheckTimeout -->|No| UseExisting[Use Existing Session]

    CreateConvID --> StoreMapping[Store Session Mapping]
    StoreMapping --> Process[Process Conversation]
    UseExisting --> Process

    Process --> UpdateActivity[Update Session Activity]
    UpdateActivity --> Done[Done]
```

---

## Integration Points

### Home Assistant Integration

1. **Conversation Platform**: Registers as `AbstractConversationAgent`
2. **Entity Registry**: Accesses entity states and metadata
3. **Service Calls**: Executes Home Assistant services via tools
4. **Storage**: Uses `Store` helper for persistence
5. **Events**: Fires custom events for observability
6. **Config Flow**: Provides UI configuration

### External Integrations

1. **LLM Providers**: OpenAI-compatible API (OpenAI, Ollama, LocalAI, etc.)
2. **ChromaDB**: Vector database for semantic search and memory storage
3. **Embeddings**: OpenAI embeddings API or Ollama for vector generation

---

## Performance Considerations

### Optimization Strategies

1. **Parallel Context Retrieval**: Entity and memory context fetched simultaneously
2. **Context Caching**: Cache context based on mode and input
3. **Debounced Saves**: Reduce I/O with delayed persistence
4. **Token Optimization**: Compress and truncate context to stay within limits
5. **Streaming Responses**: ~10x faster response time for voice assistants
6. **Lazy Tool Registration**: Defer registration until first use

### Scalability

- **Memory Limits**: Configurable max memories (default: 1000)
- **History Limits**: Configurable max messages and tokens
- **Tool Timeouts**: Prevent hanging tool executions (default: 30s)
- **Session Expiration**: Automatic cleanup of inactive sessions (default: 1 hour)
- **Periodic Cleanup**: Background task for memory maintenance

---

## Error Handling

### Exception Hierarchy

```
HomeAgentError (base)
├── ContextInjectionError
├── ToolExecutionError
├── ValidationError
├── TokenLimitExceeded
└── LLMError
```

### Fallback Strategies

1. **Streaming Failure**: Falls back to synchronous mode
2. **ChromaDB Unavailable**: Falls back to store-only mode for memory
3. **Context Too Large**: Truncates with warning
4. **Tool Timeout**: Returns error to LLM, continues conversation
5. **Memory Extraction Failure**: Logs error, conversation continues

---

## Event System

Home Agent emits events for observability and automation:

- `home_agent.conversation.started`
- `home_agent.conversation.finished` (with metrics)
- `home_agent.context.injected`
- `home_agent.context.optimized`
- `home_agent.tool.executed`
- `home_agent.tool.progress`
- `home_agent.error`
- `home_agent.streaming.error`
- `home_agent.memory.extracted`
- `home_agent.history.saved`

Events can be used for:
- Monitoring and alerting
- Triggering automations
- Collecting metrics (Prometheus, InfluxDB)
- Debugging and troubleshooting

---

## Configuration Architecture

Configuration is managed through multiple layers:

1. **Config Entry**: Primary configuration via UI
2. **Options Flow**: Runtime reconfiguration via UI
3. **YAML Config**: Custom tools and advanced settings
4. **Runtime Updates**: Live config updates without restart

### Configuration Flow

```mermaid
flowchart LR
    UI[UI Configuration] --> ConfigEntry[Config Entry]
    YAML[configuration.yaml] --> YAMLConfig[YAML Config]

    ConfigEntry --> Merge[Merge Configs]
    YAMLConfig --> Merge

    Merge --> Agent[HomeAgent]
    Merge --> Components[Components]

    Agent --> Validate[Validate]
    Components --> Validate

    Validate --> Apply[Apply Config]
```

---

## Security Considerations

1. **API Key Storage**: Stored securely in Home Assistant config entry
2. **Entity Exposure**: Respects Home Assistant's exposure settings
3. **Tool Validation**: Validates all tool calls before execution
4. **Timeout Enforcement**: Prevents runaway tool executions
5. **Template Sandboxing**: Uses Home Assistant's template engine
6. **Service Call Security**: Uses Home Assistant's permission system

---

## Further Reading

- [Configuration Reference](CONFIGURATION.md)
- [Custom Tools Guide](CUSTOM_TOOLS.md)
- [Memory System](MEMORY_SYSTEM.md)
- [API Reference](API_REFERENCE.md)
- [Development Standards](.claude/docs/DEVELOPMENT.md)

# Memory System User Guide

## Overview

The Home Agent memory system provides long-term memory capabilities that persist facts, preferences, context, and events across conversations. This enables the agent to remember user preferences, past interactions, and important information over time.

### What is the Memory System?

The memory system automatically extracts and stores important information from conversations, making it available in future interactions. Think of it as giving your home agent a persistent memory that improves over time.

**Key Features**:
- **Automatic extraction** from conversations
- **Semantic search** for relevant recall
- **Type-based organization** (facts, preferences, context, events)
- **Importance scoring** with access-based reinforcement
- **Deduplication** to avoid redundant memories
- **Privacy controls** with full on/off toggle
- **Retention policies** with configurable TTL per type

### How It Works (Architecture)

```
┌─────────────────────────────────────────────────────┐
│ User Conversation with Home Agent                  │
│ "I prefer the bedroom at 68°F for sleeping"        │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│ Primary LLM Response                                │
│ "I'll remember that preference"                    │
└────────────┬────────────────────────────────────────┘
             │
             ▼ (after conversation completes)
┌─────────────────────────────────────────────────────┐
│ Automatic Memory Extraction (if enabled)           │
│ - Uses External LLM OR Local LLM                   │
│ - Analyzes conversation for important info         │
│ - Extracts structured memories                     │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│ Memory Manager Storage                              │
│ ├─ Home Assistant Store (.storage/)                │
│ │  └─ Persistent JSON storage                      │
│ └─ ChromaDB (semantic indexing)                    │
│    └─ Vector embeddings for search                 │
└────────────┬────────────────────────────────────────┘
             │
             ▼ (next conversation)
┌─────────────────────────────────────────────────────┐
│ User: "What temperature do I like for sleeping?"   │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│ Memory Context Provider                             │
│ - Semantic search: "sleeping temperature"          │
│ - Retrieves: "User prefers 68°F for bedroom"       │
└────────────┬────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│ Primary LLM (with memory context)                  │
│ "You prefer the bedroom at 68°F for sleeping"      │
└─────────────────────────────────────────────────────┘
```

### Use Cases

**Personal Preferences**:
- Temperature preferences for different times of day
- Lighting preferences for different activities
- Music and media preferences
- Coffee brewing settings

**Important Facts**:
- Allergy information
- Schedule and routine information
- Pet names and care requirements
- Guest preferences

**Contextual Information**:
- Recent conversations and decisions
- Temporary notes and reminders
- Short-term state tracking

**Events**:
- Maintenance schedules
- Important occurrences
- Temporary conditions

### Privacy Implications

**What is Stored**:
- Extracted facts, preferences, and events from conversations
- Source conversation ID and timestamp
- Importance scores and access metadata
- User-provided context and topics

**What is NOT Stored**:
- Full conversation transcripts (unless extraction fails)
- Sensitive authentication data
- Personal identifiable information (unless explicitly mentioned by user)

**Privacy Controls**:
- **Master Toggle**: Disable the entire memory system with one setting
- **Automatic Extraction Toggle**: Disable automatic extraction (manual only)
- **Data Location**: All data stored locally in `.storage/home_agent.memories`
- **Manual Deletion**: Clear all memories with one service call
- **GDPR Compliance**: All data is local and user-controlled

**Memory Scope**:
- Memories are **global** (shared across all users and conversations)
- No per-user memory isolation currently implemented
- Consider this for multi-user households

## Configuration

### Enabling the Memory System

1. **Open Home Agent Configuration**

   Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure**

2. **Select Memory Settings**

   Choose **Memory Settings** from the menu

3. **Configure Memory System**

   | Field | Description | Default |
   |-------|-------------|---------|
   | **Memory Enabled** | Master toggle for the entire memory system | `On` |
   | **Automatic Extraction Enabled** | Enable automatic memory extraction after conversations | `On` |
   | **Extraction LLM** | Which LLM to use for extraction: `external` or `local` | `external` |
   | **Max Memories** | Maximum number of memories to store (oldest pruned) | `100` |
   | **Minimum Importance** | Minimum importance score to keep (0.0-1.0) | `0.3` |
   | **Context Top K** | Number of memories to inject into conversations | `5` |
   | **Collection Name** | ChromaDB collection name for memory storage | `home_agent_memories` |

### Configuration Options Explained

#### CONF_MEMORY_ENABLED (Privacy Toggle)

**Type**: Boolean
**Default**: `True`

Master switch for the entire memory system. When disabled:
- No memories are extracted
- No memories are recalled
- Existing memories remain in storage but are not used
- All memory services remain available for manual management

**Use Case**: Disable for privacy-sensitive conversations or testing.

#### CONF_MEMORY_EXTRACTION_ENABLED

**Type**: Boolean
**Default**: `True`

Controls automatic memory extraction after conversations. When disabled:
- Conversations complete normally
- No automatic extraction occurs
- Manual memory storage via `store_memory` tool still works
- Existing memories can still be recalled

**Use Case**: Disable automatic extraction if you prefer manual control or want to reduce LLM API calls.

#### CONF_MEMORY_EXTRACTION_LLM

**Type**: String (`external` or `local`)
**Default**: `external`

Chooses which LLM to use for memory extraction:

- **`external`**: Uses the configured external LLM (must be enabled in External LLM settings)
  - **Pros**: Higher quality extraction, better reasoning
  - **Cons**: Requires external LLM configured, API costs
  - **Best for**: Production use with quality extraction

- **`local`**: Uses the primary LLM (same as conversation LLM)
  - **Pros**: No additional LLM required, no extra costs
  - **Cons**: May be lower quality for smaller models
  - **Best for**: Self-hosted setups, cost optimization

**Note**: If `external` is selected but external LLM is not enabled, configuration will show an error.

#### CONF_MEMORY_MAX_MEMORIES

**Type**: Integer (10-1000)
**Default**: `100`

Maximum number of memories to store. When exceeded:
- Oldest/least important memories are pruned
- Importance score and last access time determine pruning order
- Recently accessed memories are preserved

**Tuning**:
- Small household: 50-100 memories
- Medium household: 100-200 memories
- Large household: 200-500 memories
- Power users: 500-1000 memories

#### CONF_MEMORY_MIN_IMPORTANCE

**Type**: Float (0.0-1.0)
**Default**: `0.3`

Minimum importance threshold for keeping memories. Memories below this score are not stored or are pruned during cleanup.

**Importance Levels**:
- `0.0-0.2`: Trivial information, usually discarded
- `0.3-0.5`: Moderate importance (default threshold)
- `0.6-0.8`: Important information
- `0.9-1.0`: Critical information

**Tuning**:
- **Higher threshold** (0.5-0.7): Only keep very important information, reduces storage
- **Lower threshold** (0.1-0.3): Keep more information, increases storage

#### CONF_MEMORY_CONTEXT_TOP_K

**Type**: Integer (1-20)
**Default**: `5`

Number of memories to inject into conversation context when relevant memories are found.

**Impact**:
- **Lower values** (1-3): Minimal context, focused relevance, lower token usage
- **Higher values** (10-20): Comprehensive context, may include less relevant memories, higher token usage

**Tuning**:
- Start with `5`
- Increase if LLM seems to lack context
- Decrease if getting too much irrelevant information

#### CONF_MEMORY_COLLECTION_NAME

**Type**: String
**Default**: `home_agent_memories`

ChromaDB collection name for memory storage. Change if you want separate memory databases.

**Use Cases**:
- `home_agent_memories_main` vs `home_agent_memories_test`
- Different collections for different homes/locations
- Backup and restore scenarios

#### Advanced Configuration Constants

These are set in `const.py` and not exposed in the UI (advanced users only):

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| **CONF_MEMORY_IMPORTANCE_DECAY** | Float | `0.0` | Decay rate for importance scores over time (0.0 = no decay) |
| **CONF_MEMORY_DEDUP_THRESHOLD** | Float | `0.95` | Similarity threshold for duplicate detection (0.0-1.0) |
| **CONF_MEMORY_EVENT_TTL** | Integer | `300` | Time-to-live for event memories (seconds, 5 minutes) |
| **CONF_MEMORY_FACT_TTL** | Integer/None | `None` | Time-to-live for fact memories (no expiration) |
| **CONF_MEMORY_PREFERENCE_TTL** | Integer | `7776000` | Time-to-live for preference memories (90 days) |
| **CONF_MEMORY_CLEANUP_INTERVAL** | Integer | `300` | Cleanup task interval (seconds, 5 minutes) |

## Using Memory Features

### Automatic Memory Extraction

When enabled, Home Agent automatically extracts memories from conversations.

**How it Works**:
1. User has a conversation with Home Agent
2. Conversation completes successfully
3. Memory extraction LLM analyzes the conversation
4. Important information is extracted as structured memories
5. Memories are stored in both Home Assistant Store and ChromaDB
6. Future conversations can recall these memories

**Example Conversation**:

```
User: "I prefer the bedroom temperature at 68 degrees when sleeping"
Agent: "I'll remember that you prefer the bedroom at 68°F for sleeping."

[Behind the scenes]
- Extraction LLM analyzes conversation
- Extracts:
  - Type: preference
  - Content: "User prefers bedroom temperature at 68°F for sleeping"
  - Importance: 0.8
  - Topics: ["temperature", "bedroom", "sleep", "preferences"]
```

**Next Conversation**:

```
User: "What temperature should I set the bedroom to?"
Agent: "Based on your preferences, you like the bedroom at 68°F for sleeping."

[Behind the scenes]
- Query embedded: "bedroom temperature setting"
- ChromaDB searches memories
- Retrieves: "User prefers bedroom at 68°F for sleeping"
- Injected into conversation context
```

### Manual Memory Storage with `store_memory` Tool

The LLM can explicitly store memories during conversations using the `store_memory` tool.

**Tool Definition**:
```json
{
  "name": "store_memory",
  "description": "Store important information for future reference",
  "parameters": {
    "content": "The information to remember",
    "type": "fact|preference|context|event",
    "importance": 0.0-1.0 (optional)
  }
}
```

**Example LLM Tool Call**:

```json
{
  "tool": "store_memory",
  "parameters": {
    "content": "User's dog is named Max and is allergic to chicken",
    "type": "fact",
    "importance": 0.9
  }
}
```

**Response**:
```json
{
  "success": true,
  "result": "Memory stored successfully",
  "memory_id": "abc-123-def-456"
}
```

### Memory Recall with `recall_memory` Tool

The LLM can explicitly search for relevant memories using the `recall_memory` tool.

**Tool Definition**:
```json
{
  "name": "recall_memory",
  "description": "Search for relevant memories based on a query",
  "parameters": {
    "query": "What to search for",
    "limit": 5 (optional),
    "memory_types": ["fact", "preference"] (optional)
  }
}
```

**Example LLM Tool Call**:

```json
{
  "tool": "recall_memory",
  "parameters": {
    "query": "dog allergies",
    "limit": 3
  }
}
```

**Response**:
```json
{
  "success": true,
  "result": [
    {
      "id": "abc-123",
      "content": "User's dog Max is allergic to chicken",
      "type": "fact",
      "importance": 0.9,
      "extracted_at": "2025-11-01T10:30:00Z"
    }
  ]
}
```

### Memory Types

The system supports four memory types, each with different retention policies:

#### fact
**Purpose**: Permanent factual information
**TTL**: None (permanent by default)
**Examples**:
- "User's dog is named Max"
- "Living room light switch is by the front door"
- "Garbage pickup is on Thursdays"

**Best For**: Information that doesn't change or rarely changes.

#### preference
**Purpose**: User preferences and settings
**TTL**: 90 days (configurable)
**Examples**:
- "User prefers bedroom temperature at 68°F for sleeping"
- "User likes bright lights in the morning"
- "User prefers jazz music in the evening"

**Best For**: User preferences that may evolve over time.

#### context
**Purpose**: Conversational context and temporary notes
**TTL**: 5 minutes (same as events)
**Examples**:
- "Currently discussing home automation ideas"
- "User is planning a party this weekend"
- "User mentioned feeling cold today"

**Best For**: Short-term context that's only relevant for recent conversations.

#### event
**Purpose**: Timestamped events and occurrences
**TTL**: 5 minutes (configurable)
**Examples**:
- "Furnace filter was changed on 2025-11-01"
- "Package delivered at 2:30 PM"
- "Door left open for 10 minutes"

**Best For**: Time-sensitive events that expire quickly.

## Managing Memories

### List Memories (`home_agent.list_memories`)

View all stored memories with optional filtering.

**Service Parameters**:
```yaml
service: home_agent.list_memories
data:
  memory_type: preference  # Optional: fact, preference, context, event
  limit: 50                # Optional: maximum number to return
```

**Example Response**:
```json
{
  "memories": [
    {
      "id": "abc-123-def-456",
      "type": "preference",
      "content": "User prefers bedroom at 68°F for sleeping",
      "importance": 0.8,
      "extracted_at": "2025-11-01T10:30:00Z",
      "last_accessed": "2025-11-02T08:15:00Z",
      "source_conversation_id": "conv_123",
      "metadata": {
        "entities_involved": ["climate.bedroom"],
        "topics": ["temperature", "sleep"]
      }
    }
  ]
}
```

### Search Memories (`home_agent.search_memories`)

Semantically search for relevant memories.

**Service Parameters**:
```yaml
service: home_agent.search_memories
data:
  query: "temperature preferences"  # Required
  limit: 10                          # Optional, default: 10
  min_importance: 0.5                # Optional, default: 0.0
```

**Use Cases**:
- Find related memories before manual edits
- Debug memory retrieval
- Export memories for backup
- Analyze what the agent knows

### Add Memory Manually (`home_agent.add_memory`)

Manually add a memory without going through conversation extraction.

**Service Parameters**:
```yaml
service: home_agent.add_memory
data:
  content: "User's cat Felix is on a prescription diet"  # Required
  type: fact                                              # Optional: fact, preference, context, event
  importance: 0.8                                         # Optional: 0.0-1.0, default: 0.5
```

**Use Cases**:
- Seeding initial knowledge
- Importing memories from other sources
- Manual data entry for specific facts

### Delete Memory (`home_agent.delete_memory`)

Delete a specific memory by ID.

**Service Parameters**:
```yaml
service: home_agent.delete_memory
data:
  memory_id: "abc-123-def-456"  # Required: get from list_memories
```

**Use Cases**:
- Remove incorrect or outdated information
- Privacy: delete specific sensitive memories
- Clean up after testing

### Clear All Memories (`home_agent.clear_memories`)

Delete all stored memories at once.

**Service Parameters**:
```yaml
service: home_agent.clear_memories
data:
  confirm: true  # Required: must be true to confirm deletion
```

**Warning**: This action is irreversible. All memories are permanently deleted from both Home Assistant Store and ChromaDB.

**Use Cases**:
- Complete reset
- Privacy: full data deletion
- Testing and development

## Privacy and Data Retention

### Storage Location

All memory data is stored locally in your Home Assistant instance:

**Home Assistant Store**:
- Path: `.storage/home_agent.memories`
- Format: JSON
- Contains: All memory content, metadata, and timestamps

**ChromaDB**:
- Collection: `home_agent_memories` (configurable)
- Format: Vector embeddings + metadata
- Contains: Embeddings for semantic search

### How to Delete All Data

**Option 1: Use Clear Memories Service**

```yaml
service: home_agent.clear_memories
data:
  confirm: true
```

**Option 2: Manually Delete Files**

```bash
# Stop Home Assistant
ha core stop

# Delete memory storage file
rm /config/.storage/home_agent.memories

# Delete ChromaDB collection (if using Docker)
docker exec chromadb rm -rf /chroma/chroma/home_agent_memories

# Restart Home Assistant
ha core start
```

**Option 3: Disable Memory System**

Navigate to **Memory Settings** and set **Memory Enabled** to `Off`.

### GDPR Considerations

**Data Controller**: You (the Home Assistant instance owner)
**Data Location**: Local storage only (no cloud transmission unless using external LLM for extraction)
**Data Subject Rights**:
- **Right to Access**: Use `list_memories` and `search_memories` services
- **Right to Erasure**: Use `delete_memory` or `clear_memories` services
- **Right to Portability**: Export via `list_memories` (JSON response)

**Compliance Tips**:
1. Inform household members that memories are stored globally (all users)
2. Provide access to memory management services
3. Regularly review stored memories with `list_memories`
4. Document your memory retention policies
5. Consider disabling automatic extraction for sensitive conversations

### Deduplication Behavior

The memory system automatically detects and merges duplicate memories to prevent redundant storage.

**How it Works**:
1. New memory content is embedded
2. ChromaDB searches for similar existing memories
3. If similarity exceeds threshold (default: 0.95), memories are merged:
   - **Content**: If new content is longer/more specific, it replaces old content
   - **Importance**: Boosted by 10% of new importance (reinforcement learning)
   - **Last Accessed**: Updated to current time
   - **Metadata**: Merged (union of both)

**Example**:

```
Existing Memory: "User likes bedroom at 68 degrees"
New Memory: "User prefers bedroom temperature at 68°F for sleeping"

Result:
- Content updated to more specific version
- Importance: 0.7 -> 0.75 (if new importance was 0.5)
- Merged into single memory
```

**Configuration**:
- **Threshold**: `CONF_MEMORY_DEDUP_THRESHOLD` (default: 0.95)
- **Higher threshold**: More strict deduplication (only very similar memories merged)
- **Lower threshold**: More lenient deduplication (more memories merged)

## Examples

### Example: Memory Extraction

**Conversation**:
```
User: "I always like the thermostat set to 72 during the day and 68 at night"
Agent: "Got it! I'll remember your temperature preferences."
```

**Extracted Memories**:
```json
[
  {
    "type": "preference",
    "content": "User prefers thermostat at 72°F during daytime",
    "importance": 0.8,
    "metadata": {
      "topics": ["temperature", "daytime", "thermostat"],
      "entities_involved": ["climate.thermostat"]
    }
  },
  {
    "type": "preference",
    "content": "User prefers thermostat at 68°F during nighttime",
    "importance": 0.8,
    "metadata": {
      "topics": ["temperature", "nighttime", "thermostat"],
      "entities_involved": ["climate.thermostat"]
    }
  }
]
```

### Example: Manual Memory Storage via LLM

**Conversation**:
```
User: "Remember that my cat Felix is on a prescription diet"
Agent: [Calls store_memory tool]
  {
    "content": "User's cat Felix is on a prescription diet",
    "type": "fact",
    "importance": 0.9
  }
Agent: "I'll remember that Felix is on a prescription diet."
```

### Example: Memory Recall

**Later Conversation**:
```
User: "What can my cat eat?"
Agent: [Calls recall_memory tool]
  {
    "query": "cat diet",
    "limit": 3
  }
Agent: [Receives memories]
  [
    {
      "content": "User's cat Felix is on a prescription diet",
      "importance": 0.9,
      "type": "fact"
    }
  ]
Agent: "Felix is on a prescription diet, so you should consult with your vet about what foods are appropriate."
```

### Example: Automation with Memories

**Use Case**: Automatically adjust temperature based on time and stored preferences

```yaml
automation:
  - alias: "Smart Temperature Adjustment"
    trigger:
      - platform: time
        at: "06:00:00"  # Morning
      - platform: time
        at: "22:00:00"  # Night
    action:
      - service: home_agent.search_memories
        data:
          query: "temperature preferences {{ 'daytime' if now().hour < 20 else 'nighttime' }}"
          limit: 1
        response_variable: memories

      - service: home_agent.process
        data:
          text: "Set the thermostat according to my preferences for {{ 'daytime' if now().hour < 20 else 'nighttime' }}"
```

### Example: Privacy-Focused Conversation

**Disable memory for sensitive conversation**:

```yaml
automation:
  - alias: "Private Conversation Mode"
    trigger:
      - platform: state
        entity_id: input_boolean.private_mode
        to: "on"
    action:
      # Temporarily disable memory system
      - service: homeassistant.set_config_entry_options
        data:
          config_entry_id: "<your_home_agent_config_entry_id>"
          options:
            memory_enabled: false
```

**Re-enable after private conversation**:

```yaml
automation:
  - alias: "Normal Conversation Mode"
    trigger:
      - platform: state
        entity_id: input_boolean.private_mode
        to: "off"
    action:
      - service: homeassistant.set_config_entry_options
        data:
          config_entry_id: "<your_home_agent_config_entry_id>"
          options:
            memory_enabled: true
```

## Troubleshooting

### No Memories Being Extracted

**Symptoms**: Conversations complete but no memories are stored

**Solutions**:
1. Verify **Memory Enabled** is `On`
2. Verify **Automatic Extraction Enabled** is `On`
3. Check if External LLM is configured (if using `extraction_llm: external`)
4. Review logs for extraction errors:
   ```
   [home_agent.memory_manager] Failed to extract memories: ...
   ```
5. Ensure ChromaDB is running and accessible
6. Try manual storage with `add_memory` service to test storage functionality

### Memories Not Being Recalled

**Symptoms**: Relevant memories exist but aren't used in conversations

**Solutions**:
1. Verify **Memory Enabled** is `On`
2. Check **Context Top K** setting (increase if too low)
3. Verify **Minimum Importance** threshold (lower if too high)
4. Test semantic search with `search_memories` service
5. Review logs for memory retrieval:
   ```
   [home_agent.context_manager] Retrieved 3 memories for context
   ```
6. Ensure memories have adequate importance scores

### ChromaDB Connection Issues

**Symptoms**: "Failed to connect to ChromaDB" or memory storage failures

**Solutions**:
1. Verify ChromaDB is running:
   ```bash
   curl http://localhost:8000/api/v1/heartbeat
   ```
2. Check ChromaDB configuration in Vector DB settings
3. Review ChromaDB logs:
   ```bash
   docker logs chromadb
   ```
4. See [Vector DB Setup Guide](VECTOR_DB_SETUP.md) for detailed troubleshooting

### Duplicate Memories

**Symptoms**: Similar memories stored multiple times

**Solutions**:
1. Check **Deduplication Threshold** (increase for more aggressive dedup)
2. Manually delete duplicates with `delete_memory`
3. Review logs for deduplication events:
   ```
   [home_agent.memory_manager] Duplicate found, merging: ...
   ```
4. Consider adjusting importance scores to prioritize better memories

### High Storage Usage

**Symptoms**: Large `.storage/home_agent.memories` file or ChromaDB storage

**Solutions**:
1. Reduce **Max Memories** setting
2. Increase **Minimum Importance** threshold
3. Configure shorter TTLs for memory types (advanced)
4. Periodically run `clear_memories` to reset
5. Review stored memories with `list_memories` and delete unnecessary ones

## Advanced Topics

### Importance Scoring and Access Boost

Every memory has an importance score (0.0-1.0) that determines:
- Whether the memory is stored (must exceed minimum importance)
- Pruning order when storage limit is reached
- Recall relevance in semantic search

**Access Boost**:
Each time a memory is accessed (via recall or retrieval), its importance increases by 0.05:

```python
new_importance = min(1.0, old_importance + 0.05)
```

This implements **reinforcement learning**: frequently accessed memories become more important over time.

### Memory Retention Policies

Each memory type has a configurable Time-To-Live (TTL):

| Type | Default TTL | Purpose |
|------|-------------|---------|
| **event** | 5 minutes | Short-term events that expire quickly |
| **context** | 5 minutes | Conversational context |
| **preference** | 90 days | User preferences that may evolve |
| **fact** | Never | Permanent factual information |

**Cleanup Process**:
- Runs every 5 minutes (configurable via `CONF_MEMORY_CLEANUP_INTERVAL`)
- Removes expired memories based on TTL
- Logs cleanup activity for monitoring

**Custom TTL** (advanced):
Edit `const.py` to change default TTL values:
```python
DEFAULT_MEMORY_EVENT_TTL = 300  # 5 minutes
DEFAULT_MEMORY_FACT_TTL = None  # Never expire
DEFAULT_MEMORY_PREFERENCE_TTL = 7776000  # 90 days
```

### Importance Decay

Optionally, importance scores can decay over time to gradually forget old information:

**Configuration**: `CONF_MEMORY_IMPORTANCE_DECAY` (default: 0.0, no decay)

**Example**:
```python
# 1% decay per cleanup interval
CONF_MEMORY_IMPORTANCE_DECAY = 0.01

# After cleanup:
new_importance = old_importance * (1.0 - 0.01)
```

**Use Case**: Gradually forget outdated preferences while keeping recently accessed memories.

**Warning**: Enabling decay may cause important memories to fade if not regularly accessed.

## Next Steps

After setting up the memory system:

1. **Monitor Memory Storage**
   - Periodically run `list_memories` to review what's stored
   - Check `.storage/home_agent.memories` file size
   - Monitor ChromaDB storage usage

2. **Fine-Tune Settings**
   - Adjust **Max Memories** based on usage patterns
   - Tune **Minimum Importance** for better relevance
   - Configure **Context Top K** for optimal recall

3. **Integrate with Automations**
   - Use memories in smart automations
   - Build context-aware scripts
   - Create personalized experiences

4. **Privacy Audits**
   - Regularly review stored memories
   - Delete outdated or unnecessary memories
   - Document your memory policies

## Additional Resources

- [Installation Guide](INSTALLATION.md) - Initial setup
- [Vector DB Setup Guide](VECTOR_DB_SETUP.md) - ChromaDB configuration (required for memories)
- [Project Specification](PROJECT_SPEC.md) - Technical architecture
- [README](../README.md) - Feature overview and examples

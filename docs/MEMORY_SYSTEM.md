# Memory System - Quick Start

## What is the Memory System?

The memory system gives Home Agent persistent long-term memory across conversations. It automatically extracts and stores important facts, preferences, and events, then recalls them when relevant.

**Think of it as:** Giving your home agent a memory that improves over time, remembering your preferences, past interactions, and important information.

**Key capabilities:**
- Automatic extraction from conversations
- Semantic search for relevant recall
- Four memory types (facts, preferences, context, events)
- Privacy controls with full on/off toggle
- Manual memory management services

**Example:**
```
You: "I prefer the bedroom at 68°F for sleeping"
Agent: "I'll remember that preference"

[Later...]
You: "What temperature should I set the bedroom to?"
Agent: "Based on your preferences, you like the bedroom at 68°F for sleeping"
```

## Quick Enable

### Prerequisites

- ChromaDB must be installed and configured (see [VECTOR_DB_SETUP.md](VECTOR_DB_SETUP.md))
- Embedding provider configured (Ollama or OpenAI)

### Enable Memory

1. Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure**
2. Select **Memory Settings**
3. Configure:

| Field | Recommended Value |
|-------|-------------------|
| Memory Enabled | `On` |
| Automatic Extraction Enabled | `On` |
| Extraction LLM | `external` (better quality) or `local` (no extra LLM needed) |
| Max Memories | `100` |
| Minimum Importance | `0.3` |
| Context Top K | `5` |
| Collection Name | `home_agent_memories` |

4. Save configuration

If using `external` extraction LLM, also configure External LLM settings with your preferred extraction model.

## Basic Configuration

### Essential Settings

**Memory Enabled:**
- Master on/off switch
- Disables all memory features when off
- Existing memories remain but aren't used

**Automatic Extraction Enabled:**
- Controls automatic extraction after conversations
- Manual memory storage still works when disabled

**Extraction LLM:**
- `external` - Uses configured external LLM (better quality)
- `local` - Uses primary conversation LLM (simpler, lower cost)

**Max Memories:**
- Maximum memories to store
- Oldest/least important pruned when exceeded
- Start with `100`, adjust based on usage

**Minimum Importance (0.0-1.0):**
- Memories below this threshold not stored
- `0.3` (default) - Moderate importance
- Higher = only keep very important info
- Lower = keep more information

**Context Top K:**
- Number of memories to inject into conversations
- `5` (recommended) - Balanced relevance and context
- Increase if LLM lacks context, decrease if too much irrelevant info

## Try It Out

### Example Conversation

**Store a preference:**
```
You: "I always like the living room lights at 50% brightness in the evening"
Agent: "I'll remember your lighting preference"
```

**Later, recall it:**
```
You: "How bright should I set the living room lights?"
Agent: "You prefer the living room lights at 50% brightness in the evening"
```

**Store a fact:**
```
You: "Remember that my dog Max is allergic to chicken"
Agent: "I'll remember that Max is allergic to chicken"
```

**Recall it:**
```
You: "What can my dog eat?"
Agent: "Max is allergic to chicken, so avoid chicken-based foods..."
```

## Managing Memories

### Available Services

**List all memories:**
```yaml
service: home_agent.list_memories
data:
  memory_type: preference  # Optional: fact, preference, context, event
  limit: 50                # Optional
```

**Search memories:**
```yaml
service: home_agent.search_memories
data:
  query: "temperature preferences"
  limit: 10
  min_importance: 0.5  # Optional
```

**Add memory manually:**
```yaml
service: home_agent.add_memory
data:
  content: "User's cat Felix is on a prescription diet"
  type: fact
  importance: 0.8  # Optional, default: 0.5
```

**Delete specific memory:**
```yaml
service: home_agent.delete_memory
data:
  memory_id: "abc-123-def-456"  # Get from list_memories
```

**Clear all memories:**
```yaml
service: home_agent.clear_memories
data:
  confirm: true  # Required
```

### Memory Types

**fact** - Permanent factual information
- No expiration
- Examples: "Dog named Max", "Garbage pickup on Thursdays"

**preference** - User preferences and settings
- Expires after 90 days (configurable)
- Examples: "Prefers 68°F for sleeping", "Likes jazz in evening"

**context** - Conversational context
- Expires after 5 minutes
- Examples: "Currently discussing automation", "User planning party"

**event** - Timestamped events
- Expires after 5 minutes
- Examples: "Filter changed on 2025-11-01", "Package delivered"

## Privacy Note

**What's stored:**
- Extracted facts, preferences, and events
- Timestamps and importance scores
- No full conversation transcripts

**Where it's stored:**
- Locally in `.storage/home_agent.memories`
- ChromaDB collection (local)
- Never sent to cloud except for embedding generation (if using OpenAI)

**Your controls:**
- Master toggle to disable entirely
- Manual deletion of specific memories
- Full clear with one service call
- All data local and user-controlled

**Multi-user consideration:**
- Memories are global (shared across all users)
- No per-user isolation currently
- Consider this for multi-user households

**Complete deletion:**
```yaml
service: home_agent.clear_memories
data:
  confirm: true
```

Or manually delete:
```bash
rm /config/.storage/home_agent.memories
```

## Automatic Behavior

**Memory extraction happens automatically:**
1. You have a conversation with Home Agent
2. Conversation completes successfully
3. Extraction LLM analyzes the conversation
4. Important information extracted as structured memories
5. Memories stored in Home Assistant and ChromaDB
6. Future conversations can recall these memories

**Memory recall happens automatically:**
1. You ask a question
2. Query is embedded and searched in ChromaDB
3. Top K relevant memories retrieved
4. Memories injected into conversation context
5. LLM responds with memory-aware answer

**No action needed** - it all happens in the background!

## Advanced: Manual LLM Tools

The LLM can explicitly manage memories during conversations:

**Store memory:**
```json
{
  "tool": "store_memory",
  "parameters": {
    "content": "User's cat Felix is on prescription diet",
    "type": "fact",
    "importance": 0.9
  }
}
```

**Recall memory:**
```json
{
  "tool": "recall_memory",
  "parameters": {
    "query": "dog allergies",
    "limit": 3
  }
}
```

These tools are available to the LLM automatically when memory is enabled.

## Troubleshooting

**No memories being extracted:**
- Verify Memory Enabled and Automatic Extraction Enabled are both On
- Check External LLM is configured (if using extraction_llm: external)
- Review logs for extraction errors
- Ensure ChromaDB is running

**Memories not being recalled:**
- Verify Memory Enabled is On
- Increase Context Top K if too low
- Lower Minimum Importance threshold
- Test with search_memories service

**ChromaDB connection issues:**
- Verify ChromaDB running: `curl http://localhost:8000/api/v1/heartbeat`
- Check Vector DB settings configuration
- See [VECTOR_DB_SETUP.md](VECTOR_DB_SETUP.md) for troubleshooting

## Need More Details?

See the [Complete Memory System Reference](reference/MEMORY_SYSTEM.md) for:
- Detailed architecture and data flow
- All configuration constants
- Advanced importance scoring
- Retention policies and TTL settings
- Deduplication behavior
- GDPR compliance considerations
- Performance tuning
- Integration examples

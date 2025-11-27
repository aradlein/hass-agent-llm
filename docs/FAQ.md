# Frequently Asked Questions (FAQ)

Quick answers to common questions. For comprehensive coverage, see the [Complete FAQ](reference/FAQ.md).

---

## General

### What is Home Agent?

Home Agent is a custom Home Assistant integration providing intelligent conversational AI with OpenAI-compatible LLMs. It enables natural language control with advanced features like custom tools, memory system, multi-LLM workflows, and streaming responses.

### How is it different from extended_openai_conversation?

**Key advantages:**
- **Memory system**: Remembers preferences and facts across conversations
- **Streaming responses**: ~10x faster perceived response time for voice
- **Dual-LLM support**: Use fast local LLM for control, powerful cloud LLM for analysis
- **Vector DB context**: Smart entity retrieval for large setups

See [MIGRATION.md](MIGRATION.md) for migration guide.

### Which LLM providers are supported?

Any OpenAI-compatible API:
- **Cloud**: OpenAI, Anthropic (via proxy), Groq, OpenRouter
- **Local**: Ollama, LM Studio, LocalAI, vLLM

**Configuration:**
```yaml
# OpenAI
LLM Base URL: https://api.openai.com/v1

# Ollama (local)
LLM Base URL: http://localhost:11434/v1
```

### Can I use local models only?

**Yes!** Works completely offline with Ollama or other local models.

**Setup:**
```yaml
LLM Base URL: http://localhost:11434/v1
Model: llama2:13b
External LLM: false
Memory Extraction LLM: local
```

**Hardware requirements:**
- Minimum: 16GB RAM, modern CPU
- Recommended: 32GB RAM, NVIDIA GPU

### What's the minimum Home Assistant version?

**Minimum**: Home Assistant 2024.1.0
**Recommended**: Home Assistant 2024.6.0+

---

## Features

### What is the memory system?

Automatically extracts and recalls facts, preferences, and context across conversations.

**How it works:**
1. LLM extracts important info after each conversation
2. Stored locally in `.storage` directory
3. Semantically indexed in ChromaDB
4. Relevant memories auto-injected in future chats

**Example:**
```
User: "I prefer bedroom at 68 degrees for sleeping"
→ Memory stored

[Next day]
User: "I'm going to bed"
→ "Setting bedroom to 68 degrees, as you prefer"
```

**Memory types**: fact, preference, context, event

### Should I enable memory?

**Enable if:**
- You want personalized responses
- You use routines and patterns
- You're comfortable with local data storage

**Disable if:**
- You prioritize privacy
- Simple device control only
- Shared Home Assistant

**Privacy**: All stored locally unless using cloud LLM for extraction.

### Direct vs Vector DB mode?

| Aspect | Direct | Vector DB |
|--------|--------|-----------|
| Setup | Easy | Requires ChromaDB |
| Best for | <100 entities | 100+ entities |
| Context | Fixed entities | Dynamic retrieval |
| Token efficiency | Lower | Higher |

**Recommendation**: Start with direct, upgrade to vector DB for large setups.

### Can I use multiple LLMs?

**Yes!** Dual-LLM strategy:

**Primary LLM** (local/fast):
- Handles most conversations
- Executes tools
- Example: Ollama llama2 or gpt-4o-mini

**External LLM** (cloud/powerful):
- Tool for complex queries
- Used when primary needs help
- Example: GPT-4o or Claude

**Cost benefit**: 80% queries use cheap local, 20% use expensive cloud only when needed.

### What are custom tools?

**Built-in tools**: `ha_control`, `ha_query`, `query_external_llm`, `store_memory`, `recall_memory`

**Custom tools**: User-defined in `configuration.yaml`
- REST APIs (weather, calendars, external services)
- Home Assistant services (automations, scenes)

**Example:**
```yaml
home_agent:
  custom_tools:
    - name: check_weather
      handler:
        type: rest
        url: "https://api.weather.com/..."
    - name: movie_mode
      handler:
        type: service
        service: scene.turn_on
```

See [CUSTOM_TOOLS.md](CUSTOM_TOOLS.md) for details.

---

## Performance & Cost

### How much do API calls cost?

**OpenAI GPT-4o-mini** (recommended):
- ~$0.0002 per query (0.02 cents)
- ~$0.60/month for 100 queries/day

**OpenAI GPT-4o**:
- ~$0.004 per query (0.4 cents)
- ~$12/month for 100 queries/day

**Local Ollama**: $0/month (free)

**Cost optimization**: Use local primary LLM + external LLM only for complex queries

### Why are responses slow?

**Common causes & fixes:**
1. **Large context**: Reduce entities, use vector DB mode
2. **Slow LLM**: Use gpt-4o-mini or local model
3. **Multiple tools**: Limit max_calls_per_turn
4. **Network latency**: Use local LLM, enable streaming

**Enable debug logging to identify bottlenecks:**
```yaml
Debug Logging: true
```

### Local vs cloud LLM tradeoffs?

| Aspect | Local (Ollama) | Cloud (OpenAI) |
|--------|---------------|----------------|
| Cost | Free | Pay per token |
| Privacy | Complete | Sent to provider |
| Offline | Works offline | Requires internet |
| Quality | Model dependent | Very high |
| Latency | Very low | Higher |
| Hardware | Good CPU/GPU needed | Minimal |

---

## Privacy & Security

### Where is data stored?

**Conversation history**: `.storage/home_agent.history` (local only)
**Memories**: `.storage/home_agent.memories` (local only)
**Vector DB**: Local ChromaDB instance (optional)

**Control:**
```yaml
# Disable history
History Enabled: false

# Clear all data
service: home_agent.clear_memories
data:
  confirm: true
---
service: home_agent.clear_history
```

### Can I disable memory for privacy?

**Yes:**
```yaml
# Via UI
Settings → Home Agent → Configure → Memory System
Memory Enabled: false

# Or via service
service: home_agent.clear_memories
data:
  confirm: true
```

**Privacy-focused config:**
```yaml
Memory Enabled: false
History Enabled: true
History Persist: false  # Don't save across restarts
```

### How do I delete all stored data?

**Complete deletion:**
```yaml
# 1. Clear memories
service: home_agent.clear_memories
data:
  confirm: true

# 2. Clear history
service: home_agent.clear_history

# 3. Disable future collection
Memory Enabled: false
History Enabled: false
```

---

## Compatibility

### Which voice assistants work?

**Supported (via Home Assistant):**
- Home Assistant Voice Pipeline (Assist)
- HA Companion App (Android/iOS)
- ESPHome voice devices
- Wyoming satellite devices

**Not directly supported:**
- Google Home / Alexa (use HA Companion App instead)

### Does streaming work with all voice assistants?

**Requires:**
- Home Assistant Voice Pipeline
- Wyoming Protocol TTS (Piper recommended)
- Streaming enabled in settings

**Performance**: ~500ms until first audio vs 5+ seconds without streaming (10x faster)

**Compatible TTS**: Piper, Coqui TTS, MaryTTS
**Not compatible**: Google Cloud TTS, Amazon Polly, Azure TTS

---

## Troubleshooting

### "LLM connection failed"

**Check:**
- Base URL is correct for your provider
- API key is valid and not expired
- Service is running (for local models)
- Network connectivity

**Test:**
```bash
# OpenAI
curl https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_KEY"

# Ollama
curl http://localhost:11434/api/tags
```

### Tools not working?

**Checklist:**
1. Verify entities are exposed: Settings → Voice assistants → Expose
2. Check tool name spelling
3. Validate YAML syntax: Settings → System → Configuration Validation
4. Test manually:
   ```yaml
   service: home_agent.execute_tool
   data:
     tool_name: ha_query
     parameters:
       entity_id: light.living_room
   ```
5. Enable debug logging to see LLM tool calls

### How do I debug issues?

**1. Enable debug logging:**
```yaml
# configuration.yaml
logger:
  logs:
    custom_components.home_agent: debug
```

**2. View logs:**
```
Settings → System → Logs
Filter: "home_agent"
```

**3. Monitor events:**
```
Developer Tools → Events
Listen to: home_agent.*
```

**4. Test tools manually:**
```yaml
service: home_agent.execute_tool
data:
  tool_name: ha_control
  parameters:
    action: turn_on
    entity_id: light.living_room
```

---

## Configuration

### What's a good starting configuration?

**Beginner setup:**
```yaml
# LLM
LLM Base URL: https://api.openai.com/v1
API Key: sk-your-key
Model: gpt-4o-mini
Temperature: 0.7
Max Tokens: 300

# Context
Context Mode: direct
Entities: light.*, climate.*, sensor.temperature

# History
History Enabled: true
Max Messages: 10

# Optional features
Memory Enabled: false  # Enable later
Streaming: false  # Enable for voice
External LLM: false
```

### How many entities should I include?

**Small (<20)**: Use `*.*` (all entities)
**Medium (20-100)**: Use domain wildcards `light.*, climate.*`
**Large (100+)**: Use vector DB mode with Top K: 5

**Token cost**: ~50-100 tokens per entity

### What temperature should I use?

- **0.0-0.3**: Deterministic, consistent (device control)
- **0.4-0.7**: Balanced (recommended for general use)
- **0.8-1.2**: Creative (varied responses)

**Recommended**: 0.6-0.7 for general assistant use

### How many messages in history?

- **3 messages**: Minimal context (fast, cheap)
- **10 messages**: Standard (recommended)
- **20 messages**: Extended context (complex dialogs)

**More messages = better context but higher token cost**

---

## Need More Help?

**Documentation:**
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Troubleshooting guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration guide
- [CUSTOM_TOOLS.md](CUSTOM_TOOLS.md) - Custom tool guide
- [EXAMPLES.md](EXAMPLES.md) - Practical examples

**Support:**
- GitHub Issues: Bug reports
- Home Assistant Forums: Community help
- GitHub Discussions: Questions

**Complete FAQ**: See [reference/FAQ.md](reference/FAQ.md) for comprehensive answers including GDPR compliance, API key security, browser compatibility, and more.

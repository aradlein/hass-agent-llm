# Frequently Asked Questions (FAQ)

Common questions and answers about Home Agent.

## Table of Contents

- [General Questions](#general-questions)
- [Features](#features)
- [Performance and Cost](#performance-and-cost)
- [Privacy and Security](#privacy-and-security)
- [Compatibility](#compatibility)
- [Troubleshooting](#troubleshooting)
- [Configuration](#configuration)

---

## General Questions

### What is Home Agent?

Home Agent is a custom Home Assistant integration that provides intelligent conversational AI capabilities using OpenAI-compatible LLMs. It enables natural language control of your smart home with advanced features like custom tools, memory, multi-LLM workflows, and streaming responses.

**Key capabilities:**
- Natural language device control
- Custom tool integration (REST APIs, services)
- Conversation memory and context
- Multi-LLM strategy (local + cloud)
- Streaming for low-latency voice responses

### How is it different from extended_openai_conversation?

While both integrations provide LLM-based conversation, Home Agent offers several enhancements:

| Feature | extended_openai_conversation | Home Agent |
|---------|------------------------------|------------|
| Multi-LLM Support | ❌ No | ✅ Yes (dual-LLM) |
| Memory System | ❌ No | ✅ Yes (automatic extraction) |
| Streaming Responses | ❌ No | ✅ Yes |
| Vector DB Context | ❌ No | ✅ Yes (ChromaDB) |
| Script Execution | ✅ Yes | ❌ No (security) |
| History Persistence | ⚠️ Basic | ✅ Advanced |
| Context Optimization | ⚠️ Limited | ✅ Smart compression |

**Main advantages:**
- **Memory**: Remembers preferences and facts across conversations
- **Performance**: Streaming for instant voice responses
- **Flexibility**: Dual-LLM strategy for cost/performance optimization
- **Security**: No script execution, better isolation

**Migration guide:** See [MIGRATION.md](MIGRATION.md)

### Which LLM providers are supported?

Home Agent supports any **OpenAI-compatible API endpoint**:

**Cloud Providers:**
- ✅ OpenAI (GPT-4, GPT-4o, GPT-4o-mini, etc.)
- ✅ Anthropic Claude (via OpenAI-compatible proxy)
- ✅ Google Gemini (via OpenAI-compatible proxy)
- ✅ OpenRouter (multi-provider service)
- ✅ Groq (fast inference)

**Local/Self-Hosted:**
- ✅ Ollama (llama2, mistral, codellama, etc.)
- ✅ LM Studio
- ✅ LocalAI
- ✅ Text generation web UI (oobabooga)
- ✅ vLLM
- ✅ Any OpenAI-compatible endpoint

**Configuration:**
```yaml
# OpenAI
LLM Base URL: https://api.openai.com/v1

# Ollama (local)
LLM Base URL: http://localhost:11434/v1

# OpenRouter
LLM Base URL: https://openrouter.ai/api/v1
```

### Can I use local models only?

**Yes!** Home Agent works completely offline with local models.

**Recommended setup:**
```yaml
# Primary LLM (Ollama)
LLM Base URL: http://localhost:11434/v1
Model: llama2:13b  # or mistral, codellama, etc.

# No external LLM needed
External LLM Enabled: false

# Disable features requiring cloud:
Memory Extraction LLM: local  # Use local LLM for memory extraction
Vector DB: local ChromaDB  # No cloud embedding service
```

**Hardware requirements for good performance:**
- **Minimum**: 16GB RAM, modern CPU
- **Recommended**: 32GB RAM, GPU (NVIDIA RTX 3060+)
- **Optimal**: 64GB RAM, high-end GPU (RTX 4090, etc.)

**Performance:**
- Small models (7B): Fast on CPU, very fast on GPU
- Medium models (13B): Requires good CPU or GPU
- Large models (70B): Requires powerful GPU or multiple GPUs

### Does it work offline?

**Partially**, depending on configuration:

**Fully Offline Capable:**
- ✅ Device control (ha_control, ha_query)
- ✅ Local LLM (Ollama, LM Studio)
- ✅ Direct entity context
- ✅ Conversation history
- ✅ Local ChromaDB for memory/vector DB

**Requires Internet:**
- ❌ Cloud LLM providers (OpenAI, Claude)
- ❌ External LLM tool (if using cloud model)
- ❌ REST custom tools calling external APIs
- ❌ Cloud embedding services (for vector DB)

**Recommended offline setup:**
```yaml
LLM Base URL: http://localhost:11434/v1  # Ollama
Model: llama2:13b
External LLM Enabled: false
Context Mode: direct  # No cloud embeddings needed
Memory Enabled: true
Memory Extraction LLM: local
```

### What's the minimum Home Assistant version?

**Minimum version**: Home Assistant 2024.1.0

**Recommended version**: Home Assistant 2024.6.0+

**Why:**
- 2024.1.0+: Native LLM conversation API
- 2024.4.0+: Enhanced streaming support
- 2024.6.0+: Improved Wyoming protocol for TTS

**Check your version:**
Settings > System > About > Current Version

**Upgrade:**
Settings > System > Updates

---

## Features

### What is the memory system?

The memory system allows Home Agent to remember facts, preferences, and context across conversations.

**How it works:**
1. **Automatic extraction**: After each conversation, an LLM extracts important information
2. **Storage**: Memories stored in Home Assistant's `.storage` directory
3. **Indexing**: Semantic indexing in ChromaDB for efficient retrieval
4. **Recall**: Relevant memories automatically injected into future conversations

**Example:**
```
User: "I prefer the bedroom at 68 degrees for sleeping"
→ Memory extracted and stored

[Next day]
User: "I'm going to bed"
→ Memory recalled: "Setting bedroom to 68 degrees, as you prefer for sleeping"
```

**Memory types:**
- **fact**: Objective information (allergies, family members, schedules)
- **preference**: User preferences (temperature, lighting, music)
- **context**: Situational information (recent events, temporary states)
- **event**: Time-bound occurrences (appointments, reminders)

**Configuration:**
```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: true
Automatic Extraction: true
Max Memories: 100
```

**Documentation:** See project spec Phase 3.5

### Should I enable memory?

**Enable memory if:**
- ✅ You want personalized, context-aware responses
- ✅ You frequently reference past preferences
- ✅ You use Home Agent for routines and patterns
- ✅ You're comfortable with local data storage

**Disable memory if:**
- ❌ You prioritize privacy and minimal data storage
- ❌ You only use Home Agent for simple device control
- ❌ You share Home Assistant with others
- ❌ You want faster, simpler responses

**Privacy note:** All memories stored locally in Home Assistant. No cloud storage unless you use cloud LLM for extraction.

### What's the difference between direct and vector DB modes?

**Direct Mode** (default):
- Injects specific entities you configure
- Simpler setup, no additional services
- Predictable context size
- Best for: Small number of entities, basic setups

**Vector DB Mode** (advanced):
- Dynamically retrieves relevant entities using semantic search
- Requires ChromaDB setup
- More efficient token usage
- Best for: Large smart homes, advanced users

**Comparison:**

| Aspect | Direct Mode | Vector DB Mode |
|--------|-------------|----------------|
| Setup | Easy | Advanced |
| Dependencies | None | ChromaDB, embeddings |
| Context | Fixed entities | Dynamic retrieval |
| Token Efficiency | Lower | Higher |
| Relevance | Manual selection | Automatic semantic matching |

**Example (Direct):**
```yaml
Context Entities: light.living_room, sensor.temperature
→ Always includes these, whether relevant or not
```

**Example (Vector DB):**
```
User: "What's the bedroom temperature?"
→ ChromaDB finds: sensor.bedroom_temperature
→ Only relevant entity included
```

**Recommendation:** Start with direct mode, upgrade to vector DB if you have 50+ entities.

### Can I use multiple LLMs?

**Yes!** Home Agent supports a dual-LLM strategy:

**Primary LLM**:
- Handles most conversations
- Executes tools (ha_control, ha_query, custom tools)
- Fast and cost-effective
- Example: Local Ollama llama2:13b or OpenAI gpt-4o-mini

**External LLM** (optional):
- Exposed as a tool for complex queries
- Used when primary LLM needs help
- More powerful but costlier
- Example: OpenAI gpt-4o or Claude 3.5 Sonnet

**Example workflow:**
```
User: "Analyze my energy usage and suggest optimizations"

Primary LLM (local):
  1. Calls ha_query for energy data
  2. Recognizes need for complex analysis
  3. Calls query_external_llm tool

External LLM (GPT-4):
  - Performs detailed analysis
  - Returns recommendations

Primary LLM:
  - Formats response for user
```

**Cost optimization:** 80% of queries handled by cheap/free local model, 20% use expensive cloud model only when needed.

**Configuration:** Settings > Home Agent > Configure > External LLM

### Does streaming work with all voice assistants?

**Requires:**
- ✅ Home Assistant Voice Pipeline (Assist)
- ✅ Wyoming Protocol TTS (e.g., Piper)
- ✅ Streaming enabled in Home Agent settings

**Supported setups:**
- ✅ HA Companion App (Android/iOS)
- ✅ ESPHome devices with voice
- ✅ Wyoming satellite devices
- ✅ Browser-based voice control
- ❌ Google Home (not supported by HA streaming)
- ❌ Amazon Alexa (not supported by HA streaming)

**Configuration:**
```yaml
# Settings > Home Agent > Configure > Debug Settings
Streaming Responses: true
```

**Performance improvement:**
- Without streaming: 5+ seconds until first audio
- With streaming: ~500ms until first audio
- **~10x faster perceived response time**

### What are custom tools?

Custom tools extend the LLM's capabilities beyond basic device control.

**Built-in tools:**
- `ha_control`: Control devices (turn on/off, set values)
- `ha_query`: Query device states and history
- `query_external_llm`: Delegate to external LLM (if enabled)
- `store_memory`: Manually store memories (if memory enabled)
- `recall_memory`: Search stored memories (if memory enabled)

**Custom tools:**
User-defined tools in `configuration.yaml` for:
- **REST APIs**: External services (weather, calendars, IoT platforms)
- **Home Assistant services**: Automations, scripts, scenes

**Example custom tools:**
```yaml
home_agent:
  tools_custom:
    # Call weather API
    - name: check_weather
      handler:
        type: rest
        url: "https://api.weather.com/..."

    # Trigger automation
    - name: run_morning_routine
      handler:
        type: service
        service: automation.trigger

    # Activate scene
    - name: movie_mode
      handler:
        type: service
        service: scene.turn_on
```

**Documentation:** See [CUSTOM_TOOLS.md](CUSTOM_TOOLS.md)

---

## Performance and Cost

### How much do API calls cost?

**Cost depends on:**
- LLM provider (OpenAI, local, etc.)
- Model choice (GPT-4o vs GPT-4o-mini)
- Response length (max_tokens)
- Frequency of use
- External LLM usage

**Example costs (OpenAI GPT-4o-mini):**
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens
- Average query: ~500 input, ~150 output tokens
- **Cost per query: ~$0.0002 (0.02 cents)**

**Example costs (OpenAI GPT-4o):**
- Input: $2.50 per 1M tokens
- Output: $10.00 per 1M tokens
- Average query: ~500 input, ~300 output tokens
- **Cost per query: ~$0.004 (0.4 cents)**

**Monthly estimates (100 queries/day):**
- GPT-4o-mini: ~$0.60/month
- GPT-4o: ~$12/month
- Local Ollama: $0/month (free)

**Cost optimization:**
1. Use local models for primary LLM
2. Only enable external LLM for complex queries
3. Reduce max_tokens
4. Limit tool calls per turn

### How can I reduce token usage?

**1. Reduce max_tokens:**
```yaml
# Primary LLM
Max Tokens: 150  # Instead of 500
```

**2. Limit context:**
```yaml
# Include fewer entities
Context Entities: light.*, sensor.temperature
# Instead of: *.*
```

**3. Use vector DB mode:**
```yaml
# Only retrieve relevant entities
Context Mode: vector_db
Top K: 3  # Instead of 5
```

**4. Shorten conversation history:**
```yaml
Max Messages: 5  # Instead of 10
```

**5. Disable memory extraction (if not needed):**
```yaml
Memory Extraction Enabled: false
```

**6. Use smaller models:**
```yaml
Model: gpt-4o-mini  # Instead of gpt-4o
# or
Model: llama2:7b  # Instead of llama2:70b
```

**7. Limit external LLM calls:**
```yaml
Max Tool Calls Per Turn: 2  # Instead of 5
```

**Monitoring:**
```yaml
automation:
  - alias: "Track Token Usage"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: logbook.log
        data:
          message: "Tokens: {{ trigger.event.data.tokens.total }}"
```

### Why are responses slow?

**Common causes:**

**1. Large context size**
- **Symptom**: Initial processing delay
- **Solution**: Reduce entities, use vector DB mode, limit history

**2. Slow LLM provider**
- **Symptom**: Long wait for responses
- **Solution**: Use faster model (gpt-4o-mini, local Ollama), enable streaming

**3. Multiple tool calls**
- **Symptom**: Delays during execution
- **Solution**: Limit max_calls_per_turn, optimize tool complexity

**4. Network latency**
- **Symptom**: Delays with cloud LLMs
- **Solution**: Use local LLM (Ollama), check internet connection

**5. ChromaDB queries**
- **Symptom**: Delays before LLM response
- **Solution**: Reduce vector_db_top_k, optimize collection size

**Benchmarking:**
```yaml
# Enable detailed event monitoring
automation:
  - alias: "Log Performance"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: logbook.log
        data:
          message: >
            Duration: {{ trigger.event.data.duration_ms }}ms
            LLM: {{ trigger.event.data.performance.llm_latency_ms }}ms
            Tools: {{ trigger.event.data.performance.tool_latency_ms }}ms
```

### Best practices for performance

**1. Enable streaming** (for voice assistants):
```yaml
Streaming Enabled: true
# 10x faster perceived response time
```

**2. Use local LLM for primary**:
```yaml
LLM Base URL: http://localhost:11434/v1
Model: mistral:7b-instruct  # Fast local model
```

**3. Optimize context**:
```yaml
# Use vector DB for large setups
Context Mode: vector_db
Top K: 3

# Or use focused direct context
Context Entities: light.*, climate.*, sensor.temperature
```

**4. Limit complexity**:
```yaml
Max Tokens: 200  # Shorter responses
Max Tool Calls Per Turn: 3  # Fewer tool executions
```

**5. Cache frequently used data**:
- Enable history persistence
- Use memory for common facts
- Leverage context optimization

**6. Monitor and tune**:
- Track response times via events
- Identify bottlenecks (LLM, tools, context)
- Adjust configuration based on metrics

### Local vs cloud LLM tradeoffs

| Aspect | Local LLM (Ollama) | Cloud LLM (OpenAI) |
|--------|-------------------|-------------------|
| **Cost** | Free (hardware only) | Pay per token |
| **Speed** | Fast (with good hardware) | Variable (network dependent) |
| **Privacy** | Complete privacy | Data sent to provider |
| **Offline** | Works offline | Requires internet |
| **Quality** | Depends on model size | Generally very high |
| **Hardware** | Requires good CPU/GPU | Minimal local resources |
| **Maintenance** | Self-managed updates | Provider managed |
| **Latency** | Very low (local) | Higher (network) |

**Recommended configurations:**

**Privacy-focused (Local only):**
```yaml
LLM Base URL: http://localhost:11434/v1
Model: llama2:13b
External LLM Enabled: false
```

**Balanced (Local + Cloud):**
```yaml
# Primary: Local for control
LLM Base URL: http://localhost:11434/v1
Model: mistral:7b

# External: Cloud for analysis
External LLM Enabled: true
External LLM Model: gpt-4o-mini
```

**Quality-focused (Cloud):**
```yaml
# Primary: Fast cloud model
Model: gpt-4o-mini

# External: Best cloud model
External LLM Model: gpt-4o
```

---

## Privacy and Security

### Where is conversation history stored?

**Location**: Home Assistant's `.storage` directory (local storage only)

**File**: `.storage/home_agent.history`

**Format**: JSON file with conversation messages

**Retention**:
- Configurable via max_messages or max_tokens
- Automatic cleanup based on settings
- Can be manually cleared via service call

**Access**:
- Only accessible to Home Assistant
- Not transmitted to external services (except LLM for processing)
- Standard file system permissions

**Control:**
```yaml
# Disable history
History Enabled: false

# Don't persist across restarts
History Persist: false

# Clear history
service: home_agent.clear_history
```

### How secure is memory storage?

**Storage location**: `.storage/home_agent.memories` (local only)

**Security measures:**
- ✅ Local-only storage (no cloud sync)
- ✅ Standard Home Assistant file permissions
- ✅ ChromaDB local instance (optional, also local)
- ✅ No external transmission (except for extraction if using cloud LLM)

**Data in memories:**
- User preferences
- Facts mentioned in conversations
- Context from past interactions
- Metadata (entities, topics, timestamps)

**Risks to consider:**
- Memories accessible if someone has HA access
- Extraction using cloud LLM sends conversation to provider
- ChromaDB collection accessible locally

**Enhanced security:**
```yaml
# Use local LLM for extraction (no cloud transmission)
Memory Extraction LLM: local

# Disable memory entirely
Memory Enabled: false

# Set short retention periods
Memory Fact TTL: 2592000  # 30 days
Memory Preference TTL: 7776000  # 90 days
```

### Can I disable memory for privacy?

**Yes**, memory can be completely disabled:

**Option 1: Disable via UI**
```
Settings > Home Agent > Configure > Memory System
Memory Enabled: false
```

**Option 2: Clear existing memories**
```yaml
service: home_agent.clear_memories
data:
  confirm: true
```

**Option 3: Use extraction but not storage**
```yaml
# Extract memories but review before storing
Memory Automatic Extraction: false
# Manually approve each memory via store_memory tool
```

**Privacy-focused configuration:**
```yaml
Memory Enabled: false  # No memory at all
History Enabled: true
History Persist: false  # Don't save across restarts
```

**Complete data deletion:**
```yaml
# Delete all stored data
service: home_agent.clear_memories
data:
  confirm: true
---
service: home_agent.clear_history
```

### How do I delete all stored data?

**Complete data deletion steps:**

**1. Clear memories:**
```yaml
service: home_agent.clear_memories
data:
  confirm: true
```

**2. Clear conversation history:**
```yaml
service: home_agent.clear_history
```

**3. Remove vector DB data (if using):**
```yaml
service: home_agent.reindex_entities
data:
  clear_existing: true
# Then don't add any entities
```

**4. Delete storage files (manual):**
```bash
# SSH into Home Assistant or use File Editor
rm /config/.storage/home_agent.memories
rm /config/.storage/home_agent.history
# Restart Home Assistant
```

**5. Disable future data collection:**
```yaml
Memory Enabled: false
History Enabled: false
```

**GDPR-compliant automation:**
```yaml
automation:
  - alias: "Quarterly Data Cleanup"
    trigger:
      - platform: time
        at: "03:00:00"
    condition:
      - condition: template
        value_template: >
          {{ now().month in [1, 4, 7, 10] and now().day == 1 }}
    action:
      - service: home_agent.clear_memories
        data:
          confirm: true
          older_than_days: 90
      - service: home_agent.clear_history
```

### GDPR compliance considerations

Home Agent can be configured for GDPR compliance:

**Data minimization:**
```yaml
# Only collect what's needed
Memory Enabled: false  # If not necessary
History Max Messages: 5  # Minimum for functionality
Context Entities: light.*, climate.*  # Only essential entities
```

**Purpose limitation:**
```yaml
# Configure memory for specific purpose
Memory Types Allowed: preference  # Only preferences, not facts
```

**Storage limitation:**
```yaml
# Automatic deletion after period
Memory Fact TTL: 2592000  # 30 days
Memory Preference TTL: 7776000  # 90 days
Memory Event TTL: 300  # 5 minutes
```

**Right to erasure:**
```yaml
# Easy deletion via service call
service: home_agent.clear_memories
data:
  confirm: true
```

**Data portability:**
```yaml
# Export memories (can be implemented via script)
service: home_agent.list_memories
# Returns all memories in structured format
```

**Security:**
- All data stored locally
- No third-party data sharing (except LLM for processing)
- Standard HA authentication required
- Optional: Use local LLM to avoid cloud transmission

**Note:** You are responsible for implementing appropriate data protection measures based on your jurisdiction and use case.

### API key security

**Best practices:**

**1. Use secrets.yaml (never hardcode):**
```yaml
# ❌ BAD: configuration.yaml
API Key: sk-abc123...

# ✅ GOOD: configuration.yaml
API Key: !secret openai_api_key

# secrets.yaml
openai_api_key: sk-abc123...
```

**2. Restrict file permissions:**
```bash
chmod 600 secrets.yaml
# Only HA user can read
```

**3. Use separate keys per service:**
```yaml
# Don't reuse same API key everywhere
openai_api_key: sk-primary-key
openai_external_key: sk-external-key  # Different key
weather_api_key: wx-weather-key
```

**4. Monitor API usage:**
- Check provider dashboards regularly
- Set spending limits
- Enable usage alerts

**5. Rotate keys periodically:**
- Generate new API keys every 90 days
- Update secrets.yaml
- Restart Home Assistant

**6. Revoke compromised keys immediately:**
- If key exposed, revoke in provider dashboard
- Generate new key
- Update configuration

**7. Don't commit secrets to version control:**
```bash
# .gitignore
secrets.yaml
*.db
.storage/
```

---

## Compatibility

### Which voice assistants are supported?

**Natively supported (via Home Assistant):**
- ✅ Home Assistant Voice Pipeline (Assist)
- ✅ HA Companion App (Android/iOS)
- ✅ ESPHome voice devices
- ✅ Wyoming satellite devices
- ✅ Browser-based voice control

**Not directly supported:**
- ❌ Google Home / Google Assistant
- ❌ Amazon Alexa
- ❌ Apple Siri / HomePod

**Workarounds for Google/Alexa:**
- Use HA Companion App instead
- Use ESPHome voice devices
- Trigger via automation (not voice)

**Setup:**
1. Configure Wyoming Protocol TTS (Piper)
2. Create Voice Assistant pipeline
3. Set Conversation Agent to "Home Agent"
4. Enable streaming (optional, for low latency)

### Can I use with Google Home/Alexa?

**No direct integration**, but alternatives exist:

**Option 1: Home Assistant Companion App**
- Install HA app on phone
- Use built-in voice assistant
- Full Home Agent integration

**Option 2: ESPHome voice device**
- Build ESP32-based voice assistant
- Uses Wyoming protocol
- Fully local, privacy-friendly

**Option 3: Trigger via routine (limited)**
```yaml
# Google Home routine → HA automation → Home Agent
automation:
  - alias: "Google Home to Home Agent"
    trigger:
      - platform: webhook
        webhook_id: google_home_trigger
    action:
      - service: home_agent.process
        data:
          text: "{{ trigger.json.command }}"
```

**Why not direct?**
- Google/Alexa use proprietary protocols
- Not compatible with HA conversation platform
- Would require cloud relay service

**Recommendation:** Use HA Companion App for best experience.

### Browser compatibility for streaming?

**Streaming works in:**
- ✅ Chrome/Edge (90+)
- ✅ Firefox (88+)
- ✅ Safari (14.1+)
- ✅ Opera (76+)

**Requirements:**
- Modern browser with WebSocket support
- Home Assistant accessible (local network or HTTPS)
- Wyoming TTS integration configured

**Not required:**
- Special plugins
- Browser extensions
- Additional downloads

**Testing streaming:**
1. Go to Home Assistant web UI
2. Open Voice Assistant (microphone icon)
3. Select pipeline using Home Agent
4. Speak command
5. Should hear audio start immediately (~500ms)

**If not working:**
- Check browser console for errors
- Verify Wyoming TTS is running
- Ensure streaming enabled in settings

### Which TTS systems work with streaming?

**Compatible TTS (Wyoming Protocol):**
- ✅ Piper (recommended, fast, offline)
- ✅ Coqui TTS (high quality)
- ✅ MaryTTS (offline)

**Not compatible:**
- ❌ Google Cloud TTS (not Wyoming protocol)
- ❌ Amazon Polly (not Wyoming protocol)
- ❌ Azure TTS (not Wyoming protocol)

**Setup Piper (recommended):**
```yaml
# Add-on store: Piper
# Configure voice and language
# Add to voice pipeline
```

**Why Piper?**
- Fast (real-time generation)
- Offline (privacy)
- Good quality
- Multiple languages/voices
- Low resource usage

**Alternative: Coqui TTS**
```yaml
# Better quality, slower
# Higher resource usage
# Still offline
```

---

## Troubleshooting

### Common error messages

#### "LLM connection failed"
**Cause:** Cannot reach LLM endpoint
**Solutions:**
- Verify base URL is correct
- Check API key is valid
- Test network connectivity
- Verify LLM service is running (for local)

#### "Tool execution timeout"
**Cause:** Tool took longer than timeout limit
**Solutions:**
- Increase timeout: `Tools Timeout: 60`
- Optimize tool (simplify API calls)
- Check external service response time

#### "Context exceeds token limit"
**Cause:** Too much context for model
**Solutions:**
- Reduce context entities
- Enable context optimization
- Use vector DB mode
- Reduce history max_messages

#### "Memory extraction failed"
**Cause:** Extraction LLM error or ChromaDB issue
**Solutions:**
- Check ChromaDB is running
- Verify extraction LLM credentials
- Check logs for specific error
- Disable memory temporarily

#### "Service not found"
**Cause:** Service doesn't exist
**Solutions:**
- Verify service name format: `domain.service_name`
- Check integration is loaded
- Test in Developer Tools > Services

### Why aren't my tools working?

**Checklist:**

**1. Verify tool registration:**
```yaml
# Check logs for:
"Registered custom tool: my_tool"
```

**2. Check YAML syntax:**
```yaml
# Use Configuration Validation
# Settings > System > Configuration Validation
```

**3. Test tool manually:**
```yaml
service: home_agent.execute_tool
data:
  tool_name: my_tool
  parameters:
    param1: "value"
```

**4. Verify secrets:**
```yaml
# If using {{ secrets.api_key }}
# Check secrets.yaml contains the key
```

**5. Check LLM is calling tool:**
```yaml
# Enable debug logging
logger:
  logs:
    custom_components.home_agent: debug

# Look for tool call attempts in logs
```

**6. Verify parameters:**
```yaml
# Ensure parameter schema matches usage
parameters:
  properties:
    location:  # This name must be used in templates
      type: string
```

### How do I debug issues?

**1. Enable debug logging:**
```yaml
# configuration.yaml
logger:
  default: info
  logs:
    custom_components.home_agent: debug
```

**2. Monitor events:**
```yaml
# Developer Tools > Events
# Listen to: home_agent.*

# Look for:
# - home_agent.error
# - home_agent.tool.executed
# - home_agent.conversation.finished
```

**3. Check service responses:**
```yaml
# Developer Tools > Services
service: home_agent.process
data:
  text: "Test query"

# View response in UI
```

**4. Test tools individually:**
```yaml
service: home_agent.execute_tool
data:
  tool_name: ha_query
  parameters:
    entity_id: light.living_room
```

**5. Review logs:**
```bash
# Home Assistant logs
# Settings > System > Logs
# Filter: home_agent
```

**6. Validate configuration:**
```bash
# Settings > System
# Configuration Validation
```

**7. Test LLM connection:**
```bash
# Test API directly
curl -X POST https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"test"}]}'
```

### Where are the logs?

**View in Home Assistant UI:**
```
Settings > System > Logs
Filter by: "home_agent"
```

**Log file location:**
```
/config/home-assistant.log
```

**Enable debug logging:**
```yaml
# configuration.yaml
logger:
  default: warning
  logs:
    custom_components.home_agent: debug
    custom_components.home_agent.agent: debug
    custom_components.home_agent.tools: debug
    custom_components.home_agent.memory_manager: debug
```

**View specific component logs:**
```yaml
# Agent (main conversation)
custom_components.home_agent.agent: debug

# Tool execution
custom_components.home_agent.tool_handler: debug

# Custom tools
custom_components.home_agent.tools.custom: debug

# Memory system
custom_components.home_agent.memory_manager: debug

# Context management
custom_components.home_agent.context_manager: debug
```

**Filter logs:**
```bash
# SSH or Terminal
grep "home_agent" /config/home-assistant.log

# Last 100 lines
tail -100 /config/home-assistant.log | grep "home_agent"
```

---

## Configuration

### What's a good starting configuration?

**Beginner-friendly setup:**
```yaml
# Settings > Add Integration > Home Agent

# LLM Settings
Name: Home Agent
LLM Base URL: https://api.openai.com/v1
API Key: sk-your-key
Model: gpt-4o-mini  # Cheap, fast, capable
Temperature: 0.7
Max Tokens: 300

# Context Settings
Context Mode: direct
Entities: light.*, climate.*, sensor.temperature

# Conversation History
History Enabled: true
Max Messages: 10
History Persist: true

# System Prompt
Use Default: true

# Tool Configuration
Max Tool Calls Per Turn: 5
Tool Timeout: 30

# Features (optional)
External LLM Enabled: false
Memory Enabled: false  # Enable later if desired
Streaming Enabled: false  # Enable for voice
```

**No custom tools needed to start!** Built-in tools (ha_control, ha_query) cover basic use.

### How many entities should I include?

**Recommendations by setup size:**

**Small setup (<20 entities):**
```yaml
Context Mode: direct
Entities: *.*  # Include all
```

**Medium setup (20-100 entities):**
```yaml
Context Mode: direct
Entities: light.*, climate.*, sensor.temperature, sensor.humidity
# Include only frequently used
```

**Large setup (100+ entities):**
```yaml
Context Mode: vector_db  # Recommended
Top K: 5
# Only relevant entities retrieved automatically
```

**Token considerations:**
- Each entity ~50-100 tokens
- Model limits: 4K-8K for smaller models
- Leave room for history and response

**Best practice:**
```yaml
# Use domain wildcards
Entities: light.*, climate.*, media_player.living_room

# Not: light.living_room, light.bedroom, light.kitchen...
```

### What temperature should I use?

**Temperature controls randomness/creativity:**

**0.0 - 0.3: Deterministic**
```yaml
Temperature: 0.2
# Use for: Device control, factual queries
# Behavior: Consistent, predictable responses
```

**0.4 - 0.7: Balanced** (recommended)
```yaml
Temperature: 0.6
# Use for: General conversation, mixed tasks
# Behavior: Good balance of accuracy and variety
```

**0.8 - 1.2: Creative**
```yaml
Temperature: 1.0
# Use for: Creative suggestions, varied responses
# Behavior: More diverse, less predictable
```

**1.3+: Very creative**
```yaml
Temperature: 1.5
# Use for: Brainstorming, exploration
# Behavior: Highly varied, potentially less accurate
```

**Recommendations:**
- **Device control**: 0.3-0.5
- **General assistant**: 0.6-0.8
- **Analysis/recommendations**: 0.7-0.9

### How many messages should I keep in history?

**Recommendations:**

**Minimal context (fast, cheap):**
```yaml
Max Messages: 3
# Last 1.5 conversation turns
# Use for: Simple device control
```

**Standard context (recommended):**
```yaml
Max Messages: 10
# Last 5 conversation turns
# Use for: General conversation
```

**Extended context (complex dialogs):**
```yaml
Max Messages: 20
# Last 10 conversation turns
# Use for: Multi-step planning, analysis
```

**Token-based limit (advanced):**
```yaml
Max Tokens: 2000
# Limit by tokens instead of message count
# More precise control
```

**Considerations:**
- More history = more tokens = higher cost
- More history = better context understanding
- Older messages provide diminishing value
- Enable persistence to maintain across restarts

**Best practice:**
```yaml
# Most users
Max Messages: 10
Max Tokens: 4000  # Safety limit
Persist: true
```

---

## Need More Help?

**Documentation:**
- [EXAMPLES.md](EXAMPLES.md) - Practical examples and recipes
- [CUSTOM_TOOLS.md](CUSTOM_TOOLS.md) - Custom tool guide
- [EXTERNAL_LLM.md](EXTERNAL_LLM.md) - Multi-LLM setup
- [MIGRATION.md](MIGRATION.md) - Migration from other integrations
- [PROJECT_SPEC.md](PROJECT_SPEC.md) - Complete technical specification

**Support:**
- GitHub Issues: Bug reports and feature requests
- Home Assistant Forums: Community discussion
- GitHub Discussions: Questions and sharing

**Contributing:**
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development guidelines
- Pull requests welcome!

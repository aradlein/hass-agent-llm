# Configuration Guide

Essential configuration settings for Home Agent. For complete options, see the [Configuration Reference](reference/CONFIGURATION.md).

**Looking for ready-to-use configurations?** See [Example Configurations](EXAMPLE_CONFIGS.md) for comprehensive setup guides for OpenAI, Ollama, LocalAI, multi-LLM, memory system, and vector DB.

## Configuration Overview

**Access configuration:**
```
Settings → Devices & Services → Home Agent → Configure
```

**Configuration locations:**
- **UI Settings**: Most options (LLM, context, history, memory)
- **configuration.yaml**: Custom tools only
- **Storage**: History and memories (`.storage/` directory)

## Essential Settings

| Setting | Default | When to Change |
|---------|---------|----------------|
| **LLM Base URL** | (required) | Set to your LLM provider endpoint |
| **API Key** | (required) | Set to your provider's key |
| **Model** | `gpt-4o-mini` | Use faster/cheaper model or local model |
| **Temperature** | `0.7` | Lower (0.3-0.5) for consistent control, higher (0.8-1.0) for creativity |
| **Max Tokens** | `500` | Reduce to 150-300 for voice, increase for detailed responses |
| **Context Mode** | `direct` | Use `vector_db` for 100+ entities |
| **Max Messages** | `10` | Lower to 5 for speed/cost, increase to 20 for complex dialogs |
| **Memory Enabled** | `true` | Disable for privacy/simplicity |
| **Streaming** | `false` | Enable for voice assistants (requires Wyoming TTS) |
| **External LLM** | `false` | Enable for dual-LLM strategy |

## Recommended Starting Configuration

**Beginner-friendly setup (OpenAI):**
```yaml
# Primary LLM Settings
LLM Base URL: https://api.openai.com/v1
API Key: sk-your-key-here
Model: gpt-4o-mini
Temperature: 0.7
Max Tokens: 300

# Context Settings
Context Mode: direct
Context Entities: light.*, climate.*, sensor.temperature

# History Settings
History Enabled: true
Max Messages: 10
History Persist: true

# Tool Settings
Max Tool Calls Per Turn: 5
Tool Timeout: 30

# Memory (Optional)
Memory Enabled: false  # Enable later if desired

# External LLM
External LLM Enabled: false

# Streaming
Streaming Enabled: false  # Enable for voice
```

**Local-only setup (Ollama):**
```yaml
# Primary LLM Settings
LLM Base URL: http://localhost:11434/v1
API Key: (leave empty)
Model: llama2
Temperature: 0.7
Max Tokens: 300

# Everything else: use defaults above
# No external LLM needed
# Memory extraction uses local LLM
```

**Advanced dual-LLM setup:**
```yaml
# Primary LLM (Fast/Local)
LLM Base URL: http://localhost:11434/v1
Model: mistral
Temperature: 0.5
Max Tokens: 200

# External LLM (Powerful/Cloud)
External LLM Enabled: true
External LLM Base URL: https://api.openai.com/v1
External LLM API Key: sk-your-key-here
External LLM Model: gpt-4o
External LLM Temperature: 0.8
External LLM Max Tokens: 1000

# Context (Vector DB)
Context Mode: vector_db
Vector DB Enabled: true
Vector DB Host: localhost
Vector DB Port: 8000
Vector DB Top K: 5

# History
Max Messages: 20
Max Tokens: 6000

# Memory
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: external  # Use GPT-4 for quality
Memory Max Memories: 500

# Streaming
Streaming Enabled: true
```

## Configuration Tips

### 1. LLM Provider Selection
- **OpenAI (gpt-4o-mini)**: Best balance of speed, quality, cost
- **OpenAI (gpt-4o)**: Highest quality but expensive
- **Ollama (local)**: Free, private, requires good hardware
- **LocalAI/LM Studio**: Alternative local options

### 2. Context Management
- **Small setups (<20 entities)**: Use direct mode with all entities
- **Medium setups (20-100)**: Use direct mode with domain wildcards (`light.*`)
- **Large setups (100+)**: Use vector DB mode for automatic relevance

### 3. Token Optimization
- Keep `Max Tokens` low for voice (150-300)
- Use `Max Messages: 5` for simple control tasks
- Enable vector DB to only include relevant entities
- Consider local LLM to eliminate per-token costs

### 4. Memory System
- **Enable** if you want personalized, context-aware responses
- **Disable** for privacy, shared systems, or simple control
- Use `local` extraction LLM to avoid cloud calls
- Set appropriate TTLs for different memory types

### 5. Performance Tuning
- Enable streaming for voice assistants (10x faster perceived response)
- Use local LLM for primary to reduce latency
- Reduce history and entities for faster responses
- Monitor token usage via events

## Common Configurations by Use Case

### Voice Assistant (Fast Response)
```yaml
Model: gpt-4o-mini  # Fast cloud model
Temperature: 0.5  # Consistent
Max Tokens: 150  # Short responses
Max Messages: 5  # Minimal history
Streaming: true  # Essential for voice
Memory: false  # Optional
```

### Privacy-Focused (Fully Local)
```yaml
LLM Base URL: http://localhost:11434/v1
Model: llama2
External LLM: false  # No cloud
Memory Extraction LLM: local  # No cloud
Context Mode: direct  # No cloud embeddings
```

### Cost-Optimized (Minimal Tokens)
```yaml
Model: gpt-4o-mini  # Cheapest capable model
Max Tokens: 150  # Short responses
Max Messages: 5  # Minimal history
Context Mode: vector_db  # Only relevant entities
Top K: 3  # Fewer entities
Memory Extraction: false  # Save extraction tokens
```

### Quality-Focused (Best Responses)
```yaml
# Primary
Model: gpt-4o-mini  # Fast for control

# External LLM
External LLM Enabled: true
External LLM Model: gpt-4o  # Best for analysis

# Full context
Max Messages: 20
Context Mode: vector_db
Top K: 10

# Memory
Memory Enabled: true
Memory Extraction LLM: external  # Quality extraction
```

## Custom Tools (configuration.yaml)

**Add custom tools in `configuration.yaml`:**
```yaml
home_agent:
  custom_tools:
    # REST API example
    - name: check_weather
      description: "Get weather forecast for a location"
      parameters:
        type: object
        properties:
          location:
            type: string
            description: "City name"
        required:
          - location
      handler:
        type: rest
        url: "https://api.weather.com/v1/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ secrets.weather_api_key }}"
        params:
          q: "{{ location }}"

    # Home Assistant service example
    - name: activate_scene
      description: "Activate a scene by name"
      parameters:
        type: object
        properties:
          scene_name:
            type: string
      handler:
        type: service
        service: scene.turn_on
        data:
          entity_id: "scene.{{ scene_name }}"
```

**Handler types:**
- `rest`: External HTTP APIs
- `service`: Home Assistant services

**Use secrets.yaml for API keys:**
```yaml
# secrets.yaml
weather_api_key: your-key-here
openai_api_key: sk-your-key-here
```

## Quick Configuration Changes

### Switch to Local LLM
```yaml
LLM Base URL: http://localhost:11434/v1
API Key: (leave empty)
Model: mistral  # or llama2, codellama
```

### Enable Memory
```yaml
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: local  # or external
```

### Enable Streaming
```yaml
# 1. Install Wyoming TTS (Piper recommended)
# 2. Configure voice pipeline
# 3. Enable in Home Agent:
Streaming Enabled: true
```

### Enable Vector DB
```yaml
# 1. Start ChromaDB: docker run -p 8000:8000 chromadb/chroma
# 2. Configure in Home Agent:
Context Mode: vector_db
Vector DB Enabled: true
Vector DB Host: localhost
Vector DB Port: 8000
```

### Add External LLM
```yaml
External LLM Enabled: true
External LLM Base URL: https://api.openai.com/v1
External LLM API Key: sk-your-key-here
External LLM Model: gpt-4o
```

### Configure Proxy Headers (Advanced)

Custom HTTP headers for routing requests through proxies or load balancers. Useful for:
- Multi-backend Ollama setups
- Custom routing in proxy scenarios
- Load balancing across GPU clusters
- A/B testing different model configurations

**Basic example (Ollama backend selection):**
```json
{
  "X-Ollama-Backend": "llama-cpp"
}
```

**Advanced example (multiple headers):**
```json
{
  "X-Ollama-Backend": "vllm-server",
  "X-Custom-Router": "gpu-cluster-1",
  "X-Model-Tier": "premium",
  "X-Request-Priority": "high"
}
```

**Common use cases:**

1. **Ollama Backend Selection:**
   ```json
   {"X-Ollama-Backend": "llama-cpp"}
   ```
   Routes requests to specific Ollama backend (llama-cpp, vllm-server, ollama-gpu)

2. **Load Balancer Routing:**
   ```json
   {"X-Target-Server": "gpu-node-2"}
   ```
   Direct requests to specific backend servers

3. **Multi-tier Model Selection:**
   ```json
   {
     "X-Model-Tier": "premium",
     "X-Priority": "high"
   }
   ```
   Route to different model tiers or priorities

**Configuration in UI:**
1. Go to Settings → Devices & Services → Home Agent → Configure
2. Select "LLM Settings"
3. Enter JSON in "Proxy Headers" field
4. Leave empty if not needed

**Validation rules:**
- Must be valid JSON format
- Header names: alphanumeric, hyphens, underscores only (RFC 7230)
- Header values: must be strings
- Empty/null is valid (no headers added)

**Migration from legacy backend setting:**

The old `llm_backend` dropdown is deprecated. It's automatically migrated:
- `llm_backend: "llama-cpp"` → `{"X-Ollama-Backend": "llama-cpp"}`
- Proxy headers take precedence if both are set
- Legacy backend still works for backward compatibility

## Validation

**Check configuration:**
```
Settings → System → Configuration Validation
```

**Common validation errors:**
- Invalid URL format
- Missing required fields (base_url, model, api_key)
- Out of range values (temperature 0.0-2.0, top_p 0.0-1.0)
- Invalid YAML syntax in custom tools

**If configuration not saving:**
1. Check validation errors in logs
2. Verify numeric values in valid ranges
3. Check YAML syntax in configuration.yaml
4. Try minimal configuration first

## Need More Details?

See the [Complete Configuration Reference](reference/CONFIGURATION.md) for comprehensive coverage including:
- All configuration constants and defaults
- Vector DB settings and embedding providers
- History optimization settings
- Memory TTL and cleanup configuration
- Advanced examples and scenarios
- Troubleshooting configuration issues

# Example Configurations

This guide provides comprehensive, ready-to-use configurations for common LLM providers and use cases. Each example includes complete settings, recommendations, and troubleshooting tips.

## Table of Contents

1. [OpenAI Setup](#1-openai-setup)
2. [Ollama Local Setup](#2-ollama-local-setup)
3. [LocalAI Setup](#3-localai-setup)
4. [Multi-LLM Configuration](#4-multi-llm-configuration)
5. [Memory System Setup](#5-memory-system-setup)
6. [Vector DB with ChromaDB](#6-vector-db-with-chromadb)

---

## 1. OpenAI Setup

Complete configuration for using OpenAI's GPT models with Home Agent.

### Basic Configuration

**Via Home Assistant UI:**

Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure**

**LLM Settings:**
```yaml
LLM Base URL: https://api.openai.com/v1
API Key: sk-proj-your-key-here
Model: gpt-4o-mini
Temperature: 0.7
Max Tokens: 500
Keep Alive: 5m  # Not used for OpenAI
```

**Context Settings:**
```yaml
Context Mode: direct
Context Format: json
Direct Entities: light.*, climate.*, sensor.temperature*, binary_sensor.motion*
```

**History Settings:**
```yaml
History Enabled: true
Max Messages: 10
Max Tokens: 4000
Session Persistence: true
Session Timeout: 60  # minutes
```

### Model Recommendations

| Model | Use Case | Cost (per 1M tokens) | Speed | Quality |
|-------|----------|---------------------|-------|---------|
| `gpt-4o-mini` | **Recommended** - Voice control, general use | Input: $0.15, Output: $0.60 | Fast | High |
| `gpt-4o` | Complex analysis, detailed responses | Input: $2.50, Output: $10.00 | Medium | Highest |
| `gpt-3.5-turbo` | Budget option, simple commands | Input: $0.50, Output: $1.50 | Very Fast | Good |

### Cost Optimization Tips

**1. Reduce max_tokens for voice:**
```yaml
Max Tokens: 150  # Short, concise responses
```
Average cost per interaction: ~$0.001-0.003

**2. Use vector DB mode for large setups:**
```yaml
Context Mode: vector_db  # Only includes relevant entities
Vector DB Top K: 5       # Limit context size
```

**3. Limit conversation history:**
```yaml
Max Messages: 5          # Recent context only
History Max Tokens: 2000 # Reduce token usage
```

**4. Disable memory extraction for cost savings:**
```yaml
Memory Extraction Enabled: false  # No additional extraction calls
```

### Voice Assistant Configuration

Optimized for fast, low-cost voice interactions:

```yaml
# LLM Settings
Model: gpt-4o-mini
Temperature: 0.5  # More consistent
Max Tokens: 150   # Short responses

# History
Max Messages: 5   # Minimal history
Session Timeout: 60

# Streaming
Streaming Enabled: true  # Essential for voice

# Memory (optional)
Memory Enabled: false  # Faster, lower cost
```

### Troubleshooting

**Authentication errors:**
- Verify API key starts with `sk-proj-` or `sk-`
- Check key hasn't expired at [OpenAI Platform](https://platform.openai.com/api-keys)
- Ensure you have credits/billing enabled

**Rate limit errors:**
- OpenAI enforces rate limits (tier-based)
- Reduce `tools_max_calls_per_turn` to limit tool usage
- Add retry logic in automations
- Upgrade to higher tier if needed

**High costs:**
- Monitor usage at [OpenAI Usage Dashboard](https://platform.openai.com/usage)
- Set spending limits in OpenAI account settings
- Switch to gpt-4o-mini for most tasks
- Use external LLM only for complex queries

**Slow responses:**
- Enable streaming for perceived faster response
- Reduce max_tokens
- Use gpt-4o-mini (fastest)
- Minimize context (fewer entities, shorter history)

### Best Practices

1. **Always use secrets.yaml for API keys:**
   ```yaml
   # configuration.yaml
   home_agent:
     llm_api_key: !secret openai_api_key

   # secrets.yaml
   openai_api_key: sk-proj-your-key-here
   ```

2. **Set up usage monitoring:**
   ```yaml
   automation:
     - alias: "Monitor OpenAI Usage"
       trigger:
         - platform: event
           event_type: home_agent.conversation.finished
       action:
         - service: counter.increment
           target:
             entity_id: counter.openai_api_calls
   ```

3. **Use appropriate models:**
   - Voice control: `gpt-4o-mini`
   - Complex analysis: `gpt-4o` (via external LLM)
   - Testing: `gpt-3.5-turbo`

---

## 2. Ollama Local Setup

Run LLMs locally on your own hardware for complete privacy and zero per-query costs.

### Installation

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download installer from [ollama.ai](https://ollama.ai/download)

**Docker:**
```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama:latest
```

**Verify installation:**
```bash
ollama --version
curl http://localhost:11434/api/version
```

### Model Selection

Pull a model before use:

```bash
# Recommended models for home automation
ollama pull mistral:7b-instruct    # Best balance (4GB RAM)
ollama pull llama3.2:3b            # Fastest, smallest (2GB RAM)
ollama pull llama3.1:8b            # Better quality (5GB RAM)
ollama pull qwen2.5:7b             # Good tool calling (4GB RAM)

# For embedding (required for vector DB)
ollama pull nomic-embed-text       # 274MB, excellent quality
```

**Model Comparison:**

| Model | RAM Required | Speed | Quality | Best For |
|-------|-------------|-------|---------|----------|
| `llama3.2:3b` | 2GB | Fastest | Good | Voice, simple control |
| `mistral:7b-instruct` | 4GB | Fast | Very Good | General use, recommended |
| `llama3.1:8b` | 5GB | Medium | Excellent | Complex queries |
| `qwen2.5:7b` | 4GB | Fast | Very Good | Tool calling focus |

### Basic Configuration

**Via Home Assistant UI:**

```yaml
# LLM Settings
LLM Base URL: http://localhost:11434/v1
API Key: (leave empty)
Model: mistral:7b-instruct
Temperature: 0.7
Max Tokens: 500
Keep Alive: 5m  # Keep model in memory

# Context Settings
Context Mode: direct
Context Format: json
Direct Entities: light.*, climate.*, sensor.*

# History
History Enabled: true
Max Messages: 10
Session Timeout: 60

# Memory
Memory Enabled: true
Memory Extraction LLM: local  # Use same LLM
```

### Performance Optimization

**1. Keep model loaded with keep_alive:**
```yaml
Keep Alive: 5m  # Default
Keep Alive: 15m # For frequent use
Keep Alive: -1  # Keep loaded indefinitely
```

**2. Adjust context size based on hardware:**
```yaml
# For limited RAM
Max Tokens: 300
Max Messages: 5
Context Mode: vector_db
Vector DB Top K: 3

# For powerful hardware
Max Tokens: 1000
Max Messages: 20
Vector DB Top K: 10
```

**3. Use quantized models for speed:**
```bash
# Standard (best quality)
ollama pull llama3.1:8b

# Quantized (faster, less RAM)
ollama pull llama3.1:8b-q4_0
ollama pull llama3.1:8b-q4_K_M
```

**4. GPU Acceleration:**
Ollama automatically uses GPU if available (NVIDIA, AMD, Apple Silicon).

Check GPU usage:
```bash
# NVIDIA
nvidia-smi

# Check Ollama is using GPU
ollama ps  # Shows VRAM usage
```

**5. Backend selection for advanced users:**
```json
# In Proxy Headers field
{"X-Ollama-Backend": "llama-cpp"}  # Default, best compatibility
{"X-Ollama-Backend": "vllm-server"}  # Faster, requires setup
{"X-Ollama-Backend": "ollama-gpu"}  # GPU-optimized
```

### Hardware Recommendations

**Minimum (3B models):**
- CPU: 4 cores
- RAM: 8GB (2GB for model + 6GB system)
- Storage: 5GB
- Response time: 2-5 seconds

**Recommended (7B models):**
- CPU: 6-8 cores or GPU
- RAM: 16GB (4-6GB for model + system)
- Storage: 10GB
- Response time: 1-3 seconds

**Optimal (7B+ models):**
- CPU: Modern 8+ core or RTX 3060+
- RAM: 32GB
- GPU: 6GB+ VRAM
- Storage: 20GB SSD
- Response time: <1 second

### Troubleshooting

**Ollama not responding:**
```bash
# Check service status
systemctl status ollama  # Linux

# Check logs
journalctl -u ollama -f  # Linux
docker logs ollama       # Docker

# Restart service
systemctl restart ollama  # Linux
docker restart ollama     # Docker
```

**Model not found:**
```bash
# List installed models
ollama list

# Pull missing model
ollama pull mistral:7b-instruct

# Remove unused models to save space
ollama rm model-name
```

**Out of memory errors:**
- Use smaller model (3B instead of 7B)
- Reduce `Max Tokens`
- Use quantized version
- Close other applications
- Increase system swap

**Slow responses:**
- Ensure model is preloaded (`Keep Alive: 5m`)
- Use smaller model
- Use quantized version
- Enable GPU if available
- Reduce context size

**Tool calling issues:**
- Some models better at tool calling (mistral, qwen2.5)
- Update to latest Ollama version
- Try temperature 0.5-0.7 for consistency
- Check model supports function calling

### Complete Privacy Setup

Zero cloud dependency:

```yaml
# Primary LLM - Local
LLM Base URL: http://localhost:11434/v1
Model: mistral:7b-instruct

# External LLM - Disabled
External LLM Enabled: false

# Vector DB - Local embeddings
Context Mode: vector_db
Embedding Provider: ollama
Embedding Base URL: http://localhost:11434
Embedding Model: nomic-embed-text

# Memory - Local extraction
Memory Enabled: true
Memory Extraction LLM: local
```

**Data location:**
- Models: `~/.ollama/models` (Linux/Mac) or `%USERPROFILE%\.ollama\models` (Windows)
- All inference: 100% local
- No internet connection required after model download

---

## 3. LocalAI Setup

LocalAI provides OpenAI-compatible API for local models with broader model support than Ollama.

### Installation

**Docker Compose (Recommended):**

Create `docker-compose.yml`:

```yaml
version: '3.9'
services:
  localai:
    image: quay.io/go-skynet/local-ai:latest
    container_name: localai
    ports:
      - "8080:8080"
    environment:
      - DEBUG=true
      - CONTEXT_SIZE=4096
      - THREADS=4
    volumes:
      - ./models:/models
      - ./config:/config
    restart: unless-stopped
    # Uncomment for GPU support (NVIDIA)
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

Start LocalAI:
```bash
mkdir -p models config
docker-compose up -d
```

**Verify installation:**
```bash
curl http://localhost:8080/readyz
curl http://localhost:8080/v1/models
```

### Model Setup

**1. Download a model:**

LocalAI supports multiple model formats (GGUF, GGML, PyTorch, etc.).

```bash
# Create models directory
mkdir -p models

# Download a GGUF model (Hugging Face)
cd models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**2. Create model configuration:**

Create `config/mistral.yaml`:

```yaml
name: mistral-7b-instruct
backend: llama
parameters:
  model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
  temperature: 0.7
  top_k: 40
  top_p: 0.95
  max_tokens: 2048
context_size: 4096
threads: 4
f16: true
stopwords:
  - "User:"
  - "Assistant:"
template:
  chat: |
    <s>[INST] {{ .Input }} [/INST]
  completion: |
    {{ .Input }}
```

**3. Test the model:**
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b-instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Home Agent Configuration

**Via Home Assistant UI:**

```yaml
# LLM Settings
LLM Base URL: http://localhost:8080/v1
API Key: (leave empty or set if configured)
Model: mistral-7b-instruct  # Must match name in config
Temperature: 0.7
Max Tokens: 500
Keep Alive: 5m  # Not applicable for LocalAI

# Context Settings
Context Mode: direct
Context Format: json
Direct Entities: light.*, climate.*, sensor.*

# History
History Enabled: true
Max Messages: 10

# Memory
Memory Enabled: true
Memory Extraction LLM: local
```

### Advanced Configuration

**GPU Acceleration (NVIDIA):**

Update `docker-compose.yml`:
```yaml
services:
  localai:
    # ... existing config ...
    environment:
      - GPU_LAYERS=35  # Offload layers to GPU
      - GPU_MEMORY_UTILIZATION=0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Multiple models:**

```yaml
# config/gpt-3.5-turbo.yaml (alias for compatibility)
name: gpt-3.5-turbo
backend: llama
parameters:
  model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
  # ... rest of config ...
```

**Embedding model for vector DB:**

Download embedding model:
```bash
cd models
wget https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf
```

Create `config/nomic-embed.yaml`:
```yaml
name: nomic-embed-text
backend: bert-embeddings
parameters:
  model: nomic-embed-text-v1.5.Q4_K_M.gguf
embeddings: true
```

Configure in Home Agent:
```yaml
Embedding Provider: ollama  # LocalAI uses OpenAI-compatible API
Embedding Base URL: http://localhost:8080/v1
Embedding Model: nomic-embed-text
```

### Performance Tuning

**Optimize for speed:**
```yaml
# In model config YAML
parameters:
  threads: 8  # Match CPU cores
  batch_size: 512
  gpu_layers: 35  # If GPU available
context_size: 2048  # Smaller context
```

**Optimize for quality:**
```yaml
parameters:
  temperature: 0.8
  top_p: 0.95
  top_k: 50
  repeat_penalty: 1.1
context_size: 4096
```

### Troubleshooting

**LocalAI not starting:**
```bash
# Check logs
docker logs localai -f

# Common issues:
# - Port 8080 already in use (change port)
# - Insufficient RAM (use smaller model)
# - Model file not found (check volumes)
```

**Model loading errors:**
```bash
# Verify model file exists
ls -lh models/

# Check model config syntax
cat config/mistral.yaml

# Test model directly
curl http://localhost:8080/v1/models
```

**Poor response quality:**
- Adjust temperature (0.7-0.9)
- Check prompt template in config
- Try different model quantization
- Increase context_size

**Out of memory:**
- Use smaller quantized model (Q4 instead of Q6)
- Reduce context_size
- Lower gpu_layers
- Increase system swap

**Integration config file:**

Create `config/integration.yaml` for Home Agent compatibility:
```yaml
name: gpt-4o-mini  # Alias for Home Agent
backend: llama
parameters:
  model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
  temperature: 0.7
  max_tokens: 500
context_size: 4096
```

### Model Recommendations

**For home automation:**
- Mistral 7B Instruct (Q4_K_M) - Best balance
- Llama 3.1 8B Instruct - Better quality
- Phi-3 Mini - Fastest, smallest
- Qwen 2.5 7B - Good tool calling

**Download sources:**
- [TheBloke on Hugging Face](https://huggingface.co/TheBloke) - Wide selection of GGUF models
- [Hugging Face Model Hub](https://huggingface.co/models) - All model types
- [LocalAI Model Gallery](https://localai.io/models/) - Pre-configured models

---

## 4. Multi-LLM Configuration

Dual-LLM strategy: Fast local model for control, powerful cloud model for analysis.

### Architecture Overview

```
User Query
    ↓
Fast Local LLM (Ollama/LocalAI)
├─ Simple commands → Direct response
├─ Tool execution → ha_control, ha_query
└─ Complex analysis → Delegates to External LLM
                          ↓
                   Powerful Cloud LLM (OpenAI/Claude)
                          ↓
                   Detailed analysis/response
```

### Benefits

- **Cost-effective**: Most queries handled by free local LLM
- **Fast**: Local LLM responds in <1 second
- **Powerful**: Cloud LLM available for complex tasks
- **Flexible**: Best tool for each job

### Complete Configuration

**Primary LLM (Local - Fast):**
```yaml
# LLM Settings
LLM Base URL: http://localhost:11434/v1
API Key: (leave empty)
Model: mistral:7b-instruct
Temperature: 0.5  # Lower for consistent control
Max Tokens: 300   # Short responses
Keep Alive: 5m

# Context
Context Mode: vector_db  # Efficient context
Vector DB Top K: 5

# History
Max Messages: 10
History Max Tokens: 3000
```

**External LLM (Cloud - Powerful):**
```yaml
# External LLM Settings
External LLM Enabled: true
External LLM Base URL: https://api.openai.com/v1
External LLM API Key: sk-proj-your-key-here
External LLM Model: gpt-4o
External LLM Temperature: 0.8  # Higher for creativity
External LLM Max Tokens: 1500
External LLM Keep Alive: 5m

# Tool Description
External LLM Tool Description: |
  Use this powerful external AI for:
  - Detailed analysis of energy usage, climate patterns, or trends
  - Complex recommendations requiring multi-step reasoning
  - Creative suggestions for automations or scenes
  - Explaining concepts or providing detailed instructions

  Do NOT use for:
  - Simple device control (turn on/off)
  - Quick status queries
  - Routine tool execution

# Auto-include context
Auto Include Context: true  # Pass entity context automatically
```

**Vector DB (Shared):**
```yaml
# Vector DB Settings
Vector DB Host: localhost
Vector DB Port: 8000
Collection: home_entities
Embedding Provider: ollama
Embedding Base URL: http://localhost:11434
Embedding Model: nomic-embed-text
Top K: 5
Similarity Threshold: 250.0
```

**Memory (Use external for quality):**
```yaml
# Memory Settings
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: external  # Use GPT-4 for quality extraction
Memory Max Memories: 500
Memory Min Importance: 0.3
Memory Context Top K: 5
```

### Use Case Examples

**Example 1: Simple Control (Local LLM Only)**
```
User: "Turn on the living room lights"
Local LLM: [Uses ha_control tool] → Done, no external LLM needed
Cost: $0.00
```

**Example 2: Status Query (Local LLM Only)**
```
User: "What's the temperature in the bedroom?"
Local LLM: [Reads from context CSV] → "72°F"
Cost: $0.00
```

**Example 3: Complex Analysis (Delegates to External)**
```
User: "Analyze my energy usage this week and suggest ways to save money"
Local LLM:
  1. Calls ha_query to get energy data
  2. Calls query_external_llm with data
External LLM: [Analyzes data, provides detailed recommendations]
Local LLM: Formats and presents response
Cost: ~$0.02 (only external LLM call)
```

**Example 4: Automation Planning (Delegates to External)**
```
User: "Help me create a smart lighting automation for my office"
Local LLM: Recognizes complex planning task
  → Calls query_external_llm
External LLM: Provides detailed automation design
Local LLM: Guides user through implementation
Cost: ~$0.01-0.03
```

### Cost Optimization

**Typical usage pattern:**
- 90% queries: Local LLM (free)
- 10% queries: External LLM ($0.01-0.05 each)
- Average monthly cost: $1-5

**Optimization strategies:**

**1. Tune the tool description:**
Be specific about when to use external LLM:
```yaml
External LLM Tool Description: |
  ONLY use for:
  - Energy/climate analysis requiring >5 data points
  - Automation design with 3+ steps
  - Detailed explanations >200 words

  Use local LLM for everything else.
```

**2. Limit external LLM tokens:**
```yaml
External LLM Max Tokens: 800  # Shorter responses
```

**3. Set tool call limits:**
```yaml
Tools Max Calls Per Turn: 3  # Prevent excessive delegation
Tools Timeout: 30
```

**4. Monitor usage:**
```yaml
automation:
  - alias: "Track External LLM Usage"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
        event_data:
          tool_name: query_external_llm
    action:
      - service: counter.increment
        target:
          entity_id: counter.external_llm_calls_daily
      - service: logbook.log
        data:
          name: "External LLM"
          message: "Query: {{ trigger.event.data.parameters.prompt[:100] }}"
```

### Alternative Configurations

**Budget-conscious:**
```yaml
# Primary: Local
Model: mistral:7b-instruct

# External: Cheaper cloud
External LLM Model: gpt-4o-mini  # $0.15/$0.60 per 1M tokens
External LLM Max Tokens: 500
```

**Privacy-focused:**
```yaml
# Primary: Local
Model: mistral:7b-instruct

# External: Larger local model
External LLM Enabled: true
External LLM Base URL: http://localhost:11434/v1
External LLM Model: llama3.1:70b  # Requires powerful hardware
```

**Quality-focused:**
```yaml
# Primary: Better local model
Model: qwen2.5:14b

# External: Best cloud model
External LLM Model: gpt-4o
External LLM Temperature: 0.9
External LLM Max Tokens: 2000
```

### Troubleshooting

**External LLM never called:**
- Check `External LLM Enabled: true`
- Verify API key is correct
- Try asking more complex questions
- Review tool description (may be too restrictive)

**External LLM called too often:**
- Refine tool description to be more restrictive
- Increase local LLM temperature for better responses
- Use better local model
- Check local LLM logs for errors

**High costs:**
- Review external LLM call logs
- Reduce `External LLM Max Tokens`
- Make tool description more specific
- Set up daily spending alerts
- Use gpt-4o-mini instead of gpt-4o

**Poor response quality:**
- Local LLM may not be passing enough context
- Increase `External LLM Max Tokens`
- Adjust `External LLM Temperature`
- Try different external model
- Enable `Auto Include Context`

---

## 5. Memory System Setup

Configure persistent long-term memory for personalized, context-aware interactions.

### Prerequisites

**Required:**
- ChromaDB installed and running (see [Vector DB setup](#6-vector-db-with-chromadb))
- Embedding provider configured (Ollama or OpenAI)
- Vector DB mode enabled (or at least ChromaDB accessible)

### Basic Setup

**1. Ensure ChromaDB is running:**
```bash
curl http://localhost:8000/api/v1/heartbeat
# Expected: {"nanosecond heartbeat": ...}
```

**2. Configure Memory Settings:**

Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure** > **Memory Settings**

```yaml
# Memory Settings
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: external  # Options: external, local
Memory Max Memories: 100
Memory Min Importance: 0.3
Memory Context Top K: 5
Memory Collection Name: home_agent_memories
```

**3. Configure Embedding Provider (if not already done):**

**Option A - Ollama (Local):**
```bash
ollama pull nomic-embed-text
```

```yaml
# In Vector DB Settings
Embedding Provider: ollama
Embedding Base URL: http://localhost:11434
Embedding Model: nomic-embed-text
```

**Option B - OpenAI (Cloud):**
```yaml
# In Vector DB Settings
Embedding Provider: openai
Embedding Base URL: https://api.openai.com/v1
Embedding Model: text-embedding-3-small
OpenAI API Key: sk-proj-your-key-here
```

**4. Set up extraction LLM:**

**If using external extraction:**
```yaml
# External LLM Settings
External LLM Enabled: true
External LLM Base URL: https://api.openai.com/v1
External LLM API Key: sk-proj-your-key-here
External LLM Model: gpt-4o-mini  # Good balance of quality and cost
```

**If using local extraction:**
```yaml
# Uses your primary LLM configuration
# No additional setup needed
# Memory extraction uses same model as conversations
```

### Configuration Details

**Memory Enabled:**
- Master switch for all memory features
- When off: No extraction, no recall, memories not used
- Existing memories preserved but inactive

**Memory Extraction Enabled:**
- Controls automatic extraction after conversations
- When on: Extracts facts, preferences, events automatically
- When off: Manual memory storage still works via services
- Recommended: `true` for automatic behavior

**Memory Extraction LLM:**
- `external`: Uses configured external LLM (better quality, may cost money)
- `local`: Uses primary conversation LLM (simpler, free for local LLMs)
- Recommendation: `external` with gpt-4o-mini for best results

**Memory Max Memories:**
- Maximum number of memories to store
- When exceeded: Oldest/least important memories pruned
- Recommended: `100` (start), adjust to `500+` for heavy use

**Memory Min Importance (0.0-1.0):**
- Memories below this threshold not stored
- `0.3` (default): Moderate importance filter
- `0.5`: More selective (only important info)
- `0.1`: Less selective (keep more info)

**Memory Context Top K:**
- Number of memories to inject into each conversation
- `5` (recommended): Balanced relevance
- `3`: Minimal context (faster)
- `10`: Rich context (may add noise)

### Privacy Configuration

**Minimal data collection:**
```yaml
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: local  # No cloud calls
Memory Max Memories: 50       # Limit storage
Memory Min Importance: 0.7    # Only very important info
Memory Context Top K: 3       # Minimal injection

# Use local embeddings
Embedding Provider: ollama
Embedding Model: nomic-embed-text
```

**Complete privacy (all local):**
```yaml
# Primary LLM: Local
LLM Base URL: http://localhost:11434/v1
Model: mistral:7b-instruct

# Memory: Local extraction
Memory Extraction LLM: local

# Embeddings: Local
Embedding Provider: ollama
Embedding Base URL: http://localhost:11434
Embedding Model: nomic-embed-text

# ChromaDB: Local
Vector DB Host: localhost
Vector DB Port: 8000
```

Result: All processing local, no cloud API calls, complete privacy.

### Quality-Focused Configuration

**Best memory quality:**
```yaml
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: external  # Use powerful model
Memory Max Memories: 500         # Store more
Memory Min Importance: 0.2       # Capture more info
Memory Context Top K: 8          # Rich context

# External LLM
External LLM Model: gpt-4o       # Best extraction quality
External LLM Temperature: 0.7

# Embeddings
Embedding Provider: openai
Embedding Model: text-embedding-3-large  # Better embeddings
```

### Testing Memory

**1. Store a preference:**
```yaml
service: home_agent.process
data:
  text: "I prefer the bedroom at 68°F for sleeping"
```

Check logs for extraction:
```
[home_agent.memory] Extracted 1 memories from conversation
[home_agent.memory] Stored memory: User prefers bedroom at 68°F for sleeping (type=preference, importance=0.8)
```

**2. Later, test recall:**
```yaml
service: home_agent.process
data:
  text: "What temperature should I set for the bedroom?"
```

Expected response:
```
"Based on your preference, you like the bedroom at 68°F for sleeping."
```

**3. Manually check memories:**
```yaml
service: home_agent.list_memories
data:
  limit: 10
```

### Manual Memory Management

**Add memory directly:**
```yaml
service: home_agent.add_memory
data:
  content: "Garbage pickup is every Thursday morning"
  type: fact
  importance: 0.7
```

**Search memories:**
```yaml
service: home_agent.search_memories
data:
  query: "temperature preferences"
  limit: 5
  min_importance: 0.5
```

**Delete specific memory:**
```yaml
service: home_agent.delete_memory
data:
  memory_id: "abc-123-def-456"  # From list_memories
```

**Clear all memories:**
```yaml
service: home_agent.clear_memories
data:
  confirm: true  # Required safety check
```

### Memory Types and TTL

Home Agent supports four memory types with different retention:

| Type | Description | Default TTL | Example |
|------|-------------|-------------|---------|
| `fact` | Permanent factual info | Never expires | "Dog named Max", "WiFi password is..." |
| `preference` | User preferences | 90 days | "Prefers 68°F for sleep" |
| `context` | Conversational context | 5 minutes | "Currently planning party" |
| `event` | Timestamped events | 5 minutes | "Changed filter on 2024-01-15" |

**Configure custom TTL (advanced):**

Via `configuration.yaml`:
```yaml
home_agent:
  memory_event_ttl: 300          # 5 minutes (default)
  memory_fact_ttl: null          # Never expire (default)
  memory_preference_ttl: 7776000 # 90 days (default)
  memory_cleanup_interval: 300   # Run cleanup every 5 minutes
```

### Troubleshooting

**No memories extracted:**
- Check `Memory Enabled: true` and `Memory Extraction Enabled: true`
- Verify ChromaDB is running: `curl http://localhost:8000/api/v1/heartbeat`
- Check external LLM configured if using `extraction_llm: external`
- Review logs for extraction errors
- Try: Have conversation with clear facts ("I like X", "Remember that Y")

**Memories not recalled:**
- Verify `Memory Enabled: true`
- Check memories exist: `service: home_agent.list_memories`
- Increase `Memory Context Top K`
- Lower `Memory Min Importance` threshold
- Test search: `service: home_agent.search_memories`

**Poor extraction quality:**
- Use `Memory Extraction LLM: external` with good model (gpt-4o-mini or better)
- Check extraction model has enough max_tokens (500+)
- Review extracted memories with `list_memories`
- Some LLMs better at extraction than others

**ChromaDB connection errors:**
```bash
# Verify ChromaDB running
docker ps | grep chroma
curl http://localhost:8000/api/v1/heartbeat

# Restart if needed
docker restart chromadb

# Check logs
docker logs chromadb
```

**Embedding errors:**
```bash
# Ollama
ollama list
ollama pull nomic-embed-text

# Test embedding
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test"}'
```

**Storage location:**
- Memories: `.storage/home_agent.memories`
- Vectors: ChromaDB volume (configured in Docker)

### Best Practices

**1. Start conservative, expand as needed:**
```yaml
# Start
Memory Max Memories: 100
Memory Min Importance: 0.4
Memory Context Top K: 5

# After testing, if working well
Memory Max Memories: 500
Memory Min Importance: 0.3
Memory Context Top K: 8
```

**2. Use appropriate extraction LLM:**
- Testing/development: `local`
- Production/quality: `external` with gpt-4o-mini
- Best quality: `external` with gpt-4o

**3. Monitor memory storage:**
```yaml
automation:
  - alias: "Memory Storage Monitor"
    trigger:
      - platform: time
        at: "00:00:00"  # Daily at midnight
    action:
      - service: home_agent.list_memories
        response_variable: memories
      - service: notify.admin
        data:
          message: "Total memories: {{ memories.memories | length }}"
```

**4. Periodic cleanup:**
```yaml
automation:
  - alias: "Monthly Memory Cleanup"
    trigger:
      - platform: time
        at: "03:00:00"
    condition:
      - condition: template
        value_template: "{{ now().day == 1 }}"  # First of month
    action:
      - service: home_agent.search_memories
        data:
          query: "outdated temporary"
          min_importance: 0.0
        response_variable: old_memories
      # Review and delete as needed
```

**5. Privacy considerations:**
- Memories are global (shared across all users in household)
- No per-user isolation currently
- Consider for shared/multi-user homes
- Use `clear_memories` before demos/guests
- All storage is local unless using OpenAI embeddings

---

## 6. Vector DB with ChromaDB

Complete setup guide for semantic entity search and memory system foundation.

### What is Vector DB Mode?

Vector DB mode uses ChromaDB to store entity states as vector embeddings, enabling semantic similarity search. Instead of manually specifying entities, the system automatically finds relevant entities based on meaning.

**Example:**
```
Query: "Is it warm in my bedroom?"
Direct mode: Must configure "sensor.bedroom_temperature" explicitly
Vector DB: Automatically finds bedroom temperature sensor via semantic search
```

### Benefits

- **Automatic relevance**: Finds entities based on meaning, not exact names
- **Scalable**: Handles 100+ entities efficiently
- **Token efficient**: Only includes relevant entities in context
- **Foundation for memory**: Required for long-term memory system
- **Flexible queries**: Works with varied phrasing

### Installation

**1. Install ChromaDB:**

**Docker (Recommended):**
```bash
# Create persistent storage directory
mkdir -p /config/chromadb

# Run ChromaDB container
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v /config/chromadb:/chroma/chroma \
  -e ANONYMIZED_TELEMETRY=False \
  -e ALLOW_RESET=True \
  --restart unless-stopped \
  chromadb/chroma:latest
```

**Docker Compose:**

Create `docker-compose.yml`:
```yaml
version: '3.9'
services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    ports:
      - "8000:8000"
    volumes:
      - ./chromadb:/chroma/chroma
    environment:
      - ANONYMIZED_TELEMETRY=False
      - ALLOW_RESET=True
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_PORT=8000
    restart: unless-stopped
```

Start:
```bash
docker-compose up -d
```

**Python (Alternative):**
```bash
pip install chromadb
chromadb run --host 0.0.0.0 --port 8000
```

**2. Verify ChromaDB:**
```bash
curl http://localhost:8000/api/v1/heartbeat
# Expected: {"nanosecond heartbeat": <timestamp>}

curl http://localhost:8000/api/v1/version
# Expected: version information
```

### Embedding Provider Setup

ChromaDB needs an embedding model to convert text to vectors.

**Option A: Ollama (Local, Free, Recommended)**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull embedding model
ollama pull nomic-embed-text

# Verify
ollama list
# Should show: nomic-embed-text

# Test embedding
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test query"}'
```

**Option B: OpenAI (Cloud, Paid)**

Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)

Cost: ~$0.02 per 1M tokens (~$0.01-0.05/month for typical home automation)

### Configure Home Agent

**1. Configure Vector DB Connection:**

Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure** > **Vector DB Settings**

**For Ollama (Local):**
```yaml
Vector DB Host: localhost
Vector DB Port: 8000
Collection Name: home_entities
Embedding Provider: ollama
Embedding Base URL: http://localhost:11434
Embedding Model: nomic-embed-text
OpenAI API Key: (leave empty)
Top K: 5
Similarity Threshold: 250.0
Keep Alive: 5m
```

**For OpenAI (Cloud):**
```yaml
Vector DB Host: localhost
Vector DB Port: 8000
Collection Name: home_entities
Embedding Provider: openai
Embedding Base URL: https://api.openai.com/v1
Embedding Model: text-embedding-3-small
OpenAI API Key: sk-proj-your-key-here
Top K: 5
Similarity Threshold: 250.0
```

**2. Enable Vector DB Mode:**

Navigate to **Configure** > **Context Settings**

```yaml
Context Mode: vector_db  # Switch from "direct" to "vector_db"
Context Format: json     # or natural_language
```

Click **Submit**

### Index Your Entities

After configuration, index your Home Assistant entities:

**Via Developer Tools:**
1. Open **Developer Tools** > **Services**
2. Select `home_agent.reindex_entities`
3. Click **Call Service**

**Via YAML:**
```yaml
service: home_agent.reindex_entities
```

**Monitor progress in logs:**
```
[home_agent.vector_db] Starting entity indexing...
[home_agent.vector_db] Indexing 127 entities...
[home_agent.vector_db] Successfully indexed 127 entities in 12.4 seconds
```

**When to reindex:**
- After initial setup (required)
- After adding/removing significant entities
- After entity renames
- Weekly/monthly for large setups (optional)

### Verify It Works

**Test query:**
```yaml
service: home_agent.process
data:
  text: "What's the temperature in the bedroom?"
```

**Check logs for vector search:**
```
[home_agent.vector_db] Searching for entities related to: "bedroom temperature"
[home_agent.vector_db] Retrieved 3 entities (scores: 45.2, 87.4, 123.6)
[home_agent.vector_db] Entities: sensor.bedroom_temperature, climate.bedroom_ac, sensor.bedroom_humidity
```

**Expected behavior:**
- LLM receives only bedroom-related sensors
- Response references correct entity
- No irrelevant entities included

### Configuration Tuning

**Top K (Number of Results):**

Controls how many entities to retrieve per query.

```yaml
Top K: 3  # Minimal context
Top K: 5  # Balanced (recommended)
Top K: 10 # Rich context
Top K: 20 # Very comprehensive
```

**Recommendations:**
- Small homes (<50 entities): `Top K: 3-5`
- Medium homes (50-150 entities): `Top K: 5-8`
- Large homes (150+ entities): `Top K: 8-15`

**Symptoms:**
- Too few results: LLM lacks context, can't find entities
- Too many results: Irrelevant entities, increased tokens, slower

**Similarity Threshold (L2 Distance):**

Controls minimum similarity required. Lower = more similar required.

```yaml
Similarity Threshold: 100.0  # Very strict (highly relevant only)
Similarity Threshold: 250.0  # Balanced (recommended)
Similarity Threshold: 500.0  # Lenient (more entities)
Similarity Threshold: 1000.0 # Very lenient (debugging)
```

**L2 Distance explained:**
- Lower distance = more similar
- Typical useful range: 50-500
- >500: May include irrelevant entities
- <100: May exclude relevant entities

**Recommendations:**
- Start with `250.0`
- If too few entities returned: Increase to `350.0` or `500.0`
- If irrelevant entities: Decrease to `150.0` or `100.0`

**Log distance scores to calibrate:**
```yaml
# In configuration.yaml (enable debug logging)
logger:
  logs:
    custom_components.home_agent.vector_db: debug
```

Check logs for actual distance scores and adjust threshold accordingly.

### Additional Collections

You can query multiple ChromaDB collections (for custom data, memories, etc.).

**Configure:**
```yaml
# In Vector DB Settings
Additional Collections: custom_data,home_memories  # Comma-separated
Additional Top K: 5
Additional L2 Distance Threshold: 250.0
```

**Use case:** Store custom information (recipes, notes, schedules) in separate collections and query alongside entities.

### Performance Optimization

**1. Use appropriate embedding model:**

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `nomic-embed-text` (Ollama) | 274MB | Fast | Excellent | Recommended |
| `text-embedding-3-small` (OpenAI) | N/A | Fast | Excellent | Cloud option |
| `text-embedding-3-large` (OpenAI) | N/A | Medium | Best | Quality-focused |
| `all-minilm-l6-v2` | Small | Fastest | Good | Resource-constrained |

**2. Optimize ChromaDB storage:**

```bash
# Use SSD storage for ChromaDB volume
# Allocate adequate RAM (2GB+ recommended)
# Regular backups
docker exec chromadb tar -czf /backup/chromadb.tar.gz /chroma/chroma
```

**3. Batch reindexing for large setups:**

Instead of full reindex, update incrementally:
```yaml
# Not currently supported - use full reindex
# Feature request: Incremental entity updates
```

**4. Reduce query latency:**

```yaml
# Fewer results
Top K: 3

# Local embeddings (no network delay)
Embedding Provider: ollama

# Smaller embedding model
Embedding Model: all-minilm-l6-v2
```

### Privacy Configuration

**Fully local (no cloud):**
```yaml
# Embedding: Local
Embedding Provider: ollama
Embedding Base URL: http://localhost:11434
Embedding Model: nomic-embed-text

# ChromaDB: Local
Vector DB Host: localhost
Vector DB Port: 8000

# All data stored locally
# No external API calls
# Complete privacy
```

**Data location:**
- ChromaDB data: Configured volume (`/config/chromadb` in example)
- Embeddings: Generated locally by Ollama or sent to OpenAI
- Entity data: Never leaves your network (except embeddings if using OpenAI)

### Troubleshooting

**ChromaDB connection failed:**
```bash
# Check if running
docker ps | grep chromadb

# Check logs
docker logs chromadb

# Restart
docker restart chromadb

# Test connection
curl http://localhost:8000/api/v1/heartbeat

# Check firewall/port
netstat -tlnp | grep 8000
```

**Ollama embedding errors:**
```bash
# Verify Ollama running
systemctl status ollama  # or `docker ps` if Docker

# Check model pulled
ollama list | grep nomic

# Pull if missing
ollama pull nomic-embed-text

# Test embedding
ollama embed nomic-embed-text "test query"
```

**OpenAI embedding errors:**
```bash
# Test API key
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "test",
    "model": "text-embedding-3-small"
  }'

# Check key permissions and credits
# Visit: https://platform.openai.com/usage
```

**No entities retrieved:**
1. Verify entities indexed: Check logs for "Successfully indexed X entities"
2. Re-run indexing: `service: home_agent.reindex_entities`
3. Increase `Top K` or `Similarity Threshold`
4. Check embedding model working
5. Enable debug logging and review distance scores

**Irrelevant entities retrieved:**
1. Decrease `Similarity Threshold`
2. Decrease `Top K`
3. Reindex with better entity names/aliases
4. Review distance scores in debug logs

**Slow queries:**
1. Reduce `Top K`
2. Use local embeddings (Ollama)
3. Ensure ChromaDB has adequate resources
4. Use faster embedding model
5. Consider SSD for ChromaDB storage

**Collection not found:**
```bash
# Check collections in ChromaDB
curl http://localhost:8000/api/v1/collections

# Reset and reindex
service: home_agent.reindex_entities
```

**Indexing fails:**
1. Check ChromaDB logs for errors
2. Verify embedding provider configured correctly
3. Test embedding generation separately
4. Check disk space for ChromaDB volume
5. Review Home Agent logs for specific error

### Backup and Restore

**Backup ChromaDB data:**
```bash
# Stop ChromaDB
docker stop chromadb

# Backup data directory
tar -czf chromadb-backup-$(date +%Y%m%d).tar.gz /config/chromadb

# Restart ChromaDB
docker start chromadb
```

**Restore ChromaDB data:**
```bash
# Stop ChromaDB
docker stop chromadb

# Restore data
tar -xzf chromadb-backup-20240115.tar.gz -C /

# Restart ChromaDB
docker start chromadb

# Verify
curl http://localhost:8000/api/v1/heartbeat
```

**Automated backup:**
```yaml
automation:
  - alias: "Weekly ChromaDB Backup"
    trigger:
      - platform: time
        at: "03:00:00"
    condition:
      - condition: template
        value_template: "{{ now().weekday() == 0 }}"  # Monday
    action:
      - service: shell_command.backup_chromadb

shell_command:
  backup_chromadb: >
    docker exec chromadb tar -czf /backup/chromadb-$(date +\%Y\%m\%d).tar.gz /chroma/chroma
```

### Advanced: Multiple Collections

Create and query custom collections for specialized data.

**Use case:** Store custom knowledge base, documentation, or reference data.

**Create collection via API:**
```bash
curl -X POST http://localhost:8000/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "custom_knowledge",
    "metadata": {"description": "Custom knowledge base"}
  }'
```

**Add documents:**
```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection("custom_knowledge")

collection.add(
    documents=["Home garbage pickup is on Thursday mornings"],
    ids=["info_1"],
    metadatas=[{"type": "schedule", "importance": 0.8}]
)
```

**Configure Home Agent to query:**
```yaml
# In Vector DB Settings
Additional Collections: custom_knowledge
Additional Top K: 3
```

**Result:** Home Agent will search both `home_entities` and `custom_knowledge` collections.

---

## Summary Table

Quick reference for choosing configurations:

| Configuration | Best For | Cost | Setup Complexity | Privacy |
|---------------|----------|------|------------------|---------|
| OpenAI | Quick start, voice control | $1-5/month | Low | Moderate (cloud) |
| Ollama | Local control, privacy | Free | Medium | Excellent (local) |
| LocalAI | Advanced local, flexibility | Free | High | Excellent (local) |
| Multi-LLM | Best of both, cost-effective | $1-10/month | Medium | Mixed |
| Memory System | Personalized experience | Varies | Medium | Depends on setup |
| Vector DB | Large setups, semantic search | Low | Medium | Depends on embeddings |

## Next Steps

After setting up your configuration:

1. **Test basic functionality:** Simple commands, status queries
2. **Add custom tools:** Extend with REST APIs or Home Assistant services (see [CUSTOM_TOOLS.md](CUSTOM_TOOLS.md))
3. **Set up automations:** Integrate with voice assistants, triggers (see [EXAMPLES.md](EXAMPLES.md))
4. **Monitor performance:** Enable debug logging, track token usage
5. **Optimize for your use case:** Adjust settings based on actual usage patterns

## Additional Resources

- [Configuration Guide](CONFIGURATION.md) - Complete configuration reference
- [Vector DB Setup](VECTOR_DB_SETUP.md) - Detailed vector database guide
- [Memory System](MEMORY_SYSTEM.md) - Comprehensive memory documentation
- [External LLM](EXTERNAL_LLM.md) - Multi-LLM strategies
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [FAQ](FAQ.md) - Frequently asked questions

## Contributing

Found an issue with these examples or have a suggestion? Please:
- Open an issue on [GitHub](https://github.com/aradlein/hass-agent-llm/issues)
- Submit a pull request with improvements
- Share your configuration in [Discussions](https://github.com/aradlein/hass-agent-llm/discussions)

---

**Need help?** Join the discussion on GitHub or check the [FAQ](FAQ.md) for quick answers.

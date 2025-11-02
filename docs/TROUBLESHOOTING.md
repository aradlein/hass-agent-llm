# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Home Agent integration.

## Table of Contents

- [LLM Connection Issues](#llm-connection-issues)
- [Tool Execution Errors](#tool-execution-errors)
- [Memory System Issues](#memory-system-issues)
- [Vector DB Issues](#vector-db-issues)
- [Streaming Issues](#streaming-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [Debugging Techniques](#debugging-techniques)

---

## LLM Connection Issues

### Authentication Error (401)

**Symptoms:**
- Error: "LLM API authentication failed. Check your API key."
- 401 status code in logs

**Solutions:**
1. Verify your API key is correct in configuration
2. Check if API key has expired or been revoked
3. For OpenAI: Ensure you're using the correct key format (starts with `sk-`)
4. For local models (Ollama): Check if API key is required at all
5. Verify base URL is correct for your provider

**Test:**
```bash
# Test OpenAI connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"

# Test Ollama connection
curl http://localhost:11434/api/tags
```

### API Endpoint Unreachable

**Symptoms:**
- Error: "Failed to connect to LLM API"
- Connection timeout or refused

**Solutions:**
1. Check base URL configuration:
   - OpenAI: `https://api.openai.com/v1`
   - Ollama: `http://localhost:11434/v1`
   - LocalAI: `http://localhost:8080/v1`
2. Verify service is running (for local models)
3. Check firewall settings
4. Ensure network connectivity
5. For Docker/containers: Verify network configuration

**Test:**
```bash
# Check if endpoint is accessible
curl -I http://localhost:11434/v1/models
```

### Timeout Errors

**Symptoms:**
- Error: "LLM API request timed out"
- Long delays before error

**Solutions:**
1. Increase HTTP timeout in configuration (`HTTP_TIMEOUT = 60`)
2. Check network latency
3. For local models: Verify adequate system resources (RAM, CPU)
4. Reduce max_tokens to speed up generation
5. Use faster model (e.g., gpt-4o-mini instead of gpt-4)

### Rate Limiting (429)

**Symptoms:**
- Error: "Rate limit exceeded"
- 429 status code in logs

**Solutions:**
1. Reduce conversation frequency
2. Implement delays between requests
3. Upgrade API plan with provider
4. Use local model to avoid rate limits
5. Check provider dashboard for quota usage

### Model Not Found

**Symptoms:**
- Error: "Model not found" or 404 status
- Invalid model name in logs

**Solutions:**
1. Verify model name spelling in configuration
2. Check available models:
   - OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
   - Ollama: Run `ollama list` to see installed models
3. For Ollama: Pull model first: `ollama pull llama2`
4. Ensure model is compatible with your API tier

---

## Tool Execution Errors

### Tool Not Found

**Symptoms:**
- Error: "Tool 'tool_name' not found"
- Available tools listed in error

**Solutions:**
1. Verify tool name spelling matches exactly
2. Check if custom tools are properly defined in `configuration.yaml`
3. For memory tools: Ensure memory system is enabled
4. For external LLM: Verify `CONF_EXTERNAL_LLM_ENABLED = true`
5. Restart Home Assistant after adding custom tools

**Available Built-in Tools:**
- `ha_control` - Control devices
- `ha_query` - Query device states
- `query_external_llm` - Query external LLM (if enabled)
- `store_memory` - Store memory (if enabled)
- `recall_memory` - Recall memories (if enabled)

### Permission Denied

**Symptoms:**
- Error: "Entity is not accessible"
- PermissionDenied exception
- Tool execution fails for specific entities

**Solutions:**
1. **Expose entities to voice assistants:**
   - Go to Settings → Voice assistants → Expose
   - Select entities to expose
   - Or expose individual entities in their settings
2. Check entity exists and is not disabled
3. Verify entity ID format: `domain.entity_name`
4. Check entity domain is supported by tool
5. Review Home Assistant logs for permission errors

### Entity Not Exposed

**Symptoms:**
- Tool executes but entity not found
- Empty results from ha_query

**Solutions:**
1. Verify entity is exposed to conversation:
   ```python
   # Check in Developer Tools → Services
   service: homeassistant.expose_entity
   data:
     entity_id: light.living_room
     assistants: conversation
   ```
2. Check entity exists: Developer Tools → States
3. Restart Home Assistant after exposing entities
4. Verify entity is not in a disabled state

### Custom Tool Configuration Errors

**Symptoms:**
- Custom tool not registered
- ValidationError during startup
- Tool definition errors in logs

**Solutions:**
1. **Validate YAML syntax:**
   ```yaml
   home_agent:
     custom_tools:
       - name: my_tool
         description: "Tool description"
         parameters:
           type: object
           properties:
             param_name:
               type: string
         handler:
           type: rest  # or service
           url: "https://api.example.com"
   ```
2. Check handler type is valid: `rest` or `service`
3. Ensure all required fields are present
4. Validate JSON schema for parameters
5. Check handler configuration for type-specific fields

**Common mistakes:**
- Missing quotes around URLs
- Invalid JSON schema
- Using unsupported handler type
- Missing required handler fields

### External LLM Tool Failures

**Symptoms:**
- Error: "Failed to query external LLM"
- Timeout or connection error when using external LLM

**Solutions:**
1. Verify external LLM configuration:
   - `CONF_EXTERNAL_LLM_ENABLED = true`
   - Correct base URL and API key
   - Valid model name
2. Check external LLM is accessible
3. Increase `HTTP_TIMEOUT_EXTERNAL_LLM` (default: 90s)
4. Review external LLM provider status
5. Test external endpoint separately

---

## Memory System Issues

### Memory Extraction Not Working

**Symptoms:**
- No memories being extracted from conversations
- No `home_agent.memory.extracted` events fired

**Solutions:**
1. **Enable memory extraction:**
   ```yaml
   CONF_MEMORY_ENABLED: true
   CONF_MEMORY_EXTRACTION_ENABLED: true
   ```
2. Check which LLM is configured for extraction:
   - `CONF_MEMORY_EXTRACTION_LLM: "external"` - Requires external LLM enabled
   - `CONF_MEMORY_EXTRACTION_LLM: "local"` - Uses primary LLM
3. Review logs for extraction errors
4. Verify extraction prompt is generating valid JSON
5. Check importance threshold isn't filtering all memories

### ChromaDB Connection Errors

**Symptoms:**
- Error: "Failed to connect to ChromaDB"
- Memory storage/retrieval fails
- Vector DB query errors

**Solutions:**
1. **Verify ChromaDB is running:**
   ```bash
   # Check if ChromaDB is accessible
   curl http://localhost:8000/api/v1/heartbeat
   ```
2. Check configuration:
   ```yaml
   CONF_VECTOR_DB_HOST: "localhost"
   CONF_VECTOR_DB_PORT: 8000
   ```
3. Install ChromaDB dependencies: `chromadb >= 0.4.0`
4. Check ChromaDB logs for errors
5. Verify collection name is valid

### Memories Not Recalled

**Symptoms:**
- Memories stored but not appearing in conversations
- Search returns no results

**Solutions:**
1. Check importance threshold:
   ```yaml
   CONF_MEMORY_MIN_IMPORTANCE: 0.3  # Lower to include more memories
   ```
2. Verify memory collection name matches configuration
3. Check ChromaDB indexing completed successfully
4. Review memory metadata and topics for relevance
5. Test memory search manually:
   ```yaml
   service: home_agent.search_memories
   data:
     query: "temperature preferences"
     limit: 10
     min_importance: 0.0
   ```

### Memory Services Errors

**Symptoms:**
- Memory services return errors
- Unable to list/delete/search memories

**Solutions:**
1. Verify memory system is enabled
2. Check memory storage file exists: `.storage/home_agent.memories`
3. Ensure memory manager is initialized
4. Review logs for MemoryManager errors
5. Validate memory IDs when deleting

---

## Vector DB Issues

### ChromaDB Connection Failures

**Symptoms:**
- Cannot connect to ChromaDB
- Vector DB queries timing out

**Solutions:**
1. **Start ChromaDB:**
   ```bash
   # Using Docker
   docker run -p 8000:8000 chromadb/chroma

   # Or install locally
   pip install chromadb
   chroma run --path /path/to/chroma/data
   ```
2. Check host and port configuration
3. Verify firewall allows port 8000
4. Check ChromaDB version compatibility
5. Review ChromaDB logs

### Embedding Provider Errors

**Symptoms:**
- Error generating embeddings
- Cannot index entities or memories

**Solutions:**
1. **For OpenAI embeddings:**
   ```yaml
   CONF_VECTOR_DB_EMBEDDING_PROVIDER: "openai"
   CONF_OPENAI_API_KEY: "sk-..."
   CONF_VECTOR_DB_EMBEDDING_MODEL: "text-embedding-3-small"
   ```
2. **For Ollama embeddings:**
   ```yaml
   CONF_VECTOR_DB_EMBEDDING_PROVIDER: "ollama"
   CONF_VECTOR_DB_EMBEDDING_BASE_URL: "http://localhost:11434"
   ```
3. Install required dependencies:
   - OpenAI: `openai >= 1.3.8`
4. Test embedding generation separately
5. Verify embedding model is available

### Slow Searches

**Symptoms:**
- Vector DB queries take too long
- Context retrieval delays

**Solutions:**
1. Reduce `CONF_VECTOR_DB_TOP_K` (default: 5)
2. Increase similarity threshold to filter results
3. Optimize ChromaDB configuration
4. Check system resources (RAM, CPU)
5. Consider using faster embedding model

### Index Not Updating

**Symptoms:**
- New entities not appearing in searches
- Stale entity data in vector DB

**Solutions:**
1. **Manually reindex:**
   ```yaml
   service: home_agent.reindex_entities
   ```
2. **Index specific entity:**
   ```yaml
   service: home_agent.index_entity
   data:
     entity_id: light.living_room
   ```
3. Check auto-indexing is enabled
4. Verify ChromaDB collection is accessible
5. Review indexing logs for errors

---

## Streaming Issues

### Streaming Not Working

**Symptoms:**
- No streaming output from assistant
- Responses appear all at once

**Solutions:**
1. **Enable streaming:**
   ```yaml
   CONF_STREAMING_ENABLED: true
   ```
2. Verify assist pipeline supports streaming:
   - Wyoming TTS integration required
   - Voice assistant pipeline configured
3. Check ChatLog is available during conversation
4. Review streaming prerequisites in configuration
5. Test with voice assistant, not just service calls

### Fallback to Synchronous

**Symptoms:**
- Warning: "Streaming failed, falling back to synchronous mode"
- `home_agent.streaming.error` event fired

**Solutions:**
1. Review error details in `streaming.error` event
2. Check LLM endpoint supports SSE streaming
3. Verify streaming format compatibility
4. Increase timeouts if stream is slow
5. Check for network issues during streaming

### Audio Delays

**Symptoms:**
- Long delay before audio starts
- Choppy or interrupted audio

**Solutions:**
1. Check TTS processing speed
2. Verify network latency to LLM API
3. Use faster model for better streaming performance
4. Optimize LLM parameters (lower max_tokens)
5. Check Home Assistant system resources

---

## Performance Issues

### Slow Responses

**Symptoms:**
- Response time exceeds 5-10 seconds
- Timeout warnings in logs

**Solutions:**
1. **Optimize context:**
   - Reduce number of exposed entities
   - Use vector DB mode instead of direct injection
   - Lower `CONF_HISTORY_MAX_MESSAGES` (default: 10)
2. **Optimize LLM:**
   - Use faster model (gpt-4o-mini vs gpt-4)
   - Reduce `CONF_LLM_MAX_TOKENS`
   - Lower temperature for faster generation
3. **Enable caching** (when available)
4. Use local model for faster responses
5. Reduce tool execution timeout

### High Token Usage

**Symptoms:**
- Expensive API bills
- Token limit warnings
- Context window exceeded errors

**Solutions:**
1. **Reduce context size:**
   ```yaml
   CONF_HISTORY_MAX_TOKENS: 4000  # Lower from default
   CONF_COMPRESSION_LEVEL: "high"
   ```
2. Enable context optimization:
   ```yaml
   CONF_SUMMARIZATION_ENABLED: true
   ```
3. Use shorter system prompt
4. Reduce exposed entity count
5. Clear old conversation history

### Context Window Exceeded

**Symptoms:**
- Error: "Context size exceeds limit"
- TokenLimitExceeded exception

**Solutions:**
1. **Reduce context components:**
   - Lower history messages: `CONF_HISTORY_MAX_MESSAGES`
   - Reduce entity count
   - Compress or summarize history
2. Use model with larger context window
3. Enable context optimization
4. Clear conversation history more frequently
5. Avoid very long custom system prompts

---

## Configuration Issues

### Validation Errors

**Symptoms:**
- ValidationError during setup
- Configuration rejected by integration

**Solutions:**
1. **Check required fields:**
   - `CONF_LLM_BASE_URL` - Must be valid URL
   - `CONF_LLM_API_KEY` - Required for most providers
   - `CONF_LLM_MODEL` - Must match provider's models
2. Validate data types:
   - Temperature: float 0.0-2.0
   - Max tokens: integer > 0
   - Top P: float 0.0-1.0
3. Review error message for specific field
4. Use default values if unsure

### YAML Syntax Errors

**Symptoms:**
- Configuration not loading
- YAML parse errors in logs

**Solutions:**
1. **Validate YAML syntax:**
   - Use YAML validator online
   - Check indentation (use spaces, not tabs)
   - Ensure quotes around special characters
2. Common mistakes:
   ```yaml
   # WRONG
   custom_tools:
   - name: tool1
     description: Tool description

   # CORRECT
   custom_tools:
     - name: tool1
       description: "Tool description"
   ```
3. Restart Home Assistant after fixing
4. Check Configuration → Info → Check configuration

### Invalid Parameters

**Symptoms:**
- Parameter value rejected
- Out of range errors

**Solutions:**
1. **Review valid ranges:**
   - Temperature: 0.0 - 2.0
   - Top P: 0.0 - 1.0
   - Max tokens: 1 - model_limit
   - History max messages: > 0
2. Use recommended values from defaults
3. Check provider-specific limitations
4. Review const.py for valid options

---

## Debugging Techniques

### Enable Debug Logging

**Method 1: Configuration**
```yaml
CONF_DEBUG_LOGGING: true
```

**Method 2: logger configuration**
```yaml
# configuration.yaml
logger:
  default: info
  logs:
    custom_components.home_agent: debug
    custom_components.home_agent.agent: debug
    custom_components.home_agent.tool_handler: debug
    custom_components.home_agent.context_manager: debug
    custom_components.home_agent.memory_manager: debug
```

**What it shows:**
- LLM request/response details
- Tool execution parameters
- Context injection details
- Memory extraction process
- Token usage statistics

### Using Home Assistant Logs

**View logs:**
1. Settings → System → Logs
2. Look for `custom_components.home_agent` entries
3. Filter by error level

**Common log patterns:**
```
DEBUG: Calling LLM at https://api.openai.com/v1 with 3 messages
INFO: Tool 'ha_control' executed successfully in 45.2ms
ERROR: Tool execution failed: Entity not found
WARNING: Context size 8500 approaching limit 10000
```

### Event Monitoring

**Monitor events in real-time:**

**Developer Tools → Events:**
1. Listen to event type: `home_agent.*`
2. Watch for:
   - `home_agent.conversation.started`
   - `home_agent.tool.executed`
   - `home_agent.error`
   - `home_agent.memory.extracted`

**Automation for logging:**
```yaml
automation:
  - alias: "Log Home Agent Errors"
    trigger:
      - platform: event
        event_type: home_agent.error
    action:
      - service: system_log.write
        data:
          message: "Home Agent Error: {{ trigger.event.data }}"
          level: error
```

### Manual Tool Testing

**Test tools directly:**

```yaml
# Test ha_query
service: home_agent.execute_tool
data:
  tool_name: ha_query
  parameters:
    entity_id: light.living_room

# Test ha_control
service: home_agent.execute_tool
data:
  tool_name: ha_control
  parameters:
    action: turn_on
    entity_id: light.living_room
    parameters:
      brightness: 128

# Test memory search
service: home_agent.search_memories
data:
  query: "temperature preferences"
  limit: 10
```

### Metrics and Performance Tracking

**Monitor performance:**
1. Enable events: `CONF_EMIT_EVENTS: true`
2. Track `home_agent.conversation.finished` events
3. Review metrics:
   - `duration_ms` - Total conversation time
   - `tokens.total` - Token usage
   - `tool_calls` - Number of tools executed
   - `performance.llm_latency_ms` - LLM response time
   - `performance.tool_latency_ms` - Tool execution time

**Create dashboard:**
```yaml
# Track response times
sensor:
  - platform: template
    sensors:
      home_agent_avg_response_time:
        friendly_name: "Home Agent Avg Response Time"
        value_template: "{{ states.sensor.home_agent_last_conversation.attributes.duration_ms }}"
        unit_of_measurement: "ms"
```

### Testing Checklist

Before reporting an issue, verify:

- [ ] Configuration is valid and complete
- [ ] LLM endpoint is accessible
- [ ] API key is valid and not expired
- [ ] Entities are exposed to conversation
- [ ] Debug logging is enabled
- [ ] Home Assistant is up to date
- [ ] Integration is latest version
- [ ] No conflicting integrations
- [ ] System has adequate resources
- [ ] Network connectivity is stable

### Getting Help

If issues persist:

1. **Gather diagnostic info:**
   - Home Assistant version
   - Integration version
   - LLM provider and model
   - Full error logs with debug enabled
   - Configuration (redact API keys)
   - Steps to reproduce

2. **Where to report:**
   - GitHub Issues: For bugs and feature requests
   - Home Assistant Forums: For general help
   - Discord: For quick questions

3. **Include in report:**
   - Clear description of issue
   - Expected vs actual behavior
   - Relevant logs
   - Configuration details
   - Steps to reproduce

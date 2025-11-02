# Troubleshooting Quick Reference

Quick fixes for common Home Agent issues. For detailed diagnostics, see the [Complete Troubleshooting Guide](reference/TROUBLESHOOTING.md).

## Quick Fixes

| Issue | Solution |
|-------|----------|
| **"LLM connection failed"** | Check base URL and API key in configuration |
| **Slow responses** | Enable streaming, reduce entities, use faster model |
| **Tool not found** | Verify spelling, ensure entities are exposed to voice assistants |
| **Memory not working** | Enable in config, verify ChromaDB is running |
| **Context size exceeded** | Reduce history messages, use vector DB mode |

## LLM Connection Issues

### Authentication Error (401)
- Verify API key is correct and not expired
- For OpenAI: Key should start with `sk-`
- For Ollama: Usually no key needed

**Test connection:**
```bash
# OpenAI
curl https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_KEY"

# Ollama
curl http://localhost:11434/api/tags
```

### API Endpoint Unreachable
- Verify base URL matches provider:
  - OpenAI: `https://api.openai.com/v1`
  - Ollama: `http://localhost:11434/v1`
  - LocalAI: `http://localhost:8080/v1`
- Check service is running (for local models)
- Verify firewall settings

### Timeout Errors
- Increase timeout in configuration (`HTTP_TIMEOUT = 60`)
- Use faster model (gpt-4o-mini instead of gpt-4)
- Reduce max_tokens for faster generation
- Check system resources for local models

## Tool Execution Errors

### Tool Not Found
**Available built-in tools:** `ha_control`, `ha_query`, `query_external_llm`, `store_memory`, `recall_memory`

**Solutions:**
- Verify tool name spelling matches exactly
- Check custom tools in `configuration.yaml`
- Restart Home Assistant after adding tools

### Entity Not Accessible
**Most common issue:** Entities not exposed to voice assistants

**Fix:**
1. Go to Settings → Voice assistants → Expose
2. Select entities to expose
3. Or expose in individual entity settings

### Custom Tool Errors
**Common mistakes:**
- Invalid YAML syntax (check indentation)
- Missing quotes around URLs
- Invalid JSON schema in parameters
- Wrong handler type (use `rest` or `service`)

**Validate configuration:**
```
Settings → System → Configuration Validation
```

## Performance Issues

### Slow Responses
**Quick fixes:**
1. Enable streaming: `Streaming Enabled: true`
2. Use faster model: `gpt-4o-mini` or local Ollama
3. Reduce entities in context
4. Lower `Max Messages` in history (try 5 instead of 10)
5. Use vector DB mode for large setups

### High Token Usage
**Reduce costs:**
```yaml
Max Tokens: 150  # Instead of 500
Max Messages: 5  # Instead of 10
Context Mode: vector_db  # Only relevant entities
Memory Extraction: false  # If not needed
```

### Context Window Exceeded
**Solutions:**
- Reduce history: Lower `Max Messages`
- Use fewer entities in context
- Enable context optimization
- Use model with larger context window

## Memory System Issues

### Memory Not Extracting
**Check configuration:**
```yaml
Memory Enabled: true
Memory Extraction Enabled: true
Memory Extraction LLM: "local"  # or "external"
```

**If using external LLM:**
- Verify `External LLM Enabled: true`
- Check external LLM credentials

### ChromaDB Connection Errors
**Verify ChromaDB is running:**
```bash
curl http://localhost:8000/api/v1/heartbeat
```

**Check configuration:**
```yaml
Vector DB Host: localhost
Vector DB Port: 8000
```

### Memories Not Recalled
**Solutions:**
- Lower importance threshold: `Min Importance: 0.0`
- Test search manually:
  ```yaml
  service: home_agent.search_memories
  data:
    query: "temperature preferences"
    limit: 10
    min_importance: 0.0
  ```

## Getting Help

### Enable Debug Logging

**Option 1: Configuration**
```yaml
Debug Logging: true
```

**Option 2: logger configuration**
```yaml
# configuration.yaml
logger:
  logs:
    custom_components.home_agent: debug
```

**What it shows:**
- LLM request/response details
- Tool execution parameters
- Context injection details
- Memory extraction process
- Token usage statistics

### View Logs

**In Home Assistant UI:**
```
Settings → System → Logs
Filter: "home_agent"
```

**Log file location:**
```
/config/home-assistant.log
```

### Event Monitoring

**Monitor in Developer Tools → Events:**
Listen to: `home_agent.*`

**Key events:**
- `home_agent.error` - Errors
- `home_agent.tool.executed` - Tool results
- `home_agent.conversation.finished` - Performance metrics
- `home_agent.memory.extracted` - Memory events

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
```

### Testing Checklist

Before reporting an issue:

- [ ] Configuration is valid and complete
- [ ] LLM endpoint is accessible
- [ ] API key is valid and not expired
- [ ] Entities are exposed to conversation
- [ ] Debug logging is enabled
- [ ] Home Assistant is up to date
- [ ] Integration is latest version
- [ ] System has adequate resources

### Report Issues

**Gather this information:**
- Home Assistant version
- Integration version
- LLM provider and model
- Full error logs with debug enabled
- Configuration (redact API keys)
- Steps to reproduce

**Where to report:**
- GitHub Issues: Bug reports and feature requests
- Home Assistant Forums: General help
- Discord/Discussions: Quick questions

## Need More Details?

See the [Complete Troubleshooting Guide](reference/TROUBLESHOOTING.md) for comprehensive coverage including:
- Vector DB issues
- Streaming configuration
- Advanced debugging techniques
- Performance tracking
- Configuration validation

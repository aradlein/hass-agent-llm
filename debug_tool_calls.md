# Tool Call Debugging Guide

## Expected Tool Call Format

The integration expects OpenAI-compatible tool calling:

### Request (to LLM):
```json
{
  "model": "gpt-4",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "ha_control",
        "description": "Control Home Assistant devices...",
        "parameters": {
          "type": "object",
          "properties": {...},
          "required": [...]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Response (from LLM):
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "ha_control",
            "arguments": "{\"action\":\"turn_on\",\"entity_id\":\"light.living_room\"}"
          }
        }
      ]
    }
  }]
}
```

## Diagnostic Steps

### 1. Enable Debug Logging

In Home Agent options:
- Go to: Settings > Devices & Services > Home Agent > Configure
- Enable "Debug Logging"

### 2. Check Logs

Look for these log messages:

**Tools being registered:**
```
[home_agent.tool_handler] Registered tool: ha_control
[home_agent.tool_handler] Registered tool: ha_query
```

**Tools being sent to LLM:**
```
[home_agent.agent] Calling LLM at [REDACTED] with N messages and 2 tools
```

**Tool calls being detected:**
```
[home_agent.agent] LLM requested N tool calls
```

**Tool execution:**
```
[home_agent.tool_handler] Executing tool 'ha_control' with timeout 30s
[home_agent.tool_handler] Tool 'ha_control' executed successfully in XXms
```

### 3. Common Issues

#### Issue: LLM doesn't call tools
**Symptoms:** LLM responds with text instead of using tools
**Causes:**
- Model doesn't support function calling (check if your model supports it)
- Ollama: Need to enable tools support explicitly
- LocalAI: May need specific configuration

**Fix for Ollama:**
Some Ollama models need explicit prompting. Check that you're using a model that supports function calling like:
- `llama3.1:8b` or higher
- `mistral:7b-instruct` or higher

#### Issue: Tool parsing fails
**Symptoms:** Logs show `Tool execution failed: ...`
**Check:**
```python
# In agent.py line 665
tool_args = json.loads(tool_args_str)
```

This might fail if LLM returns invalid JSON in arguments.

**Debug:** Look for logs like:
```
[home_agent.agent] Tool execution failed: ...
```

#### Issue: Tool execution times out
**Symptoms:** `Tool execution timed out after 30s`
**Fix:** Increase timeout in options:
- Go to: Settings > Devices & Services > Home Agent > Configure
- Tool Configuration > Tool Timeout (increase from 30)

### 4. Test Tool Execution Directly

Use the `home_agent.execute_tool` service to test if tools work independently:

```yaml
service: home_agent.execute_tool
data:
  tool_name: "ha_control"
  parameters:
    action: "turn_on"
    entity_id: "light.living_room"
```

### 5. Check Tool Definitions

The tool definitions should match this format in logs (when debug enabled):

```
[home_agent.tool_handler] Added definition for tool: ha_control
[home_agent.tool_handler] Returning 2 tool definitions
```

## Code Path

1. `agent.py:595` - `_call_llm()` sends tools in payload
2. `agent.py:618` - Parse `tool_calls` from LLM response
3. `agent.py:659-665` - Extract tool name and arguments
4. `agent.py:669` - Execute via `tool_handler.execute_tool()`
5. `tool_handler.py:268` - Call `tool.execute(**parameters)`

## Verify Format Compatibility

Run this automation to see what your LLM returns:

```yaml
automation:
  - alias: "Debug Tool Calls"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: persistent_notification.create
        data:
          title: "Tool Call Debug"
          message: >
            Tool calls: {{ trigger.event.data.tool_calls }}
            Duration: {{ trigger.event.data.duration_ms }}ms
```

If `tool_calls` is always 0, the LLM isn't using tools.

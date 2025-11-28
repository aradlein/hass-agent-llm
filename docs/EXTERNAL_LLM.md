# External LLM Guide

This guide explains how to configure and use the External LLM feature in Home Agent, which enables a dual-LLM strategy where a fast primary LLM can delegate complex queries to a more powerful external LLM.

## Table of Contents

- [Overview](#overview)
- [Dual-LLM Strategy](#dual-llm-strategy)
- [Use Cases](#use-cases)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Cost Management](#cost-management)
- [Troubleshooting](#troubleshooting)

## Overview

The External LLM feature provides a powerful dual-LLM architecture that combines the efficiency of a fast local model with the capabilities of a powerful cloud model. The primary LLM handles tool execution and most conversations, but can delegate complex analysis or reasoning tasks to an external LLM when needed.

### Key Benefits

- **Cost Efficiency**: Use inexpensive local models for most tasks, only calling expensive models when necessary
- **Performance**: Fast response times for simple queries while still having access to advanced capabilities
- **Flexibility**: Choose different models for different strengths (speed vs. capability)
- **Transparency**: Primary LLM explicitly decides when to use external LLM, making the decision process clear

## Dual-LLM Strategy

### Architecture

```
User Query
    ↓
Primary LLM (e.g., llama2, gpt-4o-mini)
    ↓
    ├─→ Simple query → Direct response
    │
    ├─→ Tool execution → ha_control, ha_query, custom tools
    │
    └─→ Complex analysis → query_external_llm
                               ↓
                        External LLM (e.g., gpt-4o, claude-3-5-sonnet)
                               ↓
                        Detailed response
                               ↓
                        Primary LLM → Formatted response to user
```

### Design Philosophy

1. **Primary LLM**: Fast, efficient, handles tool calling and direct queries
2. **External LLM**: Powerful, used only when needed for complex reasoning
3. **Explicit Delegation**: Primary LLM chooses when to use external LLM
4. **Context Isolation**: External LLM only receives what primary LLM sends (not full conversation history)

## Use Cases

### When to Use External LLM

The external LLM is ideal for:

#### 1. Complex Analysis

**Example**: Analyzing energy usage patterns

```
User: "Analyze my energy consumption over the past week and suggest optimizations"

Primary LLM:
  1. Calls ha_query to get energy data
  2. Calls query_external_llm with data and analysis request
  3. External LLM performs detailed analysis
  4. Primary LLM formats and presents recommendations
```

#### 2. Detailed Explanations

**Example**: Understanding home automation concepts

```
User: "Explain how I can optimize my heating schedule for maximum efficiency"

Primary LLM:
  1. Recognizes this needs detailed explanation
  2. Calls query_external_llm for comprehensive explanation
  3. External LLM provides detailed breakdown with examples
  4. Primary LLM presents to user
```

#### 3. Multi-Step Reasoning

**Example**: Planning automations

```
User: "Help me create an automation that adjusts lights based on time of day and occupancy"

Primary LLM:
  1. Calls query_external_llm for automation design advice
  2. External LLM suggests logic and structure
  3. Primary LLM uses response to guide user through implementation
```

#### 4. Creative Tasks

**Example**: Naming and organizing

```
User: "Suggest creative names for my smart home scenes and explain when to use each"

Primary LLM:
  1. Calls query_external_llm for creative suggestions
  2. External LLM generates creative, descriptive names
  3. Primary LLM presents organized list
```

### When NOT to Use External LLM

The primary LLM should handle directly:

- **Simple queries**: "Turn on the lights", "What's the temperature?"
- **Tool execution**: Controlling devices, querying sensors
- **Quick lookups**: Current states, recent history
- **Direct commands**: Clear, single-step actions

## Configuration

### Enabling External LLM

Configure via Home Assistant UI or `configuration.yaml`:

#### UI Configuration

1. Go to **Settings** > **Devices & Services** > **Home Agent**
2. Click **Configure**
3. Select **External LLM** from the menu
4. Fill in the configuration:
   - **Enable External LLM**: Toggle on
   - **Base URL**: External LLM API endpoint
   - **API Key**: Your API key
   - **Model**: Model name
   - **Tool Description**: When primary LLM should use it (optional)
   - **Temperature**: Creativity level (optional, default: 0.8)
   - **Max Tokens**: Maximum response length (optional, default: 1000)

#### YAML Configuration (Alternative)

```yaml
home_agent:
  external_llm_enabled: true
  external_llm_base_url: "https://api.openai.com/v1"
  external_llm_api_key: !secret openai_api_key
  external_llm_model: "gpt-4o"
  external_llm_temperature: 0.8
  external_llm_max_tokens: 1000
  external_llm_tool_description: |
    Use this tool for complex analysis, detailed explanations, or tasks
    requiring advanced reasoning. Provide the specific question as the
    prompt and include relevant context data.
```

**secrets.yaml:**
```yaml
openai_api_key: sk-your-api-key-here
```

### Configuration Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `external_llm_enabled` | Yes | `false` | Enable the external LLM tool |
| `external_llm_base_url` | Yes | - | API endpoint (OpenAI-compatible) |
| `external_llm_api_key` | Yes | - | Authentication key |
| `external_llm_model` | No | `gpt-4o` | Model name |
| `external_llm_temperature` | No | `0.8` | Creativity (0.0-2.0) |
| `external_llm_max_tokens` | No | `1000` | Max response length |
| `external_llm_tool_description` | No | Built-in | When to use the tool |
| `tools_timeout` | No | `30` | Timeout in seconds |

### Choosing an External LLM

Popular options:

#### OpenAI GPT-4

```yaml
external_llm_base_url: "https://api.openai.com/v1"
external_llm_model: "gpt-4o"
```

**Pros**: Excellent reasoning, good at analysis, reliable
**Cons**: Higher cost, requires internet

#### Anthropic Claude

```yaml
external_llm_base_url: "https://api.anthropic.com/v1"  # Via OpenAI-compatible proxy
external_llm_model: "claude-3-5-sonnet-20241022"
```

**Pros**: Excellent at analysis and explanation, large context window
**Cons**: Requires compatible proxy, higher cost

#### OpenRouter (Multiple Models)

```yaml
external_llm_base_url: "https://openrouter.ai/api/v1"
external_llm_model: "anthropic/claude-3.5-sonnet"  # or other models
```

**Pros**: Access to many models, flexible pricing
**Cons**: Requires OpenRouter account

#### Self-Hosted Models

```yaml
external_llm_base_url: "http://localhost:11434/v1"  # Ollama
external_llm_model: "llama2:70b"  # Larger model than primary
```

**Pros**: Free, private, no internet required
**Cons**: Requires powerful hardware, may be slower

## How It Works

### The `query_external_llm` Tool

When you enable External LLM, the primary LLM gains access to a new tool:

```json
{
  "name": "query_external_llm",
  "description": "Use for complex analysis, detailed explanations, or advanced reasoning",
  "parameters": {
    "prompt": "The question or task for the external LLM (required)",
    "context": "Additional context data (optional)"
  }
}
```

### Conversation Flow

1. **User sends query** to Home Agent
2. **Primary LLM processes** the query
3. **Primary LLM decides** if external LLM is needed
4. If needed:
   - Primary LLM constructs `prompt` parameter
   - Primary LLM includes relevant `context` (entity data, tool results, etc.)
   - Primary LLM calls `query_external_llm`
5. **External LLM receives**:
   - The prompt (question/task)
   - The context (if provided)
   - **NOT** the full conversation history (for efficiency)
6. **External LLM responds** with analysis/explanation
7. **Primary LLM formats** the response for the user

### Context Handling

**Important**: The external LLM does NOT automatically receive:
- Full conversation history
- Previous tool calls
- Original user query

The primary LLM must **explicitly include** all necessary information in the `prompt` and `context` parameters.

#### Example Context Passing

```python
# What primary LLM sends to external LLM
{
  "prompt": "Analyze this energy usage data and suggest optimizations",
  "context": {
    "energy_data": {
      "sensor.energy_usage": [
        {"time": "2024-01-01T00:00:00", "value": 150},
        {"time": "2024-01-01T01:00:00", "value": 160},
        ...
      ]
    },
    "current_rate": "$0.12 per kWh",
    "peak_hours": "4pm-9pm"
  }
}
```

## Best Practices

### 1. Customize the Tool Description

Help the primary LLM understand when to use the external LLM:

```yaml
external_llm_tool_description: |
  Delegate to this tool when:
  - User asks for detailed analysis or recommendations
  - Task requires multi-step reasoning or planning
  - Creative suggestions or explanations are needed
  - Complex data interpretation is required

  Do NOT use for:
  - Simple device control
  - Quick status queries
  - Direct tool execution
```

### 2. Choose Complementary Models

Select models with different strengths:

**Primary LLM**: Fast, efficient, good at tool calling
- `gpt-4o-mini`
- `llama2:13b`
- `mistral`

**External LLM**: Powerful, good at reasoning
- `gpt-4o`
- `claude-3-5-sonnet`
- `llama2:70b`

### 3. Optimize Prompts

The primary LLM should provide clear, focused prompts:

❌ **Too vague**:
```json
{
  "prompt": "Help with energy"
}
```

✅ **Clear and specific**:
```json
{
  "prompt": "Analyze the energy consumption data and identify: 1) Peak usage times, 2) Unusual spikes, 3) Optimization opportunities",
  "context": { ...energy data... }
}
```

### 4. Monitor Costs

Track external LLM usage to manage costs:

```yaml
automation:
  - alias: "Log External LLM Usage"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
        event_data:
          tool_name: query_external_llm
    action:
      - service: counter.increment
        target:
          entity_id: counter.external_llm_calls
      - service: notify.admin
        data:
          message: "External LLM called: {{ trigger.event.data.parameters.prompt[:50] }}..."
```

### 5. Set Appropriate Limits

Configure tool call limits to prevent excessive costs:

```yaml
home_agent:
  tools_max_calls_per_turn: 3  # Limit tool calls per conversation turn
  tools_timeout: 30  # Timeout for external LLM calls
```

### 6. Test with Different Temperatures

Adjust temperature based on task:

- **Analysis (0.3-0.5)**: More focused, deterministic
- **Explanations (0.7-0.8)**: Balanced creativity and accuracy
- **Creative tasks (0.9-1.2)**: More creative and varied

## Examples

### Example 1: Energy Analysis

**User**: "Analyze my energy usage this week and suggest ways to save money"

**Primary LLM**:
1. Calls `ha_query` to get energy sensor data
2. Receives data for past week
3. Calls `query_external_llm`:
   ```json
   {
     "prompt": "Analyze this energy consumption data and provide specific recommendations to reduce costs",
     "context": {
       "energy_data": [...],
       "electricity_rate": "$0.12/kWh",
       "peak_rate_hours": "4pm-9pm at $0.18/kWh"
     }
   }
   ```
4. External LLM responds with detailed analysis
5. Primary LLM formats response: "Based on your energy usage, here are three ways to save money: ..."

### Example 2: Automation Design

**User**: "Help me create a smart lighting automation for my home office"

**Primary LLM**:
1. Calls `query_external_llm`:
   ```json
   {
     "prompt": "Design a smart lighting automation for a home office that considers: time of day, occupancy, natural light, and work/focus modes. Provide specific logic and triggers.",
     "context": {
       "available_lights": ["light.desk", "light.overhead", "light.ambient"],
       "sensors": ["binary_sensor.office_occupied", "sensor.office_light_level"],
       "user_preferences": "Prefers warm lighting for video calls, bright light for detailed work"
     }
   }
   ```
2. External LLM provides detailed automation design
3. Primary LLM guides user through implementation

### Example 3: Climate Optimization

**User**: "Optimize my heating schedule for maximum comfort and efficiency"

**Primary LLM**:
1. Calls `ha_query` to get climate history and preferences
2. Calls `query_external_llm`:
   ```json
   {
     "prompt": "Create an optimized heating schedule that balances comfort and efficiency, considering occupancy patterns and outdoor temperature variations",
     "context": {
       "current_schedule": [...],
       "occupancy_pattern": "Home 6pm-8am weekdays, all day weekends",
       "temperature_preferences": "68°F when home, 62°F when away",
       "historical_usage": [...],
       "outdoor_temps": [...]
     }
   }
   ```
3. External LLM creates optimized schedule
4. Primary LLM presents schedule and helps implement it

## Cost Management

### Estimating Costs

External LLM costs depend on:
- Model pricing (per 1K tokens)
- Frequency of use
- Response length (`max_tokens`)
- Context size

**Example calculation (GPT-4o)**:
- Input: $2.50 per 1M tokens
- Output: $10.00 per 1M tokens
- Average query: ~500 input tokens, ~300 output tokens
- Cost per query: ~$0.004

### Cost Optimization Tips

1. **Use external LLM selectively**
   - Let primary LLM handle simple tasks
   - Only delegate truly complex queries

2. **Limit max_tokens**
   ```yaml
   external_llm_max_tokens: 500  # Shorter, more focused responses
   ```

3. **Reduce context size**
   - Only include essential data in `context`
   - Summarize large datasets before passing

4. **Choose cost-effective models**
   - OpenAI GPT-4o-mini instead of GPT-4o
   - Self-hosted models for no per-query cost

5. **Set usage limits**
   ```yaml
   tools_max_calls_per_turn: 2  # Limit tools per conversation
   ```

6. **Monitor and alert**
   ```yaml
   automation:
     - alias: "External LLM Usage Alert"
       trigger:
         - platform: numeric_state
           entity_id: counter.external_llm_calls_today
           above: 50
       action:
         - service: notify.admin
           data:
             message: "High external LLM usage today: {{ states('counter.external_llm_calls_today') }} calls"
   ```

## Troubleshooting

### External LLM Tool Not Available

**Symptoms**: Primary LLM doesn't call `query_external_llm`

**Solutions**:
1. Verify `external_llm_enabled: true` in configuration
2. Restart Home Assistant after configuration changes
3. Check logs for registration errors
4. Verify tool appears in tool list (Developer Tools > Services)

### Authentication Errors

**Symptoms**: "External LLM authentication failed" errors

**Solutions**:
1. Verify API key is correct and not expired
2. Check API key has sufficient permissions/credits
3. Ensure API key is properly set in secrets.yaml
4. Test API key directly with curl:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer YOUR_API_KEY"
   ```

### Timeout Errors

**Symptoms**: "External LLM request timed out" errors

**Solutions**:
1. Increase timeout:
   ```yaml
   tools_timeout: 60  # Increase to 60 seconds
   ```
2. Reduce `max_tokens` for faster responses
3. Check network connectivity
4. Verify external LLM service is operational

### High Costs

**Symptoms**: Unexpected API charges

**Solutions**:
1. Check usage metrics in API provider dashboard
2. Review logs for excessive calls
3. Reduce `max_tokens`
4. Adjust tool description to be more selective
5. Implement usage limits and monitoring

### Poor Response Quality

**Symptoms**: External LLM responses are not helpful

**Solutions**:
1. **Improve prompts**: Primary LLM may not be providing enough context
2. **Adjust temperature**: Lower for more focused, higher for creative
3. **Increase max_tokens**: May be cutting off responses
4. **Try different model**: Some models excel at different tasks
5. **Customize tool description**: Help primary LLM understand when/how to use it

### Context Not Being Passed

**Symptoms**: External LLM seems unaware of relevant information

**Remember**: External LLM only receives what's in `prompt` and `context` parameters

**Solutions**:
1. Check primary LLM is including context in tool call
2. Verify context data is being retrieved correctly
3. Review tool execution logs to see what's being sent
4. Adjust primary LLM's system prompt to encourage context passing

### Debug Logging

Enable detailed logging for troubleshooting:

```yaml
logger:
  logs:
    custom_components.home_agent: debug
    custom_components.home_agent.tools.external_llm: debug
```

Check logs for:
- Tool registration
- API calls and responses
- Error messages
- Token usage
- Response times

## Further Reading

- [Custom Tools Guide](CUSTOM_TOOLS.md) - Learn about custom tool creation
- [Project Specification](PROJECT_SPEC.md) - Technical details and architecture
- [Home Agent README](../README.md) - General setup and usage
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference/)

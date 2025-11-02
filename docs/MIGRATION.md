# Migration Guide

Quick guide to migrating from extended_openai_conversation to Home Agent.

## Should You Migrate?

| Feature | extended_openai_conversation | Home Agent |
|---------|------------------------------|------------|
| **OpenAI/Local LLMs** | ✅ Yes | ✅ Yes |
| **Custom Functions** | ✅ REST, Service, Script | ✅ REST, Service (no script) |
| **Multi-LLM Support** | ❌ No | ✅ Yes (Primary + External) |
| **Memory System** | ❌ No | ✅ Yes (automatic extraction) |
| **Streaming Responses** | ❌ No | ✅ Yes (low-latency TTS) |
| **Configuration** | UI + YAML | UI (LLM) + YAML (tools) |

**Key Benefit:** Enhanced capabilities (memory, dual-LLM, streaming) while maintaining compatibility with existing workflows.

## Quick Migration (5 Steps)

### Step 1: Install Home Agent

**Via HACS (when available):**
1. Open HACS > Search "Home Agent" > Install
2. Restart Home Assistant

**Manual:**
1. Download latest release from GitHub
2. Copy `custom_components/home_agent` to config folder
3. Restart Home Assistant

### Step 2: Configure LLM Settings

1. Go to **Settings > Devices & Services > Add Integration > Home Agent**
2. Enter configuration:
   - **LLM Base URL**: Your endpoint (OpenAI: `https://api.openai.com/v1`, Ollama: `http://localhost:11434/v1`)
   - **API Key**: Your API key (if required)
   - **Model**: Model name (e.g., `gpt-4o-mini`, `llama2`)
   - **Temperature**: 0.7
   - **Max Tokens**: 500

### Step 3: Convert Functions to Tools

Edit `configuration.yaml` and convert your functions using the table below.

### Step 4: Configure Context

1. Go to **Settings > Home Agent > Configure > Context Settings**
2. Enter same entities from your old integration
3. Use wildcards for groups: `light.*` instead of listing each light

### Step 5: Test and Remove Old Integration

1. Test basic queries: "What's the temperature?"
2. Test device control: "Turn on the living room lights"
3. Test custom tools: "What's the weather like?"
4. Only after everything works: Remove extended_openai_conversation

## Function to Tool Conversion

### REST Function Example

**Before (extended_openai_conversation):**
```yaml
functions:
  - spec:
      name: get_weather
      description: "Get weather forecast"
      parameters:
        type: object
        properties:
          location:
            type: string
      function:
        type: rest
        url: "https://api.weather.com/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ api_key }}"
```

**After (Home Agent):**
```yaml
# configuration.yaml
home_agent:
  tools_custom:
    - name: get_weather
      description: "Get weather forecast"
      parameters:
        type: object
        properties:
          location:
            type: string
      handler:
        type: rest
        url: "https://api.weather.com/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ secrets.weather_api_key }}"
```

**Key Changes:**
- `functions` → `tools_custom`
- `spec.function` → `handler`
- Use `{{ secrets.key_name }}` for API keys in `secrets.yaml`

### Service Function Example

**Before:**
```yaml
functions:
  - spec:
      name: trigger_routine
      function:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine
```

**After:**
```yaml
home_agent:
  tools_custom:
    - name: trigger_routine
      description: "Trigger morning routine automation"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine
```

**Key Changes:**
- Add explicit `description` field
- `function.type` → `handler.type`

### Script Functions (Not Supported)

**Before:**
```yaml
functions:
  - spec:
      name: custom_sequence
      function:
        type: script
        sequence:
          - service: light.turn_on
            target:
              entity_id: light.living_room
          - delay: 00:00:05
          - service: light.turn_off
            target:
              entity_id: light.living_room
```

**After - Alternative 1 (Create HA Script):**

Create script in `scripts.yaml`:
```yaml
custom_sequence:
  alias: "Custom Sequence"
  sequence:
    - service: light.turn_on
      target:
        entity_id: light.living_room
    - delay: 00:00:05
    - service: light.turn_off
      target:
        entity_id: light.living_room
```

Reference in `configuration.yaml`:
```yaml
home_agent:
  tools_custom:
    - name: run_custom_sequence
      description: "Run custom light sequence"
      handler:
        type: service
        service: script.custom_sequence
```

**After - Alternative 2 (Use Automation):**

Create automation in `automations.yaml`:
```yaml
- id: custom_sequence
  alias: "Custom Sequence"
  trigger:
    - platform: event
      event_type: custom_sequence_trigger
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room
    - delay: 00:00:05
    - service: light.turn_off
      target:
        entity_id: light.living_room
```

Reference in `configuration.yaml`:
```yaml
home_agent:
  tools_custom:
    - name: trigger_custom_sequence
      description: "Trigger custom light sequence"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.custom_sequence
```

## Testing Checklist

After migration, verify:

### Basic Functionality
- [ ] Integration loads without errors
- [ ] Simple queries return responses
- [ ] Device control works (lights, switches)
- [ ] Entity status queries work
- [ ] Conversation history persists

### Custom Tools
- [ ] All REST tools execute successfully
- [ ] All service tools execute successfully
- [ ] Tools return expected results
- [ ] API keys work correctly (check secrets.yaml)

### Integration Points
- [ ] Voice assistant integration works
- [ ] Automations trigger correctly
- [ ] Events fire as expected

### Optional Features
- [ ] External LLM delegation works (if configured)
- [ ] Memory extraction and recall works (if enabled)
- [ ] Streaming responses work (if enabled)

### Before Removing Old Integration
- [ ] All custom tools work correctly
- [ ] Voice assistant integration works
- [ ] You have a backup of your configuration

## Common Issues

### Issue 1: Custom Tools Not Appearing

**Solutions:**
1. Check YAML syntax: Settings > System > Configuration Validation
2. Verify configuration is at root of `configuration.yaml`:
   ```yaml
   home_agent:  # Correct location at root
     tools_custom:
       - name: my_tool
   ```
3. Check logs for errors:
   ```yaml
   logger:
     logs:
       custom_components.home_agent: debug
   ```
4. Verify secrets are defined in `secrets.yaml`

### Issue 2: Template Rendering Errors

**Solutions:**
1. Check template syntax:
   ```yaml
   # Correct
   url: "https://api.example.com/{{ location }}"

   # Incorrect
   url: "https://api.example.com/{ location }"
   ```
2. Verify parameter names match between definition and usage
3. Test templates in Developer Tools > Template

### Issue 3: Service Not Found

**Solutions:**
1. Verify service exists in Developer Tools > Services
2. Check service name format:
   ```yaml
   # Correct
   service: automation.trigger

   # Incorrect
   service: trigger automation
   service: automation_trigger
   ```
3. Ensure required integration is installed and loaded

### Issue 4: REST API Authentication Failures

**Solutions:**
1. Verify API key in `secrets.yaml` is correct
2. Check header format:
   ```yaml
   # Bearer token
   headers:
     Authorization: "Bearer {{ secrets.api_key }}"

   # API key header
   headers:
     X-API-Key: "{{ secrets.api_key }}"
   ```
3. Test API directly with curl to verify credentials

### Issue 5: External LLM Not Being Called

**Solutions:**
1. Verify external LLM is enabled in Settings
2. Make tool description clear about when to use:
   ```yaml
   Tool Description: |
     Use for complex analysis, detailed explanations, and recommendations.
     Examples: energy analysis, troubleshooting, planning automations.
   ```
3. Test explicitly:
   ```yaml
   service: home_agent.execute_tool
   data:
     tool_name: query_external_llm
     parameters:
       prompt: "Test query"
   ```

## Need More Details?

See the [Complete Reference](reference/MIGRATION.md) for comprehensive migration information and advanced troubleshooting.

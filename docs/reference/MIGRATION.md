# Migration Guide

This guide helps you migrate to Home Agent from other conversational AI integrations or upgrade between Home Agent versions.

## Table of Contents

- [Introduction](#introduction)
- [Migrating from extended_openai_conversation](#migrating-from-extended_openai_conversation)
- [Version Migrations](#version-migrations)
- [Common Migration Issues](#common-migration-issues)

---

## Introduction

### Who Should Use This Guide

This migration guide is for users who are:

- **Migrating from extended_openai_conversation** to Home Agent
- **Upgrading between major versions** of Home Agent
- **Evaluating Home Agent** as a replacement for their current setup

### What to Expect

Migration to Home Agent typically involves:

1. Installing the Home Agent integration
2. Configuring your LLM connection settings
3. Converting existing function/tool definitions
4. Testing functionality to ensure feature parity
5. Removing the old integration

Most migrations can be completed in **30-60 minutes** depending on complexity.

---

## Migrating from extended_openai_conversation

### Feature Comparison Table

| Feature | extended_openai_conversation | Home Agent | Notes |
|---------|------------------------------|------------|-------|
| **OpenAI API** | ✅ Yes | ✅ Yes | Full compatibility |
| **Local LLMs (Ollama)** | ✅ Yes | ✅ Yes | Both support OpenAI-compatible endpoints |
| **Custom Functions** | ✅ REST, Service, Script | ✅ REST, Service | Script execution not supported for security |
| **Conversation History** | ✅ Yes | ✅ Yes (Enhanced) | Persistent across restarts in Home Agent |
| **Entity Context** | ✅ Direct injection | ✅ Direct + Vector DB | Home Agent adds semantic search option |
| **Tool Calling** | ✅ Function calling | ✅ Native tools + Custom | Home Agent uses HA's native LLM API |
| **Multi-LLM Support** | ❌ No | ✅ Yes | Dual-LLM strategy (Primary + External) |
| **Memory System** | ❌ No | ✅ Yes | Automatic extraction and recall |
| **Streaming Responses** | ❌ No | ✅ Yes | Low-latency TTS integration |
| **Vector Database** | ❌ No | ✅ Yes (ChromaDB) | Semantic entity search |
| **Context Optimization** | ⚠️ Basic | ✅ Advanced | Smart compression and prioritization |
| **Event System** | ⚠️ Limited | ✅ Comprehensive | Detailed metrics and monitoring |

### Breaking Changes

#### 1. Script Functions Not Supported

**extended_openai_conversation:**
```yaml
functions:
  - spec:
      name: run_script
      function:
        type: script
        sequence:
          - service: light.turn_on
          - delay: 5
```

**Home Agent:**
Script execution is **not supported** for security reasons. Use service tools or create a Home Assistant script entity instead.

**Alternative:**
```yaml
# Create a script in scripts.yaml
my_custom_script:
  sequence:
    - service: light.turn_on
    - delay: 5

# Reference it in Home Agent
home_agent:
  tools_custom:
    - name: run_my_script
      description: "Execute my custom script"
      handler:
        type: service
        service: script.my_custom_script
```

#### 2. Function Definition Format

**extended_openai_conversation:**
```yaml
functions:
  - spec:
      name: check_weather
      description: "Get weather"
      parameters:
        # JSON schema
      function:
        type: rest
        url: "..."
```

**Home Agent:**
```yaml
home_agent:
  tools_custom:
    - name: check_weather
      description: "Get weather"
      parameters:
        # JSON schema (same format)
      handler:
        type: rest
        url: "..."
```

**Key Changes:**
- `functions` → `tools_custom`
- `spec.function` → `handler`
- Configuration in `configuration.yaml` instead of integration UI

#### 3. Configuration Location

**extended_openai_conversation:**
- Most configuration via Home Assistant UI
- Functions sometimes in `configuration.yaml`

**Home Agent:**
- LLM settings via Home Assistant UI
- Custom tools **must** be in `configuration.yaml`
- More consistent configuration approach

### Configuration Differences

#### LLM Connection Settings

**extended_openai_conversation:**
```yaml
# Settings in UI
API Key: sk-...
Model: gpt-4
Max Tokens: 500
```

**Home Agent:**
```yaml
# Settings in UI
LLM Base URL: https://api.openai.com/v1
API Key: sk-...
Model: gpt-4o-mini
Temperature: 0.7
Max Tokens: 500
Top P: 1.0
```

**Migration Steps:**
1. Note your current API key and model
2. Add integration: Settings > Devices & Services > Add Integration > Home Agent
3. Enter same credentials
4. Add `/v1` to base URL if using OpenAI

#### Context Injection

**extended_openai_conversation:**
```yaml
# UI configuration - select entities
Exposed Entities:
  - light.living_room
  - sensor.temperature
```

**Home Agent:**
```yaml
# UI: Settings > Home Agent > Configure > Context Settings
Context Mode: direct
Entities to Include: light.living_room, sensor.temperature
```

**Migration Steps:**
1. List all exposed entities from old integration
2. Configure same entities in Home Agent context settings
3. Consider using wildcards: `light.*` instead of individual lights

### Function/Tool Conversion

#### REST Function → REST Custom Tool

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
- `function` → `handler`
- Use `{{ secrets.key_name }}` for API keys
- Place in `configuration.yaml` under `home_agent:`

#### Service Function → Service Custom Tool

**Before (extended_openai_conversation):**
```yaml
functions:
  - spec:
      name: trigger_automation
      function:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine
```

**After (Home Agent):**
```yaml
home_agent:
  tools_custom:
    - name: trigger_automation
      description: "Trigger morning routine"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine
```

**Key Changes:**
- Add explicit `description` field (helps LLM understand when to use)
- `function.type` → `handler.type`
- Same service call format

#### Script Functions (Not Supported)

**Before (extended_openai_conversation):**
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

**After (Home Agent) - Alternative 1: Create Script Entity:**
```yaml
# scripts.yaml
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

# configuration.yaml
home_agent:
  tools_custom:
    - name: run_custom_sequence
      description: "Run custom light sequence"
      handler:
        type: service
        service: script.custom_sequence
```

**After (Home Agent) - Alternative 2: Use Automation:**
```yaml
# automations.yaml
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

# configuration.yaml
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

### Step-by-Step Migration

#### Step 1: Install Home Agent

**Option A: HACS (Recommended - when available)**
1. Open HACS in Home Assistant
2. Search for "Home Agent"
3. Click Install
4. Restart Home Assistant

**Option B: Manual Installation**
1. Download the latest release from GitHub
2. Copy `custom_components/home_agent` to your HA config folder
3. Restart Home Assistant

#### Step 2: Configure LLM Settings

1. Go to **Settings > Devices & Services**
2. Click **Add Integration**
3. Search for "Home Agent"
4. Enter your LLM configuration:
   - **Name**: Home Agent
   - **LLM Base URL**: Your OpenAI-compatible endpoint
     - OpenAI: `https://api.openai.com/v1`
     - Ollama: `http://localhost:11434/v1`
   - **API Key**: Your API key (if required)
   - **Model**: Model name (e.g., `gpt-4o-mini`, `llama2`)
   - **Temperature**: 0.7 (or your preference)
   - **Max Tokens**: 500 (or your preference)

#### Step 3: Convert Function Definitions

1. **Document existing functions**:
   - List all functions from extended_openai_conversation
   - Note their types (REST, service, script)
   - Note any parameters and configurations

2. **Create custom tools configuration**:
   - Edit `configuration.yaml`
   - Add `home_agent:` section
   - Convert each function following the patterns above

3. **Add secrets** (if using API keys):
   - Edit `secrets.yaml`
   - Add all API keys referenced in tools
   ```yaml
   # secrets.yaml
   weather_api_key: your_key_here
   todoist_api_key: another_key_here
   ```

4. **Validate YAML**:
   - Check YAML syntax
   - Restart Home Assistant
   - Check logs for any errors

**Example Full Conversion:**

**Before (extended_openai_conversation functions):**
```yaml
# Documented from UI
Function 1: get_weather (REST)
Function 2: trigger_routine (Service)
Function 3: custom_script (Script - 3 steps)
```

**After (configuration.yaml):**
```yaml
home_agent:
  tools_custom:
    # Function 1: REST tool
    - name: get_weather
      description: "Get weather forecast for location"
      parameters:
        type: object
        properties:
          location:
            type: string
      handler:
        type: rest
        url: "https://api.weather.com/v1/forecast"
        method: GET
        headers:
          Authorization: "Bearer {{ secrets.weather_api_key }}"
        query_params:
          location: "{{ location }}"

    # Function 2: Service tool
    - name: trigger_routine
      description: "Trigger morning routine automation"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine

    # Function 3: Script → Convert to HA script entity first
    - name: run_custom_script
      description: "Run custom light script"
      handler:
        type: service
        service: script.custom_light_sequence
```

**scripts.yaml (for converted script):**
```yaml
custom_light_sequence:
  alias: "Custom Light Sequence"
  sequence:
    - service: light.turn_on
      target:
        entity_id: light.living_room
    - delay: 00:00:05
    - service: light.turn_off
      target:
        entity_id: light.living_room
```

#### Step 4: Test Functionality

1. **Test basic queries**:
   ```yaml
   service: home_agent.process
   data:
     text: "What's the temperature?"
   ```

2. **Test device control**:
   ```yaml
   service: home_agent.process
   data:
     text: "Turn on the living room lights"
   ```

3. **Test custom tools**:
   ```yaml
   service: home_agent.process
   data:
     text: "What's the weather like?"
   ```

4. **Test via voice** (if using voice assistant):
   - Set up Voice Assistant pipeline with Home Agent
   - Test voice commands

5. **Check logs**:
   ```yaml
   # Enable debug logging
   logger:
     logs:
       custom_components.home_agent: debug
   ```

6. **Verify events**:
   - Go to Developer Tools > Events
   - Listen to `home_agent.*` events
   - Trigger conversations and verify events fire

#### Step 5: Remove Old Integration

**Only after confirming everything works!**

1. Go to **Settings > Devices & Services**
2. Find **Extended OpenAI Conversation**
3. Click the three dots > **Delete**
4. Confirm deletion
5. Restart Home Assistant

**Before removing:**
- [ ] All custom tools work correctly
- [ ] Voice assistant integration works
- [ ] Automations using the old integration are updated
- [ ] You have a backup of your configuration

### Testing Checklist

Use this checklist to verify migration success:

#### Basic Functionality
- [ ] Integration loads without errors
- [ ] LLM connection works (check logs)
- [ ] Simple queries return responses
- [ ] Device control works (turn on/off lights)
- [ ] Entity status queries work

#### Custom Tools
- [ ] All REST tools execute successfully
- [ ] All service tools execute successfully
- [ ] Tools return expected results
- [ ] Error handling works (test with invalid parameters)

#### Context and History
- [ ] Entity context is injected correctly
- [ ] Conversation history persists across messages
- [ ] History persists across HA restarts (if enabled)

#### Advanced Features
- [ ] External LLM delegation works (if configured)
- [ ] Memory extraction and recall works (if enabled)
- [ ] Streaming responses work (if enabled)
- [ ] Vector DB integration works (if configured)

#### Integration Points
- [ ] Voice assistant integration works
- [ ] Automations trigger correctly
- [ ] Events fire as expected
- [ ] Services respond correctly

#### Performance
- [ ] Response times are acceptable
- [ ] No memory leaks over extended use
- [ ] Token usage is reasonable
- [ ] API costs are within expectations

---

## Version Migrations

### Upgrading Between Home Agent Versions

#### From 0.1.0 (Phase 1) → 0.2.0 (Phase 2)

**New Features:**
- Vector DB (ChromaDB) support
- Conversation history persistence
- Context optimization
- Enhanced event system

**Breaking Changes:**
- None

**Migration Steps:**
1. Update integration via HACS or manual replacement
2. Restart Home Assistant
3. (Optional) Configure Vector DB in integration options
4. (Optional) Enable history persistence

**Configuration Changes:**
```yaml
# New optional configuration
# Settings > Home Agent > Configure

# Context Settings (new option)
Context Mode: vector_db  # NEW
Vector DB Host: localhost
Vector DB Port: 8000
Vector DB Collection: home_entities

# Conversation History (enhanced)
History Persist: true  # NEW - save across restarts
```

#### From 0.2.0 (Phase 2) → 0.3.0 (Phase 3)

**New Features:**
- Custom tools framework (REST + Service handlers)
- External LLM tool for dual-LLM strategy
- Standardized tool response format

**Breaking Changes:**
- Custom tools must be defined in `configuration.yaml` (not UI)

**Migration Steps:**
1. Update integration
2. Add custom tools to `configuration.yaml`
3. Configure external LLM (optional)
4. Restart Home Assistant

**New Configuration Required:**
```yaml
# configuration.yaml - NEW SECTION
home_agent:
  tools_custom:
    - name: my_tool
      description: "..."
      handler:
        type: rest  # or service
        # ... configuration
```

**External LLM Configuration:**
```yaml
# Settings > Home Agent > Configure > External LLM
External LLM Enabled: true
Base URL: https://api.openai.com/v1
API Key: sk-...
Model: gpt-4o
```

#### From 0.3.0 (Phase 3) → 0.4.0 (Phase 3.5)

**New Features:**
- Long-term memory system
- Automatic memory extraction
- Manual memory tools (store_memory, recall_memory)
- Memory management services

**Breaking Changes:**
- None

**Migration Steps:**
1. Update integration
2. Configure memory settings
3. Restart Home Assistant
4. (Optional) Migrate to ChromaDB if not already using it

**New Configuration:**
```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: true
Automatic Extraction: true
Extraction LLM: external  # or "local"
Max Memories: 100
Min Importance: 0.3
```

#### From 0.4.0 (Phase 3.5) → 0.5.0 (Phase 4)

**New Features:**
- Streaming response support
- Low-latency TTS integration
- Tool progress indicators

**Breaking Changes:**
- None

**Migration Steps:**
1. Update integration
2. Enable streaming in debug settings (optional)
3. Configure Voice Assistant pipeline with Wyoming TTS

**New Configuration:**
```yaml
# Settings > Home Agent > Configure > Debug Settings
Streaming Responses: true  # NEW
```

**Voice Pipeline Setup:**
- Requires Wyoming Protocol TTS (Piper recommended)
- Configure Assist pipeline with Home Agent

### Configuration Schema Changes

#### History of Schema Versions

| Version | Schema Changes | Migration Required |
|---------|----------------|-------------------|
| 0.1.0 | Initial schema | N/A |
| 0.2.0 | Added vector_db settings, history_persist | No - backward compatible |
| 0.3.0 | Added external_llm settings | No - backward compatible |
| 0.3.5 | Added memory settings | No - backward compatible |
| 0.4.0 | Added streaming settings | No - backward compatible |

All versions maintain **backward compatibility** - old configurations continue to work with new versions.

---

## Common Migration Issues

### Issue 1: Custom Tools Not Appearing

**Symptoms:**
- Tools defined in `configuration.yaml` don't show up
- LLM doesn't have access to custom tools

**Solutions:**

1. **Check YAML syntax**:
   ```bash
   # Use Home Assistant configuration checker
   # Settings > System > Configuration Validation
   ```

2. **Verify configuration location**:
   ```yaml
   # Correct location: root of configuration.yaml
   home_agent:
     tools_custom:
       - name: my_tool

   # Incorrect: nested under another integration
   # some_other_integration:
   #   home_agent:  # WRONG!
   ```

3. **Check logs for errors**:
   ```yaml
   # Enable debug logging
   logger:
     logs:
       custom_components.home_agent: debug

   # Restart HA and check for YAML parsing errors
   ```

4. **Verify secrets are defined**:
   ```yaml
   # If using {{ secrets.api_key }}
   # Check secrets.yaml contains:
   api_key: your_actual_key
   ```

### Issue 2: Template Rendering Errors

**Symptoms:**
- Errors mentioning template rendering
- Tools fail with "invalid template" messages

**Solutions:**

1. **Check template syntax**:
   ```yaml
   # Correct
   url: "https://api.example.com/{{ location }}"

   # Incorrect
   url: "https://api.example.com/{ location }"  # Missing second brace
   ```

2. **Verify parameter names match**:
   ```yaml
   parameters:
     properties:
       location:  # Parameter name
         type: string

   handler:
     query_params:
       city: "{{ location }}"  # Use same name
   ```

3. **Test templates manually**:
   - Developer Tools > Template
   - Test Jinja2 syntax before using in configuration

### Issue 3: Service Not Found

**Symptoms:**
- "ServiceNotFound" errors
- Tools using service handler fail

**Solutions:**

1. **Verify service exists**:
   - Developer Tools > Services
   - Search for the service (e.g., `automation.trigger`)
   - Verify it's available

2. **Check service name format**:
   ```yaml
   # Correct
   service: automation.trigger

   # Incorrect
   service: trigger automation  # Wrong format
   service: automation_trigger  # Wrong separator
   ```

3. **Ensure integration is loaded**:
   - Some services require specific integrations
   - Check integration is installed and loaded

### Issue 4: REST API Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- 403 Forbidden errors
- REST tools fail with authentication errors

**Solutions:**

1. **Verify API key is correct**:
   ```yaml
   # Check secrets.yaml
   weather_api_key: sk-...  # Verify this is correct
   ```

2. **Check header format**:
   ```yaml
   # Common formats:

   # Bearer token
   headers:
     Authorization: "Bearer {{ secrets.api_key }}"

   # API key header
   headers:
     X-API-Key: "{{ secrets.api_key }}"

   # Basic auth (less common)
   headers:
     Authorization: "Basic {{ secrets.basic_auth }}"
   ```

3. **Test API directly**:
   ```bash
   # Use curl to verify credentials work
   curl -H "Authorization: Bearer YOUR_KEY" \
        https://api.example.com/endpoint
   ```

### Issue 5: Conversation History Not Persisting

**Symptoms:**
- History resets after HA restart
- Conversation context lost

**Solutions:**

1. **Enable history persistence**:
   ```yaml
   # Settings > Home Agent > Configure > Conversation History
   History Persist: true
   ```

2. **Check storage permissions**:
   - Verify HA can write to `.storage/` directory
   - Check disk space availability

3. **Verify conversation_id is consistent**:
   ```yaml
   # Use same conversation_id for related messages
   service: home_agent.process
   data:
     text: "Turn on lights"
     conversation_id: "my_conversation"  # Must be same for history
   ```

### Issue 6: External LLM Not Being Called

**Symptoms:**
- External LLM never gets invoked
- Complex queries handled by primary LLM only

**Solutions:**

1. **Verify external LLM is enabled**:
   ```yaml
   # Settings > Home Agent > Configure > External LLM
   External LLM Enabled: true
   ```

2. **Check tool description**:
   - Make it clear when to use external LLM
   - Be specific about use cases
   ```yaml
   Tool Description: |
     Use for complex analysis, detailed explanations, and recommendations.
     Examples: energy analysis, troubleshooting, planning automations.
   ```

3. **Test explicitly**:
   ```yaml
   service: home_agent.execute_tool
   data:
     tool_name: query_external_llm
     parameters:
       prompt: "Test query"
   ```

### Issue 7: High API Costs

**Symptoms:**
- Unexpected high API charges
- Excessive token usage

**Solutions:**

1. **Reduce max_tokens**:
   ```yaml
   # Primary LLM
   Max Tokens: 200  # Lower for cheaper responses

   # External LLM
   Max Tokens: 500  # Lower than default
   ```

2. **Limit tool calls**:
   ```yaml
   # Settings > Home Agent > Configure > Tool Configuration
   Max Tool Calls Per Turn: 3  # Reduce from default 5
   ```

3. **Use cheaper models**:
   ```yaml
   # Primary
   Model: gpt-4o-mini  # Instead of gpt-4o

   # External
   Model: gpt-4o-mini  # Instead of gpt-4o (for less critical tasks)
   ```

4. **Monitor usage**:
   ```yaml
   automation:
     - alias: "Monitor LLM Usage"
       trigger:
         - platform: event
           event_type: home_agent.conversation.finished
       action:
         - service: logbook.log
           data:
             message: "Tokens used: {{ trigger.event.data.tokens.total }}"
   ```

### Issue 8: Memory Not Being Recalled

**Symptoms:**
- Stored memories not used in responses
- Agent doesn't remember previous conversations

**Solutions:**

1. **Verify memory is enabled**:
   ```yaml
   # Settings > Home Agent > Configure > Memory System
   Memory Enabled: true
   ```

2. **Check ChromaDB is running**:
   - Memory requires ChromaDB for semantic search
   - Verify Vector DB configuration is correct

3. **Verify importance threshold**:
   ```yaml
   # Lower threshold to include more memories
   Min Importance: 0.1  # Instead of 0.5
   ```

4. **Check memory was actually stored**:
   ```yaml
   service: home_agent.list_memories
   # Verify memories exist in the list
   ```

5. **Increase context_top_k**:
   ```yaml
   # Retrieve more memories per query
   Memory Context Top K: 10  # Instead of default 5
   ```

### Feature Parity Checklist

After migration, verify you have equivalent functionality:

#### From extended_openai_conversation

- [ ] All REST functions converted to REST custom tools
- [ ] All service functions converted to service custom tools
- [ ] Script functions converted to HA script entities + service tools
- [ ] Entity context configured (same entities exposed)
- [ ] Conversation history working
- [ ] Voice assistant integration working
- [ ] All automations updated to use Home Agent

#### New Capabilities to Consider

- [ ] External LLM configured for complex queries (optional)
- [ ] Memory system enabled for personalized responses (optional)
- [ ] Vector DB for semantic entity search (optional)
- [ ] Streaming enabled for voice assistants (optional)
- [ ] Enhanced event monitoring set up (optional)

### Getting Help

If you encounter issues not covered here:

1. **Check logs**: Enable debug logging and review error messages
2. **GitHub Issues**: Search existing issues or create a new one
3. **Home Assistant Forums**: Post in the integrations category
4. **Documentation**: Review detailed docs in `/docs` folder

---

## Summary

Migrating to Home Agent provides enhanced capabilities while maintaining compatibility with existing workflows. Most migrations follow these key steps:

1. Install Home Agent
2. Configure LLM settings
3. Convert custom functions to tools
4. Test thoroughly
5. Remove old integration

The process is straightforward, and the additional features (memory, streaming, dual-LLM, etc.) make the migration worthwhile for most users.

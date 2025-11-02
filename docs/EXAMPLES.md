# Examples

Quick reference of the most useful Home Agent examples to get you started.

## Voice Assistant Examples

### Basic Voice Control

Set up Home Agent for simple voice commands.

```yaml
# Settings > Devices & Services > Add Integration > Home Agent
Name: Home Agent
LLM Base URL: https://api.openai.com/v1
API Key: sk-your-api-key-here
Model: gpt-4o-mini
Temperature: 0.7
Max Tokens: 500
```

**Voice Commands:**
- "Turn on the living room lights"
- "Set bedroom temperature to 72 degrees"
- "Are the doors locked?"

### Voice Assistant with Memory

Enable personalized, context-aware responses.

```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: true
Automatic Extraction: true
Extraction LLM: external
Max Memories: 100
Context Top K: 5
```

**Example Interaction:**
```
User: "I prefer the bedroom at 68 degrees for sleeping"
Assistant: "I'll remember that you prefer 68 degrees in the bedroom for sleeping."

[Later]
User: "Set bedroom to my preferred temperature"
Assistant: "Setting bedroom to 68 degrees as you prefer."
```

## Custom Tool Examples

### Weather API Tool

Call external weather APIs.

```yaml
# configuration.yaml
home_agent:
  tools_custom:
    - name: check_weather
      description: "Get current weather and 3-day forecast for any location"
      parameters:
        type: object
        properties:
          location:
            type: string
            description: "City name (e.g., 'Seattle' or 'London, UK')"
        required:
          - location
      handler:
        type: rest
        url: "https://api.open-meteo.com/v1/forecast"
        method: GET
        query_params:
          latitude: "47.6788491"
          longitude: "-122.3971093"
          forecast_days: 3
          current: "temperature_2m,precipitation"
```

**Usage:** "What's the weather like?"

### Service Trigger Tool

Trigger Home Assistant automations and scripts.

```yaml
home_agent:
  tools_custom:
    - name: run_morning_routine
      description: "Trigger the morning routine automation"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine

    - name: notify_family
      description: "Send notification to family members"
      parameters:
        type: object
        properties:
          message:
            type: string
        required:
          - message
      handler:
        type: service
        service: notify.family
        data:
          message: "{{ message }}"
```

**Usage:** "Run the morning routine" or "Tell everyone dinner is ready"

## Automation Examples

### Trigger on Tool Execution

Log or react when specific tools are executed.

```yaml
automation:
  - alias: "Log Device Control Events"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
        event_data:
          tool_name: ha_control
    condition:
      - condition: template
        value_template: "{{ trigger.event.data.success }}"
    action:
      - service: logbook.log
        data:
          name: Home Agent
          message: >
            Controlled {{ trigger.event.data.parameters.entity_id }}
            via voice: {{ trigger.event.data.parameters.action }}
```

### Smart Morning Routine

Context-aware routine with memory and custom tools.

```yaml
automation:
  - alias: "Smart Morning Routine"
    trigger:
      - platform: time
        at: "07:00:00"
    action:
      - service: home_agent.process
        data:
          text: >
            Good morning! Please prepare for the day based on my preferences,
            today's weather, and calendar events.
          conversation_id: morning_routine
```

**What Happens:**
1. Recalls user preferences from memory (e.g., bedroom temperature)
2. Checks weather via custom tool
3. Reviews calendar for events
4. Executes personalized routine

## Memory System Examples

### Automatic Memory Extraction

Memories are automatically extracted after conversations.

```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: true
Automatic Extraction: true
Extraction LLM: external
Max Memories: 100
```

**Example:**
```
User: "I really like the living room warm in the evenings, around 72 degrees"
Assistant: "I'll keep the living room at 72 degrees in the evenings. Got it!"

[Memory stored automatically]
Type: preference
Content: "User prefers living room at 72°F in the evenings"

[Later]
User: "Set the living room for the evening"
Assistant: "Setting living room to 72 degrees, as you prefer for evenings."
```

### Manual Memory Storage

Explicitly store information during conversation.

```
User: "Remember that I'm allergic to pollen"
Assistant: [Calls store_memory tool]
"I've made a note that you're allergic to pollen."

[Later]
User: "Should I open the windows?"
Assistant: "The pollen count is high today. Since you're allergic to pollen,
I'd recommend keeping windows closed and using air conditioning instead."
```

## Multi-LLM Examples

### Local Model + GPT-4 for Analysis

Use a fast local model for control and delegate complex analysis to GPT-4.

```yaml
# Primary LLM (Local Ollama)
LLM Base URL: http://localhost:11434/v1
Model: llama2:13b
Temperature: 0.5
Max Tokens: 300

# External LLM (GPT-4)
External LLM Enabled: true
External LLM Base URL: https://api.openai.com/v1
External LLM API Key: sk-your-api-key
External LLM Model: gpt-4o
Temperature: 0.8
Max Tokens: 1000
```

**Example Workflow:**
```
User: "Analyze my energy usage this week and suggest optimizations"

Primary LLM (Ollama):
  1. Calls ha_query to get energy data
  2. Recognizes this needs complex analysis
  3. Calls query_external_llm with data

External LLM (GPT-4):
  - Performs detailed analysis
  - Identifies patterns and anomalies
  - Provides specific recommendations

Primary LLM:
  - Returns formatted response to user
```

**Cost Optimization:** Simple queries handled by free local model; complex analysis only calls paid GPT-4 when needed. Typical ratio: 80% local, 20% external.

### Cost Optimization Strategy

Minimize API costs while maintaining functionality.

```yaml
# Customize external LLM tool description
External LLM Tool Description: |
  Use this tool ONLY for:
  - Detailed multi-step analysis
  - Recommendations requiring complex reasoning
  - Creative suggestions (naming, organizing)

  Do NOT use for:
  - Simple device control
  - Status queries
  - Quick calculations

# Use cheaper models
Model: gpt-4o-mini  # Primary
External LLM Model: gpt-4o  # Only when needed

# Reduce token limits
Max Tokens: 200  # Primary
External Max Tokens: 500  # External
```

## Complete Smart Home Setup

Combining multiple features.

**configuration.yaml:**
```yaml
home_agent:
  tools_custom:
    # Weather tool
    - name: check_weather
      description: "Get weather forecast"
      handler:
        type: rest
        url: "https://api.open-meteo.com/v1/forecast"
        method: GET
        query_params:
          latitude: "40.7128"
          longitude: "-74.0060"
          forecast_days: 3
          current: "temperature_2m,precipitation"

    # Morning routine
    - name: trigger_morning_routine
      description: "Start the morning routine"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine

    # Family notification
    - name: notify_family
      description: "Send message to family"
      parameters:
        type: object
        properties:
          message:
            type: string
        required:
          - message
      handler:
        type: service
        service: notify.family
        data:
          message: "{{ message }}"
```

**automations.yaml:**
```yaml
# Proactive morning greeting
- alias: "Morning Greeting"
  trigger:
    - platform: state
      entity_id: binary_sensor.bedroom_motion
      to: "on"
    - platform: time
      at: "07:00:00"
  condition:
    - condition: time
      after: "06:00:00"
      before: "09:00:00"
  action:
    - service: home_agent.process
      data:
        text: >
          Good morning! Check the weather and suggest what to prepare for the day.
        conversation_id: morning

# Security monitoring
- alias: "Unexpected Entry Alert"
  trigger:
    - platform: state
      entity_id: binary_sensor.front_door
      to: "on"
  condition:
    - condition: state
      entity_id: alarm_control_panel.home
      state: "armed_away"
  action:
    - service: camera.snapshot
      target:
        entity_id: camera.front_door
    - service: home_agent.process
      data:
        text: >
          Front door opened while alarm is armed. Analyze if this matches
          expected patterns and recommend immediate actions.
        conversation_id: security
```

## Need More Details?

See the [Complete Reference](reference/EXAMPLES.md) for comprehensive examples and advanced usage.

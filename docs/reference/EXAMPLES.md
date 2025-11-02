# Home Agent Examples Guide

This guide provides comprehensive examples for using Home Agent in various scenarios, from basic voice control to advanced multi-LLM workflows and memory-enhanced automations.

## Table of Contents

- [Voice Assistant Examples](#voice-assistant-examples)
- [Custom Tool Examples](#custom-tool-examples)
- [Multi-LLM Workflow Examples](#multi-llm-workflow-examples)
- [Automation Trigger Examples](#automation-trigger-examples)
- [Advanced Use Cases](#advanced-use-cases)
- [Memory System Examples](#memory-system-examples)

---

## Voice Assistant Examples

### Basic Voice Control Setup

Set up Home Agent as your voice assistant for simple device control.

**Configuration:**

```yaml
# In Home Assistant UI: Settings > Devices & Services > Add Integration > Home Agent

Name: Home Agent
LLM Base URL: https://api.openai.com/v1
API Key: sk-your-api-key-here
Model: gpt-4o-mini
Temperature: 0.7
Max Tokens: 500
```

**Voice Commands:**

```
"Turn on the living room lights"
"Set bedroom temperature to 72 degrees"
"Are the doors locked?"
"What's the current temperature in the kitchen?"
```

### Multi-Room Voice Assistant

Configure separate instances for different rooms with room-specific context.

**Living Room Configuration:**

```yaml
# Settings > Home Agent > Configure > Context Settings
Context Entities:
  light.living_room_*,
  media_player.living_room,
  climate.living_room,
  sensor.living_room_temperature
```

**Bedroom Configuration:**

```yaml
# Create second instance: Settings > Add Integration > Home Agent
Name: Bedroom Assistant
Context Entities:
  light.bedroom_*,
  climate.bedroom,
  sensor.bedroom_temperature,
  media_player.bedroom
```

**Voice Commands (Living Room):**

```
"Turn on the lights" → Controls living room lights only
"What's the temperature?" → Returns living room sensor
"Play some music" → Controls living room media player
```

### Voice Assistant with Memory Enabled

Enable memory for personalized, context-aware responses.

**Configuration:**

```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: true
Automatic Extraction: true
Extraction LLM: external
Max Memories: 100
Context Top K: 5
```

**Example Interactions:**

```
User: "I prefer the bedroom at 68 degrees for sleeping"
Assistant: "I'll remember that you prefer 68 degrees in the bedroom for sleeping."
[Memory stored: preference type]

User: "Set bedroom to my preferred temperature"
Assistant: "Setting bedroom to 68 degrees as you prefer."
[Memory recalled and used]

User: "What are my temperature preferences?"
Assistant: "You prefer the bedroom at 68 degrees for sleeping."
[Memory searched and retrieved]
```

### Streaming Setup Example

Configure streaming for low-latency voice responses.

**Prerequisites:**

1. Wyoming Protocol TTS (Piper) installed
2. Voice Assistant pipeline configured

**Configuration:**

```yaml
# Settings > Home Agent > Configure > Debug Settings
Streaming Responses: true
```

**Voice Assistant Pipeline Setup:**

1. Navigate to Settings > Voice Assistants > Add Pipeline
2. Configure:
   - Name: Home Agent Streaming
   - Conversation Agent: Home Agent
   - Speech-to-Text: Whisper
   - Text-to-Speech: Piper
   - Wake Word: (optional)

**Result:**

- First audio chunk: ~500ms (vs 5+ seconds without streaming)
- Real-time feel with immediate audio playback
- Tool progress indicators during execution

---

## Custom Tool Examples

### Weather API Integration (Working Example)

Call external weather API using REST handler.

**Configuration (configuration.yaml):**

```yaml
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
          latitude: "47.6788491"  # Example: Seattle coordinates
          longitude: "-122.3971093"
          forecast_days: 3
          precipitation_unit: "inch"
          current: "temperature_2m,precipitation"
          hourly: "temperature_2m,precipitation,showers"
```

**Voice Command:**

```
User: "What's the weather like?"
Assistant: [Calls check_weather tool]
"Currently 65°F with light rain. The forecast shows..."
```

**Alternative with API Key:**

```yaml
- name: openweather_forecast
  description: "Get detailed weather forecast from OpenWeather API"
  parameters:
    type: object
    properties:
      city:
        type: string
        description: "City name"
      days:
        type: integer
        minimum: 1
        maximum: 5
    required:
      - city
  handler:
    type: rest
    url: "https://api.openweathermap.org/data/2.5/forecast"
    method: GET
    headers:
      Accept: "application/json"
    query_params:
      q: "{{ city }}"
      cnt: "{{ days * 8 }}"  # API returns 3-hour intervals
      appid: "{{ secrets.openweather_api_key }}"
```

**secrets.yaml:**

```yaml
openweather_api_key: your_api_key_here
```

### Calendar Integration

Create and query calendar events.

**Configuration:**

```yaml
home_agent:
  tools_custom:
    - name: add_calendar_event
      description: "Add a new event to the family calendar"
      parameters:
        type: object
        properties:
          summary:
            type: string
            description: "Event title"
          start_time:
            type: string
            description: "Start time (ISO format)"
          end_time:
            type: string
            description: "End time (ISO format)"
        required:
          - summary
          - start_time
      handler:
        type: service
        service: calendar.create_event
        data:
          entity_id: calendar.family
          summary: "{{ summary }}"
          start_date_time: "{{ start_time }}"
          end_date_time: "{{ end_time }}"

    - name: check_calendar
      description: "Check upcoming calendar events"
      handler:
        type: service
        service: calendar.get_events
        data:
          entity_id: calendar.family
          duration:
            hours: 24
```

**Usage:**

```
User: "Add a dentist appointment tomorrow at 2pm"
Assistant: [Calls add_calendar_event]
"I've added the dentist appointment to your calendar for tomorrow at 2pm."

User: "What's on my calendar today?"
Assistant: [Calls check_calendar]
"You have 3 events today: Team meeting at 10am, Lunch with Sarah at 12pm..."
```

### Smart Device Control via API

Control smart devices through manufacturer APIs.

**Configuration:**

```yaml
home_agent:
  tools_custom:
    - name: control_philips_hue
      description: "Control Philips Hue lights via their API"
      parameters:
        type: object
        properties:
          light_id:
            type: string
            description: "Hue light identifier"
          action:
            type: string
            enum: [on, off, color, brightness]
          value:
            type: string
            description: "Action value (color hex or brightness 0-255)"
      handler:
        type: rest
        url: "https://{{ secrets.hue_bridge_ip }}/api/{{ secrets.hue_api_key }}/lights/{{ light_id }}/state"
        method: PUT
        headers:
          Content-Type: "application/json"
        body:
          on: "{{ 'true' if action == 'on' else 'false' if action == 'off' else state_attr('light.hue_' + light_id, 'state') }}"
          bri: "{{ value if action == 'brightness' else '' }}"
          hue: "{{ value if action == 'color' else '' }}"
```

### Service Trigger Examples

Trigger Home Assistant services, automations, and scripts.

**Configuration:**

```yaml
home_agent:
  tools_custom:
    # Trigger automation
    - name: run_morning_routine
      description: "Trigger the morning routine automation"
      handler:
        type: service
        service: automation.trigger
        data:
          entity_id: automation.morning_routine

    # Run script with parameters
    - name: notify_family
      description: "Send notification to family members"
      parameters:
        type: object
        properties:
          message:
            type: string
          priority:
            type: string
            enum: [low, normal, high]
        required:
          - message
      handler:
        type: service
        service: notify.family
        data:
          message: "{{ message }}"
          data:
            priority: "{{ priority | default('normal') }}"

    # Activate scene
    - name: set_scene
      description: "Activate a predefined scene"
      parameters:
        type: object
        properties:
          scene_name:
            type: string
            enum: [morning, evening, movie, party, bedtime]
      handler:
        type: service
        service: scene.turn_on
        target:
          entity_id: "scene.{{ scene_name }}"

    # Climate control with area targeting
    - name: set_room_temperature
      description: "Set temperature for a specific room"
      parameters:
        type: object
        properties:
          room:
            type: string
            enum: [living_room, bedroom, office]
          temperature:
            type: number
            minimum: 60
            maximum: 80
      handler:
        type: service
        service: climate.set_temperature
        target:
          area_id: "{{ room }}"
        data:
          temperature: "{{ temperature }}"
```

**Usage:**

```
User: "Run the morning routine"
Assistant: [Calls run_morning_routine]
"Starting your morning routine now."

User: "Tell everyone dinner is ready"
Assistant: [Calls notify_family]
"I've sent a notification to the family."

User: "Set the living room to 72 degrees"
Assistant: [Calls set_room_temperature]
"Setting living room temperature to 72 degrees."
```

---

## Multi-LLM Workflow Examples

### Local Model for Control + GPT-4 for Analysis

Use a fast local model for device control and delegate complex analysis to GPT-4.

**Configuration:**

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
  1. Calls ha_query(entity_id="sensor.energy_*", history={duration: "7d"})
  2. Receives energy data
  3. Recognizes this needs complex analysis
  4. Calls query_external_llm:
     {
       "prompt": "Analyze this energy data and provide optimization suggestions",
       "context": {
         "energy_data": [...],
         "current_rate": "$0.12/kWh",
         "peak_hours": "4pm-9pm"
       }
     }

External LLM (GPT-4):
  - Performs detailed analysis
  - Identifies patterns and anomalies
  - Provides specific recommendations

Primary LLM:
  - Formats GPT-4's response
  - Returns to user: "Based on your energy usage, here are 3 ways to save..."
```

**Cost Optimization:**

- Simple queries: Handled by free local model
- Complex analysis: Only calls paid GPT-4 when needed
- Typical ratio: 80% local, 20% external

### Cost Optimization Strategies

Minimize API costs while maintaining functionality.

**Strategy 1: Selective External LLM Usage**

Customize the tool description to be very specific:

```yaml
External LLM Tool Description: |
  Use this tool ONLY for:
  - Detailed multi-step analysis
  - Recommendations requiring complex reasoning
  - Creative suggestions (naming, organizing)

  Do NOT use for:
  - Simple device control
  - Status queries
  - Quick calculations
  - Formatting responses
```

**Strategy 2: Token Limits**

Reduce token consumption:

```yaml
# Primary LLM
Max Tokens: 200  # Keep responses concise

# External LLM
Max Tokens: 500  # Limit analysis length
```

**Strategy 3: Use Cheaper Models**

```yaml
# Primary: GPT-4o-mini (very cheap)
Model: gpt-4o-mini
Temperature: 0.5

# External: GPT-4o (only when needed)
External LLM Model: gpt-4o
```

**Cost Tracking Automation:**

```yaml
automation:
  - alias: "Track LLM Costs"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
        event_data:
          tool_name: query_external_llm
    action:
      - service: counter.increment
        target:
          entity_id: counter.external_llm_calls
      - service: input_number.set_value
        target:
          entity_id: input_number.estimated_llm_cost
        data:
          value: "{{ states('input_number.estimated_llm_cost') | float + 0.004 }}"
```

### Performance Tuning

Optimize response times and quality.

**Fast Response Configuration:**

```yaml
# Primary LLM: Local Ollama for speed
LLM Base URL: http://localhost:11434/v1
Model: mistral:7b-instruct  # Fast small model
Temperature: 0.3  # More deterministic
Max Tokens: 150  # Short responses
Streaming: true  # Enable for voice assistants

# External LLM: Only for complex tasks
External LLM Enabled: true
External LLM Model: gpt-4o-mini  # Fast cloud model
```

**Quality-Focused Configuration:**

```yaml
# Primary LLM: Better model for understanding
Model: llama2:70b  # Larger model if hardware allows
Temperature: 0.7
Max Tokens: 500

# External LLM: Most capable model
External LLM Model: gpt-4o  # Best reasoning
Temperature: 0.8
Max Tokens: 1500
```

**Balanced Configuration:**

```yaml
# Primary: Good balance
Model: gpt-4o-mini
Temperature: 0.6
Max Tokens: 300
Tools Max Calls Per Turn: 3  # Limit complexity

# External: Powerful but constrained
External LLM Model: gpt-4o
Max Tokens: 800
Tools Timeout: 45  # Reasonable timeout
```

---

## Automation Trigger Examples

### Trigger on Tool Execution

Run automations when specific tools are executed.

**Example: Log Device Controls**

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

**Example: Security Alert on Lock Control**

```yaml
automation:
  - alias: "Alert on Lock Control"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
    condition:
      - condition: template
        value_template: >
          {{ trigger.event.data.tool_name == 'ha_control' and
             trigger.event.data.parameters.entity_id.startswith('lock.') }}
    action:
      - service: notify.admin
        data:
          title: "Lock Control Alert"
          message: >
            Lock {{ trigger.event.data.parameters.entity_id }} was
            {{ trigger.event.data.parameters.action }} via Home Agent
```

### Trigger on Memory Extraction

React when new memories are stored.

**Example: Log Important Memories**

```yaml
automation:
  - alias: "Log Important Memories"
    trigger:
      - platform: event
        event_type: home_agent.memory.extracted
    condition:
      - condition: template
        value_template: >
          {{ trigger.event.data.memories | length > 0 }}
    action:
      - service: notify.admin
        data:
          title: "New Memories Stored"
          message: >
            Home Agent stored {{ trigger.event.data.memories | length }}
            new memories from conversation {{ trigger.event.data.conversation_id }}
```

**Example: Review Preference Changes**

```yaml
automation:
  - alias: "Review Preference Changes"
    trigger:
      - platform: event
        event_type: home_agent.memory.extracted
    condition:
      - condition: template
        value_template: >
          {{ trigger.event.data.memories | selectattr('type', 'eq', 'preference') | list | length > 0 }}
    action:
      - service: persistent_notification.create
        data:
          title: "New Preference Detected"
          message: >
            Home Agent learned a new preference. Review in the memory manager.
```

### Trigger on Conversation Completion

Execute actions after conversations finish.

**Example: Track Conversation Metrics**

```yaml
automation:
  - alias: "Track Conversation Metrics"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: input_number.set_value
        target:
          entity_id: input_number.total_tool_calls
        data:
          value: >
            {{ states('input_number.total_tool_calls') | int +
               trigger.event.data.tool_calls | int }}

      - service: input_number.set_value
        target:
          entity_id: input_number.total_tokens_used
        data:
          value: >
            {{ states('input_number.total_tokens_used') | int +
               trigger.event.data.tokens.total | int }}
```

### Conditional Automation Based on Events

Complex conditional automations triggered by Home Agent events.

**Example: Context-Aware Follow-Up**

```yaml
automation:
  - alias: "Context-Aware Follow-Up"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    condition:
      # Only if external LLM was used
      - condition: template
        value_template: "{{ trigger.event.data.used_external_llm }}"
      # And conversation took longer than 5 seconds
      - condition: template
        value_template: "{{ trigger.event.data.duration_ms > 5000 }}"
    action:
      - delay:
          seconds: 2
      - service: tts.speak
        data:
          entity_id: media_player.living_room
          message: "I used my advanced analysis for that. Let me know if you need more details."
```

**Example: Automated Cleanup**

```yaml
automation:
  - alias: "Clean Old Conversations"
    trigger:
      - platform: time
        at: "03:00:00"  # 3 AM daily
    action:
      - service: home_agent.clear_history
        data:
          # Clear conversations older than 7 days
          older_than_days: 7
```

---

## Advanced Use Cases

### Context-Aware Routines

Create smart routines that adapt based on context and memory.

**Morning Routine with Context:**

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

1. Home Agent recalls memory: User prefers bedroom at 68°F
2. Checks weather via custom tool: Rain expected
3. Checks calendar: Early meeting at 9 AM
4. Executes:
   - Sets bedroom to preferred temperature
   - Suggests umbrella reminder
   - Activates "focus mode" scene for early meeting
   - Starts coffee maker

### Proactive Notifications with Memory

Use memory to provide proactive, personalized notifications.

**Example: Temperature Comfort Alerts**

```yaml
automation:
  - alias: "Proactive Temperature Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.bedroom_temperature
        below: 66  # Below user's preference
        for:
          minutes: 10
    action:
      - service: home_agent.process
        data:
          text: >
            The bedroom temperature is {{ states('sensor.bedroom_temperature') }}°F,
            which is below my preference. Should I adjust it?
          conversation_id: temperature_management
```

**What Happens:**

1. Home Agent recalls: User prefers 68°F for bedroom
2. Compares current (66°F) to preference
3. Proactively offers to adjust
4. If user agrees, executes climate control

**Example: Calendar-Aware Reminders**

```yaml
automation:
  - alias: "Smart Meeting Reminder"
    trigger:
      - platform: calendar
        event: start
        entity_id: calendar.work
        offset: "-00:15:00"  # 15 min before
    action:
      - service: home_agent.process
        data:
          text: >
            I have a meeting starting in 15 minutes: {{ trigger.calendar_event.summary }}.
            Should I prepare the home office?
```

### Energy Optimization with Analysis

Leverage external LLM for detailed energy analysis.

**Configuration:**

```yaml
# Custom tool for energy data
home_agent:
  tools_custom:
    - name: get_energy_details
      description: "Get detailed energy consumption data"
      handler:
        type: service
        service: recorder.get_statistics
        data:
          entity_id:
            - sensor.energy_living_room
            - sensor.energy_bedroom
            - sensor.energy_kitchen
          duration:
            days: 7
```

**Weekly Analysis Automation:**

```yaml
automation:
  - alias: "Weekly Energy Analysis"
    trigger:
      - platform: time
        at: "09:00:00"
      - platform: template
        value_template: "{{ now().weekday() == 6 }}"  # Sunday
    action:
      - service: home_agent.process
        data:
          text: >
            Analyze my energy consumption for the past week and provide
            specific recommendations to reduce costs. Include peak usage
            times and suggest automation adjustments.
          conversation_id: energy_optimization
```

**What Happens:**

1. Primary LLM calls get_energy_details tool
2. Receives 7 days of data
3. Calls query_external_llm with data and analysis request
4. External LLM (GPT-4):
   - Identifies peak usage patterns
   - Spots inefficiencies
   - Suggests specific automations
   - Calculates potential savings
5. Response delivered via notification or voice

### Security Monitoring

Monitor security events and respond intelligently.

**Configuration:**

```yaml
automation:
  - alias: "Security Event Analysis"
    trigger:
      - platform: state
        entity_id:
          - binary_sensor.front_door
          - binary_sensor.back_door
          - binary_sensor.garage_door
        to: "on"
    condition:
      # Only when house is in away mode
      - condition: state
        entity_id: input_select.house_mode
        state: "away"
    action:
      - service: camera.snapshot
        target:
          entity_id: camera.front_door
        data:
          filename: /config/www/security/latest.jpg

      - service: home_agent.process
        data:
          text: >
            Security alert: {{ trigger.to_state.attributes.friendly_name }}
            opened while house is in away mode. Evaluate if this is expected
            based on my schedule and usual patterns, and recommend actions.
          conversation_id: security_monitoring
```

**What Happens:**

1. Door opened during away mode
2. Snapshot captured
3. Home Agent:
   - Recalls memory: User typically returns at 6 PM
   - Checks current time
   - Reviews calendar for scheduled visitors
   - Analyzes pattern
4. Response:
   - Expected: "Welcome home! You're back earlier than usual."
   - Unexpected: "Unexpected entry detected. I've captured a photo and notified security contacts."

---

## Memory System Examples

### Automatic Memory Extraction Workflow

Memory is automatically extracted after each conversation when enabled.

**Configuration:**

```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: true
Automatic Extraction: true
Extraction LLM: external  # or "local"
Max Memories: 100
```

**Example Conversation:**

```
User: "I really like the living room warm in the evenings, around 72 degrees"
Assistant: "I'll keep the living room at 72 degrees in the evenings. Got it!"

[After conversation ends]
→ EVENT: home_agent.conversation.finished
→ Memory extraction triggered
→ External LLM extracts:
  {
    "type": "preference",
    "content": "User prefers living room at 72°F in the evenings",
    "importance": 0.8,
    "metadata": {
      "entities": ["climate.living_room"],
      "time_of_day": "evening",
      "temperature": 72
    }
  }
→ Memory stored and indexed in ChromaDB
```

**Subsequent Interaction:**

```
User: "Set the living room for the evening"
Assistant: "Setting living room to 72 degrees, as you prefer for evenings."
[Memory recalled from context injection]
```

### Manual Memory Storage During Conversation

Use the store_memory tool to explicitly save information.

**Example:**

```
User: "Remember that I'm allergic to pollen"
Assistant: [Calls store_memory tool]
{
  "content": "User is allergic to pollen",
  "type": "fact",
  "importance": 0.9
}
"I've made a note that you're allergic to pollen."

[Later conversation]
User: "Should I open the windows?"
Assistant: [Recalls pollen allergy memory]
"The pollen count is high today. Since you're allergic to pollen,
I'd recommend keeping windows closed and using air conditioning instead."
```

### Memory-Enhanced Responses

Memories automatically enhance responses when relevant.

**Example: Temperature Preferences**

```
[Stored memory: "User prefers 68°F for sleeping"]

User: "I'm going to bed"
Assistant: [Memory injected into context]
"Good night! I've set the bedroom to 68 degrees, your preferred sleeping temperature."
```

**Example: Routine Patterns**

```
[Stored memories:
 - "User leaves for work at 7:30 AM on weekdays"
 - "User prefers house at 65°F when away"]

User: "I'm leaving for work"
Assistant: [Recalls memories]
"Have a great day! I've set the house to 65 degrees and activated away mode.
I'll prepare things for your return around 5:30 PM."
```

### Privacy-Focused Memory Configuration

Configure memory with privacy and data retention in mind.

**Minimal Memory Configuration:**

```yaml
# Only store essential preferences, no personal facts
Memory Enabled: true
Automatic Extraction: false  # Manual only
Max Memories: 50
Memory Types Allowed:
  - preference  # Only preferences
Memory Fact TTL: 2592000  # 30 days
Memory Preference TTL: 7776000  # 90 days
```

**Disable Memory Completely:**

```yaml
# Settings > Home Agent > Configure > Memory System
Memory Enabled: false
```

**GDPR-Compliant Setup:**

```yaml
automation:
  - alias: "Monthly Memory Cleanup"
    trigger:
      - platform: time
        at: "02:00:00"
      - platform: template
        value_template: "{{ now().day == 1 }}"  # First of month
    action:
      # Clear all memories older than 90 days
      - service: home_agent.clear_memories
        data:
          confirm: true
          older_than_days: 90

      - service: persistent_notification.create
        data:
          title: "Memory Cleanup Complete"
          message: "Memories older than 90 days have been removed per data retention policy."
```

**User-Initiated Data Deletion:**

```yaml
script:
  delete_all_my_data:
    sequence:
      - service: home_agent.clear_memories
        data:
          confirm: true

      - service: home_agent.clear_history

      - service: persistent_notification.create
        data:
          title: "All Data Deleted"
          message: "All conversation history and memories have been permanently deleted."
```

---

## Complete Example: Smart Home Setup

Here's a complete configuration combining multiple features.

**configuration.yaml:**

```yaml
home_agent:
  # Custom weather tool
  tools_custom:
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

    # Morning routine automation
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
    - condition: state
      entity_id: input_boolean.morning_greeting_said
      state: "off"
  action:
    - service: home_agent.process
      data:
        text: >
          Good morning! Check the weather and suggest what to prepare for the day.
        conversation_id: morning
    - service: input_boolean.turn_on
      target:
        entity_id: input_boolean.morning_greeting_said

# Weekly energy report
- alias: "Weekly Energy Report"
  trigger:
    - platform: time
      at: "20:00:00"
    - platform: template
      value_template: "{{ now().weekday() == 6 }}"
  action:
    - service: home_agent.process
      data:
        text: >
          Analyze my energy usage this week, compare to previous weeks,
          and suggest ways to reduce consumption and save money.
        conversation_id: energy_analysis

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

This comprehensive example demonstrates the power of combining custom tools, multi-LLM workflows, memory, and automations for a fully intelligent home assistant experience.

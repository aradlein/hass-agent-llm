# Home Agent Observability with InfluxDB + Grafana

Complete monitoring setup for Home Agent using your existing InfluxDB integration and Grafana.

## 📊 Overview

Since you already have InfluxDB integrated with Home Assistant, this guide shows you how to:
1. Configure Home Assistant to send Home Agent events to InfluxDB
2. Connect Grafana directly to InfluxDB
3. Import pre-built dashboards for instant visualization

**No Prometheus needed!** This is the simplest monitoring setup.

## 🎯 What You Can Monitor

- **Conversations**: Total count, rate, duration
- **Token Usage**: Prompt, completion, and total tokens consumed
- **Performance**: LLM latency, tool execution time, context retrieval speed
- **Tool Execution**: Success rates, call frequency, execution duration
- **External LLM Usage**: When and how often external LLM is used
- **Errors**: Error rates, types, and patterns

## 🚀 Quick Start (3 Steps)

### Step 1: Configure Home Assistant to Track Events

**📁 EASIEST METHOD: Use the example files**
1. Copy from [`configuration.yaml.example`](configuration.yaml.example) → your `configuration.yaml`
2. Copy from [`automations.yaml.example`](automations.yaml.example) → your `automations.yaml`
3. Restart Home Assistant
4. Skip to Step 2

---

**Or configure manually:**

Add the template sensors, counters, and automations to your `configuration.yaml`, then configure InfluxDB to include them.

**Add to `configuration.yaml`:**

```yaml
# Template sensors to track Home Agent metrics
template:
  - trigger:
      - platform: event
        event_type: home_agent.conversation.finished
      - platform: event
        event_type: home_agent.tool.executed
      - platform: event
        event_type: home_agent.error
    sensor:
      # Conversation metrics
      - name: "Home Agent Last Conversation Duration"
        unique_id: home_agent_last_conversation_duration
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.duration_ms }}
          {% else %}
            {% set current = states('sensor.home_agent_last_conversation_duration') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Conversation Tokens"
        unique_id: home_agent_last_conversation_tokens
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.tokens.total }}
          {% else %}
            {% set current = states('sensor.home_agent_last_conversation_tokens') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "tokens"
        state_class: measurement
        attributes:
          prompt_tokens: >
            {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
              {{ trigger.event.data.tokens.prompt }}
            {% endif %}
          completion_tokens: >
            {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
              {{ trigger.event.data.tokens.completion }}
            {% endif %}

      - name: "Home Agent Last LLM Latency"
        unique_id: home_agent_last_llm_latency
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.performance.llm_latency_ms }}
          {% else %}
            {% set current = states('sensor.home_agent_last_llm_latency') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Tool Latency"
        unique_id: home_agent_last_tool_latency
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.performance.tool_latency_ms }}
          {% else %}
            {% set current = states('sensor.home_agent_last_tool_latency') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Context Latency"
        unique_id: home_agent_last_context_latency
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.performance.context_latency_ms }}
          {% else %}
            {% set current = states('sensor.home_agent_last_context_latency') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last TTFT"
        unique_id: home_agent_last_ttft
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.performance.ttft_ms }}
          {% else %}
            {% set current = states('sensor.home_agent_last_ttft') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Tool Calls"
        unique_id: home_agent_last_tool_calls
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.tool_calls }}
          {% else %}
            {% set current = states('sensor.home_agent_last_tool_calls') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "calls"
        state_class: measurement

      - name: "Home Agent Used External LLM"
        unique_id: home_agent_used_external_llm
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ 1 if trigger.event.data.used_external_llm else 0 }}
          {% else %}
            {% set current = states('sensor.home_agent_used_external_llm') %}
            {{ current if current not in ['unknown', 'unavailable'] else 0 }}
          {% endif %}
        unit_of_measurement: "boolean"
        state_class: measurement

# Counter sensors (persistent across restarts)
counter:
  home_agent_conversations_total:
    name: "Home Agent Conversations Total"
    icon: mdi:chat
  home_agent_tool_successes:
    name: "Home Agent Tool Successes"
    icon: mdi:check-circle
  home_agent_tool_failures:
    name: "Home Agent Tool Failures"
    icon: mdi:alert-circle
  home_agent_errors_total:
    name: "Home Agent Errors Total"
    icon: mdi:alert

# Automations to increment counters
# IMPORTANT: If you have automations in a separate automations.yaml file,
# add these to that file instead. Do NOT add "automation:" to configuration.yaml
# if you're already using automations.yaml, as it will override your existing automations.

# Option 1: If you use automations.yaml (most common), add these to that file:
# Just copy the automation entries below into your automations.yaml

# Option 2: If you configure automations in configuration.yaml, use this format:
automation:
  - id: home_agent_conversation_counter
    alias: "Home Agent: Increment Conversation Counter"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: counter.increment
        target:
          entity_id: counter.home_agent_conversations_total

  - id: home_agent_tool_counter
    alias: "Home Agent: Track Tool Success/Failure"
    trigger:
      - platform: event
        event_type: home_agent.tool.executed
    action:
      - choose:
          - conditions:
              - condition: template
                value_template: "{{ trigger.event.data.success }}"
            sequence:
              - service: counter.increment
                target:
                  entity_id: counter.home_agent_tool_successes
        default:
          - service: counter.increment
            target:
              entity_id: counter.home_agent_tool_failures

  - id: home_agent_error_counter
    alias: "Home Agent: Increment Error Counter"
    trigger:
      - platform: event
        event_type: home_agent.error
    action:
      - service: counter.increment
        target:
          entity_id: counter.home_agent_errors_total
```

### Configure InfluxDB to Include Home Agent Sensors

**IMPORTANT:** You must tell InfluxDB to capture these sensors. Add this to your InfluxDB configuration:

```yaml
influxdb:
  # ... your existing InfluxDB settings (host, port, token, organization, bucket, etc.)

  # Add these entities to be included
  include:
    entities:
      - sensor.home_agent_last_conversation_duration
      - sensor.home_agent_last_conversation_tokens
      - sensor.home_agent_last_llm_latency
      - sensor.home_agent_last_tool_latency
      - sensor.home_agent_last_context_latency
      - sensor.home_agent_last_ttft
      - sensor.home_agent_last_tool_calls
      - sensor.home_agent_used_external_llm
      - counter.home_agent_conversations_total
      - counter.home_agent_tool_successes
      - counter.home_agent_tool_failures
      - counter.home_agent_errors_total
```

**Alternative (if you want to include all Home Agent entities):**

```yaml
influxdb:
  # ... your existing settings
  include:
    entity_globs:
      - sensor.home_agent_*
      - counter.home_agent_*
```

**Then restart Home Assistant.**

> **⚠️ Important Note on Automations:**
>
> Most Home Assistant installations use a separate `automations.yaml` file. If you do:
> - **DO NOT** add the `automation:` section to `configuration.yaml`
> - **Instead**, copy just the three automation entries (without the `automation:` line) and paste them into your `automations.yaml` file
> - Or, use the Home Assistant UI: **Settings** → **Automations & Scenes** → **Create Automation** → **Skip** → **Edit in YAML** and paste each automation
>
> If you add `automation:` to `configuration.yaml` when you're already using `automations.yaml`, it will override and remove all your existing automations!

### Step 2: Verify Sensors Are Working

After restarting Home Assistant, the sensors will show `0` initially (this prevents errors).

To verify everything is working:

1. Go to **Developer Tools** → **States**
2. Search for `sensor.home_agent_` - you should see 7 sensors (all showing `0` initially)
3. **Have a conversation with Home Agent** (via voice or text)
4. Refresh the States page - sensors should now show real values from that conversation

Your existing InfluxDB integration will automatically capture these sensor values once they start updating.

In InfluxDB, you should see measurements like:
- `home_agent_last_conversation_duration`
- `home_agent_last_conversation_tokens`
- `home_agent_last_llm_latency`
- etc.

### Step 3: Import Grafana Dashboard

1. Open Grafana
2. Go to **Dashboards** → **Import**
3. Upload `grafana/home_agent_influxdb_dashboard.json`
4. Select your InfluxDB data source
5. Click **Import**

**Done!** You now have full observability.

## 📁 Folder Structure

```
observability/
├── README.md                                    # This file
├── configuration.yaml.example                   # ⭐ Copy to your configuration.yaml
├── automations.yaml.example                     # ⭐ Copy to your automations.yaml
├── grafana/
│   └── home_agent_influxdb_dashboard.json      # Grafana dashboard with Flux queries
└── influxdb/
    └── flux_queries.md                          # 30+ example Flux queries
```

## 📊 Available Metrics

| Metric | Measurement Name | Description |
|--------|------------------|-------------|
| Conversation Duration | `home_agent_last_conversation_duration` | Time to process conversation (ms) |
| Token Usage | `home_agent_last_conversation_tokens` | Total tokens per conversation |
| Prompt Tokens | `home_agent_last_conversation_tokens.prompt_tokens` | Tokens in prompt |
| Completion Tokens | `home_agent_last_conversation_tokens.completion_tokens` | Tokens in completion |
| LLM Latency | `home_agent_last_llm_latency` | LLM API response time (ms) |
| Tool Latency | `home_agent_last_tool_latency` | Tool execution time (ms) |
| Context Latency | `home_agent_last_context_latency` | Context retrieval time (ms) |
| TTFT | `home_agent_last_ttft` | Time to First Token from LLM (ms) |
| Tool Calls | `home_agent_last_tool_calls` | Number of tools called |
| External LLM Used | `home_agent_used_external_llm` | 1 if external LLM was used, 0 otherwise |
| Total Conversations | `counter.home_agent_conversations_total` | Cumulative conversation count |
| Tool Successes | `counter.home_agent_tool_successes` | Successful tool executions |
| Tool Failures | `counter.home_agent_tool_failures` | Failed tool executions |
| Total Errors | `counter.home_agent_errors_total` | Cumulative error count |

## 🎙️ Voice Pipeline Metrics (STT/TTS)

Home Agent tracks **TTFT (Time to First Token)** natively as part of its performance metrics. However, **STT (Speech-to-Text) and TTS (Text-to-Speech) timing** is handled by Home Assistant's assist pipeline, which operates outside Home Agent's scope.

### Monitoring STT/TTS Performance

To track complete voice pipeline performance including STT and TTS metrics:

#### Option 1: StreamAssist (Recommended)

Install **[StreamAssist](https://github.com/AlexxIT/StreamAssist)** from HACS for comprehensive voice pipeline monitoring:

1. Install StreamAssist via HACS
2. StreamAssist provides sensors for each pipeline stage:
   - **WAKE**: Wake word detection timing
   - **STT**: Speech-to-text conversion timing
   - **INTENT**: Intent processing timing (includes Home Agent processing)
   - **TTS**: Text-to-speech generation timing

3. Each sensor includes timing attributes that can be added to your monitoring:
   ```yaml
   influxdb:
     include:
       entities:
         - sensor.streamassist_wake
         - sensor.streamassist_stt
         - sensor.streamassist_intent
         - sensor.streamassist_tts
   ```

4. Add these sensors to your Grafana dashboard for end-to-end voice pipeline visualization

**StreamAssist Benefits:**
- Complete pipeline visibility (wake word to final speech output)
- Historical timing data for each stage
- Easy integration with existing InfluxDB + Grafana setup
- Identifies bottlenecks in your voice pipeline

#### Option 2: Home Assistant Debug View

For quick debugging without installing additional components:

1. Go to **Settings** → **Voice assistants**
2. Select your pipeline
3. Click **Debug**
4. Run a test conversation
5. View timing breakdown for each stage

This provides real-time insight but doesn't persist historical data for trend analysis.

### Complete Voice Pipeline Timing

When using StreamAssist alongside Home Agent observability, you can track:

| Stage | Metric Source | What It Measures |
|-------|--------------|------------------|
| Wake Word | StreamAssist | Time to detect wake word |
| STT | StreamAssist | Speech → text conversion |
| Intent + LLM | Home Agent + StreamAssist | Combined: context retrieval, LLM processing, tool execution |
| TTFT | Home Agent | LLM time to first token (streaming responsiveness) |
| TTS | StreamAssist | Text → speech generation |
| **Total** | StreamAssist | End-to-end pipeline duration |

This complete picture helps you optimize the entire voice experience, not just the Home Agent portion.

## 🎨 Dashboard Features

The Grafana dashboard includes:

### Overview Row
- **Total Conversations** (stat)
- **Average Duration** (stat with sparkline)
- **Total Tokens** (stat with trend)
- **Error Rate** (gauge)

### Performance Row
- **Latency Breakdown** (timeseries) - LLM, tool, and context latency
- **Conversation Duration Over Time** (timeseries)

### Token Usage Row
- **Token Consumption Over Time** (timeseries) - prompt vs completion
- **Tokens per Conversation** (stat)

### Tool Analytics Row
- **Tool Success Rate** (gauge)
- **Tool Calls Over Time** (timeseries)
- **Tool Execution Duration** (histogram)

### External LLM Row
- **External LLM Usage** (stat) - percentage of conversations
- **External LLM Calls Over Time** (timeseries)

### Error Tracking Row
- **Errors Over Time** (timeseries)
- **Error Rate** (stat)

## 🔧 Customization

### Add Custom Metrics

Add new template sensors to track additional event data:

```yaml
template:
  - trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    sensor:
      - name: "Home Agent Custom Metric"
        state: "{{ trigger.event.data.your_custom_field }}"
```

### Modify Dashboard Queries

All dashboard panels use Flux queries. Example:

```flux
from(bucket: "home_assistant")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
```

Edit panels in Grafana to customize:
- Time ranges
- Aggregation functions
- Visualization types
- Thresholds and colors

### Calculate Cost

Add a cost tracking sensor:

```yaml
template:
  - trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    sensor:
      - name: "Home Agent Conversation Cost"
        state: >
          {% set prompt = trigger.event.data.tokens.prompt %}
          {% set completion = trigger.event.data.tokens.completion %}
          {% set prompt_cost = (prompt / 1000000) * 0.15 %}
          {% set completion_cost = (completion / 1000000) * 0.60 %}
          {{ (prompt_cost + completion_cost) | round(4) }}
        unit_of_measurement: "USD"
        state_class: measurement
```

Then create a cumulative cost counter using `utility_meter` integration.

## 🐛 Troubleshooting

### Sensors Showing "Unknown" State or Errors

**Initial State:** Trigger-based template sensors will show `0` initially (instead of "unknown") and update to real values after the first Home Agent conversation.

**If you see errors like "ValueError: invalid literal for int() with base 10: 'unknown'":**
- This means you're using an older version of the template configuration
- Update your `configuration.yaml` with the corrected templates from this README (they now handle unknown states properly)
- Restart Home Assistant after updating

**To populate the sensors with real data:**
1. Have a conversation with Home Agent (talk to it via the conversation integration)
2. After the conversation completes, the sensors will immediately update with actual values
3. The sensors will then show real metrics from your conversations

**If sensors remain "unknown" after a conversation:**

1. **Verify events are enabled in Home Agent**:
   - Check your Home Agent configuration
   - Ensure event emission is not disabled (it's enabled by default)

2. **Check event is actually firing**:
   - Go to **Developer Tools** → **Events** → **Listen to events**
   - Type: `home_agent.conversation.finished`
   - Click **Start Listening**
   - Have a conversation with Home Agent
   - You should see the event with all the data fields

3. **Verify template sensors exist**:
   - Go to **Developer Tools** → **States**
   - Search for: `sensor.home_agent_`
   - You should see all 7 sensors listed (even if "unknown")

4. **Check Home Assistant logs for template errors**:
   ```bash
   tail -f /config/home-assistant.log | grep -i "template\|home_agent"
   ```

5. **Verify configuration.yaml syntax**:
   - Go to **Developer Tools** → **YAML** → **Check Configuration**
   - Fix any errors before restarting

### Sensors Not Updating

1. **Check event is firing**:
   - Developer Tools → Events
   - Listen for: `home_agent.conversation.finished`
   - Trigger a conversation

2. **Verify template sensors exist**:
   - Developer Tools → States
   - Search for: `sensor.home_agent_`

3. **Check Home Assistant logs**:
   ```bash
   tail -f /config/home-assistant.log | grep home_agent
   ```

### Data Not Appearing in InfluxDB

1. **Verify InfluxDB integration is running**:
   - Settings → Integrations → InfluxDB
   - Check status is "OK"

2. **Check InfluxDB configuration**:
   ```yaml
   influxdb:
     host: localhost
     port: 8086
     database: home_assistant
     # ... other settings
   ```

3. **Verify sensors are included**:
   ```yaml
   influxdb:
     include:
       entities:
         - sensor.home_agent_last_conversation_duration
         # ... other sensors
   ```

4. **Check InfluxDB logs** for write errors

### Grafana Dashboard Not Loading

1. **Verify InfluxDB data source**:
   - Grafana → Configuration → Data Sources
   - Test & Save

2. **Check bucket/database name matches**:
   - Update dashboard variables if needed

3. **Verify Flux queries**:
   - Open dashboard panel
   - Edit query
   - Run query manually

### Missing Data Points

1. **Check sensor state_class**:
   - Must be `measurement` for time-series data

2. **Verify InfluxDB retention policy**:
   - Data may have been deleted if policy is too short

3. **Check time range in Grafana**:
   - Ensure time range includes when data was recorded

## 📚 Example Flux Queries

See `influxdb/flux_queries.md` for detailed examples including:
- Average conversation duration
- Token consumption trends
- Tool success rate calculations
- Error rate monitoring
- Cost analysis

## 🔄 Advanced: Real-Time Event Logging

For detailed event logging (not just aggregated metrics), add this automation:

```yaml
automation:
  - id: home_agent_event_logger
    alias: "Home Agent: Log Detailed Events"
    trigger:
      - platform: event
        event_type: home_agent.conversation.finished
    action:
      - service: influxdb.write
        data:
          bucket: home_agent_events
          measurement: conversation
          tags:
            user_id: "{{ trigger.event.data.user_id }}"
            conversation_id: "{{ trigger.event.data.conversation_id }}"
            used_external_llm: "{{ trigger.event.data.used_external_llm }}"
          fields:
            duration_ms: "{{ trigger.event.data.duration_ms }}"
            prompt_tokens: "{{ trigger.event.data.tokens.prompt }}"
            completion_tokens: "{{ trigger.event.data.tokens.completion }}"
            total_tokens: "{{ trigger.event.data.tokens.total }}"
            llm_latency: "{{ trigger.event.data.performance.llm_latency_ms }}"
            tool_latency: "{{ trigger.event.data.performance.tool_latency_ms }}"
            context_latency: "{{ trigger.event.data.performance.context_latency_ms }}"
            tool_calls: "{{ trigger.event.data.tool_calls }}"
```

This creates a separate `conversation` measurement with full event details for deep analysis.

## 📈 Dashboard Maintenance

### Update Dashboard

When new metrics are added:
1. Export your customized dashboard
2. Import updated dashboard
3. Merge customizations

### Backup Dashboard

```bash
# Export from Grafana UI or via API
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/dashboards/uid/home-agent-influxdb \
  > backup_$(date +%Y%m%d).json
```

## 💡 Best Practices

1. **Set appropriate retention policies** in InfluxDB to manage disk space
2. **Use downsampling** for long-term trends (aggregate hourly/daily)
3. **Create alerts** in Grafana for critical metrics (high error rate, latency spikes)
4. **Monitor InfluxDB performance** if you have high conversation volume
5. **Regular backups** of Grafana dashboards and InfluxDB data

## 🤝 Contributing

Found a better query or dashboard layout? Contributions welcome!

## 📄 License

Same license as Home Agent project.

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/aradlein/home-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aradlein/home-agent/discussions)

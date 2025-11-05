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

Add this to your `configuration.yaml`:

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
            {{ states('sensor.home_agent_last_conversation_duration') }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Conversation Tokens"
        unique_id: home_agent_last_conversation_tokens
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.tokens.total }}
          {% else %}
            {{ states('sensor.home_agent_last_conversation_tokens') }}
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
            {{ states('sensor.home_agent_last_llm_latency') }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Tool Latency"
        unique_id: home_agent_last_tool_latency
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.performance.tool_latency_ms }}
          {% else %}
            {{ states('sensor.home_agent_last_tool_latency') }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Context Latency"
        unique_id: home_agent_last_context_latency
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.performance.context_latency_ms }}
          {% else %}
            {{ states('sensor.home_agent_last_context_latency') }}
          {% endif %}
        unit_of_measurement: "ms"
        state_class: measurement

      - name: "Home Agent Last Tool Calls"
        unique_id: home_agent_last_tool_calls
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ trigger.event.data.tool_calls }}
          {% else %}
            {{ states('sensor.home_agent_last_tool_calls') }}
          {% endif %}
        unit_of_measurement: "calls"
        state_class: measurement

      - name: "Home Agent Used External LLM"
        unique_id: home_agent_used_external_llm
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ 1 if trigger.event.data.used_external_llm else 0 }}
          {% else %}
            {{ states('sensor.home_agent_used_external_llm') }}
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

# Automation to increment counters
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

**Then restart Home Assistant.**

### Step 2: Verify InfluxDB is Capturing Data

Your existing InfluxDB integration should already be capturing these sensors. Verify by:

1. Go to **Developer Tools** → **States**
2. Search for `sensor.home_agent_`
3. Trigger a conversation with Home Agent
4. Confirm sensor values update

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
| Tool Calls | `home_agent_last_tool_calls` | Number of tools called |
| External LLM Used | `home_agent_used_external_llm` | 1 if external LLM was used, 0 otherwise |
| Total Conversations | `counter.home_agent_conversations_total` | Cumulative conversation count |
| Tool Successes | `counter.home_agent_tool_successes` | Successful tool executions |
| Tool Failures | `counter.home_agent_tool_failures` | Failed tool executions |
| Total Errors | `counter.home_agent_errors_total` | Cumulative error count |

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

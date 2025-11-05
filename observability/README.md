# Home Agent Observability

Complete observability setup for monitoring Home Agent performance, usage, and health using Prometheus, InfluxDB, and Grafana.

## 📊 Overview

Home Agent emits detailed events to Home Assistant's event bus that can be captured and visualized using standard observability tools. This folder contains ready-to-use configurations for:

- **Prometheus** - Metrics collection and alerting
- **InfluxDB** - Time-series data storage
- **Grafana** - Visualization dashboards
- **Home Assistant** - Event-to-metrics transformation

## 🎯 What You Can Monitor

### Conversation Metrics
- Total conversations processed
- Average conversation duration
- Conversation rate (per second/minute/hour)
- User activity patterns

### Token Usage
- Total tokens consumed (prompt + completion)
- Prompt tokens vs completion tokens
- Token consumption rate
- Cost estimation (when using paid APIs)

### Performance Metrics
- LLM API latency
- Tool execution latency
- Context retrieval latency
- End-to-end conversation duration

### Tool Execution
- Tool call frequency by tool name
- Tool success rate
- Tool execution duration
- Tool failures and errors

### External LLM Usage
- Number of external LLM calls
- Percentage of conversations using external LLM
- External LLM vs local LLM usage patterns

### Error Tracking
- Error rate over time
- Error types and categories
- Component-level error breakdown

## 🚀 Quick Start

### Option 1: Prometheus Only (Simplest)

1. **Add sensors to Home Assistant** (`configuration.yaml`):
   ```bash
   cat prometheus/home_agent_events.yaml >> /config/configuration.yaml
   ```

2. **Enable Prometheus integration**:
   ```yaml
   prometheus:
     namespace: home_agent
   ```

3. **Configure Prometheus scraping**:
   ```yaml
   scrape_configs:
     - job_name: 'homeassistant'
       static_configs:
         - targets: ['homeassistant:8123']
   ```

4. **Import Grafana dashboard**:
   - Open Grafana → Dashboards → Import
   - Upload `grafana/home_agent_dashboard.json`

### Option 2: InfluxDB + Prometheus (Recommended)

For users who already have InfluxDB running:

1. **Configure InfluxDB integration**:
   ```bash
   cat influxdb/home_agent_influxdb.yaml >> /config/configuration.yaml
   ```

2. **Set up InfluxDB credentials** in `secrets.yaml`:
   ```yaml
   influxdb_username: your_username
   influxdb_password: your_password
   influxdb_auth_header: "Token YOUR_INFLUXDB_TOKEN"
   ```

3. **Bridge InfluxDB to Prometheus**:
   - See `influxdb/influxdb_to_prometheus_bridge.md` for detailed instructions
   - Choose from Telegraf, influxdb_exporter, or native InfluxDB 2.x methods

4. **Import Grafana dashboard** as above

## 📁 Folder Structure

```
observability/
├── README.md                           # This file
├── prometheus/
│   ├── home_agent_events.yaml          # HA sensors + Prometheus export config
│   └── alerting_rules.yaml             # Prometheus alerting rules
├── influxdb/
│   ├── home_agent_influxdb.yaml        # InfluxDB integration config
│   └── influxdb_to_prometheus_bridge.md # Bridging guide
└── grafana/
    └── home_agent_dashboard.json       # Pre-built Grafana dashboard
```

## 📊 Available Metrics

| Metric Name | Type | Description | Unit |
|-------------|------|-------------|------|
| `home_agent_conversations_total` | Counter | Total conversations processed | conversations |
| `home_agent_avg_duration` | Gauge | Average conversation duration | ms |
| `home_agent_total_tokens` | Counter | Total tokens consumed | tokens |
| `home_agent_prompt_tokens` | Counter | Prompt tokens used | tokens |
| `home_agent_completion_tokens` | Counter | Completion tokens generated | tokens |
| `home_agent_tool_calls_total` | Counter | Total tool executions | calls |
| `home_agent_tool_success_rate` | Gauge | Tool execution success rate | % |
| `home_agent_llm_latency` | Gauge | LLM API response latency | ms |
| `home_agent_tool_latency` | Gauge | Tool execution latency | ms |
| `home_agent_context_latency` | Gauge | Context retrieval latency | ms |
| `home_agent_external_llm_calls` | Counter | External LLM invocations | calls |
| `home_agent_errors_total` | Counter | Total errors encountered | errors |

## 🔔 Alerting

Pre-configured Prometheus alerts are available in `prometheus/alerting_rules.yaml`:

### Warning Alerts
- High error rate (>0.1 errors/sec for 5m)
- Low tool success rate (<80% for 10m)
- High LLM latency (>2000ms for 5m)
- High tool latency (>1000ms for 5m)
- No activity for 30 minutes
- High token consumption
- High external LLM usage
- High context latency (>500ms for 10m)

### Critical Alerts
- Critical error rate (>1 error/sec for 2m)
- Critical tool success rate (<50% for 5m)
- Critical LLM latency (>5000ms for 2m)

### Add alerts to Prometheus config:

```yaml
# prometheus.yml
rule_files:
  - /path/to/observability/prometheus/alerting_rules.yaml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

## 🎨 Grafana Dashboard

The pre-built dashboard (`grafana/home_agent_dashboard.json`) includes:

### Overview Row
- Total conversations (stat)
- Average duration (stat)
- Total tokens used (stat)
- Tool success rate (gauge)

### Trends Row
- Conversations over time (timeseries)
- Token usage over time (timeseries - stacked)

### Performance Row
- LLM, tool, and context latency (timeseries - multi-line)

### Analysis Row
- Tool usage distribution (pie chart)
- External LLM usage (timeseries - stacked bars)

### Error Tracking Row
- Error rate over time (timeseries)

### Features
- Auto-refresh every 10 seconds
- Time range selector (default: last 6 hours)
- Data source selector
- Drill-down capabilities

## 🔧 Advanced Configuration

### Custom Metrics

Add custom metrics by creating template sensors in Home Assistant:

```yaml
template:
  - sensor:
      - name: "Home Agent Custom Metric"
        unique_id: home_agent_custom
        state: >
          {% if trigger.event.event_type == 'home_agent.conversation.finished' %}
            {{ your_custom_logic_here }}
          {% endif %}
```

### Metric Aggregation

Use Prometheus recording rules for pre-aggregated metrics:

```yaml
# Example: Average tokens per conversation
- record: home_agent:avg_tokens_per_conversation
  expr: |
    home_agent_total_tokens /
    home_agent_conversations_total
```

### Cost Tracking

Calculate API costs based on token usage:

```yaml
template:
  - sensor:
      - name: "Home Agent Estimated Cost"
        state: >
          {% set prompt_tokens = states('sensor.home_agent_prompt_tokens') | float(0) %}
          {% set completion_tokens = states('sensor.home_agent_completion_tokens') | float(0) %}
          {% set prompt_cost = (prompt_tokens / 1000000) * 0.15 %}  # $0.15 per 1M tokens
          {% set completion_cost = (completion_tokens / 1000000) * 0.60 %}  # $0.60 per 1M tokens
          {{ (prompt_cost + completion_cost) | round(2) }}
        unit_of_measurement: "USD"
```

## 🐛 Troubleshooting

### Metrics Not Appearing in Prometheus

1. **Verify sensors exist in Home Assistant**:
   ```bash
   # In Home Assistant Developer Tools → States
   # Search for: sensor.home_agent_
   ```

2. **Check Prometheus integration is enabled**:
   ```yaml
   # configuration.yaml
   prometheus:
   ```

3. **Verify Prometheus is scraping Home Assistant**:
   ```bash
   # Open http://prometheus:9090/targets
   # Look for homeassistant job
   ```

4. **Check Home Assistant logs**:
   ```bash
   tail -f /config/home-assistant.log | grep home_agent
   ```

### InfluxDB Not Receiving Data

1. **Test InfluxDB connectivity**:
   ```bash
   curl -I http://localhost:8086/ping
   ```

2. **Verify rest_command credentials**:
   ```bash
   # Check secrets.yaml has correct tokens
   ```

3. **Test manual write**:
   ```bash
   curl -XPOST "http://localhost:8086/write?db=home_agent_metrics" \
     -H "Authorization: Token YOUR_TOKEN" \
     --data-binary "test_metric value=1"
   ```

4. **Check automation triggers**:
   ```bash
   # Home Assistant → Developer Tools → Events
   # Listen for: home_agent.conversation.finished
   ```

### Grafana Dashboard Not Loading

1. **Verify Prometheus data source**:
   - Grafana → Configuration → Data Sources
   - Test & Save

2. **Check metric names match**:
   ```bash
   # In Prometheus, query:
   {__name__=~"home_agent.*"}
   ```

3. **Import dashboard again**:
   - Grafana → Dashboards → Import
   - Upload JSON, select Prometheus data source

### Events Not Firing

1. **Verify Home Agent integration is running**:
   ```bash
   # Home Assistant → Settings → Devices & Services → Home Agent
   ```

2. **Check event emission is enabled**:
   ```yaml
   # In Home Agent config
   emit_events: true  # Should be true (default)
   ```

3. **Listen for events manually**:
   ```bash
   # Home Assistant → Developer Tools → Events
   # Listen for: home_agent.*
   # Then trigger a conversation
   ```

## 📚 Additional Resources

### Documentation
- [Prometheus Integration](https://www.home-assistant.io/integrations/prometheus/)
- [InfluxDB Integration](https://www.home-assistant.io/integrations/influxdb/)
- [Grafana Setup](https://grafana.com/docs/grafana/latest/)
- [Home Assistant Template Sensors](https://www.home-assistant.io/integrations/template/)

### Example Queries

**Prometheus PromQL Examples**:

```promql
# Conversations per hour
rate(home_agent_conversations_total[1h]) * 3600

# Average latency (all components)
(home_agent_llm_latency + home_agent_tool_latency + home_agent_context_latency) / 3

# Tool success rate over time
100 - (home_agent_tool_success_rate)

# External LLM usage percentage
(home_agent_external_llm_calls / home_agent_conversations_total) * 100

# Token cost estimate (OpenAI GPT-4o-mini pricing)
(home_agent_prompt_tokens / 1000000 * 0.15) + (home_agent_completion_tokens / 1000000 * 0.60)
```

**InfluxDB Flux Examples**:

```flux
// Average conversation duration per hour
from(bucket: "home_agent_metrics")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_conversation")
  |> filter(fn: (r) => r._field == "duration_ms")
  |> aggregateWindow(every: 1h, fn: mean)

// Tool success rate
from(bucket: "home_agent_metrics")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_tool")
  |> filter(fn: (r) => r._field == "success")
  |> group()
  |> reduce(
      fn: (r, accumulator) => ({
        total: accumulator.total + 1,
        successes: accumulator.successes + (if r._value == "true" then 1 else 0)
      }),
      identity: {total: 0, successes: 0}
    )
  |> map(fn: (r) => ({ _value: float(v: r.successes) / float(v: r.total) * 100.0 }))
```

## 🤝 Contributing

Have improvements or additional dashboards? Contributions are welcome!

1. Test your configuration thoroughly
2. Document any new metrics or alerts
3. Update this README with usage instructions
4. Submit a pull request

## 📄 License

Same license as Home Agent project.

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/aradlein/home-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aradlein/home-agent/discussions)
- **Documentation**: See main [docs/](../docs/) directory

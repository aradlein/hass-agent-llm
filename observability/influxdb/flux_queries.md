# Flux Query Examples for Home Agent Metrics

Example Flux queries for querying Home Agent metrics from InfluxDB. Use these as starting points for custom Grafana panels or InfluxDB CLI queries.

## Basic Queries

### Get Latest Conversation Duration

```flux
from(bucket: "home_assistant")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> last()
```

### Get Average Conversation Duration (Last 24h)

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> mean()
```

### Get Token Usage Over Time

```flux
from(bucket: "home_assistant")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "value" or r._field == "prompt_tokens" or r._field == "completion_tokens")
  |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
```

## Performance Metrics

### All Latency Metrics Combined

```flux
llm = from(bucket: "home_assistant")
  |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_llm_latency")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> set(key: "latency_type", value: "llm")

tool = from(bucket: "home_assistant")
  |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_tool_latency")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> set(key: "latency_type", value: "tool")

context = from(bucket: "home_assistant")
  |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_context_latency")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
  |> set(key: "latency_type", value: "context")

union(tables: [llm, tool, context])
  |> sort(columns: ["_time"])
```

### P95 Latency (95th Percentile)

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_llm_latency")
  |> filter(fn: (r) => r._field == "value")
  |> quantile(q: 0.95, method: "estimate_tdigest")
```

### Max Latency in Last Hour

```flux
from(bucket: "home_assistant")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_llm_latency")
  |> filter(fn: (r) => r._field == "value")
  |> max()
```

## Tool Metrics

### Tool Success Rate Calculation

```flux
successes = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_tool_successes")
  |> filter(fn: (r) => r._field == "value")
  |> last()
  |> toFloat()

failures = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_tool_failures")
  |> filter(fn: (r) => r._field == "value")
  |> last()
  |> toFloat()

union(tables: [successes, failures])
  |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      success_rate: (r["counter.home_agent_tool_successes"] /
                    (r["counter.home_agent_tool_successes"] + r["counter.home_agent_tool_failures"])) * 100.0
    }))
```

### Tool Call Frequency (per hour)

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_tool_calls")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1h, fn: sum, createEmpty: false)
```

## Token Analysis

### Token Consumption by Type

```flux
prompt = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "prompt_tokens")
  |> sum()
  |> set(key: "token_type", value: "prompt")

completion = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "completion_tokens")
  |> sum()
  |> set(key: "token_type", value: "completion")

union(tables: [prompt, completion])
```

### Average Tokens Per Conversation

```flux
tokens = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "value")
  |> sum()

conversations = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_conversations_total")
  |> filter(fn: (r) => r._field == "value")
  |> last()

union(tables: [tokens, conversations])
  |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      avg_tokens: r["home_agent_last_conversation_tokens"] / r["counter.home_agent_conversations_total"]
    }))
```

### Token Usage Rate (per minute)

```flux
from(bucket: "home_assistant")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1m, fn: sum, createEmpty: false)
```

## External LLM Usage

### External LLM Usage Percentage

```flux
external = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_used_external_llm")
  |> filter(fn: (r) => r._field == "value")
  |> sum()

total = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_conversations_total")
  |> filter(fn: (r) => r._field == "value")
  |> last()

union(tables: [external, total])
  |> pivot(rowKey:["_time"], columnKey: ["_measurement"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      external_llm_percentage: (r["home_agent_used_external_llm"] / r["counter.home_agent_conversations_total"]) * 100.0
    }))
```

### External LLM Calls Over Time

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_used_external_llm")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1h, fn: sum, createEmpty: false)
```

## Error Tracking

### Error Count (Last 24h)

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_errors_total")
  |> filter(fn: (r) => r._field == "value")
  |> last()
```

### Error Rate (per minute)

```flux
from(bucket: "home_assistant")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_errors_total")
  |> filter(fn: (r) => r._field == "value")
  |> derivative(unit: 1m, nonNegative: true)
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
```

### Error Rate Spike Detection

```flux
from(bucket: "home_assistant")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_errors_total")
  |> filter(fn: (r) => r._field == "value")
  |> derivative(unit: 1m, nonNegative: true)
  |> filter(fn: (r) => r._value > 5.0)  // Alert if more than 5 errors per minute
```

## Cost Calculation

### Estimated Cost (OpenAI GPT-4o-mini Pricing)

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "prompt_tokens" or r._field == "completion_tokens")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      prompt_cost: float(v: r.prompt_tokens) / 1000000.0 * 0.15,
      completion_cost: float(v: r.completion_tokens) / 1000000.0 * 0.60,
      total_cost: (float(v: r.prompt_tokens) / 1000000.0 * 0.15) +
                  (float(v: r.completion_tokens) / 1000000.0 * 0.60)
    }))
  |> sum(column: "total_cost")
```

### Daily Cost Trend

```flux
from(bucket: "home_assistant")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "prompt_tokens" or r._field == "completion_tokens")
  |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> map(fn: (r) => ({
      _time: r._time,
      daily_cost: (float(v: r.prompt_tokens) / 1000000.0 * 0.15) +
                  (float(v: r.completion_tokens) / 1000000.0 * 0.60)
    }))
```

## Aggregations

### Hourly Statistics

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> yield(name: "mean")

from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1h, fn: max, createEmpty: false)
  |> yield(name: "max")

from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> aggregateWindow(every: 1h, fn: min, createEmpty: false)
  |> yield(name: "min")
```

### Conversations Per Hour

```flux
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "counter.home_agent_conversations_total")
  |> filter(fn: (r) => r._field == "value")
  |> derivative(unit: 1h, nonNegative: true)
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
```

## Advanced Queries

### Correlation: Duration vs Token Count

```flux
duration = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")

tokens = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_tokens")
  |> filter(fn: (r) => r._field == "value")

join(tables: {duration: duration, tokens: tokens}, on: ["_time"])
  |> map(fn: (r) => ({
      _time: r._time,
      duration: r._value_duration,
      tokens: r._value_tokens,
      ms_per_token: r._value_duration / float(v: r._value_tokens)
    }))
```

### Anomaly Detection (Simple)

Detect conversations that took significantly longer than average:

```flux
import "experimental"

// Calculate mean and stddev
stats = from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> experimental.stddev()

mean_val = stats |> mean() |> findRecord(fn: (key) => true, idx: 0)
stddev_val = stats |> last() |> findRecord(fn: (key) => true, idx: 0)

// Find outliers (> 2 standard deviations from mean)
from(bucket: "home_assistant")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> filter(fn: (r) => r._field == "value")
  |> filter(fn: (r) => r._value > mean_val._value + (2.0 * stddev_val._value))
```

### Downsampling for Long-Term Storage

```flux
from(bucket: "home_assistant")
  |> range(start: -30d)
  |> filter(fn: (r) => r._measurement =~ /home_agent_/)
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
  |> to(bucket: "home_assistant_downsampled", org: "your_org")
```

## Tips

### Use Variables in Grafana

Replace hardcoded values with variables:

```flux
from(bucket: "${bucket}")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
```

### Performance Optimization

1. **Use aggregateWindow** for time-series data:
   ```flux
   |> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)
   ```

2. **Limit data with filters early**:
   ```flux
   from(bucket: "home_assistant")
     |> range(start: -1h)  // Narrow time range first
     |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")  // Then filter
   ```

3. **Use `last()` instead of `mean()` for current values**:
   ```flux
   from(bucket: "home_assistant")
     |> range(start: -5m)
     |> filter(fn: (r) => r._measurement == "counter.home_agent_conversations_total")
     |> last()
   ```

### Debugging Queries

Add `yield()` at different stages to see intermediate results:

```flux
from(bucket: "home_assistant")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "home_agent_last_conversation_duration")
  |> yield(name: "after_filter")
  |> mean()
  |> yield(name: "after_mean")
```

## Resources

- [Flux Documentation](https://docs.influxdata.com/flux/latest/)
- [Flux Query Examples](https://docs.influxdata.com/influxdb/latest/query-data/flux/)
- [Grafana InfluxDB Data Source](https://grafana.com/docs/grafana/latest/datasources/influxdb/)

# InfluxDB to Prometheus Bridge Configuration

This guide explains how to expose InfluxDB metrics to Prometheus for unified monitoring.

## Method 1: Using Telegraf (Recommended)

Telegraf can read from InfluxDB and expose metrics in Prometheus format.

### Telegraf Configuration

Create a `telegraf.conf` file:

```toml
# Telegraf Configuration for InfluxDB -> Prometheus Bridge

[agent]
  interval = "10s"
  round_interval = true
  metric_batch_size = 1000
  metric_buffer_limit = 10000
  collection_jitter = "0s"
  flush_interval = "10s"
  flush_jitter = "0s"
  precision = ""
  hostname = ""
  omit_hostname = false

# Read from InfluxDB
[[inputs.influxdb]]
  urls = ["http://localhost:8086"]
  username = "your_username"
  password = "your_password"
  database = "home_agent_metrics"

  # Query specific measurements
  query = '''
    SELECT * FROM "home_agent_conversation" WHERE time > now() - 5m;
    SELECT * FROM "home_agent_tool" WHERE time > now() - 5m;
    SELECT * FROM "home_agent_error" WHERE time > now() - 5m;
  '''

# Expose metrics in Prometheus format
[[outputs.prometheus_client]]
  listen = ":9273"
  path = "/metrics"
  expiration_interval = "60s"
  collectors_exclude = ["gocollector", "process"]
  string_as_label = true
```

### Docker Compose Setup

```yaml
version: '3.8'

services:
  telegraf:
    image: telegraf:latest
    container_name: telegraf-influxdb-bridge
    restart: unless-stopped
    volumes:
      - ./telegraf.conf:/etc/telegraf/telegraf.conf:ro
    ports:
      - "9273:9273"
    networks:
      - monitoring

networks:
  monitoring:
    external: true
```

### Prometheus Configuration

Add this to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'home-agent-influxdb'
    static_configs:
      - targets: ['telegraf:9273']
    relabel_configs:
      - source_labels: [__name__]
        regex: 'influxdb_.*'
        target_label: __name__
        replacement: 'home_agent_${1}'
```

## Method 2: Using influxdb_exporter

The official InfluxDB exporter for Prometheus.

### Installation

```bash
# Download the exporter
wget https://github.com/prometheus/influxdb_exporter/releases/download/v0.11.5/influxdb_exporter-0.11.5.linux-amd64.tar.gz
tar xvfz influxdb_exporter-0.11.5.linux-amd64.tar.gz
cd influxdb_exporter-0.11.5.linux-amd64
```

### Configuration

Create `influxdb_exporter.yml`:

```yaml
# InfluxDB Exporter Configuration
influxdb:
  url: http://localhost:8086
  database: home_agent_metrics
  username: your_username
  password: your_password

# Queries to export
queries:
  - name: home_agent_conversations
    query: SELECT COUNT(*) FROM "home_agent_conversation" WHERE time > now() - 1h GROUP BY time(5m)
    metric_name: home_agent_conversations_total
    metric_type: counter

  - name: home_agent_avg_duration
    query: SELECT MEAN("duration_ms") FROM "home_agent_conversation" WHERE time > now() - 5m
    metric_name: home_agent_conversation_duration_avg_ms
    metric_type: gauge

  - name: home_agent_tokens
    query: SELECT SUM("total_tokens") FROM "home_agent_conversation" WHERE time > now() - 5m
    metric_name: home_agent_tokens_total
    metric_type: counter

  - name: home_agent_tool_success_rate
    query: |
      SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN success = 'true' THEN 1 ELSE 0 END) AS successes
      FROM "home_agent_tool"
      WHERE time > now() - 5m
    metric_name: home_agent_tool_success_rate
    metric_type: gauge

  - name: home_agent_errors
    query: SELECT COUNT(*) FROM "home_agent_error" WHERE time > now() - 1h GROUP BY time(5m)
    metric_name: home_agent_errors_total
    metric_type: counter
```

### Run the Exporter

```bash
./influxdb_exporter --config.file=influxdb_exporter.yml --web.listen-address=:9122
```

### Docker Setup

```yaml
version: '3.8'

services:
  influxdb-exporter:
    image: prom/influxdb-exporter:latest
    container_name: influxdb-exporter
    restart: unless-stopped
    command:
      - '--influxdb.server=http://influxdb:8086'
      - '--influxdb.database=home_agent_metrics'
      - '--influxdb.username=your_username'
      - '--influxdb.password=your_password'
    ports:
      - "9122:9122"
    networks:
      - monitoring

networks:
  monitoring:
    external: true
```

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'influxdb-exporter'
    static_configs:
      - targets: ['influxdb-exporter:9122']
```

## Method 3: Direct InfluxDB 2.x to Prometheus

InfluxDB 2.x has built-in Prometheus-compatible endpoints.

### InfluxDB 2.x Configuration

No additional configuration needed! InfluxDB 2.x exposes metrics at:
- `/api/v2/query` - Query data in Prometheus format

### Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'influxdb-v2'
    scrape_interval: 30s
    scheme: http
    static_configs:
      - targets: ['influxdb:8086']
    params:
      db: ['home_agent_metrics']
    bearer_token: 'YOUR_INFLUXDB_TOKEN'
    metrics_path: '/api/v2/query'
    body: |
      from(bucket: "home_agent_metrics")
        |> range(start: -5m)
        |> filter(fn: (r) => r._measurement =~ /home_agent_.*/)
```

## Verification

### Test Telegraf Metrics

```bash
curl http://localhost:9273/metrics | grep home_agent
```

### Test InfluxDB Exporter

```bash
curl http://localhost:9122/metrics | grep home_agent
```

### Test Prometheus Scraping

```bash
# Query Prometheus to verify metrics are being scraped
curl 'http://localhost:9090/api/v1/query?query=home_agent_conversations_total'
```

## Recommended Approach

**For most users**: Use **Method 1 (Telegraf)** as it:
- Provides the most flexibility
- Handles data transformation well
- Is well-documented and widely used
- Supports multiple output formats

**For InfluxDB 2.x users**: Use **Method 3** for simplicity and native integration.

## Troubleshooting

### Metrics Not Appearing

1. **Check InfluxDB connectivity**:
   ```bash
   curl -G http://localhost:8086/query --data-urlencode "db=home_agent_metrics" --data-urlencode "q=SHOW MEASUREMENTS"
   ```

2. **Verify Telegraf is running**:
   ```bash
   docker logs telegraf-influxdb-bridge
   ```

3. **Check Prometheus targets**:
   Open `http://localhost:9090/targets` in your browser

### Data Not Updating

1. Verify Home Assistant automations are triggering
2. Check InfluxDB write permissions
3. Ensure clock synchronization between systems

## Next Steps

- Set up Grafana dashboards (see `../grafana/README.md`)
- Configure alerting rules (see `../prometheus/alerting_rules.yaml`)
- Monitor metric cardinality to avoid performance issues

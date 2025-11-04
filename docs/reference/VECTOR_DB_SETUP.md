# Vector Database Setup Guide

## Introduction

This guide explains how to set up and configure ChromaDB for semantic entity search in Home Agent. Vector DB mode provides intelligent, context-aware entity retrieval that improves with scale.

### What is Vector DB Mode?

**Vector DB Mode** uses semantic similarity search to dynamically find the most relevant entities based on the user's query, rather than including a fixed list of entities in every request.

**How it works**:
1. Entity states are embedded (converted to numerical vectors) and stored in ChromaDB
2. When a user asks a question, the query is also embedded
3. ChromaDB finds entities semantically similar to the query
4. Only relevant entities are included in the LLM context
5. More efficient token usage and better context relevance

### Direct Mode vs Vector DB Mode

| Feature | Direct Mode | Vector DB Mode |
|---------|-------------|----------------|
| **Setup Complexity** | Simple | Moderate (requires ChromaDB) |
| **Configuration** | Fixed entity list | Dynamic semantic search |
| **Best For** | Small setups, specific use cases | Large setups, general queries |
| **Token Usage** | All configured entities every time | Only relevant entities |
| **Latency** | Lower | Slightly higher (embedding + search) |
| **Scalability** | Limited by token budget | Scales well with many entities |
| **Relevance** | Manual selection | Automatic based on query |

### When to Use Vector DB Mode

**Use Vector DB Mode when**:
- You have many entities (50+) and can't include them all in context
- Users ask diverse questions about different areas of the home
- You want the LLM to automatically find relevant entities
- You need the long-term memory system (requires ChromaDB)

**Use Direct Mode when**:
- You have a small number of entities (< 20)
- You want specific entities always included
- You need minimal latency
- You want simpler setup and configuration

## Prerequisites

### Required Components

1. **ChromaDB Server**
   - Version 0.4.0 or later
   - Can run as Docker container or standalone
   - Network accessible from Home Assistant

2. **Embedding Provider**
   - **OpenAI** (cloud-based, requires API key)
     - Model: `text-embedding-3-small` (recommended)
     - Cost: ~$0.02 per 1M tokens
   - **Ollama** (local, free)
     - Model: `nomic-embed-text` or `mxbai-embed-large`
     - Requires Ollama installation

3. **Home Agent Installation**
   - See [Installation Guide](INSTALLATION.md)
   - Basic configuration completed

### Hardware Requirements

For ChromaDB server:
- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum, 4GB+ recommended for large entity sets
- **Storage**: ~100MB for database + embeddings (scales with entity count)

For Ollama embeddings (if using local):
- **RAM**: Additional 1-2GB for embedding model
- **GPU**: Optional but significantly faster

## ChromaDB Installation

### Docker Installation (Recommended)

Docker provides the easiest installation and management of ChromaDB.

1. **Install Docker**

   If not already installed:

   - **Linux**: Follow [Docker installation guide](https://docs.docker.com/engine/install/)
   - **Home Assistant OS**: Docker is already available
   - **Home Assistant Supervised**: Docker is already available

2. **Run ChromaDB Container**

   Create a directory for persistent storage:

   ```bash
   mkdir -p /config/chromadb
   ```

   Start ChromaDB:

   ```bash
   docker run -d \
     --name chromadb \
     -p 8000:8000 \
     -v /config/chromadb:/chroma/chroma \
     --restart unless-stopped \
     chromadb/chroma:latest
   ```

   **Parameters explained**:
   - `-d`: Run in background
   - `--name chromadb`: Container name
   - `-p 8000:8000`: Expose port 8000
   - `-v /config/chromadb:/chroma/chroma`: Persistent storage
   - `--restart unless-stopped`: Auto-restart on failures

3. **Verify Installation**

   Test that ChromaDB is running:

   ```bash
   curl http://localhost:8000/api/v1/heartbeat
   ```

   Expected response: `{"nanosecond heartbeat": ...}`

4. **Configure for External Access** (if needed)

   If Home Assistant is on a different machine, expose ChromaDB on all interfaces:

   ```bash
   docker run -d \
     --name chromadb \
     -p 8000:8000 \
     -e ALLOW_RESET=true \
     -e IS_PERSISTENT=true \
     -v /config/chromadb:/chroma/chroma \
     --restart unless-stopped \
     chromadb/chroma:latest
   ```

   **Security Note**: Consider using a reverse proxy with authentication for production deployments.

### Standalone Installation

For advanced users who prefer non-Docker installations:

1. **Install Python Dependencies**

   ```bash
   pip install chromadb
   ```

2. **Create ChromaDB Server Script**

   Create `/opt/chromadb/server.py`:

   ```python
   import chromadb
   from chromadb.config import Settings

   # Configure ChromaDB
   client = chromadb.Client(Settings(
       chroma_db_impl="duckdb+parquet",
       persist_directory="/opt/chromadb/data",
       chroma_server_host="0.0.0.0",
       chroma_server_http_port=8000,
   ))

   # Start server
   print("ChromaDB server running on port 8000")
   client.run_server()
   ```

3. **Create Systemd Service** (Linux)

   Create `/etc/systemd/system/chromadb.service`:

   ```ini
   [Unit]
   Description=ChromaDB Server
   After=network.target

   [Service]
   Type=simple
   User=homeassistant
   WorkingDirectory=/opt/chromadb
   ExecStart=/usr/bin/python3 /opt/chromadb/server.py
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:

   ```bash
   sudo systemctl enable chromadb
   sudo systemctl start chromadb
   ```

### Network Configuration

#### Same Machine

If ChromaDB is on the same machine as Home Assistant:
- **Host**: `localhost` or `127.0.0.1`
- **Port**: `8000` (default)

#### Different Machine

If ChromaDB is on a different machine:
- **Host**: IP address of the ChromaDB server
- **Port**: `8000` (default)

**Firewall Configuration**:
Ensure port 8000 is open on the ChromaDB server:

```bash
# Linux (ufw)
sudo ufw allow 8000/tcp

# Linux (firewalld)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

## Home Agent Configuration

### Connecting to ChromaDB

1. **Open Home Agent Configuration**

   Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure**

2. **Select Vector DB Settings**

   Choose **Vector DB Settings** from the menu

3. **Configure Connection**

   | Field | Description | Example |
   |-------|-------------|---------|
   | **Vector DB Host** | ChromaDB server hostname/IP | `localhost` |
   | **Vector DB Port** | ChromaDB server port | `8000` |
   | **Collection Name** | ChromaDB collection for entities | `home_entities` |
   | **Embedding Provider** | Provider for generating embeddings | `ollama` or `openai` |
   | **Embedding Base URL** | Base URL for embedding API | See below |
   | **Embedding Model** | Model name for embeddings | See below |
   | **OpenAI API Key** | API key (if using OpenAI) | Your API key |
   | **Top K** | Number of entities to retrieve | `5` |
   | **Similarity Threshold** | Minimum similarity score (L2 distance) | `250.0` |

### Embedding Provider Setup

#### Option 1: Ollama (Local, Free)

Best for privacy and no API costs.

1. **Install Ollama**

   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Embedding Model**

   ```bash
   ollama pull nomic-embed-text
   ```

   Alternative models:
   - `mxbai-embed-large` (higher quality, larger)
   - `all-minilm` (smaller, faster)

3. **Configure in Home Agent**

   - **Embedding Provider**: `ollama`
   - **Embedding Base URL**: `http://localhost:11434`
   - **Embedding Model**: `nomic-embed-text`
   - **OpenAI API Key**: Leave blank

#### Option 2: OpenAI (Cloud)

Best for quality and convenience (requires API costs).

1. **Get OpenAI API Key**

   - Sign up at [OpenAI Platform](https://platform.openai.com/)
   - Generate API key from API Keys section
   - Add billing information

2. **Configure in Home Agent**

   - **Embedding Provider**: `openai`
   - **Embedding Base URL**: `https://api.openai.com/v1`
   - **Embedding Model**: `text-embedding-3-small`
   - **OpenAI API Key**: Your API key (e.g., `sk-...`)

**Cost Estimate**:
- Model: `text-embedding-3-small`
- Cost: ~$0.02 per 1M tokens
- Average entity: ~100 tokens
- 100 entities: ~10,000 tokens = $0.0002 per indexing
- Typical usage: $0.01-0.05/month for queries

### Collection Name Configuration

The collection name identifies your entity database in ChromaDB.

**Default**: `home_entities`

**Custom Collection**:
- Use if you want separate collections for different purposes
- Example: `home_entities_main` vs `home_entities_test`

### Top-K and Similarity Threshold Tuning

#### Top-K (Number of Results)

Controls how many entities are retrieved for each query.

| Top-K Value | Use Case | Token Usage |
|-------------|----------|-------------|
| `3` | Very focused queries, minimal context | Low |
| `5` | Balanced (recommended) | Medium |
| `10` | Complex queries, comprehensive context | High |
| `20+` | Special cases, very large context | Very High |

**Recommendation**: Start with `5` and adjust based on results.

#### Similarity Threshold (L2 Distance)

Controls the minimum similarity required for entities to be included.

ChromaDB uses **L2 (Euclidean) distance** by default:
- **Lower values** = higher similarity (more strict)
- **Higher values** = lower similarity (more lenient)

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| `100.0` | Very strict, only highly relevant entities | Precision-focused |
| `250.0` | Balanced (recommended) | General use |
| `500.0` | Lenient, include more entities | Comprehensive context |
| `1000.0` | Very lenient, most entities match | Testing/debugging |

**Recommendation**: Start with `250.0` and adjust based on results.

**Tuning Tips**:
- If getting too few entities: Increase threshold or increase Top-K
- If getting irrelevant entities: Decrease threshold or reduce Top-K
- Monitor via events: `home_agent.vector_db.queried` shows retrieved entities

### Enable Vector DB Mode

After configuring Vector DB settings:

1. **Go to Context Settings**

   Navigate to **Configure** > **Context Settings**

2. **Select Context Mode**

   - **Context Mode**: Select `vector_db`

3. **Save Configuration**

   Click **Submit**

## Entity Indexing

After enabling Vector DB mode, you must index your entities before semantic search will work.

### Using the Reindex Service

The `home_agent.reindex_entities` service indexes all exposed entities at once.

1. **Open Developer Tools**

   Navigate to **Developer Tools** > **Services**

2. **Call Reindex Service**

   Select `home_agent.reindex_entities` and execute:

   ```yaml
   # No parameters required
   ```

3. **Monitor Progress**

   Check Home Assistant logs for indexing progress:

   ```
   [home_agent.vector_db_manager] Indexing 127 entities...
   [home_agent.vector_db_manager] Indexed 127 entities in 12.3s
   ```

**When to Reindex**:
- After initial setup
- After adding new entities
- After modifying entity configurations
- Weekly or monthly for large setups (optional)

### Using the Single Entity Index Service

For indexing individual entities:

1. **Call Index Service**

   ```yaml
   service: home_agent.index_entity
   data:
     entity_id: "light.new_bedroom_light"
   ```

2. **Verify Indexing**

   Check logs for confirmation:

   ```
   [home_agent.vector_db_manager] Indexed entity: light.new_bedroom_light
   ```

### Automatic Indexing

Currently, entities must be manually indexed. Future versions may include:
- Automatic indexing on entity creation
- Scheduled background re-indexing
- Change detection and incremental updates

## Troubleshooting

### Connection Errors

**Symptom**: "Failed to connect to ChromaDB" or connection timeout

**Solutions**:
1. Verify ChromaDB is running:
   ```bash
   docker ps | grep chromadb
   curl http://localhost:8000/api/v1/heartbeat
   ```

2. Check network connectivity:
   ```bash
   ping <chromadb-host>
   telnet <chromadb-host> 8000
   ```

3. Verify firewall rules allow port 8000

4. Check ChromaDB logs:
   ```bash
   docker logs chromadb
   ```

### Embedding Provider Issues

#### Ollama Errors

**Symptom**: "Failed to generate embedding" or "Model not found"

**Solutions**:
1. Verify Ollama is running:
   ```bash
   ollama list
   ```

2. Pull the embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

3. Test embedding generation:
   ```bash
   ollama embed nomic-embed-text "test query"
   ```

4. Check Ollama logs:
   ```bash
   journalctl -u ollama -f
   ```

#### OpenAI Errors

**Symptom**: "Invalid API key" or "Quota exceeded"

**Solutions**:
1. Verify API key is correct and active
2. Check billing status on OpenAI platform
3. Verify quota limits haven't been exceeded
4. Test API key manually:
   ```bash
   curl https://api.openai.com/v1/embeddings \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"input": "test", "model": "text-embedding-3-small"}'
   ```

### Indexing Failures

**Symptom**: "Failed to index entities" or partial indexing

**Solutions**:
1. Check entity count:
   ```bash
   # In Home Assistant Developer Tools > States
   # Count exposed entities
   ```

2. Verify ChromaDB has space:
   ```bash
   df -h /config/chromadb
   ```

3. Check for rate limiting (OpenAI):
   - Reduce batch size (code limitation)
   - Wait and retry

4. Review logs for specific errors:
   ```
   [home_agent.vector_db_manager] Failed to index light.bedroom: ...
   ```

### Query Issues

**Symptom**: No entities retrieved or irrelevant entities

**Solutions**:
1. **Check indexing status**:
   - Verify entities are indexed in ChromaDB
   - Re-run `home_agent.reindex_entities`

2. **Adjust Top-K**:
   - Increase if getting too few results
   - Current value shown in logs

3. **Adjust similarity threshold**:
   - Increase for more lenient matching
   - Decrease for stricter matching

4. **Review query embeddings**:
   - Check logs for query embedding generation
   - Verify embedding provider is working

5. **Test manually**:
   ```yaml
   service: home_agent.process
   data:
     text: "What is the bedroom temperature?"
   ```

   Check logs for retrieved entities:
   ```
   [home_agent.vector_db_manager] Retrieved 3 entities for query (threshold=250.0)
   ```

### Performance Tuning

#### Slow Queries

**Symptom**: Vector DB queries take >2 seconds

**Solutions**:
1. **Reduce Top-K**: Lower the number of results retrieved
2. **Use local embeddings**: Switch from OpenAI to Ollama
3. **Optimize ChromaDB**:
   - Ensure ChromaDB has adequate resources (RAM/CPU)
   - Consider SSD storage for better I/O performance
4. **Index optimization**:
   - ChromaDB automatically optimizes indexes over time
   - Consider compaction for very large datasets

#### High Memory Usage

**Symptom**: ChromaDB using excessive RAM

**Solutions**:
1. **Limit collection size**: Only index necessary entities
2. **Increase storage**: Use disk-backed storage vs in-memory
3. **Restart ChromaDB**: Periodic restarts can help
4. **Upgrade hardware**: Consider more RAM for large entity sets (1000+)

## Advanced Configuration

### Multiple Collections

You can use multiple collections for different purposes:

- `home_entities` - Main entity database
- `home_agent_memories` - Long-term memory storage (automatic)
- `custom_data` - User-defined data

**Switching collections**:
Change the **Collection Name** in Vector DB settings and re-index.

### Additional Collections for Context Enhancement

Home Agent supports querying additional ChromaDB collections alongside the main entity collection. This allows you to inject supplementary context (documentation, knowledge bases, etc.) into prompts without mixing it with entity data.

#### How It Works

When additional collections are configured, Home Agent uses a **two-tier ranking system**:

1. **Entity Collection (Priority)**: Always queries the main entity collection first
   - Returns top K entities based on `CONF_TOP_K` setting
   - Filtered by `L2_DISTANCE_THRESHOLD`
   - These results always appear first in context

2. **Additional Collections (Supplementary)**: Queries all specified additional collections
   - Results from all additional collections are merged and ranked together
   - Returns top K results from the merged pool based on `CONF_ADDITIONAL_TOP_K`
   - Filtered by `CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD`
   - These results appear after entity context

#### Configuration

Additional collections can be configured through the Vector DB settings:

| Field | Description | Default |
|-------|-------------|---------|
| **Additional Collections** | Comma-separated list of collection names | (empty) |
| **Additional Top K** | Number of results from additional collections | `5` |
| **Additional Similarity Threshold** | L2 distance threshold for additional collections | `250.0` |

**Example Configuration**:
```
Additional Collections: company_docs,project_notes,faq_data
Additional Top K: 5
Additional Similarity Threshold: 250.0
```

#### Context Format

Additional collection results are injected into the prompt with a clear header:

```
[Entity embeddings results - CSV table format]

### RELEVANT ADDITIONAL CONTEXT FOR ANSWERING QUESTIONS, NOT CONTROL ###
[Additional collections results - JSON with metadata]
```

This format ensures the LLM understands that additional context is informational only and not related to Home Assistant control.

#### Use Cases

**Knowledge Base Integration**:
- Store product manuals, documentation, or FAQs in a separate collection
- Agent can reference this information when answering questions
- Example: "How do I reset my Philips Hue bulb?"

**Custom Context**:
- Add domain-specific knowledge relevant to your home
- Store information about devices, routines, or preferences
- Example: Device installation dates, warranty information, custom procedures

**Multi-Source Information**:
- Combine multiple knowledge sources (docs, notes, logs)
- Each collection can serve a different purpose
- Agent retrieves most relevant information from all sources

#### Error Handling

- If a specified collection doesn't exist, it will be skipped silently
- A warning will be logged: `Collection 'xyz' not found, skipping`
- Other collections will continue to be queried normally
- Empty additional collections list disables this feature

#### Best Practices

1. **Keep collections focused**: Each collection should serve a specific purpose
2. **Use consistent embeddings**: All collections should use the same embedding model
3. **Monitor context size**: Additional context consumes tokens, adjust Top K accordingly
4. **Separate thresholds**: Use different similarity thresholds for different content types
5. **Test queries**: Verify relevant information is being retrieved from additional collections

### Custom Embedding Models

For Ollama, you can use any embedding model:

```bash
# List available embedding models
ollama list | grep embed

# Pull a specific model
ollama pull mxbai-embed-large
```

Update Home Agent configuration:
- **Embedding Model**: `mxbai-embed-large`

### Backup and Restore

#### Backup ChromaDB Data

**Docker**:
```bash
docker exec chromadb tar -czf /tmp/chromadb-backup.tar.gz /chroma/chroma
docker cp chromadb:/tmp/chromadb-backup.tar.gz ./chromadb-backup.tar.gz
```

**Standalone**:
```bash
tar -czf chromadb-backup.tar.gz /opt/chromadb/data
```

#### Restore ChromaDB Data

**Docker**:
```bash
docker stop chromadb
docker cp chromadb-backup.tar.gz chromadb:/tmp/
docker exec chromadb tar -xzf /tmp/chromadb-backup.tar.gz -C /
docker start chromadb
```

**Standalone**:
```bash
systemctl stop chromadb
tar -xzf chromadb-backup.tar.gz -C /
systemctl start chromadb
```

## Next Steps

After setting up Vector DB mode:

1. **Enable Memory System**
   - See [Memory System Guide](MEMORY_SYSTEM.md)
   - Requires ChromaDB (already configured)
   - Adds long-term memory for facts and preferences

2. **Monitor Performance**
   - Enable debug logging to see query details
   - Monitor events: `home_agent.vector_db.queried`
   - Check token usage in `home_agent.conversation.finished` events

3. **Optimize Configuration**
   - Fine-tune Top-K based on typical queries
   - Adjust similarity threshold for better relevance
   - Consider switching embedding providers based on cost/performance

## Additional Resources

- [Installation Guide](INSTALLATION.md) - Initial setup
- [Memory System Guide](MEMORY_SYSTEM.md) - Long-term memory features
- [Project Specification](PROJECT_SPEC.md) - Technical architecture
- [ChromaDB Documentation](https://docs.trychroma.com/) - Official ChromaDB docs

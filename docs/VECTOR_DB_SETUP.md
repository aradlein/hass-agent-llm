# Vector Database Setup - Quick Start

## What is Vector DB Mode?

Vector DB mode uses semantic similarity search to automatically find the most relevant entities for each query. Instead of manually specifying which entities to include, ChromaDB embeds entity states as vectors and retrieves only what's relevant to the user's question.

**How it works:** Query is embedded → ChromaDB finds similar entities → Only relevant entities sent to LLM

**Key benefits:**
- Efficient token usage (only relevant entities included)
- Scales well with large entity counts (100+)
- No manual entity list maintenance
- Required for memory system features

## When to Use It

**Use Vector DB Mode when:**
- You have 50+ entities and can't include them all
- Users ask diverse questions about different areas
- You want automatic entity relevance detection
- You need the long-term memory system

**Use Direct Mode when:**
- You have fewer than 20 entities
- You want specific entities always included
- You need minimal latency
- You prefer simpler setup

## Quick Setup

### 1. Install ChromaDB

**Docker (Recommended):**

Create storage directory:
```bash
mkdir -p /config/chromadb
```

Run ChromaDB:
```bash
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v /config/chromadb:/chroma/chroma \
  --restart unless-stopped \
  chromadb/chroma:latest
```

Verify it's running:
```bash
curl http://localhost:8000/api/v1/heartbeat
```

Expected response: `{"nanosecond heartbeat": ...}`

### 2. Set Up Embedding Provider

**Option A: Ollama (Local, Free)**

Install and pull model:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
```

**Option B: OpenAI (Cloud)**

Get an API key from [OpenAI Platform](https://platform.openai.com/)

Cost: ~$0.02 per 1M tokens (~$0.01-0.05/month typical usage)

### 3. Configure Home Agent

1. Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure**
2. Select **Vector DB Settings**
3. Configure connection:

| Field | Value (Ollama) | Value (OpenAI) |
|-------|----------------|----------------|
| Vector DB Host | `localhost` | `localhost` |
| Vector DB Port | `8000` | `8000` |
| Collection Name | `home_entities` | `home_entities` |
| Embedding Provider | `ollama` | `openai` |
| Embedding Base URL | `http://localhost:11434` | `https://api.openai.com/v1` |
| Embedding Model | `nomic-embed-text` | `text-embedding-3-small` |
| OpenAI API Key | (leave blank) | Your API key (sk-...) |
| Top K | `5` | `5` |
| Similarity Threshold | `250.0` | `250.0` |

4. Save configuration

### 4. Enable Vector DB Mode

1. Go to **Configure** > **Context Settings**
2. Set **Context Mode** to `vector_db`
3. Click **Submit**

## Index Your Entities

After enabling Vector DB mode, index your entities:

1. Open **Developer Tools** > **Services**
2. Select `home_agent.reindex_entities`
3. Execute (no parameters required)

Monitor progress in logs:
```
[home_agent.vector_db_manager] Indexing 127 entities...
[home_agent.vector_db_manager] Indexed 127 entities in 12.3s
```

**When to reindex:**
- After initial setup
- After adding new entities
- Weekly/monthly for large setups (optional)

## Verify It's Working

Test a query:

```yaml
service: home_agent.process
data:
  text: "What is the bedroom temperature?"
```

Check logs for entity retrieval:
```
[home_agent.vector_db_manager] Retrieved 3 entities for query (threshold=250.0)
```

The LLM should receive only relevant entities (e.g., bedroom temperature sensor) rather than all entities.

## Tuning Settings

### Top K (Number of Results)

Controls how many entities to retrieve:

- `3` - Very focused, minimal context
- `5` - Balanced (recommended)
- `10` - Comprehensive context
- `20+` - Very large context

**Recommendation:** Start with `5`, increase if LLM lacks context.

### Similarity Threshold (L2 Distance)

Controls minimum similarity required:

- `100.0` - Very strict, highly relevant only
- `250.0` - Balanced (recommended)
- `500.0` - Lenient, more entities included
- `1000.0` - Very lenient (testing/debugging)

**Recommendation:** Start with `250.0`, adjust based on results.

**Tips:**
- Too few entities? Increase threshold or Top K
- Irrelevant entities? Decrease threshold or Top K

## Troubleshooting

### ChromaDB Connection Failed

1. Verify ChromaDB is running:
   ```bash
   docker ps | grep chromadb
   curl http://localhost:8000/api/v1/heartbeat
   ```

2. Check logs:
   ```bash
   docker logs chromadb
   ```

3. Verify port 8000 is accessible

### Ollama Embedding Errors

1. Verify Ollama is running:
   ```bash
   ollama list
   ```

2. Pull embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

3. Test embedding:
   ```bash
   ollama embed nomic-embed-text "test query"
   ```

### No Entities Retrieved

1. Verify entities are indexed:
   - Re-run `home_agent.reindex_entities`

2. Increase Top K or similarity threshold

3. Check logs for query details

### Slow Queries

1. Reduce Top K value
2. Switch from OpenAI to Ollama embeddings (local)
3. Ensure ChromaDB has adequate RAM/CPU
4. Use SSD storage for ChromaDB

## Privacy Note

All vector data is stored locally on your Home Assistant instance. No entity data is sent to external services except:
- Embedding generation (if using OpenAI embeddings)
- LLM queries with retrieved entity context (standard Home Agent operation)

Use local Ollama embeddings for complete privacy.

## Need More Details?

See the [Complete Vector DB Reference](reference/VECTOR_DB_SETUP.md) for:
- Detailed configuration options
- Advanced tuning strategies
- Backup and restore procedures
- Performance optimization
- Multiple collection management
- Custom embedding models

# Testing Quick Start Guide

## Overview

This guide provides a quick reference for setting up and running integration and E2E tests for Home Agent.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- ChromaDB and Ollama available (via Docker)

## Quick Setup

### 1. Start Test Infrastructure

```bash
# Start ChromaDB and Ollama
docker-compose -f tests/docker-compose.test.yml up -d

# Wait for services to be healthy (automatic health checks)
# Initialize Ollama models (one-time setup)
docker-compose -f tests/docker-compose.test.yml run --rm ollama-init
```

### 2. Configure Environment

```bash
# Copy test environment template
cp .env.test.example .env.test

# Edit if needed (defaults work for Docker Compose setup)
# CHROMADB_HOST=localhost
# CHROMADB_PORT=8000
# OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Run Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run all E2E tests
pytest tests/e2e/ -v -m e2e

# Run smoke tests (quick validation)
pytest tests/e2e/ -v -m smoke

# Run with coverage
pytest tests/integration/ --cov=custom_components/home_agent --cov-report=html
```

### 4. Cleanup

```bash
# Stop and remove containers
docker-compose -f tests/docker-compose.test.yml down -v
```

## Test Categories

### Unit Tests (Existing)
- **Location**: `tests/unit/`
- **Purpose**: Test individual components in isolation with mocks
- **Run**: `pytest tests/unit/ -v`
- **Speed**: < 1s per test
- **Coverage**: ~75%

### Integration Tests (New)
- **Location**: `tests/integration/`
- **Purpose**: Test components with real ChromaDB and Ollama
- **Run**: `pytest tests/integration/ -v -m integration`
- **Speed**: 1-10s per test
- **Coverage Target**: 60%+

### E2E Tests (New)
- **Location**: `tests/e2e/`
- **Purpose**: Test complete workflows end-to-end
- **Run**: `pytest tests/e2e/ -v -m e2e`
- **Speed**: 10-60s per test
- **Coverage Target**: 40%+

## Common Test Scenarios

### Test Real Vector DB Search

```python
@pytest.mark.integration
async def test_semantic_search(vector_manager, ollama_config):
    """Test semantic search with real embeddings."""
    # Index entities
    await vector_manager.async_index_entity("light.living_room")

    # Search
    results = await vector_manager.search("lights in the living room")

    # Verify
    assert len(results) > 0
    assert "light.living_room" in [r["entity_id"] for r in results]
```

### Test Real LLM Conversation

```python
@pytest.mark.integration
async def test_conversation_flow(agent):
    """Test conversation with real Ollama."""
    response = await agent.process_message(
        "What lights are currently on?",
        conversation_id="test"
    )

    assert response is not None
    assert len(response) > 0
```

### Test E2E Scenario

```python
@pytest.mark.e2e
async def test_complete_flow(executor):
    """Test complete conversation flow."""
    scenario = await executor.load_scenario("scenarios/basic_conversation.yaml")
    result = await executor.execute_scenario(scenario)

    assert result["success"]
    assert result["metrics"]["total_duration_ms"] < 5000
```

## Test Markers

Use pytest markers to selectively run tests:

```bash
# Integration tests only
pytest -v -m integration

# E2E tests only
pytest -v -m e2e

# Smoke tests only (quick validation)
pytest -v -m smoke

# Performance tests
pytest -v -m performance

# Exclude slow tests
pytest -v -m "not slow"

# Run integration AND e2e
pytest -v -m "integration or e2e"
```

## Debugging Tests

### View Logs

```bash
# Run with detailed logging
pytest tests/integration/ -v --log-cli-level=DEBUG

# Run single test with full output
pytest tests/integration/test_real_vector_db.py::test_semantic_search -v -s
```

### Check Service Health

```bash
# ChromaDB health
curl http://localhost:8000/api/v1/heartbeat

# Ollama health
curl http://localhost:11434/api/tags

# View ChromaDB collections
curl http://localhost:8000/api/v1/collections

# View Ollama models
curl http://localhost:11434/api/tags
```

### Inspect Docker Logs

```bash
# ChromaDB logs
docker-compose -f tests/docker-compose.test.yml logs chromadb

# Ollama logs
docker-compose -f tests/docker-compose.test.yml logs ollama
```

## CI/CD Integration

### GitHub Actions

The integration tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests

```yaml
# .github/workflows/integration-tests.yml
- name: Run integration tests
  run: pytest tests/integration/ -v --cov=custom_components/home_agent
```

### Pre-commit Hooks

Integration tests can run on `pre-push`:

```yaml
# .pre-commit-config.yaml
- id: integration-tests
  entry: scripts/run_integration_tests.sh
  stages: [pre-push]
```

## Performance Benchmarks

Target performance baselines:

| Metric | Target | Test |
|--------|--------|------|
| Simple query (p95) | < 2s | `test_latency_baseline` |
| Tool execution (p95) | < 5s | `test_tool_latency` |
| Memory extraction | < 3s | `test_memory_extraction_time` |
| Concurrent conversations | 5+ | `test_concurrent_conversations` |

## Troubleshooting

### ChromaDB Connection Failed

```bash
# Check if ChromaDB is running
docker ps | grep chromadb

# Restart ChromaDB
docker-compose -f tests/docker-compose.test.yml restart chromadb

# Check logs
docker-compose -f tests/docker-compose.test.yml logs chromadb
```

### Ollama Model Not Found

```bash
# Pull models manually
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull mxbai-embed-large

# Verify models
docker exec ollama ollama list
```

### Tests Timing Out

```bash
# Increase timeout in pytest.ini
[tool.pytest.ini_options]
timeout = 300  # 5 minutes

# Or set per-test
@pytest.mark.timeout(60)
async def test_slow_operation(...):
    ...
```

### Out of Memory

```bash
# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory

# Or reduce concurrent tests
pytest -n 1  # Single worker
```

## Next Steps

1. Review full testing strategy: `docs/TESTING_STRATEGY.md`
2. Implement Phase 1: Set up test infrastructure
3. Write first integration tests
4. Create E2E scenarios
5. Integrate with CI/CD

## Resources

- Full Testing Strategy: `docs/TESTING_STRATEGY.md`
- Test Fixtures: `tests/conftest.py`
- Integration Tests: `tests/integration/`
- E2E Tests: `tests/e2e/`
- Docker Compose: `tests/docker-compose.test.yml`

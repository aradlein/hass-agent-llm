# Home Agent Testing Strategy

## Executive Summary

This document provides a comprehensive testing strategy for the Home Agent integration, with focus on integration tests and end-to-end (E2E) tests that ensure credible releases. The strategy assumes availability of ChromaDB and Ollama backend infrastructure.

**Current State**: Strong unit test coverage (~17 files, 10,436 lines) with mocked dependencies. Limited integration tests (4 files) that still use mocks.

**Gap**: Missing real-world integration tests and E2E tests that validate actual behavior against live services.

---

## Table of Contents

1. [Current Test Coverage Analysis](#current-test-coverage-analysis)
2. [Testing Gaps Identified](#testing-gaps-identified)
3. [Proposed Test Architecture](#proposed-test-architecture)
4. [Integration Test Harness Design](#integration-test-harness-design)
5. [E2E Test Harness Design](#e2e-test-harness-design)
6. [Test Infrastructure Setup](#test-infrastructure-setup)
7. [Specific Test Cases Required](#specific-test-cases-required)
8. [CI/CD Integration](#cicd-integration)
9. [Success Criteria](#success-criteria)

---

## Current Test Coverage Analysis

### ✅ Well-Tested Areas (Unit Tests)

| Component | Coverage | Test File | Quality |
|-----------|----------|-----------|---------|
| Vector DB Manager | Good | `test_vector_db_manager.py` | Mocked ChromaDB |
| Memory Manager | Excellent | `test_memory_manager.py` | Full CRUD operations |
| Memory Extraction | Excellent | `test_memory_extraction.py` | LLM extraction logic |
| Conversation History | Good | `test_conversation.py` | Token limits, persistence |
| Context Management | Good | `test_context_manager.py` | Context injection |
| Tool Execution | Good | `test_tool_handler.py` | Tool calling loop |
| Streaming | Good | `test_streaming.py`, `test_agent_streaming.py` | SSE transformation |
| Context Providers | Good | `test_context_providers/` | All 3 strategies |
| Tools | Excellent | `test_tools/` | 8 different tool types |

### ⚠️ Partially Tested Areas (Integration Tests with Mocks)

| Test File | What It Tests | Limitation |
|-----------|---------------|------------|
| `test_phase2_vector_db.py` | Vector DB flow | Uses mocked ChromaDB client |
| `test_phase3_custom_tools.py` | Custom REST/service tools | No real HTTP calls |
| `test_phase3_external_llm.py` | Multi-LLM delegation | Mocked LLM responses |
| `test_phase4_streaming.py` | Streaming integration | Mocked SSE streams |

### ❌ Missing Test Coverage

1. **Real Service Integration**
   - No tests against actual ChromaDB instance
   - No tests against real Ollama backend
   - No tests with real embedding generation
   - No tests with real LLM inference

2. **End-to-End Workflows**
   - No full conversation flows with real services
   - No multi-turn conversations with tool calling
   - No tests of streaming with real LLM responses

3. **Performance & Load Testing**
   - No latency benchmarks
   - No concurrent conversation tests
   - No memory usage profiling
   - No token limit stress tests

4. **Error Recovery & Resilience**
   - No service unavailability tests
   - No network failure recovery tests
   - No partial failure handling tests
   - No rate limiting tests

5. **Service Integration**
   - No tests of Home Assistant service calls
   - No tests of service discovery
   - No tests of config flow UI
   - No tests of service registration/cleanup

6. **Data Integrity**
   - No tests of conversation persistence across restarts
   - No tests of memory deduplication
   - No tests of vector DB index consistency
   - No tests of concurrent memory access

---

## Testing Gaps Identified

### Critical Gaps (P0)

1. **Real Vector DB Integration**
   - **Impact**: Cannot verify semantic search works correctly
   - **Risk**: Production issues with entity retrieval
   - **Test Needed**: Real ChromaDB queries with actual embeddings

2. **Real LLM Integration**
   - **Impact**: Cannot verify tool calling works with real model responses
   - **Risk**: Tool execution failures in production
   - **Test Needed**: Full conversations with Ollama

3. **Streaming E2E**
   - **Impact**: Cannot verify streaming works end-to-end
   - **Risk**: Streaming failures in production
   - **Test Needed**: Real streaming responses with SSE

### High-Priority Gaps (P1)

4. **Error Recovery**
   - **Impact**: Unknown behavior when services fail
   - **Risk**: Poor user experience during outages
   - **Test Needed**: Service failure scenarios

5. **Concurrent Access**
   - **Impact**: Race conditions with multiple conversations
   - **Risk**: Data corruption or deadlocks
   - **Test Needed**: Multi-conversation concurrency tests

6. **Performance Baselines**
   - **Impact**: No performance regression detection
   - **Risk**: Slow releases with degraded performance
   - **Test Needed**: Latency and throughput benchmarks

### Medium-Priority Gaps (P2)

7. **Home Assistant Service Integration**
   - **Impact**: Service calls may not work correctly
   - **Risk**: Integration issues with HA ecosystem
   - **Test Needed**: Real service call execution

8. **Configuration Flow Testing**
   - **Impact**: UI configuration may break
   - **Risk**: Users cannot configure integration
   - **Test Needed**: Config flow validation

9. **Memory Lifecycle**
   - **Impact**: Memory management issues over time
   - **Risk**: Memory leaks or TTL bugs
   - **Test Needed**: Long-running memory tests

---

## Proposed Test Architecture

### Test Pyramid

```
                    ┌─────────────┐
                    │  E2E Tests  │  ← New: 10-15 tests
                    │  (Real Svcs)│
                    └─────────────┘
                  ┌─────────────────┐
                  │ Integration Tests│ ← New: 30-40 tests
                  │  (Real ChromaDB, │
                  │   Real Ollama)   │
                  └─────────────────┘
            ┌─────────────────────────┐
            │    Unit Tests (Mocked)   │ ← Existing: ~17 files
            │      Good Coverage       │
            └─────────────────────────┘
```

### Test Categories

| Category | Description | Test Environment | Duration |
|----------|-------------|-----------------|----------|
| **Unit** | Component isolation with mocks | No external deps | < 1s per test |
| **Integration** | Real services, single feature | ChromaDB + Ollama | 1-10s per test |
| **E2E** | Full workflows, all components | Full stack | 10-60s per test |
| **Performance** | Latency, throughput, load | Full stack | 30-300s per test |
| **Smoke** | Quick validation of core paths | Full stack | < 30s total |

---

## Integration Test Harness Design

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│            Integration Test Harness                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Test Fixtures│  │ Test Helpers │  │  Assertions  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Service Management Layer                  │  │
│  │  - Start/Stop ChromaDB                           │  │
│  │  - Start/Stop Ollama                             │  │
│  │  - Health Checks                                 │  │
│  │  - Cleanup                                       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Home Assistant Test Environment          │  │
│  │  - Mock HA Core                                  │  │
│  │  - Real State Management                         │  │
│  │  - Real Service Calls                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ ChromaDB │        │  Ollama  │        │ Home     │
   │ (Real)   │        │  (Real)  │        │ Assistant│
   └──────────┘        └──────────┘        └──────────┘
```

### Implementation Guide

#### 1. Test Fixtures (`tests/integration/conftest.py`)

```python
"""Integration test fixtures with real services."""
import asyncio
import httpx
import pytest
from chromadb import HttpClient
from homeassistant.core import HomeAssistant

@pytest.fixture(scope="session")
def chromadb_config():
    """ChromaDB configuration from environment."""
    return {
        "host": os.getenv("CHROMADB_HOST", "localhost"),
        "port": int(os.getenv("CHROMADB_PORT", "8000")),
    }

@pytest.fixture(scope="session")
def ollama_config():
    """Ollama configuration from environment."""
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"),
    }

@pytest.fixture(scope="session")
async def chromadb_client(chromadb_config):
    """Create real ChromaDB client with health check."""
    client = HttpClient(
        host=chromadb_config["host"],
        port=chromadb_config["port"]
    )

    # Health check
    try:
        client.heartbeat()
    except Exception as e:
        pytest.skip(f"ChromaDB not available: {e}")

    yield client

    # Cleanup: delete test collections
    for collection in client.list_collections():
        if collection.name.startswith("test_"):
            client.delete_collection(collection.name)

@pytest.fixture(scope="session")
async def ollama_client(ollama_config):
    """Create Ollama client with health check."""
    async with httpx.AsyncClient() as client:
        # Health check
        try:
            response = await client.get(f"{ollama_config['base_url']}/api/tags")
            response.raise_for_status()
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    yield ollama_config

    # No cleanup needed for Ollama

@pytest.fixture
async def test_home_assistant():
    """Create test Home Assistant instance."""
    from homeassistant import core
    from homeassistant.helpers import entity_registry as er

    hass = core.HomeAssistant()
    await hass.async_start()

    # Register test entities
    registry = er.async_get(hass)
    # ... register test entities

    yield hass

    await hass.async_stop()

@pytest.fixture
async def home_agent_instance(test_home_assistant, chromadb_config, ollama_config):
    """Create HomeAgent with real services."""
    from custom_components.home_agent.agent import HomeAgent

    config = {
        "llm": {
            "base_url": ollama_config["base_url"],
            "model": ollama_config["model"],
            "api_key": "ollama",  # Ollama doesn't need real key
        },
        "context": {
            "mode": "vector_db",
        },
        "vector_db": chromadb_config,
        # ... rest of config
    }

    agent = HomeAgent(test_home_assistant, config)

    yield agent

    await agent.close()

@pytest.fixture
def test_collection_name():
    """Generate unique collection name for test isolation."""
    import uuid
    return f"test_{uuid.uuid4().hex[:8]}"
```

#### 2. Service Health Checks (`tests/integration/health.py`)

```python
"""Service health check utilities."""
import asyncio
import httpx
from chromadb import HttpClient

async def wait_for_chromadb(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for ChromaDB to be ready."""
    start = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - start) < timeout:
        try:
            client = HttpClient(host=host, port=port)
            client.heartbeat()
            return True
        except Exception:
            await asyncio.sleep(1)

    return False

async def wait_for_ollama(base_url: str, timeout: int = 30) -> bool:
    """Wait for Ollama to be ready."""
    start = asyncio.get_event_loop().time()

    async with httpx.AsyncClient() as client:
        while (asyncio.get_event_loop().time() - start) < timeout:
            try:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(1)

    return False

async def check_ollama_model_available(base_url: str, model: str) -> bool:
    """Check if specific model is available in Ollama."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/api/tags")
        models = response.json().get("models", [])
        return any(m["name"] == model for m in models)
```

#### 3. Test Helpers (`tests/integration/helpers.py`)

```python
"""Integration test helper functions."""
from typing import Any, Dict, List
import json

async def index_test_entities(vector_manager, entities: List[Dict[str, Any]]):
    """Index test entities into ChromaDB."""
    for entity in entities:
        await vector_manager.async_index_entity(entity["entity_id"])

async def verify_semantic_search(
    vector_manager,
    query: str,
    expected_entities: List[str],
    threshold: float = 0.8
):
    """Verify semantic search returns expected entities."""
    results = await vector_manager.search(query)

    # Check that expected entities are in results
    result_ids = [r["entity_id"] for r in results]
    for entity_id in expected_entities:
        assert entity_id in result_ids, f"Expected {entity_id} in results"

async def send_message_and_wait(
    agent,
    message: str,
    conversation_id: str = "test",
    timeout: float = 10.0
) -> str:
    """Send message and wait for response with timeout."""
    import asyncio

    response = await asyncio.wait_for(
        agent.process_message(message, conversation_id),
        timeout=timeout
    )

    return response

def assert_tool_called(agent, tool_name: str, min_calls: int = 1):
    """Assert that a specific tool was called."""
    metrics = agent.tool_handler.get_metrics()
    calls = metrics.get(f"{tool_name}_executions", 0)
    assert calls >= min_calls, f"Expected {tool_name} called at least {min_calls} times, got {calls}"

def assert_response_contains(response: str, expected: str):
    """Assert response contains expected text."""
    assert expected.lower() in response.lower(), f"Expected '{expected}' in response"
```

#### 4. Custom Assertions (`tests/integration/assertions.py`)

```python
"""Custom assertions for integration tests."""

def assert_valid_embeddings(embeddings: List[float], expected_dim: int = 1024):
    """Assert embeddings are valid."""
    assert len(embeddings) == expected_dim, f"Expected {expected_dim} dimensions"
    assert all(isinstance(x, float) for x in embeddings), "All values must be floats"
    # Check normalized (L2 norm ~ 1.0 for some models)
    import math
    norm = math.sqrt(sum(x*x for x in embeddings))
    assert norm > 0, "Embeddings should be non-zero"

def assert_memory_stored(memory_manager, content_substring: str):
    """Assert memory was stored with specific content."""
    memories = await memory_manager.list_all_memories()
    assert any(
        content_substring.lower() in m["content"].lower()
        for m in memories
    ), f"No memory found with content: {content_substring}"

def assert_conversation_history_length(agent, conversation_id: str, expected_length: int):
    """Assert conversation history has expected length."""
    history = agent.conversation_manager.get_history(conversation_id)
    assert len(history) == expected_length, f"Expected {expected_length} messages, got {len(history)}"

def assert_llm_response_valid(response: Dict[str, Any]):
    """Assert LLM response has valid structure."""
    assert "choices" in response, "Response missing 'choices'"
    assert len(response["choices"]) > 0, "No choices in response"

    choice = response["choices"][0]
    assert "message" in choice, "Choice missing 'message'"

    message = choice["message"]
    assert "role" in message, "Message missing 'role'"
    assert message["role"] in ["assistant", "user"], f"Invalid role: {message['role']}"

def assert_chromadb_collection_exists(client, collection_name: str):
    """Assert ChromaDB collection exists."""
    collections = [c.name for c in client.list_collections()]
    assert collection_name in collections, f"Collection {collection_name} not found"

def assert_chromadb_collection_count(client, collection_name: str, min_count: int):
    """Assert ChromaDB collection has minimum number of documents."""
    collection = client.get_collection(collection_name)
    count = collection.count()
    assert count >= min_count, f"Expected at least {min_count} documents, got {count}"
```

---

## E2E Test Harness Design

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              E2E Test Harness                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Scenario Engine                          │  │
│  │  - Load scenario definitions                     │  │
│  │  - Execute conversation flows                    │  │
│  │  - Validate outcomes                             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         State Verification                        │  │
│  │  - Entity states                                 │  │
│  │  - Memory states                                 │  │
│  │  - Conversation history                          │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Metrics Collection                        │  │
│  │  - Latency tracking                              │  │
│  │  - Token usage                                   │  │
│  │  - Tool call counts                              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Implementation Guide

#### 1. Scenario Definitions (`tests/e2e/scenarios/`)

```yaml
# tests/e2e/scenarios/basic_conversation.yaml
name: "Basic Conversation Flow"
description: "Test simple question-answer without tool calls"

setup:
  entities:
    - entity_id: "light.living_room"
      state: "on"
      attributes:
        brightness: 255

steps:
  - user: "What lights are on?"
    expected_response_contains: "living room"
    expected_tool_calls: ["ha_query"]
    timeout: 10

  - user: "Turn off the living room light"
    expected_response_contains: "turned off"
    expected_tool_calls: ["ha_control"]
    timeout: 10
    verify:
      - entity_id: "light.living_room"
        state: "off"

assertions:
  - conversation_history_length: 4  # 2 user + 2 assistant
  - total_duration_ms: { max: 5000 }
  - total_tokens: { max: 1000 }
```

```yaml
# tests/e2e/scenarios/memory_extraction.yaml
name: "Memory Extraction Flow"
description: "Test memory extraction from conversation"

setup:
  config:
    memory_enabled: true
    memory_extraction_enabled: true

steps:
  - user: "I prefer the bedroom temperature at 68 degrees for sleeping"
    expected_response_contains: ["noted", "remember"]
    timeout: 15

  - user: "What temperature do I like in the bedroom?"
    expected_response_contains: ["68", "degrees"]
    expected_tool_calls: ["recall_memory"]
    timeout: 15

assertions:
  - memory_count: { min: 1 }
  - memory_content_contains: "bedroom"
  - memory_content_contains: "68"
```

#### 2. Scenario Executor (`tests/e2e/executor.py`)

```python
"""E2E scenario executor."""
import yaml
from typing import Any, Dict, List
from pathlib import Path

class ScenarioExecutor:
    """Executes E2E test scenarios."""

    def __init__(self, hass, agent, vector_manager=None, memory_manager=None):
        self.hass = hass
        self.agent = agent
        self.vector_manager = vector_manager
        self.memory_manager = memory_manager
        self.metrics = {}

    async def load_scenario(self, scenario_file: Path) -> Dict[str, Any]:
        """Load scenario from YAML file."""
        with open(scenario_file) as f:
            return yaml.safe_load(f)

    async def execute_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete scenario."""
        # Setup
        await self._setup_scenario(scenario.get("setup", {}))

        # Execute steps
        conversation_id = f"e2e_{scenario['name']}"
        results = []

        for step in scenario["steps"]:
            result = await self._execute_step(step, conversation_id)
            results.append(result)

        # Verify assertions
        await self._verify_assertions(scenario.get("assertions", []), conversation_id)

        return {
            "scenario": scenario["name"],
            "results": results,
            "metrics": self.metrics,
        }

    async def _setup_scenario(self, setup: Dict[str, Any]):
        """Setup scenario environment."""
        # Create test entities
        for entity_config in setup.get("entities", []):
            # Set entity state in Home Assistant
            self.hass.states.async_set(
                entity_config["entity_id"],
                entity_config["state"],
                entity_config.get("attributes", {})
            )

        # Apply config overrides
        config_overrides = setup.get("config", {})
        if config_overrides:
            await self.agent.update_config(config_overrides)

    async def _execute_step(self, step: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
        """Execute a single conversation step."""
        import time

        start = time.time()

        # Send user message
        response = await self.agent.process_message(
            text=step["user"],
            conversation_id=conversation_id,
            timeout=step.get("timeout", 10)
        )

        duration_ms = int((time.time() - start) * 1000)

        # Verify response
        expected_contains = step.get("expected_response_contains", [])
        if isinstance(expected_contains, str):
            expected_contains = [expected_contains]

        for expected in expected_contains:
            assert expected.lower() in response.lower(), \
                f"Expected '{expected}' in response: {response}"

        # Verify tool calls
        expected_tools = step.get("expected_tool_calls", [])
        for tool_name in expected_tools:
            assert_tool_called(self.agent, tool_name)

        # Verify entity states
        for verification in step.get("verify", []):
            entity_id = verification["entity_id"]
            expected_state = verification["state"]

            state = self.hass.states.get(entity_id)
            assert state.state == expected_state, \
                f"Expected {entity_id} to be {expected_state}, got {state.state}"

        return {
            "user": step["user"],
            "response": response,
            "duration_ms": duration_ms,
        }

    async def _verify_assertions(self, assertions: List[Dict[str, Any]], conversation_id: str):
        """Verify scenario-level assertions."""
        for assertion in assertions:
            if "conversation_history_length" in assertion:
                expected_length = assertion["conversation_history_length"]
                assert_conversation_history_length(self.agent, conversation_id, expected_length)

            if "memory_count" in assertion:
                memories = await self.memory_manager.list_all_memories()
                min_count = assertion["memory_count"].get("min", 0)
                assert len(memories) >= min_count, f"Expected at least {min_count} memories"

            if "memory_content_contains" in assertion:
                assert_memory_stored(self.memory_manager, assertion["memory_content_contains"])

            # Add more assertion types as needed
```

#### 3. Metrics Collector (`tests/e2e/metrics.py`)

```python
"""E2E metrics collection."""
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class ConversationMetrics:
    """Metrics for a conversation."""
    conversation_id: str
    total_turns: int = 0
    total_duration_ms: int = 0
    total_tokens: int = 0
    tool_calls: Dict[str, int] = field(default_factory=dict)
    llm_latency_ms: List[int] = field(default_factory=list)
    context_latency_ms: List[int] = field(default_factory=list)

    def add_turn(self, duration_ms: int, tokens: int, tools_used: List[str]):
        """Add metrics for a conversation turn."""
        self.total_turns += 1
        self.total_duration_ms += duration_ms
        self.total_tokens += tokens

        for tool in tools_used:
            self.tool_calls[tool] = self.tool_calls.get(tool, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.total_duration_ms // max(self.total_turns, 1),
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens // max(self.total_turns, 1),
            "tool_calls": self.tool_calls,
            "avg_llm_latency_ms": sum(self.llm_latency_ms) // max(len(self.llm_latency_ms), 1),
        }

class MetricsCollector:
    """Collects metrics across E2E tests."""

    def __init__(self):
        self.conversations: Dict[str, ConversationMetrics] = {}

    def start_conversation(self, conversation_id: str):
        """Start tracking a new conversation."""
        self.conversations[conversation_id] = ConversationMetrics(conversation_id)

    def record_turn(self, conversation_id: str, duration_ms: int, tokens: int, tools: List[str]):
        """Record metrics for a conversation turn."""
        if conversation_id not in self.conversations:
            self.start_conversation(conversation_id)

        self.conversations[conversation_id].add_turn(duration_ms, tokens, tools)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        all_durations = []
        all_tokens = []

        for conv in self.conversations.values():
            all_durations.append(conv.total_duration_ms)
            all_tokens.append(conv.total_tokens)

        return {
            "total_conversations": len(self.conversations),
            "avg_conversation_duration_ms": sum(all_durations) // max(len(all_durations), 1),
            "avg_tokens_per_conversation": sum(all_tokens) // max(len(all_tokens), 1),
            "conversations": [c.to_dict() for c in self.conversations.values()],
        }
```

---

## Test Infrastructure Setup

### Docker Compose Configuration

Create `tests/docker-compose.test.yml`:

```yaml
version: '3.8'

services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    environment:
      - ANONYMIZED_TELEMETRY=False
    volumes:
      - chromadb_data:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 10s
      timeout: 5s
      retries: 5

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Model initialization service
  ollama-init:
    image: ollama/ollama:latest
    depends_on:
      ollama:
        condition: service_healthy
    volumes:
      - ollama_data:/root/.ollama
    entrypoint: /bin/sh
    command: >
      -c "
      ollama pull llama3.2 &&
      ollama pull mxbai-embed-large &&
      echo 'Models pulled successfully'
      "

volumes:
  chromadb_data:
  ollama_data:
```

### Test Environment Variables

Create `.env.test`:

```bash
# ChromaDB Configuration
CHROMADB_HOST=localhost
CHROMADB_PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large

# Test Configuration
TEST_TIMEOUT=300
TEST_PARALLEL_CONVERSATIONS=5
TEST_ENABLE_PERFORMANCE=true
TEST_ENABLE_LOAD=false
```

### Test Runner Script

Create `scripts/run_integration_tests.sh`:

```bash
#!/bin/bash
set -e

echo "Starting test infrastructure..."
docker-compose -f tests/docker-compose.test.yml up -d

echo "Waiting for services to be healthy..."
docker-compose -f tests/docker-compose.test.yml exec -T chromadb curl -f http://localhost:8000/api/v1/heartbeat || sleep 10
docker-compose -f tests/docker-compose.test.yml exec -T ollama curl -f http://localhost:11434/api/tags || sleep 10

echo "Initializing Ollama models..."
docker-compose -f tests/docker-compose.test.yml run --rm ollama-init

echo "Running integration tests..."
pytest tests/integration/ -v --tb=short --log-cli-level=INFO

echo "Running E2E tests..."
pytest tests/e2e/ -v --tb=short --log-cli-level=INFO

echo "Cleaning up..."
docker-compose -f tests/docker-compose.test.yml down

echo "Tests complete!"
```

---

## Specific Test Cases Required

### Integration Tests

#### Vector DB Integration (`tests/integration/test_real_vector_db.py`)

```python
"""Integration tests with real ChromaDB."""

class TestRealVectorDBIntegration:
    """Test vector DB with real ChromaDB instance."""

    @pytest.mark.integration
    async def test_entity_indexing_real_embeddings(self, chromadb_client, ollama_config):
        """Test entity indexing with real embedding generation."""
        # Test that entities are indexed with real embeddings
        # Verify embedding dimensions are correct
        # Check that similar entities cluster together
        pass

    @pytest.mark.integration
    async def test_semantic_search_accuracy(self, vector_manager, ollama_config):
        """Test semantic search returns relevant entities."""
        # Index diverse set of entities
        # Query with natural language
        # Verify top results are semantically relevant
        pass

    @pytest.mark.integration
    async def test_l2_distance_threshold(self, vector_manager):
        """Test L2 distance filtering with real embeddings."""
        # Query with varying thresholds
        # Verify results respect threshold
        # Test edge cases (no results, all results)
        pass

    @pytest.mark.integration
    async def test_incremental_indexing(self, vector_manager):
        """Test incremental entity updates."""
        # Index entity
        # Update entity state
        # Verify index is updated
        # Check no duplicates
        pass

    @pytest.mark.integration
    async def test_collection_persistence(self, chromadb_client):
        """Test collection persists across restarts."""
        # Create collection and index entities
        # Restart ChromaDB
        # Verify collection still exists with same data
        pass

    @pytest.mark.integration
    async def test_multiple_collections(self, vector_manager):
        """Test querying multiple ChromaDB collections."""
        # Create multiple collections
        # Query across collections
        # Verify results from all collections
        pass

#### LLM Integration (`tests/integration/test_real_llm.py`)

```python
"""Integration tests with real Ollama backend."""

class TestRealLLMIntegration:
    """Test LLM integration with real Ollama."""

    @pytest.mark.integration
    async def test_basic_conversation(self, agent, ollama_config):
        """Test basic conversation with real LLM."""
        # Send simple query
        # Verify response is coherent
        # Check token usage is reported
        pass

    @pytest.mark.integration
    async def test_tool_calling(self, agent):
        """Test LLM triggers tool calls correctly."""
        # Send query requiring tool use
        # Verify correct tool is called
        # Verify tool arguments are correct
        # Check response incorporates tool result
        pass

    @pytest.mark.integration
    async def test_multi_turn_conversation(self, agent):
        """Test conversation context is maintained."""
        # Send multiple related messages
        # Verify agent remembers context
        # Check conversation history is used
        pass

    @pytest.mark.integration
    async def test_streaming_response(self, agent):
        """Test streaming with real LLM."""
        # Enable streaming
        # Send query
        # Verify SSE stream is received
        # Check complete response is correct
        pass

    @pytest.mark.integration
    async def test_context_injection(self, agent, vector_manager):
        """Test context is properly injected into LLM prompt."""
        # Index relevant entities
        # Send query about those entities
        # Verify LLM uses injected context
        # Check response is accurate
        pass

    @pytest.mark.integration
    async def test_external_llm_delegation(self, agent):
        """Test delegation to external LLM."""
        # Configure external LLM
        # Send complex query
        # Verify external LLM is called
        # Check response quality
        pass

#### Memory Integration (`tests/integration/test_real_memory.py`)

```python
"""Integration tests for memory system."""

class TestRealMemoryIntegration:
    """Test memory system with real vector DB."""

    @pytest.mark.integration
    async def test_memory_extraction_flow(self, agent, memory_manager):
        """Test automatic memory extraction."""
        # Have conversation with memorable content
        # Wait for extraction to complete
        # Verify memory is stored
        # Check memory quality filters work
        pass

    @pytest.mark.integration
    async def test_memory_recall(self, agent, memory_manager):
        """Test memory recall in conversation."""
        # Store memory
        # Have new conversation referencing memory
        # Verify memory is recalled and used
        pass

    @pytest.mark.integration
    async def test_memory_search_accuracy(self, memory_manager):
        """Test semantic memory search."""
        # Store diverse memories
        # Search with various queries
        # Verify relevant memories are returned
        pass

    @pytest.mark.integration
    async def test_memory_ttl_expiration(self, memory_manager):
        """Test memory TTL and expiration."""
        # Store memory with short TTL
        # Wait for expiration
        # Verify memory is removed
        pass

    @pytest.mark.integration
    async def test_memory_importance_filtering(self, memory_manager):
        """Test importance-based filtering."""
        # Store memories with varying importance
        # Query with importance threshold
        # Verify only high-importance memories returned
        pass

### E2E Tests (`tests/e2e/`)

```python
"""End-to-end tests for complete workflows."""

class TestE2EConversationFlows:
    """Test complete conversation flows."""

    @pytest.mark.e2e
    async def test_simple_query_flow(self, executor):
        """Test simple question-answer flow."""
        scenario = await executor.load_scenario("scenarios/simple_query.yaml")
        result = await executor.execute_scenario(scenario)
        assert result["success"]

    @pytest.mark.e2e
    async def test_tool_execution_flow(self, executor):
        """Test flow with tool execution."""
        scenario = await executor.load_scenario("scenarios/tool_execution.yaml")
        result = await executor.execute_scenario(scenario)
        assert result["success"]

    @pytest.mark.e2e
    async def test_multi_turn_flow(self, executor):
        """Test multi-turn conversation."""
        scenario = await executor.load_scenario("scenarios/multi_turn.yaml")
        result = await executor.execute_scenario(scenario)
        assert result["success"]

    @pytest.mark.e2e
    async def test_memory_flow(self, executor):
        """Test conversation with memory."""
        scenario = await executor.load_scenario("scenarios/memory_flow.yaml")
        result = await executor.execute_scenario(scenario)
        assert result["success"]

    @pytest.mark.e2e
    async def test_streaming_flow(self, executor):
        """Test streaming conversation."""
        scenario = await executor.load_scenario("scenarios/streaming_flow.yaml")
        result = await executor.execute_scenario(scenario)
        assert result["success"]

    @pytest.mark.e2e
    async def test_error_recovery_flow(self, executor):
        """Test error recovery."""
        scenario = await executor.load_scenario("scenarios/error_recovery.yaml")
        result = await executor.execute_scenario(scenario)
        assert result["success"]

class TestE2EPerformance:
    """Test performance characteristics."""

    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_latency_baseline(self, agent, metrics_collector):
        """Test response latency is within acceptable range."""
        # Send 10 queries
        # Measure latency for each
        # Verify p50, p95, p99 latencies
        # Assert < 2s for p95
        pass

    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_concurrent_conversations(self, agent):
        """Test concurrent conversation handling."""
        # Start 5 concurrent conversations
        # Send messages to each
        # Verify no interference
        # Check all complete successfully
        pass

    @pytest.mark.e2e
    @pytest.mark.load
    async def test_sustained_load(self, agent):
        """Test agent under sustained load."""
        # Run 100 conversations over 5 minutes
        # Monitor memory usage
        # Check for memory leaks
        # Verify performance doesn't degrade
        pass
```

### Error Recovery Tests (`tests/integration/test_error_recovery.py`)

```python
"""Integration tests for error recovery."""

class TestErrorRecovery:
    """Test error handling and recovery."""

    @pytest.mark.integration
    async def test_chromadb_unavailable(self, agent):
        """Test behavior when ChromaDB is unavailable."""
        # Stop ChromaDB
        # Send query
        # Verify graceful fallback
        # Check error is logged
        pass

    @pytest.mark.integration
    async def test_ollama_unavailable(self, agent):
        """Test behavior when Ollama is unavailable."""
        # Stop Ollama
        # Send query
        # Verify error handling
        # Check user-friendly error message
        pass

    @pytest.mark.integration
    async def test_ollama_timeout(self, agent):
        """Test handling of LLM timeout."""
        # Configure short timeout
        # Send query
        # Verify timeout is handled
        # Check retry logic (if any)
        pass

    @pytest.mark.integration
    async def test_tool_execution_failure(self, agent):
        """Test recovery from tool execution failure."""
        # Trigger tool that will fail
        # Verify error is returned to LLM
        # Check LLM can continue conversation
        pass

    @pytest.mark.integration
    async def test_chromadb_connection_recovery(self, agent, vector_manager):
        """Test ChromaDB connection recovery."""
        # Establish connection
        # Kill ChromaDB
        # Restart ChromaDB
        # Verify connection is re-established
        pass

### Service Integration Tests (`tests/integration/test_services.py`)

```python
"""Integration tests for Home Assistant services."""

class TestServiceIntegration:
    """Test Home Assistant service integration."""

    @pytest.mark.integration
    async def test_process_service(self, hass, agent):
        """Test home_agent.process service."""
        # Call service with message
        # Verify response
        # Check service_data is correct
        pass

    @pytest.mark.integration
    async def test_clear_history_service(self, hass, agent):
        """Test home_agent.clear_history service."""
        # Have conversation
        # Call clear_history
        # Verify history is cleared
        pass

    @pytest.mark.integration
    async def test_reindex_entities_service(self, hass, vector_manager):
        """Test home_agent.reindex_entities service."""
        # Call reindex_entities
        # Verify all exposed entities are indexed
        # Check return value has stats
        pass

    @pytest.mark.integration
    async def test_memory_services(self, hass, memory_manager):
        """Test memory management services."""
        # Test add_memory
        # Test list_memories
        # Test search_memories
        # Test delete_memory
        pass

---

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/integration-tests.yml`:

```yaml
name: Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  integration:
    runs-on: ubuntu-latest

    services:
      chromadb:
        image: chromadb/chroma:latest
        ports:
          - 8000:8000
        options: >-
          --health-cmd "curl -f http://localhost:8000/api/v1/heartbeat"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      ollama:
        image: ollama/ollama:latest
        ports:
          - 11434:11434
        options: >-
          --health-cmd "curl -f http://localhost:11434/api/tags"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements_dev.txt

      - name: Pull Ollama models
        run: |
          docker exec ollama ollama pull llama3.2
          docker exec ollama ollama pull mxbai-embed-large

      - name: Run integration tests
        env:
          CHROMADB_HOST: localhost
          CHROMADB_PORT: 8000
          OLLAMA_BASE_URL: http://localhost:11434
        run: |
          pytest tests/integration/ -v --cov=custom_components/home_agent --cov-report=xml

      - name: Run E2E tests
        env:
          CHROMADB_HOST: localhost
          CHROMADB_PORT: 8000
          OLLAMA_BASE_URL: http://localhost:11434
        run: |
          pytest tests/e2e/ -v -m e2e

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: integration

  smoke-tests:
    runs-on: ubuntu-latest
    needs: integration

    steps:
      - uses: actions/checkout@v3

      - name: Run smoke tests
        run: |
          pytest tests/e2e/ -v -m smoke --tb=short
```

### Pre-commit Hook

Update `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: integration-tests
      name: Run integration tests
      entry: scripts/run_integration_tests.sh
      language: script
      pass_filenames: false
      stages: [pre-push]  # Only run on push, not commit
```

---

## Success Criteria

### Test Coverage Targets

| Category | Target Coverage | Current | Gap |
|----------|----------------|---------|-----|
| Unit Tests | 80%+ | ~75% | -5% |
| Integration Tests | 60%+ | ~10% | -50% |
| E2E Tests | 40%+ | 0% | -40% |
| Critical Paths | 100% | ~60% | -40% |

### Performance Baselines

| Metric | Target | Measurement |
|--------|--------|-------------|
| Simple query latency (p95) | < 2s | Integration test |
| Tool execution latency (p95) | < 5s | Integration test |
| Memory extraction time | < 3s | Integration test |
| Concurrent conversations | 5+ | Load test |
| Memory leak rate | 0 MB/hour | Long-running test |

### Quality Gates

✅ **Required for Release**:
- All unit tests pass
- 90%+ integration tests pass
- All smoke tests pass
- No critical security vulnerabilities
- Performance baselines met

⚠️ **Warning Indicators**:
- Integration test coverage < 50%
- E2E test failures > 10%
- Performance regression > 20%

### Test Execution Matrix

| Environment | Unit | Integration | E2E | Frequency |
|-------------|------|-------------|-----|-----------|
| Local Dev | ✅ | ✅ | Optional | On commit |
| CI (PR) | ✅ | ✅ | ✅ Smoke only | On push |
| CI (Main) | ✅ | ✅ | ✅ Full suite | On merge |
| Nightly | ✅ | ✅ | ✅ + Load tests | Daily |

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Set up Docker Compose test infrastructure
- [ ] Create integration test fixtures and helpers
- [ ] Implement health check utilities
- [ ] Write first 5 integration tests (ChromaDB + Ollama)

### Phase 2: Core Integration Tests (Week 3-4)

- [ ] Vector DB integration tests (10 tests)
- [ ] LLM integration tests (10 tests)
- [ ] Memory integration tests (8 tests)
- [ ] Service integration tests (6 tests)

### Phase 3: E2E Framework (Week 5-6)

- [ ] Build scenario executor
- [ ] Create scenario definitions (YAML)
- [ ] Implement metrics collector
- [ ] Write first 5 E2E scenarios

### Phase 4: Advanced Testing (Week 7-8)

- [ ] Error recovery tests (10 tests)
- [ ] Performance tests (5 tests)
- [ ] Load tests (3 tests)
- [ ] Concurrent access tests (5 tests)

### Phase 5: CI/CD Integration (Week 9-10)

- [ ] Configure GitHub Actions
- [ ] Set up test result reporting
- [ ] Implement coverage tracking
- [ ] Create release gates

---

## Appendix

### Useful Commands

```bash
# Run only integration tests
pytest tests/integration/ -v -m integration

# Run only E2E tests
pytest tests/e2e/ -v -m e2e

# Run smoke tests (quick validation)
pytest tests/e2e/ -v -m smoke

# Run with coverage
pytest tests/integration/ --cov=custom_components/home_agent --cov-report=html

# Run performance tests
pytest tests/e2e/ -v -m performance

# Run load tests (long-running)
pytest tests/e2e/ -v -m load
```

### Test Markers

```python
# In pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "integration: Integration tests with real services",
    "e2e: End-to-end tests",
    "smoke: Quick smoke tests",
    "performance: Performance benchmarks",
    "load: Load and stress tests",
    "slow: Slow-running tests (>30s)",
]
```

### Environment Setup

```bash
# Start test infrastructure
docker-compose -f tests/docker-compose.test.yml up -d

# Initialize Ollama models
docker-compose -f tests/docker-compose.test.yml run --rm ollama-init

# Run tests
pytest tests/integration/ -v

# Stop infrastructure
docker-compose -f tests/docker-compose.test.yml down -v
```

---

## Conclusion

This testing strategy provides a comprehensive roadmap for implementing integration and E2E tests for the Home Agent integration. By following this guide, the project will achieve:

1. **Confidence in releases**: Real-world testing against actual services
2. **Early bug detection**: Integration issues caught before production
3. **Performance visibility**: Continuous performance monitoring
4. **Quality gates**: Automated checks prevent regressions

The phased implementation approach allows incremental progress while delivering value at each step. Start with Phase 1 to establish the foundation, then build up comprehensive test coverage over the following phases.

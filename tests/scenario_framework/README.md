# Scenario Framework (Mock-Based Unit Tests)

> **Note**: This is NOT an end-to-end test framework. It uses mocked LLM responses and does not test real service interactions. For actual integration testing with real LLM and services, see `tests/integration/test_real_*.py`.

YAML-based scenario framework for testing Home Agent conversation workflows with mocked components.

## What This Framework Tests

This framework tests the **scenario execution infrastructure**, not actual LLM behavior:
- YAML scenario loading and parsing
- Conversation flow orchestration
- Metrics collection and aggregation
- Assertion verification logic
- Agent setup/teardown

**What it does NOT test:**
- Real LLM responses or reasoning
- Actual tool execution
- Real service calls to Home Assistant
- Network/API behavior

## Quick Start

```bash
# Run all scenario framework tests
pytest tests/scenario_framework/ -m e2e

# Run a specific test
pytest tests/scenario_framework/test_scenarios.py::test_basic_conversation_flow -v
```

## Architecture

### Components

1. **ScenarioExecutor** (`executor.py`)
   - Loads YAML scenario files
   - Sets up test environment with mocked entities
   - Executes conversation steps with **mocked LLM responses**
   - Verifies assertions

2. **MetricsCollector** (`metrics.py`)
   - Tracks conversation metrics (duration, tokens, tool calls)
   - Aggregates statistics across conversations
   - Generates summary reports

3. **Fixtures** (`conftest.py`)
   - `scenario_executor`: Instance of ScenarioExecutor
   - `metrics_collector`: Instance of MetricsCollector

4. **Scenarios** (`scenarios/`)
   - YAML files defining test workflows
   - Include **mock_response** for each step (not real LLM output)

## Scenario File Format

```yaml
name: "Scenario Name"
description: "Brief description"

setup:
  entities:
    - entity_id: "light.living_room"
      state: "on"
      attributes:
        brightness: 255
        friendly_name: "Living Room Light"
  conversation_id: "test_001"

steps:
  - user: "User's input text"
    mock_response: "Hardcoded response (NOT from real LLM)"
    tools_used:
      - "ha_control"  # Declared, not actually executed
    expected_response_contains: "text to verify"

assertions:
  - conversation_history_length: 2
  - total_tool_calls: 1
```

## Existing Scenarios

| Scenario | Steps | Purpose |
|----------|-------|---------|
| basic_conversation.yaml | 1 | Test simple Q&A flow |
| tool_execution.yaml | 2 | Test tool declaration tracking |
| memory_flow.yaml | 3 | Test multi-turn conversation |
| error_recovery.yaml | 2 | Test error handling flow |

## When to Use This vs Integration Tests

| Use Scenario Framework For | Use Integration Tests For |
|---------------------------|---------------------------|
| Testing YAML parsing | Testing real LLM responses |
| Testing metrics collection | Testing actual tool execution |
| Testing assertion logic | Testing service calls |
| CI/CD smoke tests | Validating LLM behavior |
| Quick iteration on scenarios | Real-world workflow validation |

## Known Issues

The scenario framework tests currently have failures due to Home Assistant entity registry API changes. The `EntityRegistry` mock needs updating to match newer HA versions. This is a pre-existing issue and not a priority since:
1. These tests only verify the scenario execution infrastructure
2. Real functionality is tested in `tests/integration/test_real_*.py`
3. The framework itself is mock-based and doesn't catch real integration issues

## For Real E2E Testing

Use the integration tests in `tests/integration/`:
- `test_real_llm.py` - Tests with real LLM API
- `test_real_memory.py` - Tests real memory storage
- `test_real_vector_db.py` - Tests real ChromaDB
- `test_graceful_degradation.py` - Tests real failure scenarios

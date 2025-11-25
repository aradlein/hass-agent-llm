# Test Upgrade Roadmap

> **Generated**: 2025-11-25
> **Status**: In Progress
> **Branch**: feature/integration-test-environment

---

## Executive Summary

Comprehensive analysis of test coverage revealed significant gaps across integration tests, E2E tests, configuration coverage, and assertion quality. This document tracks all findings and remediation efforts.

**Overall Assessment**:
- Integration Tests: 65% happy paths, 30% error paths, 20% integration paths
- E2E Tests: 100% mocked (not actually E2E)
- Config Coverage: 20% completely untested
- Weak Assertions: 40+ instances identified

---

## P0 - Critical Issues

### P0.1 - E2E Tests Are Completely Mocked (NOT Real E2E)

**Problem**: The E2E tests in `tests/e2e/` never call a real LLM or execute real tool calls. Everything is hardcoded in YAML.

**Evidence**:
- `tests/e2e/executor.py:203-239` - LLM response entirely mocked
- `tests/e2e/scenarios/*.yaml` - Tool calls declared, not executed
- `tests/e2e/executor.py:244-248` - Response content never actually verified

**What's Mocked**:
| Component | Real? | Issue |
|-----------|-------|-------|
| LLM API | NO | Response hardcoded in YAML |
| LLM Tool Call Extraction | NO | `"tool_calls": None` hardcoded |
| Tool Execution | NO | Tools listed in YAML, never called |
| Home Assistant Services | NO | `hass.services` never called |
| Response Validation | NO | Only checks `result is not None` |

**Resolution**: Either fix to use real services OR rename to "scenario framework unit tests"

**Status**: [ ] Not Started

---

### P0.2 - `__init__.py` Has ZERO Test Coverage

**Problem**: 11 service handlers and all setup/teardown functions are completely untested.

**File**: `custom_components/home_agent/__init__.py` (644 LOC)

**Untested Functions**:
1. `async_setup_entry()` - Entry point setup
2. `async_unload_entry()` - Cleanup on unload
3. `async_reload_entry()` - Reload handling
4. `_async_handle_add_memory()` - Service handler
5. `_async_handle_search_memories()` - Service handler
6. `_async_handle_delete_memory()` - Service handler
7. `_async_handle_clear_memories()` - Service handler
8. `_async_handle_get_memories()` - Service handler
9. `_async_handle_reindex_entities()` - Service handler
10. `_async_handle_clear_conversation()` - Service handler
11. `_async_handle_get_conversation_history()` - Service handler

**Status**: [ ] Not Started

---

### P0.3 - Placeholder `assert True` Test

**Problem**: Test that always passes without testing anything.

**File**: `tests/integration/test_phase2_vector_db.py:323`

```python
async def test_phase2_success_criteria():
    """Validation test for Phase 2 success criteria..."""
    assert True, "Phase 2 core functionality verified through manual testing"
```

**Resolution**: Replace with actual assertions or delete the test.

**Status**: [ ] Not Started

---

## P1 - High Priority Issues

### P1.1 - Tool Timeout Handling Tests Missing

**Problem**: No integration tests verify tool execution timeout behavior.

**Affected Code**: `custom_components/home_agent/tool_handler.py`

**Missing Tests**:
- Tool call exceeds `max_timeout_seconds`
- Tool timeout triggers fallback behavior
- Tool timeout doesn't crash agent
- Proper error message returned to LLM

**Current State**: Only mocked tests with `AsyncMock` without actual delays

**Status**: [ ] Not Started

---

### P1.2 - Streaming Network Failure Tests Missing

**Problem**: Streaming tests mock all network calls; no real network failure scenarios.

**File**: `tests/integration/test_phase4_streaming.py`

**Missing Scenarios**:
- Partial SSE stream (connection drops mid-response)
- Malformed SSE data
- HTTP 503 Service Unavailable mid-streaming
- Slow network causing timeout
- Invalid JSON in SSE events

**Evidence**: Lines 279-280 use `mock_response.status = 200` with mocked content iterator.

**Status**: [ ] Not Started

---

### P1.3 - Tool Parameter Validation Error Tests Missing

**Problem**: Tests don't validate what happens with invalid tool parameters at call time.

**File**: `tests/integration/test_phase3_custom_tools.py:226-252`

**Current**: Only tests invalid tool CONFIG, not invalid tool PARAMETERS

**Missing Tests**:
- Tool receives wrong parameter type (string vs number)
- Required parameter is missing at call time
- Parameter value fails regex validation
- Parameter exceeds size limits
- Nested object validation failures

**Status**: [ ] Not Started

---

### P1.4 - Memory + Vector DB Integration Tests Missing

**Problem**: No tests verify memory system interaction with vector DB context provider.

**Current State**:
- `test_real_memory.py` tests memory extraction in isolation
- `test_phase2_vector_db.py` tests vector DB context without memories
- **NO TEST** verifies memories are indexed and retrieved via semantic search

**Missing Scenario**:
1. Add memory via memory_manager
2. Verify memory is indexed in vector DB collection
3. Query vector DB with semantic search
4. Confirm memory is returned in context
5. Verify memory metadata is intact

**Status**: [ ] Not Started

---

### P1.5 - Graceful Degradation Tests Missing

**Problem**: No tests verify behavior when optional components are unavailable.

**Missing Scenarios**:
- Vector DB unavailable but direct context works
- Memory system unavailable but conversation continues
- External LLM unavailable but primary LLM works
- Tool registration partial failure
- Context provider initialization failure recovery

**Status**: [ ] Not Started

---

### P1.6 - Alternative Config Values Never Tested

**Problem**: Code supports multiple values for enum configs but only tests defaults.

**LLM Backends** - Only "default" tested:
- `llm_backend="llama-cpp"` - NOT TESTED
- `llm_backend="vllm"` - NOT TESTED
- `llm_backend="ollama-gpu"` - NOT TESTED

**Context Formats** - Only "json" tested:
- `context_format="natural_language"` - NOT TESTED
- `context_format="hybrid"` - NOT TESTED

**Embedding Providers** - Only "ollama" tested:
- `vector_db_embedding_provider="openai"` - NOT TESTED

**Memory Extraction LLM** - Only "external" tested:
- `memory_extraction_llm="local"` - NOT TESTED

**Status**: [ ] Not Started

---

### P1.7 - Boundary Value Tests Missing

**Problem**: Numeric configs have ranges but boundary values never tested.

| Config | Range | Boundaries to Test |
|--------|-------|-------------------|
| `llm_temperature` | 0.0 - 2.0 | 0.0, 2.0 |
| `llm_max_tokens` | 1 - 100k | 1, 100000 |
| `vector_db_similarity_threshold` | 0.0 - 1000.0 | 0.0, 1000.0 |
| `memory_min_importance` | 0.0 - 1.0 | 0.0, 1.0 |
| `history_max_messages` | 1 - 100 | 1, 100 |
| `history_max_tokens` | 100 - 50k | 100, 50000 |
| `tools_max_calls_per_turn` | 1 - 20 | 1, 20 |

**Status**: [ ] Not Started

---

### P1.8 - Weak Assertions Need Strengthening

**Problem**: 40+ instances of assertions that don't actually verify anything meaningful.

**Patterns to Fix**:

| Pattern | Count | Example Location |
|---------|-------|------------------|
| `assert response is not None` only | 10+ | `test_real_llm.py:71` |
| `assert len(x) > 0` without content | 7+ | `test_conversation_history.py:71` |
| Generic `Exception` catching | 3+ | `test_real_vector_db.py:318` |
| Missing side-effect verification | 15+ | Service calls not verified |

**Files Requiring Updates**:
- `tests/integration/test_real_llm.py`
- `tests/integration/test_real_memory.py`
- `tests/integration/test_real_vector_db.py`
- `tests/integration/test_conversation_history.py`
- `tests/integration/test_context_manager.py`
- `tests/integration/test_phase4_streaming.py`
- `tests/e2e/executor.py`

**Status**: [ ] Not Started

---

## P2 - Medium Priority Issues

### P2.1 - Feature Flag Combination Tests (0% coverage)

**Untested Combinations**:
- `memory_enabled=True + memory_extraction_llm="external" + external_llm_enabled=False`
- `context_mode="vector_db" + memory_enabled=True`
- `streaming_enabled=True + memory_extraction_enabled=True`

### P2.2 - Untested Config Options

These config options are defined but never tested:
- `prompt_use_default`, `prompt_custom_additions`
- `debug_logging`
- `vector_db_similarity_threshold`
- `additional_collections`, `additional_top_k`, `additional_l2_distance_threshold`
- `memory_importance_decay`, `memory_event_ttl`, `memory_fact_ttl`, `memory_preference_ttl`
- `external_llm_auto_include_context`, `external_llm_keep_alive`

### P2.3 - agent.py Core Functions Untested

**File**: `custom_components/home_agent/agent.py` (1825 LOC)

**Untested**:
- `async_process()` non-streaming path
- `_build_system_prompt()`
- `_render_template()`
- LLM error handling paths

### P2.4 - config_flow.py Under-tested

**File**: `custom_components/home_agent/config_flow.py` (1210 LOC)
**Tests**: Only 207 LOC in tests

---

## P3 - Low Priority Issues

### P3.1 - Memory TTL and Cleanup Tests
### P3.2 - Prompt Customization Path Tests
### P3.3 - Cross-Tool Data Flow Tests
### P3.4 - Cold Start Performance Tests
### P3.5 - Memory Leak Detection Under Load

---

## Test File Reference

| Type | Location | Files | Status |
|------|----------|-------|--------|
| Integration | `tests/integration/` | 13 files | Moderate coverage |
| E2E | `tests/e2e/` | 4 files | 100% mocked |
| Unit | `tests/unit/` | 37 files | Best coverage |

---

## Progress Tracking

### Completed
- [x] Initial analysis complete
- [x] **P0.1** - E2E tests analysis (documented - needs manual decision on fix approach)
- [x] **P0.2** - `__init__.py` tests created (48 tests in `tests/unit/test_init.py`)
- [x] **P0.3** - Placeholder `assert True` test removed from `test_phase2_vector_db.py`
- [x] **P1.1** - Tool timeout tests added (7 tests in `test_phase3_custom_tools.py`)
- [x] **P1.2** - Streaming failure tests created (10 tests in `tests/integration/test_phase4_streaming_failures.py`)
- [x] **P1.3** - Tool parameter validation tests created (13 tests in `tests/integration/test_tool_parameter_validation.py`)
- [x] **P1.4** - Memory+VectorDB integration tests created (6 tests in `tests/integration/test_memory_vectordb_integration.py`)
- [x] **P1.5** - Graceful degradation tests created (12 tests in `tests/integration/test_graceful_degradation.py`)
- [x] **P1.6** - Config variations tests created (12 tests in `tests/integration/test_config_variations.py`)
- [x] **P1.7** - Boundary value tests created (99 tests in `tests/unit/test_config_boundaries.py`)
- [x] **P1.8** - Weak assertions strengthened across 5 integration test files

### In Progress
- [x] **P0.1** - E2E tests renamed to `tests/scenario_framework/` with updated README clarifying they are mock-based unit tests

### Not Started
- [ ] P2 issues
- [ ] P3 issues

---

## Notes

This is a temporary roadmap file to track test improvement efforts. Remove once all critical issues are addressed.

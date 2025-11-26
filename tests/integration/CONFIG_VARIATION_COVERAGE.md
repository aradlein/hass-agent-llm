# Configuration Variation Test Coverage

## Overview

This document describes the comprehensive test coverage for alternative configuration values in the Home Agent integration. Previously, only default configuration values were tested. This test suite ensures that all enum-based configuration options are properly exercised and their code paths verified.

## Test File

**Location**: `/workspaces/home-agent/tests/integration/test_config_variations.py`

**Total Tests**: 12 comprehensive integration tests

## Configuration Options Tested

### 1. Proxy Headers (CONF_LLM_PROXY_HEADERS)

**Default Value**: `{}` (empty dictionary)

**Alternative Values Tested**:
- `{"X-Ollama-Backend": "llama-cpp"}`
- `{"X-Ollama-Backend": "vllm-server"}`
- `{"X-Ollama-Backend": "ollama-gpu"}`

**Code Paths Covered**:
- `/workspaces/home-agent/custom_components/home_agent/agent/llm.py`
  - Adds custom proxy headers to LLM API requests
- `/workspaces/home-agent/custom_components/home_agent/agent/streaming.py`
  - Adds custom proxy headers to streaming requests

**Tests**:
1. `test_proxy_headers_sent` (parametrized for 3 header values)
   - Verifies custom proxy headers are added to requests
   - Confirms header value matches configured value
   - Uses mocking to capture HTTP headers without real LLM call

2. `test_no_proxy_headers_no_custom_header`
   - Verifies no custom headers are added when proxy_headers is empty
   - Ensures backward compatibility

**Verification Method**: Mocks `aiohttp.ClientSession` to capture HTTP headers sent to LLM API

---

### 2. Context Formats (CONF_CONTEXT_FORMAT)

**Default Value**: `"json"` (CONTEXT_FORMAT_JSON)

**Alternative Values Tested**:
- `"natural_language"` (CONTEXT_FORMAT_NATURAL_LANGUAGE)
- `"hybrid"` (CONTEXT_FORMAT_HYBRID)

**Code Paths Covered**:
- `/workspaces/home-agent/custom_components/home_agent/context_manager.py:139-141`
  - Passes format to DirectContextProvider
- `/workspaces/home-agent/custom_components/home_agent/context_providers/direct.py`
  - Format-specific context generation (natural_language vs JSON vs hybrid)

**Tests**:
1. `test_context_format_variations` (parametrized for 2 formats)
   - Natural Language: Validates readable text output with minimal JSON markers
   - Hybrid: Validates mixed structured and readable content
   - Checks word count and JSON density metrics
   - Verifies provider.format_type is set correctly

2. `test_context_format_json_baseline`
   - Baseline test for JSON format
   - Verifies structured output with JSON markers

**Verification Method**: Analyzes context string output for format characteristics (word count, JSON marker density)

---

### 3. Embedding Providers (CONF_VECTOR_DB_EMBEDDING_PROVIDER)

**Default Value**: `"ollama"` (EMBEDDING_PROVIDER_OLLAMA)

**Alternative Values Tested**:
- `"openai"` (EMBEDDING_PROVIDER_OPENAI)

**Code Paths Covered**:
- `/workspaces/home-agent/custom_components/home_agent/vector_db_manager.py:528-535`
  - Embedding provider selection logic
- `/workspaces/home-agent/custom_components/home_agent/vector_db_manager.py:543-573`
  - `_embed_with_openai()` method (OpenAI API path)
- `/workspaces/home-agent/custom_components/home_agent/vector_db_manager.py:575-609`
  - `_embed_with_ollama()` method (Ollama API path)

**Tests**:
1. `test_embedding_provider_openai`
   - Verifies OpenAI provider selection
   - Confirms `_embed_with_openai` is called
   - Validates OpenAI API key requirement
   - Mocks OpenAI library to avoid real API calls

2. `test_embedding_provider_ollama`
   - Baseline test for Ollama provider
   - Confirms `_embed_with_ollama` is called
   - Validates Ollama endpoint usage

**Verification Method**: Mocks embedding methods to verify correct method is called based on provider configuration

---

### 4. Memory Extraction LLM (CONF_MEMORY_EXTRACTION_LLM)

**Default Value**: `"external"`

**Alternative Values Tested**:
- `"local"`

**Code Paths Covered**:
- `/workspaces/home-agent/custom_components/home_agent/agent.py:1754-1802`
  - Memory extraction LLM selection logic
  - Calls `tool_handler.execute_tool("query_external_llm")` for external
  - Calls `_call_primary_llm_for_extraction()` for local

**Tests**:
1. `test_memory_extraction_llm_local`
   - Verifies local LLM is used for memory extraction
   - Confirms `_call_primary_llm_for_extraction` is called
   - Validates extraction prompt contains conversation data

2. `test_memory_extraction_llm_external`
   - Baseline test for external LLM
   - Confirms external LLM tool is called
   - Validates tool parameters

**Verification Method**: Mocks LLM calls and tool execution to verify correct extraction path

---

## Cross-Configuration Integration Test

**Test**: `test_multiple_alternative_configs_together`

**Purpose**: Verifies that multiple non-default configurations can work together simultaneously without conflicts.

**Configurations Combined**:
- LLM Backend: `llama-cpp`
- Context Format: `natural_language`
- Embedding Provider: `ollama`
- Context Mode: `vector_db`

**Verification**: Ensures system stability and functionality with multiple alternatives active

---

## Testing Strategy

Each test follows this pattern:

1. **Configure**: Set up the alternative configuration value
2. **Mock Dependencies**: Isolate the code path from external services
3. **Execute**: Run functionality that uses the configuration
4. **Assert Code Path**: Verify correct method/function was called
5. **Verify Behavior**: Confirm expected behavior for that configuration

This ensures:
- Alternative values are actually used by the code
- Behavior changes as intended
- No silent fallback to defaults
- Configurations work correctly in combination

---

## Running the Tests

### Run All Configuration Variation Tests
```bash
pytest tests/integration/test_config_variations.py -v
```

### Run Specific Configuration Category
```bash
# Proxy Headers tests
pytest tests/integration/test_config_variations.py -k "proxy_headers" -v

# Context Format tests
pytest tests/integration/test_config_variations.py -k "context_format" -v

# Embedding Provider tests
pytest tests/integration/test_config_variations.py -k "embedding_provider" -v

# Memory Extraction tests
pytest tests/integration/test_config_variations.py -k "memory_extraction" -v
```

### Run Specific Proxy Headers Test
```bash
pytest tests/integration/test_config_variations.py::test_proxy_headers_sent[llama-cpp] -v
```

---

## Test Requirements

### Service Dependencies

Most tests use mocking to avoid requiring real services, but the following markers indicate optional service dependencies:

- `@pytest.mark.requires_llm`: Test may benefit from real LLM endpoint
- `@pytest.mark.requires_chromadb`: Test requires ChromaDB for vector operations
- `@pytest.mark.requires_embedding`: Test requires embedding service

Tests will automatically skip if required services are unavailable.

### Environment Variables (Optional)

For running tests with real services:

```bash
TEST_LLM_BASE_URL=http://localhost:11434
TEST_LLM_MODEL=qwen2.5:3b
TEST_CHROMADB_HOST=localhost
TEST_CHROMADB_PORT=8000
TEST_EMBEDDING_BASE_URL=http://localhost:11434
TEST_EMBEDDING_MODEL=mxbai-embed-large
```

---

## Coverage Summary

| Configuration Option | Default | Alternatives Tested | Tests | Code Coverage |
|---------------------|---------|--------------------:|------:|--------------|
| LLM Backend | default | 3 | 4 | agent.py:534-536, 598-600 |
| Context Format | json | 2 | 3 | context_manager.py:139-141, direct.py |
| Embedding Provider | ollama | 1 | 2 | vector_db_manager.py:528-609 |
| Memory Extraction LLM | external | 1 | 2 | agent.py:1754-1802 |
| **Total** | - | **7** | **12** | **Multiple modules** |

---

## Before vs After

### Before
- Only default configuration values tested
- Alternative enum values never exercised
- Risk of untested code paths
- Potential for configuration options to be non-functional

### After
- All enum-based alternatives have dedicated tests
- Code paths verified for each configuration
- Confidence that configurations work as intended
- Detection of regressions in alternative paths

---

## Future Enhancements

Potential additions to configuration testing:

1. **Compression Levels**: Test all compression_level values (none, low, medium, high)
2. **Context Modes**: Additional vector_db mode variations
3. **Custom Tool Types**: Test all custom tool handler types
4. **Performance Testing**: Measure performance differences between configurations
5. **Error Handling**: Test configuration validation and error scenarios

---

## Maintenance

When adding new configuration options:

1. Add enum values to `const.py`
2. Implement the feature with branching logic
3. Add parametrized test to `test_config_variations.py`
4. Update this coverage document
5. Verify code path is exercised in test

---

## Related Files

- Test File: `/workspaces/home-agent/tests/integration/test_config_variations.py`
- Constants: `/workspaces/home-agent/custom_components/home_agent/const.py`
- Agent: `/workspaces/home-agent/custom_components/home_agent/agent.py`
- Context Manager: `/workspaces/home-agent/custom_components/home_agent/context_manager.py`
- Vector DB Manager: `/workspaces/home-agent/custom_components/home_agent/vector_db_manager.py`
- Existing Context Tests: `/workspaces/home-agent/tests/integration/test_context_manager.py`

---

Last Updated: 2025-11-26

---

## External Service Connection Patterns (Issue #59)

This section documents the validated patterns for connecting to external services in integration tests.

### ✅ Validated Pattern: Proxy Headers (CONF_LLM_PROXY_HEADERS)

Integration tests correctly use the **proxy headers** pattern for configuring LLM connections:

```python
from custom_components.home_agent.const import CONF_LLM_PROXY_HEADERS

config = {
    CONF_LLM_BASE_URL: llm_config["base_url"],
    CONF_LLM_API_KEY: llm_config.get("api_key", "test-key"),
    CONF_LLM_MODEL: llm_config["model"],
    CONF_LLM_PROXY_HEADERS: {"X-Ollama-Backend": "llama-cpp"},  # Custom routing headers
}
```

**Key Points**:
- Use `CONF_LLM_PROXY_HEADERS` (dictionary) instead of legacy `CONF_LLM_BACKEND` (string)
- Proxy headers are flexible - any key-value pairs can be added
- Headers are applied to both streaming and non-streaming LLM requests
- Empty dict `{}` means no custom headers (production-safe default)

### ❌ Deprecated: LLM_BACKEND Setting

The `CONF_LLM_BACKEND` / `LLM_BACKEND_*` constants are **legacy** and only exist for:
1. Backwards compatibility with existing configs
2. Migration testing (in `tests/unit/test_proxy_headers.py`)

**Do NOT use in new integration tests**. The migration code (`_migrate_legacy_backend`) automatically converts:
- `LLM_BACKEND_LLAMA_CPP` → `{"X-Ollama-Backend": "llama-cpp"}`
- `LLM_BACKEND_VLLM` → `{"X-Ollama-Backend": "vllm-server"}`
- `LLM_BACKEND_OLLAMA_GPU` → `{"X-Ollama-Backend": "ollama-gpu"}`

### ✅ External Service Configuration via Environment

All external service connections are configured via `.env.test`:

```bash
# LLM Configuration
TEST_LLM_BASE_URL=http://llm.inorganic.me:8080/v1
TEST_LLM_API_KEY=your-api-key
TEST_LLM_MODEL=Qwen2.5-7B-Instruct-Q5_K_L.gguf
# TEST_LLM_PROXY_HEADERS={"X-Ollama-Backend": "llama-cpp"}

# ChromaDB Configuration
TEST_CHROMADB_HOST=db.inorganic.me
TEST_CHROMADB_PORT=8000

# Embedding Configuration
TEST_EMBEDDING_BASE_URL=http://ai.inorganic.me:11434
TEST_EMBEDDING_MODEL=mxbai-embed-large
```

Tests access these via fixtures in `conftest.py`:
- `llm_config` - Returns dict with `base_url`, `api_key`, `model`, `proxy_headers`
- `chromadb_config` - Returns dict with `host`, `port`
- `embedding_config` - Returns dict with `base_url`, `model`, `provider`

### ✅ Service Availability Markers

Tests requiring external services use pytest markers:

```python
@pytest.mark.requires_llm
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
```

Tests are automatically skipped if services are unavailable (checked via health endpoints in `health.py`).

### ✅ Example: Complete Integration Test Pattern

```python
@pytest.mark.integration
@pytest.mark.requires_llm
@pytest.mark.asyncio
async def test_llm_with_proxy_headers(test_hass, llm_config, session_manager):
    """Test LLM connection with custom proxy headers."""
    config = {
        CONF_LLM_BASE_URL: llm_config["base_url"],
        CONF_LLM_API_KEY: llm_config.get("api_key", ""),
        CONF_LLM_MODEL: llm_config["model"],
        CONF_LLM_PROXY_HEADERS: {"X-Custom-Router": "gpu-1"},
        CONF_EMIT_EVENTS: False,
    }

    agent = HomeAgent(test_hass, config, session_manager)
    # ... test logic
    await agent.close()
```

### Validation Summary (Issue #59)

| Check | Status | Notes |
|-------|--------|-------|
| Integration tests use `CONF_LLM_PROXY_HEADERS` | ✅ | No direct `LLM_BACKEND` usage in integration tests |
| `LLM_BACKEND` only used for migration tests | ✅ | Only in `tests/unit/test_proxy_headers.py` |
| Service connections via environment vars | ✅ | `.env.test` pattern followed |
| Health checks before service access | ✅ | `health.py` + pytest markers |
| Proxy headers tested for all backends | ✅ | `test_config_variations.py` covers all values |

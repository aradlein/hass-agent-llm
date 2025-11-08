# Add Configurable `keep_alive` Parameter for Model Memory Management

## Overview
Add support for the `keep_alive` parameter in LLM API calls to control how long models remain loaded in memory. This is particularly useful for Ollama deployments to optimize memory usage and response times.

## Motivation
Currently, the integration does not send the `keep_alive` parameter to the LLM API, which means:
- Ollama uses its default 5-minute timeout before unloading models
- Users cannot optimize memory usage based on their usage patterns
- No control over when embedding models are unloaded (separate concern from main LLM)

## Feature Requirements

### 1. Primary LLM Keep Alive
- **Configuration Field**: `CONF_LLM_KEEP_ALIVE`
- **Default Value**: `"5m"` (matches Ollama default)
- **Accepted Values**:
  - Duration string (e.g., `"5m"`, `"10m"`, `"1h"`)
  - `"-1"` for indefinite (never unload)
- **Behavior**: Parameter sent with all primary LLM API requests
- **Compatibility**: Ollama will use it; other providers (OpenAI, etc.) will ignore it

### 2. External LLM Keep Alive
- **Configuration Field**: `CONF_EXTERNAL_LLM_KEEP_ALIVE`
- **Default Value**: `"5m"`
- **Accepted Values**: Same as primary LLM
- **Behavior**: Parameter sent when calling external LLM tool
- **Independence**: Separate from primary LLM setting

### 3. Embedding Model Keep Alive
- **Configuration Field**: `CONF_EMBEDDING_KEEP_ALIVE`
- **Default Value**: `"5m"`
- **Accepted Values**: Same as primary LLM
- **Behavior**: Parameter sent when generating embeddings for vector DB queries
- **Independence**: Completely separate from main LLM keep_alive (different model, different usage pattern)

## Implementation Details

### Constants to Add (`const.py`)
```python
# Keep alive settings
CONF_LLM_KEEP_ALIVE = "llm_keep_alive"
CONF_EXTERNAL_LLM_KEEP_ALIVE = "external_llm_keep_alive"
CONF_EMBEDDING_KEEP_ALIVE = "embedding_keep_alive"

DEFAULT_LLM_KEEP_ALIVE = "5m"
DEFAULT_EXTERNAL_LLM_KEEP_ALIVE = "5m"
DEFAULT_EMBEDDING_KEEP_ALIVE = "5m"
```

### Code Changes

#### 1. Update `agent.py` - Primary LLM API Calls

**Location**: `_call_llm()` method (around line 527-537)

**Current Code**:
```python
payload: dict[str, Any] = {
    "model": self.config[CONF_LLM_MODEL],
    "messages": messages,
    "temperature": temperature
        if temperature is not None
        else self.config.get(CONF_LLM_TEMPERATURE, 0.7),
    "max_tokens": max_tokens
        if max_tokens is not None
        else self.config.get(CONF_LLM_MAX_TOKENS, 500),
    "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
}
```

**Updated Code**:
```python
payload: dict[str, Any] = {
    "model": self.config[CONF_LLM_MODEL],
    "messages": messages,
    "temperature": temperature
        if temperature is not None
        else self.config.get(CONF_LLM_TEMPERATURE, 0.7),
    "max_tokens": max_tokens
        if max_tokens is not None
        else self.config.get(CONF_LLM_MAX_TOKENS, 500),
    "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
    "keep_alive": self.config.get(CONF_LLM_KEEP_ALIVE, DEFAULT_LLM_KEEP_ALIVE),
}
```

#### 2. Update `agent.py` - Streaming LLM API Calls

**Location**: `_call_llm_streaming()` method (around line 585-592)

**Current Code**:
```python
payload: dict[str, Any] = {
    "model": self.config[CONF_LLM_MODEL],
    "messages": messages,
    "temperature": self.config.get(CONF_LLM_TEMPERATURE, 0.7),
    "max_tokens": self.config.get(CONF_LLM_MAX_TOKENS, 1000),
    "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
    "stream": True,  # Enable streaming!
}
```

**Updated Code**:
```python
payload: dict[str, Any] = {
    "model": self.config[CONF_LLM_MODEL],
    "messages": messages,
    "temperature": self.config.get(CONF_LLM_TEMPERATURE, 0.7),
    "max_tokens": self.config.get(CONF_LLM_MAX_TOKENS, 1000),
    "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
    "stream": True,  # Enable streaming!
    "keep_alive": self.config.get(CONF_LLM_KEEP_ALIVE, DEFAULT_LLM_KEEP_ALIVE),
}
```

#### 3. Update `tools/external_llm.py` - External LLM Tool

**Location**: `_execute()` method (search for the API call payload construction)

**Add to payload**:
```python
"keep_alive": self.config.get(CONF_EXTERNAL_LLM_KEEP_ALIVE, DEFAULT_EXTERNAL_LLM_KEEP_ALIVE),
```

#### 4. Update `vector_db_manager.py` - Embedding Generation

**Location**: Look for where embeddings are generated (OpenAI client call or similar)

**Example for OpenAI-compatible embedding calls**:
```python
# If using openai library
response = await self.embedding_client.embeddings.create(
    model=self.embedding_model,
    input=text,
    extra_body={"keep_alive": self.config.get(CONF_EMBEDDING_KEEP_ALIVE, DEFAULT_EMBEDDING_KEEP_ALIVE)}
)

# If using direct HTTP calls
payload = {
    "model": self.embedding_model,
    "input": text,
    "keep_alive": self.config.get(CONF_EMBEDDING_KEEP_ALIVE, DEFAULT_EMBEDDING_KEEP_ALIVE),
}
```

#### 5. Update `config_flow.py` - Configuration UI

**Add to schema** (in appropriate menu - likely "LLM Settings" for primary/external, "Vector DB Settings" for embedding):

```python
vol.Optional(
    CONF_LLM_KEEP_ALIVE,
    default=DEFAULT_LLM_KEEP_ALIVE,
): str,

vol.Optional(
    CONF_EXTERNAL_LLM_KEEP_ALIVE,
    default=DEFAULT_EXTERNAL_LLM_KEEP_ALIVE,
): str,

vol.Optional(
    CONF_EMBEDDING_KEEP_ALIVE,
    default=DEFAULT_EMBEDDING_KEEP_ALIVE,
): str,
```

**UI Descriptions**:
- **Primary LLM Keep Alive**: "Duration to keep model loaded in memory (e.g., '5m', '1h', '-1' for indefinite). Primarily for Ollama."
- **External LLM Keep Alive**: "Duration to keep external model loaded (e.g., '5m', '1h', '-1'). Separate from primary LLM."
- **Embedding Keep Alive**: "Duration to keep embedding model loaded (e.g., '5m', '10m', '-1'). Separate from LLM models."

### Validation (Optional but Recommended)

Add validation in `config_flow.py` to ensure valid duration strings:

```python
def validate_keep_alive(value: str) -> str:
    """Validate keep_alive parameter format.

    Args:
        value: Keep alive value (e.g., "5m", "1h", "-1")

    Returns:
        The validated value

    Raises:
        vol.Invalid: If format is invalid
    """
    if value == "-1":
        return value

    # Check format: number + unit (s, m, h)
    import re
    if not re.match(r'^\d+[smh]$', value):
        raise vol.Invalid(
            f"Invalid keep_alive format: '{value}'. "
            "Use format like '5m', '1h', '30s', or '-1' for indefinite."
        )

    return value
```

Then use in schema:
```python
vol.Optional(
    CONF_LLM_KEEP_ALIVE,
    default=DEFAULT_LLM_KEEP_ALIVE,
): validate_keep_alive,
```

## Testing Requirements

### Unit Tests

1. **Test payload construction** (`tests/unit/test_agent.py`):
   ```python
   async def test_call_llm_includes_keep_alive(self, mock_session):
       """Test that keep_alive is included in LLM API payload."""
       agent = create_test_agent(config={CONF_LLM_KEEP_ALIVE: "10m"})

       await agent._call_llm(messages=[{"role": "user", "content": "test"}])

       # Verify keep_alive in payload
       call_args = mock_session.post.call_args
       payload = call_args.kwargs["json"]
       assert payload["keep_alive"] == "10m"

   async def test_call_llm_default_keep_alive(self, mock_session):
       """Test that default keep_alive is used when not configured."""
       agent = create_test_agent(config={})  # No keep_alive specified

       await agent._call_llm(messages=[{"role": "user", "content": "test"}])

       call_args = mock_session.post.call_args
       payload = call_args.kwargs["json"]
       assert payload["keep_alive"] == "5m"  # DEFAULT_LLM_KEEP_ALIVE

   async def test_call_llm_streaming_includes_keep_alive(self, mock_session):
       """Test that keep_alive is included in streaming API payload."""
       agent = create_test_agent(config={CONF_LLM_KEEP_ALIVE: "-1"})

       async for _ in agent._call_llm_streaming(messages=[{"role": "user", "content": "test"}]):
           break

       call_args = mock_session.post.call_args
       payload = call_args.kwargs["json"]
       assert payload["keep_alive"] == "-1"
   ```

2. **Test external LLM tool** (`tests/unit/test_external_llm.py`):
   ```python
   async def test_external_llm_includes_keep_alive(self, mock_session):
       """Test that external LLM API calls include keep_alive."""
       config = {
           CONF_EXTERNAL_LLM_ENABLED: True,
           CONF_EXTERNAL_LLM_KEEP_ALIVE: "15m",
       }
       tool = ExternalLLMTool(hass, config)

       await tool._execute(prompt="test", context={})

       call_args = mock_session.post.call_args
       payload = call_args.kwargs["json"]
       assert payload["keep_alive"] == "15m"
   ```

3. **Test embedding keep_alive** (`tests/unit/test_vector_db_manager.py`):
   ```python
   async def test_embedding_includes_keep_alive(self, mock_embedding_client):
       """Test that embedding API calls include keep_alive."""
       config = {CONF_EMBEDDING_KEEP_ALIVE: "10m"}
       manager = VectorDBManager(hass, config)

       await manager.generate_embedding("test text")

       # Verify keep_alive was passed to embedding API
       # (exact assertion depends on implementation)
   ```

4. **Test validation** (`tests/unit/test_config_flow.py`):
   ```python
   def test_validate_keep_alive_valid_formats():
       """Test that valid keep_alive formats are accepted."""
       assert validate_keep_alive("5m") == "5m"
       assert validate_keep_alive("1h") == "1h"
       assert validate_keep_alive("30s") == "30s"
       assert validate_keep_alive("-1") == "-1"

   def test_validate_keep_alive_invalid_formats():
       """Test that invalid keep_alive formats are rejected."""
       with pytest.raises(vol.Invalid):
           validate_keep_alive("5")  # Missing unit
       with pytest.raises(vol.Invalid):
           validate_keep_alive("5x")  # Invalid unit
       with pytest.raises(vol.Invalid):
           validate_keep_alive("invalid")
   ```

### Integration Tests

1. **Test with Ollama** (manual or automated if Ollama is available in CI):
   - Verify model stays loaded for specified duration
   - Verify model unloads after timeout
   - Verify `-1` keeps model loaded indefinitely

### Manual Testing Checklist

- [ ] Configure primary LLM with `keep_alive: "10m"` - verify parameter sent to API
- [ ] Configure external LLM with `keep_alive: "-1"` - verify separate value used
- [ ] Configure embedding with `keep_alive: "5m"` - verify separate value used
- [ ] Test with OpenAI API - verify it doesn't break (ignores parameter)
- [ ] Test with Ollama - verify model memory behavior changes
- [ ] Verify defaults work when not configured
- [ ] Test configuration UI allows entering all three values separately
- [ ] Test invalid formats are rejected in UI

## Documentation Updates

### Files to Update:

1. **`docs/CONFIGURATION.md`**:
   - Add `keep_alive` settings to LLM configuration section
   - Explain Ollama-specific behavior
   - Provide examples of when to use different values

2. **`docs/reference/CONFIGURATION.md`**:
   - Add to configuration reference table
   - Document all three keep_alive parameters

3. **`docs/VECTOR_DB_SETUP.md`**:
   - Add `embedding_keep_alive` to vector DB configuration examples
   - Explain why embedding keep_alive is separate

4. **`docs/FAQ.md`**:
   - Add FAQ: "How do I optimize Ollama memory usage?"
   - Add FAQ: "What does keep_alive do?"
   - Add FAQ: "Should I use -1 for keep_alive?"

5. **`README.md`**:
   - Mention keep_alive in features section (if appropriate)

## Example Configuration

```yaml
# In Home Assistant UI or configuration.yaml
home_agent:
  # Primary LLM (for conversation)
  llm:
    base_url: "http://localhost:11434/v1"
    model: "llama3.2:3b"
    keep_alive: "-1"  # Keep loaded indefinitely for instant responses

  # External LLM (for complex queries)
  external_llm:
    enabled: true
    base_url: "http://localhost:11434/v1"
    model: "llama3.2:70b"
    keep_alive: "2m"  # Unload after 2 min (rarely used, save memory)

  # Vector DB / Embeddings
  vector_db:
    enabled: true
    embedding_model: "nomic-embed-text"
    embedding_keep_alive: "10m"  # Unload after 10 min (moderate usage)
```

## Acceptance Criteria

- [ ] Primary LLM API calls include `keep_alive` parameter
- [ ] Streaming LLM API calls include `keep_alive` parameter
- [ ] External LLM tool includes separate `keep_alive` parameter
- [ ] Embedding API calls include separate `keep_alive` parameter
- [ ] All three settings configurable independently via UI
- [ ] Defaults to `"5m"` when not configured
- [ ] Validation prevents invalid formats (optional but recommended)
- [ ] Works with Ollama (model memory behavior changes)
- [ ] Works with OpenAI and other providers (parameter ignored, no errors)
- [ ] Unit tests cover all three keep_alive settings
- [ ] Documentation updated with examples and explanations
- [ ] No breaking changes to existing configurations

## Related

- See updated PROJECT_SPEC.md sections:
  - Primary LLM Configuration (lines 347-352)
  - External LLM Tool (lines 363-365, 754)
  - Vector DB Configuration (lines 582-587)
  - Configuration Schema Examples (lines 1524, 1550, 1598)

## Priority

**Medium** - Nice to have for Ollama users, but not critical for functionality.

## Estimated Effort

**2-3 hours**:
- 30 min: Add constants and update agent.py
- 30 min: Update external LLM and vector DB manager
- 30 min: Update config_flow.py
- 30 min: Write unit tests
- 30 min: Update documentation
- 30 min: Manual testing and validation

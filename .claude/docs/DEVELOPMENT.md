# Development Guide

## Development Standards

This project follows strict development standards to ensure code quality, maintainability, and reliability.

### Testing Requirements

**All code must include tests before being considered complete.**

#### Unit Tests
- **Location:** `tests/unit/`
- **Coverage Target:** Minimum 80% code coverage
- **Framework:** pytest
- **Requirements:**
  - Test individual functions and classes in isolation
  - Mock external dependencies (Home Assistant, LLM APIs, etc.)
  - Test edge cases and error conditions
  - Fast execution (< 1s per test file)

#### Integration Tests
- **Location:** `tests/integration/`
- **Framework:** pytest with Home Assistant test fixtures
- **Requirements:**
  - Test component integration with Home Assistant
  - Test complete workflows (user query → tool execution → response)
  - Test configuration flows
  - Mock external APIs (OpenAI, ChromaDB, etc.)
  - May be slower but should complete in reasonable time

#### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_context_manager.py
│   ├── test_conversation.py
│   ├── test_tool_handler.py
│   ├── test_context_providers/
│   │   ├── test_direct.py
│   │   └── test_vector_db.py
│   └── test_tools/
│       ├── test_ha_control.py
│       ├── test_ha_query.py
│       └── test_custom.py
└── integration/
    ├── __init__.py
    ├── test_agent.py
    ├── test_config_flow.py
    ├── test_conversation_flow.py
    └── test_external_llm_tool.py
```

#### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=custom_components.home_agent --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_context_manager.py

# Run with verbose output
pytest -v

# Run with print statements visible
pytest -s
```

### Definition of Done

A feature/issue is **NOT complete** until:
- [ ] Code is written and follows style guidelines
- [ ] Unit tests written with >80% coverage of new code
- [ ] Integration tests written for end-to-end workflows
- [ ] All tests pass locally
- [ ] Code is documented (docstrings, comments for complex logic)
- [ ] Type hints are present for all functions
- [ ] Code reviewed (if applicable)
- [ ] Manual testing completed in development environment

### Code Quality Standards

#### Python Style
- **Style Guide:** PEP 8
- **Line Length:** 100 characters (matches Home Assistant)
- **Formatter:** Black (with line-length 100)
- **Linter:** Pylint, Flake8
- **Type Checker:** mypy with strict mode

#### Type Hints
```python
# Required for all functions
def get_entity_state(
    hass: HomeAssistant,
    entity_id: str,
    attributes: list[str] | None = None
) -> dict[str, Any]:
    """Get entity state and specified attributes."""
    ...
```

#### Docstrings
```python
def format_context(
    entities: list[EntityContext],
    format_type: Literal["json", "natural_language"]
) -> str:
    """Format entity context for LLM consumption.

    Args:
        entities: List of entity contexts to format
        format_type: Output format (json or natural_language)

    Returns:
        Formatted context string ready for LLM

    Raises:
        ValueError: If format_type is invalid

    Example:
        >>> entities = [EntityContext(...)]
        >>> format_context(entities, "json")
        '{"entities": [...]}'
    """
    ...
```

#### Error Handling
```python
# Use specific exceptions
from .exceptions import ContextInjectionError, ToolExecutionError

# Always log errors
_LOGGER.error("Failed to inject context: %s", error, exc_info=True)

# Provide helpful error messages
raise ToolExecutionError(
    f"Tool '{tool_name}' failed: {error}. "
    f"Check entity_id '{entity_id}' exists and is accessible."
)
```

#### Async Best Practices
```python
# Use async/await for I/O operations
async def query_llm(self, messages: list[dict]) -> dict:
    """Query LLM API."""
    async with self.session.post(url, json=payload) as response:
        return await response.json()

# Don't block the event loop
# BAD:
time.sleep(1)

# GOOD:
await asyncio.sleep(1)
```

### Configuration Standards

#### Constants
- **Location:** `const.py`
- **Naming:** `UPPER_CASE_WITH_UNDERSCORES`
- **Organization:** Group by category with comments

```python
# Domain and component info
DOMAIN = "home_agent"
DEFAULT_NAME = "Home Agent"

# Configuration keys
CONF_LLM_BASE_URL = "llm_base_url"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_MODEL = "llm_model"

# Context injection
CONF_CONTEXT_MODE = "context_mode"
CONTEXT_MODE_DIRECT = "direct"
CONTEXT_MODE_VECTOR_DB = "vector_db"

# Defaults
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
DEFAULT_HISTORY_MAX_MESSAGES = 10
```

#### Configuration Schemas
```python
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

CONFIG_SCHEMA = vol.Schema({
    DOMAIN: vol.Schema({
        vol.Required(CONF_LLM_BASE_URL): cv.url,
        vol.Required(CONF_LLM_API_KEY): cv.string,
        vol.Optional(CONF_LLM_MODEL, default=DEFAULT_MODEL): cv.string,
        vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=2.0)
        ),
    })
}, extra=vol.ALLOW_EXTRA)
```

### Git Workflow

#### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements
- `docs/description` - Documentation updates

#### Commit Messages
Follow Conventional Commits:
```
type(scope): short description

Longer description if needed.

- Bullet points for details
- Reference issues: #123

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`

#### Pull Request Requirements
- [ ] Descriptive title and description
- [ ] All tests pass
- [ ] Code coverage doesn't decrease
- [ ] Documentation updated if needed
- [ ] No linter warnings

### Logging Standards

```python
import logging

_LOGGER = logging.getLogger(__name__)

# Log levels:
_LOGGER.debug("Detailed debugging info: %s", data)
_LOGGER.info("Normal operations: User %s started conversation", user_id)
_LOGGER.warning("Unexpected but handled: Context size %d exceeds recommended", size)
_LOGGER.error("Error occurred: %s", error, exc_info=True)
_LOGGER.critical("System failure: Unable to initialize component")
```

### Performance Guidelines

#### Token Usage Optimization
- Cache entity states when possible
- Implement smart context truncation
- Monitor and log token consumption
- Warn users when approaching limits

#### Async Operations
- Use `asyncio.gather()` for parallel operations
- Implement timeouts for external calls
- Use connection pooling for HTTP clients

#### Memory Management
- Limit conversation history size
- Clean up old conversations periodically
- Use generators for large data sets

### Security Standards

#### API Keys
```python
# Never log API keys
_LOGGER.debug("Calling LLM at %s", base_url)  # Good
_LOGGER.debug("Calling with key %s", api_key)  # NEVER!

# Redact in error messages
def _redact_api_key(self, text: str) -> str:
    """Redact API keys from text."""
    if self.api_key in text:
        return text.replace(self.api_key, "***REDACTED***")
    return text
```

#### Entity Access Control
```python
# Always validate entity access
def validate_entity_access(
    self,
    entity_id: str,
    exposed_entities: set[str]
) -> bool:
    """Validate user can access entity."""
    if entity_id not in exposed_entities:
        _LOGGER.warning(
            "Attempted access to unexposed entity: %s",
            entity_id
        )
        raise PermissionDenied(f"Entity {entity_id} is not accessible")
    return True
```

#### Input Validation
```python
# Sanitize all user inputs
from homeassistant.helpers import entity_component

def validate_entity_id(entity_id: str) -> str:
    """Validate and sanitize entity ID."""
    if not entity_component.valid_entity_id(entity_id):
        raise ValueError(f"Invalid entity_id: {entity_id}")
    return entity_id
```

### Documentation Standards

#### Code Documentation
- Docstrings for all public functions/classes
- Inline comments for complex logic
- Type hints for all parameters and returns
- Examples in docstrings for complex functions

#### User Documentation
- README.md with installation instructions
- Configuration examples
- Troubleshooting guide
- FAQ section

#### API Documentation
- Document all services
- Document all events
- Document configuration options
- Include examples for each

### Pre-commit Checklist

Before committing code:
- [ ] Run tests: `pytest`
- [ ] Run type checker: `mypy custom_components/home_agent`
- [ ] Run linter: `pylint custom_components/home_agent`
- [ ] Run formatter: `black custom_components/home_agent`
- [ ] Check imports: `isort custom_components/home_agent`
- [ ] Review changes: `git diff`
- [ ] Write descriptive commit message

### Development Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install

# Run Home Assistant in development mode
hass -c config --debug
```

### Continuous Integration

All PRs must pass CI checks:
- Unit tests (pytest)
- Integration tests (pytest)
- Type checking (mypy)
- Linting (pylint, flake8)
- Code formatting (black, isort)
- Coverage report (>80%)

### Issue Lifecycle

1. **Created** - Issue is filed with requirements
2. **Planned** - Issue is added to project board, assigned
3. **In Progress** - Development started
4. **Testing** - Code complete, tests being written
5. **Review** - PR submitted, under review
6. **Done** - Merged, tests passing, deployed

**Remember: An issue is NOT done until tests are written and passing!**

---

## Quick Reference

### Project Structure
```
home-agent/
├── custom_components/
│   └── home_agent/
│       ├── __init__.py
│       ├── manifest.json
│       ├── const.py
│       ├── config_flow.py
│       ├── agent.py
│       ├── context_manager.py
│       ├── conversation.py
│       ├── tool_handler.py
│       ├── exceptions.py
│       ├── helpers.py
│       ├── services.py
│       ├── services.yaml
│       ├── strings.json
│       ├── context_providers/
│       ├── tools/
│       └── translations/
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── PROJECT_SPEC.md
│   ├── DEVELOPMENT.md
│   └── API.md
├── .github/
│   └── workflows/
├── requirements.txt
├── requirements_dev.txt
├── .pylintrc
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md
```

### Essential Commands
```bash
# Testing
pytest                          # Run all tests
pytest --cov                    # With coverage
pytest -k test_name            # Run specific test

# Code Quality
black .                        # Format code
isort .                        # Sort imports
pylint custom_components       # Lint code
mypy custom_components         # Type check

# Git
git checkout -b feature/name   # New feature branch
git add -p                     # Stage changes interactively
git commit                     # Commit with message
```

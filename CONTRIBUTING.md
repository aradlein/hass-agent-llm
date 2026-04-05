# Contributing to Home Agent

Thank you for your interest in contributing to Home Agent. This guide covers everything you need to get started.

## Getting Started

1. Fork the repository at [github.com/aradlein/hass-agent-llm](https://github.com/aradlein/hass-agent-llm)
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/hass-agent-llm.git
   cd hass-agent-llm
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Development Setup

**Requirements:** Python 3.13 or higher.

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements_dev.txt
   ```

2. Copy the test environment template:
   ```bash
   cp .env.test.example .env.test
   ```
   You can leave the placeholder values in `.env.test` if you only plan to run unit tests and mocked integration tests. Real service endpoints are only needed for the `test_real_*.py` integration tests.

## Running Tests

### Unit Tests

```bash
pytest tests/unit/ -v
```

### Integration Tests (Mocked)

These tests run with mocked dependencies and do not require external services:

```bash
pytest tests/integration/ -v \
  --ignore=tests/integration/test_real_llm.py \
  --ignore=tests/integration/test_real_memory.py \
  --ignore=tests/integration/test_real_vector_db.py \
  --ignore=tests/integration/test_real_ttft_metrics.py
```

### Integration Tests (Real Services)

For running against real ChromaDB, LLM, and embedding services, configure `.env.test` with valid endpoints and run:

```bash
./scripts/run_integration_tests.sh
```

### Coverage

```bash
pytest tests/unit/ tests/integration/ \
  --ignore=tests/integration/test_real_llm.py \
  --ignore=tests/integration/test_real_memory.py \
  --ignore=tests/integration/test_real_vector_db.py \
  --ignore=tests/integration/test_real_ttft_metrics.py \
  --cov=custom_components.home_agent
```

## Test Requirements for Pull Requests

- All unit tests must pass.
- All mocked integration tests must pass (test files **not** prefixed with `test_real_`).
- **New code paths must include corresponding unit or integration tests.** PRs that add functionality without test coverage will not be merged.
- `test_real_*.py` tests are optional for contributors. They require external infrastructure and are validated internally.

## Code Style

All code style settings are defined in `pyproject.toml`.

- **Formatter:** black (line-length 100)
- **Import sorting:** isort (profile: black, line-length 100)
- **Linting:** flake8

Run the following checks before submitting a PR:

```bash
black --check custom_components/ tests/
isort --check custom_components/ tests/
flake8 custom_components/ tests/
```

To auto-format your code:

```bash
black custom_components/ tests/
isort custom_components/ tests/
```

## Pull Request Process

1. Ensure all tests pass locally (unit + mocked integration).
2. Ensure code passes linting (`black`, `isort`, `flake8`).
3. Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md) completely, including:
   - Description of what the PR does and why
   - Type of change (bug fix, new feature, breaking change, docs)
   - Testing approach and coverage information
4. Link any related issues.
5. PRs require all CI status checks to pass before merge.

## Commit Messages

Use conventional commit format:

```
type: description
```

- **Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- Keep the first line under 72 characters.
- Use the imperative mood (e.g., "Add feature" not "Added feature").

Examples:

```
feat: Add support for custom embedding models
fix: Resolve memory leak in session manager
test: Add unit tests for entity extraction
docs: Update configuration examples
refactor: Simplify vector database connection logic
chore: Update dependency versions
```

## Branch Protection

The `main` branch has the following protections:

- All CI status checks must pass (unit tests, mocked integration tests, linting, hassfest, HACS validation).
- Branches must be up to date with `main` before merging.

## Questions?

Open a [Discussion](https://github.com/aradlein/hass-agent-llm/discussions) or check existing [Issues](https://github.com/aradlein/hass-agent-llm/issues).

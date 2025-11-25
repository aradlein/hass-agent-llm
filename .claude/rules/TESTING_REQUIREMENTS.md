# Testing Requirements and Standards

## Critical: Testing is Mandatory Before Any Commit

All agents working on this codebase MUST follow these testing requirements without exception.

---

## Pre-Commit Testing Requirements

### MUST Run Before Every Commit

1. **All Unit Tests** - MUST pass before committing any code changes
   ```bash
   python3 -m pytest tests/unit/ -v
   ```

2. **All Integration Tests** - MUST pass before committing any code changes
   ```bash
   ./scripts/run_integration_tests.sh
   ```

3. **Zero Tolerance for Test Failures**
   - ALL tests must pass (100% pass rate)
   - NO skipped tests unless explicitly approved by the user
   - NO ignored failures or warnings that indicate real issues

### When Making Code Changes

**REQUIRED for ANY production code changes:**

1. **Run Relevant Unit Tests**
   - Identify which unit tests cover the changed code
   - Run those specific tests first for faster feedback
   - Example: If editing `memory_manager.py`, run:
     ```bash
     python3 -m pytest tests/unit/test_memory_manager.py -v
     ```

2. **Run Relevant Integration Tests**
   - Identify which integration tests exercise the changed functionality
   - Run those specific tests to validate end-to-end behavior
   - Example: If editing memory extraction, run:
     ```bash
     ./scripts/run_integration_tests.sh -k test_real_memory
     ```

3. **Run Full Test Suite**
   - After targeted tests pass, run the complete test suite
   - This catches unexpected regressions in other areas
   - Both unit AND integration tests must pass

---

## Test Development Requirements

### When Building New Features

**REQUIRED steps when implementing new functionality:**

1. **Write Tests First (TDD Approach Preferred)**
   - Design test cases that validate the feature requirements
   - Write unit tests for individual components/functions
   - Write integration tests for end-to-end workflows
   - Tests should fail initially (red), then pass after implementation (green)

2. **Comprehensive Test Coverage**
   - **Unit Tests**: Test each function, method, and class in isolation
   - **Integration Tests**: Test complete workflows with real services
   - Cover happy paths AND edge cases
   - Test error handling and failure scenarios
   - Validate boundary conditions

3. **Test Structure**
   - Unit tests go in `tests/unit/`
   - Integration tests go in `tests/integration/`
   - Use descriptive test names that explain what is being tested
   - Follow existing test patterns and conventions

### When Editing Existing Functionality

**REQUIRED steps when modifying existing code:**

1. **Run Existing Tests First**
   - Identify all tests that cover the code being changed
   - Run them to establish a baseline (they should pass)
   - If they fail before your changes, fix them first

2. **Update Tests to Match Changes**
   - If behavior changes, update tests to reflect new expected behavior
   - If interfaces change (function signatures, etc.), update all affected tests
   - Add new test cases if the change introduces new code paths

3. **Validate Test Coverage**
   - Ensure the modified code is still fully covered by tests
   - Add new tests if coverage gaps are introduced
   - Run tests with coverage reporting:
     ```bash
     python3 -m pytest tests/unit/test_modified_file.py --cov=custom_components.home_agent.modified_module
     ```

---

## Absolute Testing Rules

### NEVER Skip Tests Without User Approval

**Rule:** Do NOT skip, disable, or comment out tests because they are:
- Hard to fix
- Taking too long to debug
- Failing for unclear reasons
- Requiring too much work

**Required Action When Tests Fail:**
1. Investigate the failure thoroughly
2. Determine root cause (test issue vs code issue)
3. Fix the underlying problem
4. If the fix is complex or unclear, **ASK THE USER** for guidance
5. Only skip tests if the user explicitly approves

**Acceptable Reasons to Skip (with user approval only):**
- External service genuinely unavailable (but integration test framework already handles this)
- Test is for deprecated functionality being removed
- Test is temporarily blocked by a known external issue (with tracking ticket)

**NEVER Acceptable:**
- "This test is too hard to fix"
- "I don't understand why it's failing"
- "It would take too long to debug"
- "The test seems flaky"

### Test-Driven Development Mindset

**When adding functionality:**
1. Write the test first (shows what the code should do)
2. Watch it fail (confirms test actually tests something)
3. Implement the code to make it pass
4. Refactor while keeping tests green

**When fixing bugs:**
1. Write a test that reproduces the bug (should fail)
2. Fix the code
3. Verify the test now passes
4. Ensure no other tests broke

---

## Test Types and Requirements

### Unit Tests (`tests/unit/`)

**Purpose:** Test individual components in isolation

**Requirements:**
- Mock external dependencies (Home Assistant, ChromaDB, LLM APIs)
- Test one component/function at a time
- Fast execution (< 1 second per test typically)
- No network calls to external services
- Deterministic (same input = same output every time)

**Must Cover:**
- Normal operation (happy path)
- Edge cases (empty inputs, null values, boundary conditions)
- Error handling (invalid inputs, exceptions)
- State transitions
- Configuration variations

**Example:**
```python
@pytest.mark.asyncio
async def test_memory_extraction_with_valid_response():
    """Test that memory extraction works with a valid LLM response."""
    # Setup, execute, assert
```

### Integration Tests (`tests/integration/`)

**Purpose:** Test complete workflows with real or realistic services

**Requirements:**
- Use real external services when available (ChromaDB, LLM, Embeddings)
- Test end-to-end functionality
- Validate service integration points
- Use proper pytest markers (`@pytest.mark.requires_chromadb`, etc.)
- Clean up test data (collections, memories) after execution

**Must Cover:**
- Complete user workflows
- Service communication and error handling
- Data persistence and retrieval
- Cross-component integration
- Real-world scenarios

**Example:**
```python
@pytest.mark.requires_chromadb
@pytest.mark.requires_embedding
@pytest.mark.asyncio
async def test_memory_storage_and_retrieval_flow():
    """Test complete flow of storing and retrieving memories from vector DB."""
    # Use real ChromaDB, real embeddings
```

---

## Test Execution Commands

### Unit Tests

```bash
# Run all unit tests
python3 -m pytest tests/unit/ -v

# Run specific test file
python3 -m pytest tests/unit/test_memory_manager.py -v

# Run with coverage
python3 -m pytest tests/unit/ --cov=custom_components.home_agent

# Run in parallel (faster)
python3 -m pytest tests/unit/ -n auto
```

### Integration Tests

```bash
# Run all integration tests (with service health checks)
./scripts/run_integration_tests.sh

# Run specific integration test
./scripts/run_integration_tests.sh -k test_real_llm

# Run smoke tests only
./scripts/run_integration_tests.sh --smoke

# Run specific test file
python3 -m pytest tests/integration/test_real_memory.py -v -p no:homeassistant
```

### Combined

```bash
# Run all tests (unit + integration)
python3 -m pytest tests/ -v
```

---

## Pre-Commit Checklist

Before committing ANY code changes, verify:

- [ ] All unit tests pass (`python3 -m pytest tests/unit/ -v`)
- [ ] All integration tests pass (`./scripts/run_integration_tests.sh`)
- [ ] New functionality has corresponding new tests
- [ ] Modified functionality has updated tests
- [ ] No tests were skipped or disabled without user approval
- [ ] Test coverage is maintained or improved
- [ ] All tests are deterministic and not flaky

**If ANY of these checks fail, DO NOT commit. Fix the issues first.**

---

## Code Review Requirements

When reviewing your own code before committing:

1. **For Each Changed File:**
   - Identify corresponding test files
   - Verify tests exist and are comprehensive
   - Run tests and confirm they pass
   - Check if new test cases are needed

2. **For New Features:**
   - Minimum: Unit tests for all new functions/methods
   - Minimum: Integration test for the feature workflow
   - Preferred: Multiple integration tests covering different scenarios

3. **For Bug Fixes:**
   - MUST include a test that would have caught the bug
   - Verify the test fails before the fix
   - Verify the test passes after the fix

---

## Testing Philosophy

### Quality Over Speed

- Taking time to write proper tests is ALWAYS worth it
- Skipping tests to "move faster" creates technical debt
- A well-tested codebase moves faster in the long run

### Fail Fast, Fix Immediately

- When a test fails, stop and fix it
- Don't accumulate test failures or "investigate later"
- Every test failure is either a code bug or test bug - both need fixing

### Ask for Help, Don't Skip

- If a test is difficult to fix, explain the issue and ask the user
- If you don't understand a failure, investigate and ask
- If you need clarification on expected behavior, ask
- NEVER skip a test silently due to difficulty

### Test Maintenance is Code Maintenance

- Tests are first-class code, not second-class
- Keep tests readable, maintainable, and well-organized
- Refactor tests when refactoring code
- Update tests when requirements change

---

## Example Workflow

### Adding a New Feature: "Enhanced Memory Search"

1. **Write Unit Tests First:**
   ```python
   # tests/unit/test_memory_search_enhancement.py
   def test_enhanced_search_with_filters():
       """Test new filtering capability."""
       # Test implementation
   ```

2. **Write Integration Test:**
   ```python
   # tests/integration/test_enhanced_memory_search.py
   @pytest.mark.requires_chromadb
   @pytest.mark.requires_embedding
   async def test_enhanced_search_end_to_end():
       """Test enhanced search with real ChromaDB."""
       # Test implementation
   ```

3. **Run Tests (should fail initially):**
   ```bash
   python3 -m pytest tests/unit/test_memory_search_enhancement.py -v
   # FAIL - feature not implemented yet
   ```

4. **Implement the Feature:**
   ```python
   # custom_components/home_agent/memory_manager.py
   def enhanced_search(self, query, filters):
       # Implementation
   ```

5. **Run Tests Again:**
   ```bash
   python3 -m pytest tests/unit/test_memory_search_enhancement.py -v
   # PASS - feature works!
   ```

6. **Run Full Test Suite:**
   ```bash
   python3 -m pytest tests/unit/ -v
   ./scripts/run_integration_tests.sh
   # All PASS - no regressions!
   ```

7. **Commit:**
   ```bash
   git add tests/unit/test_memory_search_enhancement.py
   git add tests/integration/test_enhanced_memory_search.py
   git add custom_components/home_agent/memory_manager.py
   git commit -m "feat: Add enhanced memory search with filtering"
   ```

---

## Integration Test Environment

### Service Requirements

Integration tests require these services (configured in `.env.test`):
- **ChromaDB**: Vector database for entity and memory storage
- **LLM Endpoint**: OpenAI-compatible API for language model
- **Embedding Service**: Text embedding generation (Ollama or OpenAI)

### Health Checks

The integration test script automatically:
1. Verifies service connectivity before running tests
2. Skips tests gracefully if required services are unavailable
3. Reports which services are accessible

### Running Services Locally

If services are not available, you can run them locally:

```bash
# ChromaDB
docker run -p 8000:8000 chromadb/chroma

# Ollama (for LLM and embeddings)
docker run -p 11434:11434 ollama/ollama
ollama pull qwen2.5:7b-instruct
ollama pull mxbai-embed-large
```

---

## Enforcement

**This is not optional.** These rules apply to:
- All AI agents working on this codebase
- All human developers
- All pull requests
- All commits

**Violations:**
- Commits without running tests: Revert and fix
- Skipped tests without approval: Unskip and fix
- Disabled tests without documentation: Re-enable and fix
- Missing test coverage for new features: Add tests before merging

---

## Summary

**The Golden Rule:**
> If you didn't test it, it doesn't work. If it doesn't work, don't commit it.

**When in doubt:**
1. Write more tests, not fewer
2. Ask the user, don't skip
3. Fix the problem, don't work around it
4. Test first, commit second

**Remember:**
- Tests protect the codebase from regressions
- Tests document expected behavior
- Tests enable confident refactoring
- Tests catch bugs before users do

**All tests must pass. No exceptions.**

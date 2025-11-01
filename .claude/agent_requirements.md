# Claude Agent Requirements for Home-Agent Project

## Project Context

This is a dev container environment with **two separate projects**:

1. **home-agent** (`/workspaces/home-agent/`) - **THIS IS THE PROJECT WE WORK ON**
   - Custom Home Assistant integration being developed
   - Located in separate directory from core
   - **Full permissions to modify, commit, and push**

2. **core** (`/workspaces/core/`) - Home Assistant Core (READ-ONLY)
   - External git repository (not owned by project)
   - Used for development environment and testing
   - **NEVER modify files directly in this directory**
   - The home-agent integration is symlinked into `core/custom_components/` for testing

## Permissions and Restrictions

### ✅ ALLOWED - Home-Agent Repository

**File Operations:**
- ✅ Create, read, edit, delete any file in `/workspaces/home-agent/`
- ✅ Create new directories and files as needed
- ✅ Modify code, tests, documentation, configuration

**Git Operations:**
- ✅ Create commits in home-agent repository
- ✅ Push commits to `github.com/aradlein/home-agent`
- ✅ Create and switch branches
- ✅ Merge branches
- ✅ Tag releases

**GitHub Operations:**
- ✅ Create GitHub issues in `aradlein/home-agent`
- ✅ Modify GitHub issues (labels, assignees, milestones)
- ✅ Close GitHub issues
- ✅ Create pull requests
- ✅ Comment on issues and PRs

**Testing:**
- ✅ Create test files
- ✅ Run tests (pytest, unit tests, integration tests)
- ✅ Execute linters and formatters (`ruff`, `pylint`)
- ✅ Run non-mutating system commands for validation

**Development:**
- ✅ Install Python dependencies
- ✅ Run development scripts (`./scripts/format.sh`, `./scripts/lint.sh`)
- ✅ Execute Home Assistant commands from `/workspaces/core/` for testing

### ❌ FORBIDDEN - Core Repository

**File Operations:**
- ❌ NEVER modify files in `/workspaces/core/homeassistant/` directly
- ❌ NEVER create commits in the core repository
- ❌ NEVER push to `github.com/home-assistant/core`

**Git Operations:**
- ❌ NEVER commit changes to core repository
- ❌ NEVER create branches in core repository
- ❌ NEVER push to core remote

**GitHub Operations:**
- ❌ NEVER create issues in `home-assistant/core` repository
- ❌ NEVER create PRs against Home Assistant core (unless explicitly instructed)

**Exception:** If you accidentally create an issue in the wrong repo (like #155576), close it immediately with a comment explaining it was meant for home-agent.

## Working Directory Rules

### Always Work From Home-Agent Directory

```bash
# Correct working directory for code changes
cd /workspaces/home-agent

# Edit files in home-agent
edit /workspaces/home-agent/custom_components/home_agent/agent.py

# Commit changes
git add .
git commit -m "Add feature"
git push
```

### Use Core Directory Only for Testing

```bash
# Run Home Assistant from core directory for testing
cd /workspaces/core
hass --script check_config

# Run integration tests
cd /workspaces/core
pytest tests/components/home_agent/

# Run linters on home-agent code from core
cd /workspaces/core
python3 -m ruff check custom_components/home_agent/
```

## Git Operations Best Practices

### Before Making Any File Changes

1. **Verify you're in the correct repository:**
   ```bash
   pwd  # Should be /workspaces/home-agent
   git remote -v  # Should show aradlein/home-agent
   ```

2. **Check current branch:**
   ```bash
   git branch --show-current
   ```

3. **Create feature branch if needed:**
   ```bash
   git checkout -b feature/phase-3-external-llm
   ```

### Committing Changes

1. **Stage changes:**
   ```bash
   git add <files>
   # or
   git add .
   ```

2. **Create descriptive commit:**
   ```bash
   git commit -m "feat: implement external LLM tool

   - Add ExternalLLMTool class with BaseTool interface
   - Implement prompt + context passthrough
   - Add standardized error handling
   - Include unit tests with >80% coverage

   Resolves #2"
   ```

3. **Push to remote:**
   ```bash
   git push origin feature/phase-3-external-llm
   ```

### Commit Message Format

Use conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## GitHub Issue Management

### Creating Issues

**Always specify the correct repository:**

```bash
# Correct - home-agent repository
gh issue create --repo aradlein/home-agent --title "..." --body "..."

# Wrong - would create in core repository
gh issue create --title "..."  # Uses current directory's repo
```

### Issue Templates

Include in every issue:
- Overview and user story
- Acceptance criteria (checkboxes)
- Technical specifications
- File structure (NEW/MODIFIED)
- Definition of done
- Dependencies
- References to PROJECT_SPEC.md

### Closing Issues

```bash
# Close issue with comment
gh issue close 5 --repo aradlein/home-agent --comment "Completed in PR #10"

# Close via commit message
git commit -m "feat: add feature

Closes #5"
```

## File Safety Checks

### Before Modifying ANY File

1. **Check the file path:**
   ```python
   file_path = "/workspaces/home-agent/custom_components/home_agent/agent.py"

   if "/workspaces/core/" in file_path:
       # STOP! This is in the core directory
       raise ValueError("Cannot modify core repository files")
   ```

2. **Verify it's in home-agent:**
   ```bash
   # File should start with /workspaces/home-agent/
   if not file_path.startswith("/workspaces/home-agent/"):
       # Ask user for clarification
       return
   ```

3. **If unsure, ASK the user before proceeding**

## Testing Commands

### Run from Core Directory

```bash
# Integration tests for home-agent
cd /workspaces/core
pytest tests/components/home_agent/ -v --cov

# Linting
cd /workspaces/core
ruff check custom_components/home_agent/
pylint custom_components/home_agent/

# Type checking
mypy custom_components/home_agent/
```

### Development Scripts (from home-agent)

```bash
cd /workspaces/home-agent

# Format code
./scripts/format.sh

# Lint code
./scripts/lint.sh

# Run unit tests
pytest tests/unit/ -v
```

## Quick Reference Commands

### Verify Current Repository

```bash
# Check current directory
pwd

# Check git remote
git remote -v

# Check current branch
git branch --show-current
```

### Safe Git Workflow

```bash
# 1. Navigate to home-agent
cd /workspaces/home-agent

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Run tests
pytest tests/unit/ -v

# 5. Format and lint
./scripts/format.sh

# 6. Stage and commit
git add .
git commit -m "feat: add my feature"

# 7. Push
git push origin feature/my-feature

# 8. Create PR or merge
gh pr create --title "..." --body "..."
```

### Safe GitHub Commands

```bash
# Always specify repository
gh issue create --repo aradlein/home-agent --title "..." --body "..."
gh issue list --repo aradlein/home-agent
gh issue close 5 --repo aradlein/home-agent

# Create PR
gh pr create --repo aradlein/home-agent --title "..." --body "..."
```

## Common Mistakes to Avoid

### ❌ Wrong: Modifying Core Files

```bash
# NEVER DO THIS
edit /workspaces/core/homeassistant/components/conversation/agent.py
```

### ✅ Correct: Modifying Home-Agent Files

```bash
# DO THIS
edit /workspaces/home-agent/custom_components/home_agent/agent.py
```

### ❌ Wrong: Creating Issue in Core Repo

```bash
# NEVER DO THIS (from /workspaces/core directory)
cd /workspaces/core
gh issue create --title "..."  # Goes to home-assistant/core
```

### ✅ Correct: Creating Issue in Home-Agent Repo

```bash
# DO THIS
gh issue create --repo aradlein/home-agent --title "..."
```

### ❌ Wrong: Committing to Core

```bash
# NEVER DO THIS
cd /workspaces/core
git add .
git commit -m "..."
git push
```

### ✅ Correct: Committing to Home-Agent

```bash
# DO THIS
cd /workspaces/home-agent
git add .
git commit -m "..."
git push
```

## Emergency Procedures

### If You Accidentally Modified Core

1. **Immediately discard changes:**
   ```bash
   cd /workspaces/core
   git reset --hard HEAD
   git clean -fd
   ```

2. **Verify no commits were made:**
   ```bash
   git log -1
   # Should show a commit from Home Assistant, not you
   ```

### If You Created Issue in Wrong Repo

1. **Close the issue immediately:**
   ```bash
   gh issue close <issue-number> --repo home-assistant/core \
     --comment "Wrong repository - meant for aradlein/home-agent"
   ```

2. **Create in correct repo:**
   ```bash
   gh issue create --repo aradlein/home-agent --title "..." --body "..."
   ```

## Summary

### Remember These Key Points

1. **home-agent** (`/workspaces/home-agent/`) = ✅ FULL ACCESS
2. **core** (`/workspaces/core/`) = ❌ READ-ONLY
3. Always specify `--repo aradlein/home-agent` for GitHub commands
4. Always check `pwd` and `git remote -v` before making changes
5. When in doubt, ASK the user before modifying files
6. Test from core, develop in home-agent

---

**Last Updated:** 2025-01-XX
**Status:** Active Development Guidelines

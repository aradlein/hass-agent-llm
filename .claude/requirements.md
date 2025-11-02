# Claude Code Development Requirements

This document defines mandatory requirements for working on the Home Agent project with Claude Code.

---

## 📋 Pre-Commit Checklist

Before committing any code changes, you MUST verify and update affected documentation.

### 1. Documentation Validation

When making code changes, check if any of these docs need updates:

#### User-Facing Documentation (`docs/`)
- [ ] **INSTALLATION.md** - Changed installation steps, dependencies, or prerequisites?
- [ ] **CONFIGURATION.md** - Added/changed configuration options or settings?
- [ ] **API_REFERENCE.md** - Added/modified services, events, or tools?
- [ ] **TROUBLESHOOTING.md** - Introduced new error types or common issues?
- [ ] **EXAMPLES.md** - Changed API usage patterns that affect examples?
- [ ] **MEMORY_SYSTEM.md** - Modified memory system behavior or configuration?
- [ ] **VECTOR_DB_SETUP.md** - Changed vector DB integration or setup?
- [ ] **FAQ.md** - Added features that warrant new FAQ entries?
- [ ] **MIGRATION.md** - Introduced breaking changes?

#### Feature Documentation (`docs/`)
- [ ] **CUSTOM_TOOLS.md** - Modified custom tool framework or handlers?
- [ ] **EXTERNAL_LLM.md** - Changed external LLM integration?

#### Reference Documentation (`docs/reference/`)
- [ ] Update comprehensive reference docs when quick-start guides change
- [ ] Ensure examples still work with code changes
- [ ] Update parameter schemas and configuration tables

#### Developer Documentation (`.claude/docs/`)
- [ ] **PROJECT_SPEC.md** - Completed phase tasks, changed architecture?
- [ ] **DEVELOPMENT.md** - New testing requirements or code standards?

#### Core Documentation
- [ ] **README.md** - Changed core features, requirements, or installation?

---

## 🔄 Documentation Update Rules

### Rule 1: Code + Docs = One Commit
**Documentation updates MUST be included in the same commit as the code changes.**

❌ **Bad:**
```
Commit 1: Add new memory configuration options
Commit 2: Update documentation for memory options
```

✅ **Good:**
```
Commit 1: feat: add memory TTL configuration
- Add CONF_MEMORY_TTL_* constants
- Implement TTL-based memory expiration
- Update MEMORY_SYSTEM.md with TTL settings
- Update CONFIGURATION.md reference
```

### Rule 2: Breaking Changes = Migration Guide Update
Any breaking changes MUST include:
1. Update to `docs/MIGRATION.md` with upgrade path
2. Update to `.claude/docs/PROJECT_SPEC.md` (breaking changes section)
3. Update to `README.md` changelog if applicable

### Rule 3: New Features = Documentation First
When adding new features:
1. Update `.claude/docs/PROJECT_SPEC.md` (mark task as complete)
2. Update user-facing docs in `docs/`
3. Add examples to `docs/EXAMPLES.md`
4. Update `docs/reference/` comprehensive docs
5. Add FAQ entry if the feature is commonly asked about

### Rule 4: API Changes = Reference Update
When modifying services, events, or tools:
1. Update `docs/API_REFERENCE.md` (quick reference)
2. Update `docs/reference/API_REFERENCE.md` (comprehensive)
3. Update examples that use the changed API
4. Add to `docs/TROUBLESHOOTING.md` if new errors are possible

### Rule 5: Configuration Changes = Two-Level Update
When adding/changing configuration options:
1. Update `docs/CONFIGURATION.md` (essential settings)
2. Update `docs/reference/CONFIGURATION.md` (all settings)
3. Update relevant feature guides (MEMORY_SYSTEM, VECTOR_DB_SETUP, etc.)
4. Update examples if default behavior changes

---

## 🎯 Documentation Quality Standards

### Quick-Start Docs (`docs/`)
- **Length:** 200-400 lines maximum
- **Focus:** Getting started quickly, essential information only
- **Style:** Direct, actionable, scannable (bullets, tables, code blocks)
- **Links:** Always link to comprehensive reference for details
- **Examples:** One working example per concept, copy-paste ready

### Reference Docs (`docs/reference/`)
- **Length:** No limit, comprehensive coverage
- **Focus:** Complete information, advanced scenarios, edge cases
- **Style:** Detailed explanations, multiple examples, troubleshooting
- **Links:** Cross-reference related topics
- **Examples:** Multiple variations, real-world scenarios

### Developer Docs (`.claude/docs/`)
- **Audience:** AI agents and developers building features
- **Focus:** Technical specifications, architecture, testing standards
- **PROJECT_SPEC.md:** Keep roadmap current, mark completed tasks
- **DEVELOPMENT.md:** Update when testing standards change

---

## 🚨 Validation Before Commit

Before running `git commit`, verify:

### 1. Documentation Accuracy
```bash
# Check that examples in docs still work
# Read modified docs and verify:
- Code examples match current API
- Configuration constants match const.py
- Service schemas match services.yaml
- Event schemas match agent.py definitions
```

### 2. Documentation Completeness
```bash
# Verify all affected docs are updated:
git status  # Check which files changed
# Ask yourself: Does this code change affect any docs?
# Update those docs BEFORE committing
```

### 3. Documentation Consistency
```bash
# Ensure consistency across doc levels:
- Quick-start doc updated? → Update reference doc too
- Changed API? → Update API_REFERENCE.md + examples
- New config option? → Update CONFIGURATION.md + feature guide
```

### 4. Links Still Work
```bash
# Verify internal documentation links:
- Check relative paths are correct
- Verify linked sections exist
- Test that examples reference valid files
```

---

## 📝 Commit Message Format

Include documentation updates in commit messages:

```
<type>(<scope>): <subject>

<body describing code changes>

Documentation updates:
- Updated docs/CONFIGURATION.md with new CONF_MEMORY_TTL settings
- Added TTL examples to docs/MEMORY_SYSTEM.md
- Updated docs/reference/CONFIGURATION.md comprehensive guide

<footer>
```

### Commit Types
- `feat:` - New feature (update feature docs + API reference)
- `fix:` - Bug fix (update troubleshooting if needed)
- `docs:` - Documentation only (no code changes)
- `refactor:` - Code refactoring (update if API surface changed)
- `test:` - Test changes (update DEVELOPMENT.md if test patterns changed)
- `chore:` - Build/config changes (update INSTALLATION if dependencies changed)

---

## 🔍 Review Checklist

Before pushing changes, review:

- [ ] All affected documentation files are updated
- [ ] Documentation updates are in the same commit as code changes
- [ ] Examples in documentation are tested and work
- [ ] Configuration constants in docs match `const.py`
- [ ] Service definitions in docs match `services.yaml`
- [ ] Breaking changes documented in MIGRATION.md
- [ ] PROJECT_SPEC.md roadmap is current
- [ ] Quick-start docs link to reference docs
- [ ] Reference docs are comprehensive
- [ ] No broken internal links

---

## 🎓 Examples

### Example 1: Adding a New Configuration Option

**Code Changes:**
- Add `CONF_MEMORY_AUTO_CLEANUP` to `const.py`
- Implement auto-cleanup in `memory_manager.py`

**Required Documentation Updates:**
1. `docs/CONFIGURATION.md` - Add to essential settings table
2. `docs/reference/CONFIGURATION.md` - Add full description with defaults
3. `docs/MEMORY_SYSTEM.md` - Add to configuration section
4. `docs/reference/MEMORY_SYSTEM.md` - Add detailed explanation
5. `.claude/docs/PROJECT_SPEC.md` - Mark task as complete if applicable

**Commit:**
```
feat(memory): add automatic memory cleanup configuration

Add CONF_MEMORY_AUTO_CLEANUP option to enable/disable automatic
cleanup of expired memories. Cleanup runs every 24 hours by default.

Documentation updates:
- Updated docs/CONFIGURATION.md with new setting
- Updated docs/MEMORY_SYSTEM.md with cleanup behavior
- Updated comprehensive reference docs
- Added example to memory system guide

Closes #XX
```

---

### Example 2: Fixing a Bug

**Code Changes:**
- Fix tool execution error handling in `tool_handler.py`

**Required Documentation Updates:**
1. `docs/TROUBLESHOOTING.md` - Add to common issues if user-facing
2. `docs/reference/TROUBLESHOOTING.md` - Add detailed troubleshooting
3. `.claude/docs/PROJECT_SPEC.md` - Update if this fixes a tracked issue

**Commit:**
```
fix(tools): improve error handling for failed tool executions

Tool execution errors now properly propagate to the LLM with
structured error messages instead of raising exceptions.

Documentation updates:
- Added tool execution errors to TROUBLESHOOTING.md
- Updated error handling examples in API_REFERENCE.md

Fixes #XX
```

---

### Example 3: Adding a New Service

**Code Changes:**
- Add `home_agent.export_memories` service in `services.yaml`
- Implement service handler

**Required Documentation Updates:**
1. `docs/API_REFERENCE.md` - Add service to quick reference
2. `docs/reference/API_REFERENCE.md` - Add full parameter schema
3. `docs/MEMORY_SYSTEM.md` - Add usage example
4. `docs/reference/MEMORY_SYSTEM.md` - Add detailed usage
5. `docs/EXAMPLES.md` - Add practical example if useful
6. README.md - Add to services list if it's a major feature

**Commit:**
```
feat(memory): add export_memories service

Add new service to export all memories to JSON format for backup
and migration purposes. Supports filtering by type and date range.

Documentation updates:
- Added to docs/API_REFERENCE.md quick reference
- Updated docs/MEMORY_SYSTEM.md with export example
- Added comprehensive docs in docs/reference/
- Added automation example using export service

Closes #XX
```

---

## 🚀 Quick Reference

**Before every commit, ask:**

1. ✅ Did I change any configuration options? → Update CONFIGURATION.md
2. ✅ Did I add/modify services or events? → Update API_REFERENCE.md
3. ✅ Did I add a new feature? → Update feature guide + examples
4. ✅ Did I introduce breaking changes? → Update MIGRATION.md
5. ✅ Did I complete a phase task? → Update PROJECT_SPEC.md
6. ✅ Are all examples still valid? → Test and update if needed
7. ✅ Are docs in sync with code? → Verify before committing

**Remember:** Documentation is not optional—it's part of the feature.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-02
**Applies to:** All code contributions to Home Agent

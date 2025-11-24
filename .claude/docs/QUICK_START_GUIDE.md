# Quick Start Guide - HACS Submission Preparation

**Repository:** home-agent → **hass-agent-llm** (public)
**Track:** Complete (6-8 weeks)
**Target:** Production-ready HACS submission

---

## Repository Strategy

### Two-Repository Setup

**PRIVATE (Development):** `github.com/aradlein/home-agent`
- Where you do all development work
- Keeps your private history
- Connected to public via remote

**PUBLIC (Release):** `github.com/aradlein/hass-agent-llm`
- Clean history starting from public release
- What users see and install via HACS
- Receives synced changes from private

### One-Time Setup

```bash
# 1. Create public repo on GitHub (hass-agent-llm)

# 2. Clone private repo and remove history
git clone https://github.com/aradlein/home-agent.git home-agent-public
cd home-agent-public
rm -rf .git
git init
git add .
git commit -m "Initial public release v0.6.4-beta"

# 3. Push to public
git remote add origin https://github.com/aradlein/hass-agent-llm.git
git push -u origin main
git tag v0.6.4-beta
git push origin v0.6.4-beta

# 4. In your PRIVATE repo, add public remote
cd /path/to/private/home-agent
git remote add public https://github.com/aradlein/hass-agent-llm.git
```

### Daily Workflow

**Development (in private repo):**
```bash
# Work normally
git add .
git commit -m "feat: Add cool feature"
git push origin main  # Push to PRIVATE
```

**When ready to sync to public:**
```bash
# Push to public
git push public main
git push public --tags
```

**Selective sync (if needed):**
```bash
# Create curated branch
git checkout -b public-sync
git cherry-pick <commit-hash>  # Pick specific commits
git push public public-sync:main
```

---

## Week-by-Week Plan

### Week 1: Critical Foundation (4-6 hours)
**Goal:** Get repository ready for public

- [ ] #1: Add LICENSE file (15 min)
- [ ] #2: Update manifest.json URLs (5 min)
- [ ] #3: Create info.md (30 min)
- [ ] #4: Create public repo (1 hour)
- [ ] #5: Run HACS validation (30 min)
- [ ] #6: Issue templates (1 hour)

**Deliverable:** Public repo exists and passes HACS validation

---

### Week 2: Agent Refactoring (10-14 hours)
**Goal:** Make codebase contributor-friendly

- [ ] #7: Split agent.py into 4 modules (8-12 hours)
  - `agent/core.py` - Orchestration
  - `agent/llm.py` - API calls
  - `agent/streaming.py` - Streaming
  - `agent/memory_extraction.py` - Memory
- [ ] #8: Add module docstrings (1-2 hours)

**Deliverable:** agent.py reduced from 1,826 → 400 lines per module

---

### Week 3: Features & Config (10-14 hours)
**Goal:** Add proxy headers, clean up config

- [ ] #9: Proxy headers feature (4-6 hours)
- [ ] #10: Refactor config_flow.py (4-6 hours)
- [ ] #11: Add screenshots (1-2 hours)

**Deliverable:** Flexible proxy headers + cleaner config code

---

### Week 4: Memory & Documentation (6-9 hours)
**Goal:** Clean memory code, add contributor docs

- [ ] #12: Extract MemoryValidator (3-4 hours)
- [ ] #13: Create CONTRIBUTING.md (1-2 hours)
- [ ] #14: Architecture diagram (2-3 hours)

**Deliverable:** Better memory validation + contributor guide

---

### Week 5: Security & Examples (6-9 hours)
**Goal:** Audit security, add examples

- [ ] #15: Security audit (3-4 hours)
- [ ] #16: Example configs (2-3 hours)
- [ ] #17: SECURITY.md (30 min)

**Deliverable:** Security verified + ready-to-use examples

---

### Week 6: Testing & Polish (6-8 hours)
**Goal:** Comprehensive tests, polish docs

- [ ] #18: Integration tests (4-5 hours)
- [ ] #19: PR template (30 min)
- [ ] #20: README badges (30 min)

**Deliverable:** >80% test coverage + polished README

---

### Week 7-8: Community & Release (6-10 hours)
**Goal:** Launch to community

- [ ] #21: HACS submission (1-2 hours)
- [ ] #22: Forum announcement (1-2 hours)
- [ ] #23: Demo video (2-3 hours)
- [ ] #24: Beta testing (ongoing)
- [ ] #25: v1.0.0 release (2-3 hours)

**Deliverable:** Public launch, HACS approved, v1.0.0 released

---

## Priority Labels

Use these labels to organize issues:

- `required` - Must be done before HACS submission
- `HACS` - Related to HACS compliance
- `refactoring` - Code restructuring
- `security` - Security-related
- `documentation` - Docs improvements
- `testing` - Test coverage
- `community` - Community engagement
- `enhancement` - New features
- `breaking-change` - May break existing configs

---

## Quick Commands Reference

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=custom_components.home_agent --cov-report=html

# Run specific test file
pytest tests/unit/test_agent.py -v
```

### Code Quality
```bash
# Format code
black custom_components/home_agent/
isort custom_components/home_agent/

# Lint
flake8 custom_components/home_agent/
pylint custom_components/home_agent/

# Type check
mypy custom_components/home_agent/
```

### HACS Validation
```bash
# Using GitHub Action (preferred)
# Push to repo, check Actions tab

# Manual validation
# Install HACS validate action locally
```

---

## Issue Creation Commands

### Create all issues at once (via GitHub CLI)

```bash
# Install GitHub CLI: https://cli.github.com/

# Authenticate
gh auth login

# Create issues from plan (you'll need to do this manually for each issue)
# Or use a script to batch create from GITHUB_ISSUES_PLAN.md
```

### Create issues manually

1. Go to: https://github.com/aradlein/home-agent/issues
2. Click "New Issue"
3. Copy/paste from `.claude/docs/GITHUB_ISSUES_PLAN.md`
4. Add appropriate labels
5. Set milestone (optional)
6. Assign to yourself

---

## Milestones

Suggest creating these milestones:

1. **Week 1: Foundation** (Issues #1-6)
2. **Week 2-3: Refactoring** (Issues #7-11)
3. **Week 4-5: Quality** (Issues #12-17)
4. **Week 6: Testing** (Issues #18-20)
5. **Week 7-8: Launch** (Issues #21-25)
6. **v1.0.0 Release** (All issues)

---

## Success Metrics

### Week 1
- ✅ Public repo live
- ✅ HACS validation passes
- ✅ Issue templates work

### Week 2-3
- ✅ Agent.py split into 4 modules
- ✅ Config_flow.py reduced by 50%
- ✅ Proxy headers feature working

### Week 4-5
- ✅ Security audit complete
- ✅ CONTRIBUTING.md helpful
- ✅ Example configs tested

### Week 6
- ✅ Integration tests >15
- ✅ Test coverage >80%
- ✅ README polished

### Week 7-8
- ✅ HACS PR submitted
- ✅ Forum post live
- ✅ 10+ beta testers
- ✅ v1.0.0 released

---

## Common Questions

**Q: Can I work on issues out of order?**
A: Yes! Week 1 issues are critical, but Weeks 2-8 can be flexible.

**Q: What if I don't have time for all issues?**
A: Focus on Week 1 (critical) + Week 2 (agent.py refactor). Others are enhancements.

**Q: Should I create issues in private or public repo?**
A: Start in **private** repo. Move to public when you create public repo.

**Q: When should I submit to HACS?**
A: After Week 1 (minimum) or Week 6 (recommended).

**Q: How do I handle breaking changes?**
A: Document in CHANGELOG.md, add migration guide, bump major version.

---

## Getting Help

- **Analysis:** `.claude/docs/HACS_SUBMISSION_ANALYSIS.md`
- **Full Plan:** `.claude/docs/GITHUB_ISSUES_PLAN.md`
- **Project Spec:** `.claude/docs/PROJECT_SPEC.md`
- **Ask Claude:** I'm here to help implement any of these issues!

---

## Next Steps

1. ✅ Review this guide
2. ✅ Review full issues plan
3. ⬜ Create GitHub issues from plan
4. ⬜ Set up project board (optional)
5. ⬜ Start with Issue #1 (LICENSE)
6. ⬜ Create public repository (Issue #4)
7. ⬜ Begin refactoring (Issues #7-11)

---

**Ready to start?** Let me know which issue you'd like to tackle first!

# Repository Setup & Public Release Strategy

## Overview

This project uses a **dual-repository model** for development and public release:

| Repository | Purpose | Access | URL |
|------------|---------|--------|-----|
| `home-agent` | Private development | Full access | `github.com/aradlein/home-agent` |
| `hass-agent-llm` | Public release (HACS) | Export only | `github.com/aradlein/hass-agent-llm` |

## Why Two Repositories?

1. **Clean History**: Public repo has no development noise, planning docs, or internal references
2. **Privacy**: Private infrastructure details (domains, names) stay in private repo
3. **HACS Compliance**: Public repo contains only what's needed for installation
4. **Flexibility**: Can develop freely without worrying about public visibility

## Integration Naming

The integration uses different names in different contexts:

| Context | Name | Reason |
|---------|------|--------|
| Directory name | `home_agent` | Internal Python module |
| Class names | `HomeAgent*` | Internal Python classes |
| Domain (services) | `home_agent` | HA service domain |
| **Display name** | `Home Agent` | User-facing in HA UI |
| **HACS name** | `Home Agent` | User-facing in HACS |
| **Public repo** | `hass-agent-llm` | GitHub repository name |
| **Private repo** | `home-agent` | Development repository |

## Export Process

The private repo uses `.gitattributes` with `export-ignore` patterns. When running `git archive`, these files are automatically excluded:

### Excluded from Public Repo
- `.claude/` - Agent documentation and rules
- `scripts/` - Development scripts
- `.devcontainer/` - Dev container config
- `tests/integration/` - Integration tests (contain private endpoints)
- `observability/` - Internal monitoring
- `docs/TESTING_*.md` - Development testing docs
- `.env.test*` - Test environment configs
- Planning markdown files

### Included in Public Repo
- `custom_components/home_agent/` - The actual integration
- `tests/unit/` - Unit tests
- User-facing docs (INSTALLATION, CONFIGURATION, etc.)
- `.github/workflows/` - CI/CD
- `README.md`, `info.md`, `LICENSE`

## Public Release Workflow

### Step 1: Ensure Private Repo is Clean
```bash
cd /workspaces/home-agent
git status  # Should be clean
git push    # Ensure all changes are pushed
```

### Step 2: Verify No Private Data
Before exporting, always check:
```bash
# Search for private references
grep -r "inorganic\.me" --exclude-dir=.git .
grep -ri "candace\|anton" --exclude-dir=.git --exclude-dir=.claude .
```

### Step 3: Export and Push
```bash
# Create clean export
rm -rf /tmp/hass-agent-llm
mkdir -p /tmp/hass-agent-llm
git archive --format=tar HEAD | tar -x -C /tmp/hass-agent-llm

# Verify export
cd /tmp/hass-agent-llm
grep -r "inorganic\.me" . && echo "FAIL!" || echo "Clean"

# Initialize and push
git init
git add .
git commit -m "Release vX.Y.Z-beta"
git branch -m master main
git remote add origin https://github.com/aradlein/hass-agent-llm.git
git push -u origin main --force

# Tag the release
git tag vX.Y.Z-beta
git push origin vX.Y.Z-beta
```

### Step 4: Verify on GitHub
- Check https://github.com/aradlein/hass-agent-llm
- Verify no private references visible
- Confirm release tag exists

## Safety Checklist Before Public Push

- [ ] Version in `manifest.json` matches intended release
- [ ] README.md has correct version in badges and changelog
- [ ] No `inorganic.me` references in exported files
- [ ] No personal names (Candace, Anton) in examples
- [ ] No references to private `home-agent` repo (only `hass-agent-llm`)
- [ ] All user-facing docs reference public repo URLs
- [ ] `.gitattributes` excludes all development files

## Common Mistakes to Avoid

1. **Pushing to wrong repo**: Always verify remote before pushing
   ```bash
   git remote -v  # Check before push!
   ```

2. **Forgetting to update version**: Manifest and README should match

3. **Leaking private data**: Always run grep checks before export

4. **Including test endpoints**: Integration tests contain real endpoints

## Version Synchronization

Keep versions in sync:
- `custom_components/home_agent/manifest.json` - Source of truth
- `README.md` - Badge and changelog
- Git tags - Release markers

Use `scripts/bump_version.sh` to update version consistently.

## Emergency: Accidentally Pushed Private Data

If private data is pushed to public repo:

1. **Immediately** force-push a clean version
2. Contact GitHub support to purge cached data
3. Rotate any exposed credentials
4. Review and update `.gitattributes` to prevent recurrence

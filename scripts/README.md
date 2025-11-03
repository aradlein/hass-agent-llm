# Scripts

This directory contains utility scripts for managing the Home Agent project.

## Production Deployment

### update_integration.sh

A script to safely update your Home Assistant production installation from a specific git tag.

**Usage:**
```bash
./scripts/update_integration.sh [OPTIONS] [TAG]
```

**Options:**
- `--dry-run` - Show all commands without executing them
- `--latest` - Use the latest git tag (default if no tag specified)
- `--help` - Show help message

**Examples:**
```bash
# Dry run to see what would happen with a specific tag
./scripts/update_integration.sh --dry-run v0.4.4-beta

# Update to the latest tag
./scripts/update_integration.sh --latest

# Update to a specific tag
./scripts/update_integration.sh v0.4.3-beta

# Dry run with latest tag
./scripts/update_integration.sh --dry-run --latest
```

**What it does:**
1. Validates that git repository exists at `/root/home-agent`
2. Fetches latest tags from remote
3. Determines target tag (latest or specified)
4. Stashes any local changes in the repo
5. Checks out the target tag
6. Creates a timestamped backup of current installation
7. Copies new version to `/root/config/custom_components/home_agent`
8. Verifies installation succeeded
9. Provides rollback instructions if needed

**Important:**
- Always run with `--dry-run` first to verify the commands
- Automatically creates backups before updating
- Requires Home Assistant restart after update (`ha core restart`)
- Provides rollback instructions in case of issues

**Paths:**
- Git Repository: `/root/home-agent`
- Source: `/root/home-agent/custom_components/home_agent`
- Destination: `/root/config/custom_components/home_agent`
- Backups: `/root/config/custom_components/home_agent_backup_YYYYMMDD_HHMMSS`

## Version Management

### bump_version.sh

A helper script to bump the version in `manifest.json` and create git tags.

**Usage:**
```bash
./scripts/bump_version.sh [major|minor|patch|beta|alpha|rc|release] [commit_message]
```

**Examples:**
```bash
# Bump patch version (0.4.2 -> 0.4.3)
./scripts/bump_version.sh patch

# Bump minor version (0.4.2 -> 0.5.0)
./scripts/bump_version.sh minor

# Bump major version (0.4.2 -> 1.0.0)
./scripts/bump_version.sh major

# Add beta suffix (0.4.2 -> 0.4.3-beta)
./scripts/bump_version.sh beta

# Remove suffix for release (0.4.3-beta -> 0.4.3)
./scripts/bump_version.sh release

# Custom commit message
./scripts/bump_version.sh patch "chore: release version"
```

**What it does:**
1. Reads current version from `manifest.json`
2. Calculates new version based on bump type
3. Confirms with you before making changes
4. Updates `manifest.json` with new version
5. Creates a git commit
6. Optionally creates a git tag
7. Optionally pushes to origin

**Note:** The script is interactive and will ask for confirmation at each step.

## Version Management Workflow

### Manual Version Bump (Recommended)

When you're ready to release a new version:

1. **Update the version manually:**
   ```bash
   # Edit manifest.json and change version field
   vim custom_components/home_agent/manifest.json
   ```

2. **Commit the change:**
   ```bash
   git add custom_components/home_agent/manifest.json
   git commit -m "chore: bump version to X.Y.Z"
   ```

3. **Create and push a tag:**
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push && git push --tags
   ```

4. **GitHub Actions will automatically:**
   - Validate that tag version matches manifest.json
   - Create a GitHub Release with changelog
   - Generate a release ZIP file
   - Validate HACS compatibility

### Using the Script

Alternatively, use the bump_version.sh script for automation:

```bash
./scripts/bump_version.sh patch
# Follow the interactive prompts
```

## Automated Release Process

When you push a version tag (e.g., `v0.4.3`), GitHub Actions will:

1. **Validate** - Check that tag matches manifest.json version
2. **Build** - Create release assets (ZIP file)
3. **Release** - Create GitHub Release with:
   - Automatic changelog from commits
   - Installation instructions
   - Documentation links
4. **Validate HACS** - Ensure HACS compatibility

## Version Scheme

Home Agent follows [Semantic Versioning](https://semver.org/):

- **X.Y.Z** - Stable releases
  - **X** (Major) - Breaking changes
  - **Y** (Minor) - New features (backward compatible)
  - **Z** (Patch) - Bug fixes

- **X.Y.Z-suffix** - Pre-releases
  - **-beta** - Beta releases for testing
  - **-alpha** - Alpha releases for early testing
  - **-rc** - Release candidates

**Examples:**
- `0.4.3` - Stable release
- `0.4.3-beta` - Beta version
- `0.5.0-rc1` - Release candidate 1
- `1.0.0` - Major release

## HACS Installation

Version information is read from `manifest.json`. HACS will:
- Display the current version to users
- Check for updates automatically
- Show version in the HACS UI

No manual version configuration needed in `hacs.json` - it reads from `manifest.json`.

## CI/CD Workflows

### validate-version.yml
- Runs on: PR to main, push to main (when manifest.json changes)
- Validates: Version format (semantic versioning)
- Warns: If version unchanged in PR

### release.yml
- Triggers on: Git tags matching `v*.*.*`
- Actions:
  1. Validates tag matches manifest.json
  2. Generates changelog from git commits
  3. Creates GitHub Release (pre-release if beta/alpha/rc)
  4. Uploads release ZIP
  5. Validates HACS compatibility

## Best Practices

1. **Always update manifest.json version** before tagging
2. **Use semantic versioning** - makes it clear what changed
3. **Tag format:** Always prefix with `v` (e.g., `v0.4.3`)
4. **Beta releases:** Use for testing before stable release
5. **Changelog:** Write clear commit messages - they become release notes
6. **Testing:** Test thoroughly before removing beta suffix

## Quick Reference

```bash
# Current version
jq -r '.version' custom_components/home_agent/manifest.json

# Manual bump (edit file, then):
git add custom_components/home_agent/manifest.json
git commit -m "chore: bump version to 0.4.3"
git tag -a v0.4.3 -m "Release version 0.4.3"
git push && git push --tags

# Script-assisted bump
./scripts/bump_version.sh patch

# Check latest tag
git describe --tags --abbrev=0

# List all tags
git tag -l
```

# Version Management Rules

## Critical: Always Update All Version Files

When bumping the version or tagging a new release, you MUST update the version in ALL of the following locations:

### Required Version Updates

1. **manifest.json** - Primary version source
   - Path: `custom_components/home_agent/manifest.json`
   - Field: `"version": "X.Y.Z-suffix"`
   - This is what Home Assistant displays in the UI

2. **const.py** - Python constant
   - Path: `custom_components/home_agent/const.py`
   - Constant: `VERSION: Final = "X.Y.Z-suffix"`
   - Used in code and logging

### Update Process

When tagging a new version (e.g., v0.7.0-beta):

1. **Update manifest.json**
   ```bash
   # Edit custom_components/home_agent/manifest.json
   # Change "version": "0.6.4-beta" → "version": "0.7.0-beta"
   ```

2. **Update const.py**
   ```bash
   # Edit custom_components/home_agent/const.py
   # Change VERSION: Final = "0.6.4-beta" → VERSION: Final = "0.7.0-beta"
   ```

3. **Commit both files together**
   ```bash
   git add custom_components/home_agent/manifest.json custom_components/home_agent/const.py
   git commit -m "chore: Bump version to X.Y.Z-suffix"
   ```

4. **Create and push tag**
   ```bash
   git push
   git tag vX.Y.Z-suffix
   git push origin vX.Y.Z-suffix
   ```

### Automated Script

You can use the `scripts/bump_version.sh` script to automate this process:

```bash
./scripts/bump_version.sh [major|minor|patch|beta|rc|release]
```

This script will:
- Update manifest.json
- Update const.py (if you enhance it to do so)
- Create a git commit
- Create a git tag
- Push to remote

### Version Format

Follow semantic versioning with optional suffix:
- Format: `MAJOR.MINOR.PATCH[-SUFFIX]`
- Examples:
  - `0.7.0-beta` (beta release)
  - `0.7.0-rc1` (release candidate)
  - `1.0.0` (stable release)
  - `1.2.3` (stable with patches)

### Checklist for Version Bumps

Before tagging a new version:

- [ ] Update `custom_components/home_agent/manifest.json`
- [ ] Update `custom_components/home_agent/const.py`
- [ ] Verify both files have the same version string
- [ ] Update CHANGELOG.md or release notes (if applicable)
- [ ] Commit with message: `chore: Bump version to X.Y.Z-suffix`
- [ ] Create git tag: `vX.Y.Z-suffix`
- [ ] Push commit and tag to remote

### Common Mistakes to Avoid

❌ **DON'T** update only manifest.json
❌ **DON'T** update only const.py
❌ **DON'T** forget to push the tag
❌ **DON'T** use inconsistent version strings between files

✅ **DO** update both files in the same commit
✅ **DO** use the same version string in both places
✅ **DO** follow semantic versioning
✅ **DO** push both commit and tag

### Why This Matters

- **manifest.json**: Home Assistant reads this to display version in UI
- **const.py**: Code uses this for logging, diagnostics, and runtime checks
- **Inconsistency**: Different versions in different places causes confusion and debugging issues

### Verification

After updating, verify with:

```bash
# Check manifest.json version
jq -r '.version' custom_components/home_agent/manifest.json

# Check const.py version
grep 'VERSION: Final' custom_components/home_agent/const.py

# Verify they match
```

Both should output the same version string.

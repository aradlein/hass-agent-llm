# Home Agent Production Deployment Guide

This guide explains how to deploy Home Agent updates to your Home Assistant production environment.

## Quick Start

### First-Time Setup

1. **Clone the repository on your Home Assistant OS:**
   ```bash
   cd /root
   git clone https://github.com/yourusername/home-agent.git
   ```

2. **Make the update script executable:**
   ```bash
   chmod +x /root/home-agent/scripts/update_integration.sh
   ```

3. **Create an alias for easy access (optional):**
   ```bash
   # Option 1: Create an alias
   echo 'alias update-home-agent="/root/home-agent/scripts/update_integration.sh"' >> ~/.bashrc
   source ~/.bashrc

   # Option 2: Install quick-update wrapper to system (recommended)
   cp /root/home-agent/scripts/quick-update.sh /usr/local/bin/update-home-agent
   chmod +x /usr/local/bin/update-home-agent

   # Then you can run from anywhere:
   update-home-agent --dry-run --latest
   ```

### Updating to Latest Version

**Always perform a dry-run first:**

```bash
/root/home-agent/scripts/update_integration.sh --dry-run --latest
```

**If everything looks good, run the actual update:**

```bash
/root/home-agent/scripts/update_integration.sh --latest
```

**Restart Home Assistant:**

```bash
ha core restart
```

### Updating to Specific Version

**Check available versions:**

```bash
cd /root/home-agent
git fetch --tags
git tag -l
```

**Dry-run with specific version:**

```bash
/root/home-agent/scripts/update_integration.sh --dry-run v0.4.4-beta
```

**Apply the update:**

```bash
/root/home-agent/scripts/update_integration.sh v0.4.4-beta
ha core restart
```

## Example Output

### Dry-Run Mode

```bash
$ /root/home-agent/scripts/update_integration.sh --dry-run v0.4.4-beta

========================================
Home Agent Integration Update Script
========================================

[WARNING] DRY RUN MODE - No changes will be made

[SUCCESS] Git repository found at: /root/home-agent
[INFO] Changing to git repository
[CMD] cd /root/home-agent
[INFO] Fetching latest tags from remote
[CMD] git fetch --tags
[INFO] Using specified tag: v0.4.4-beta
[INFO] Current git status:
[INFO] Stashing any local changes
[CMD] git stash
[INFO] Checking out tag: v0.4.4-beta
[CMD] git checkout v0.4.4-beta
[SUCCESS] Integration source found at: /root/home-agent/custom_components/home_agent
[INFO] Creating backup at: /root/config/custom_components/home_agent_backup_20241103_120000
[CMD] cp -r /root/config/custom_components/home_agent /root/config/custom_components/home_agent_backup_20241103_120000
[INFO] Removing old installation files
[CMD] rm -rf /root/config/custom_components/home_agent/*
[INFO] Copying new integration files
[CMD] cp -r /root/home-agent/custom_components/home_agent/* /root/config/custom_components/home_agent/
[INFO] Verifying installation...
[CMD] test -f /root/config/custom_components/home_agent/manifest.json
[CMD] grep -E '"version"|"name"' /root/config/custom_components/home_agent/manifest.json

========================================
Update Summary
========================================
Git Tag:        v0.4.4-beta
Source:         /root/home-agent/custom_components/home_agent
Destination:    /root/config/custom_components/home_agent
========================================

[WARNING] DRY RUN COMPLETED - No changes were made

[INFO] To perform the actual update, run without --dry-run flag:
  /root/home-agent/scripts/update_integration.sh v0.4.4-beta
```

### Actual Update Mode

```bash
$ /root/home-agent/scripts/update_integration.sh v0.4.4-beta

========================================
Home Agent Integration Update Script
========================================

[SUCCESS] Git repository found at: /root/home-agent
[INFO] Changing to git repository
[CMD] cd /root/home-agent
[INFO] Fetching latest tags from remote
[CMD] git fetch --tags
Already up to date.
[INFO] Using specified tag: v0.4.4-beta
[INFO] Current git status:
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
[INFO] Stashing any local changes
[CMD] git stash
No local changes to save
[INFO] Checking out tag: v0.4.4-beta
[CMD] git checkout v0.4.4-beta
Note: switching to 'v0.4.4-beta'.
HEAD is now at c248b79 fix: Correct version to 0.4.4-beta
[SUCCESS] Integration source found at: /root/home-agent/custom_components/home_agent
[INFO] Creating backup at: /root/config/custom_components/home_agent_backup_20241103_120000
[CMD] cp -r /root/config/custom_components/home_agent /root/config/custom_components/home_agent_backup_20241103_120000
[INFO] Removing old installation files
[CMD] rm -rf /root/config/custom_components/home_agent/*
[INFO] Copying new integration files
[CMD] cp -r /root/home-agent/custom_components/home_agent/* /root/config/custom_components/home_agent/
[INFO] Verifying installation...
[SUCCESS] Installation verified - manifest.json found

[INFO] Installed version information:
  "name": "Home Agent",
  "version": "0.4.4-beta",

========================================
Update Summary
========================================
Git Tag:        v0.4.4-beta
Source:         /root/home-agent/custom_components/home_agent
Destination:    /root/config/custom_components/home_agent
Backup:         /root/config/custom_components/home_agent_backup_20241103_120000
========================================

[SUCCESS] Update completed successfully!

[WARNING] IMPORTANT: Restart Home Assistant for changes to take effect

[INFO] To restart Home Assistant, run:
  ha core restart

[INFO] If you encounter issues, restore from backup:
  rm -rf /root/config/custom_components/home_agent
  mv /root/config/custom_components/home_agent_backup_20241103_120000 /root/config/custom_components/home_agent
  ha core restart
```

## Safety Features

The update script includes several safety features:

1. **Validation**
   - Verifies git repository exists
   - Confirms tag exists before checkout
   - Validates installation after copying files

2. **Backups**
   - Automatically creates timestamped backups
   - Backups stored at: `/root/config/custom_components/home_agent_backup_YYYYMMDD_HHMMSS`
   - Provides rollback instructions

3. **Dry-Run Mode**
   - Shows all commands before execution
   - No changes made to your system
   - Helps verify the update process

4. **Error Handling**
   - Exits immediately on errors
   - Provides clear error messages
   - Color-coded output for easy reading

## Troubleshooting

### Script Not Found

```bash
# Make sure the repository is in the right location
ls -la /root/home-agent

# Clone if missing
cd /root
git clone https://github.com/yourusername/home-agent.git

# Make executable
chmod +x /root/home-agent/scripts/update_integration.sh
```

### Permission Denied

```bash
# Make the script executable
chmod +x /root/home-agent/scripts/update_integration.sh

# Or run with bash explicitly
bash /root/home-agent/scripts/update_integration.sh --help
```

### Tag Not Found

```bash
# Fetch latest tags
cd /root/home-agent
git fetch --tags

# List available tags
git tag -l

# Use an existing tag
/root/home-agent/scripts/update_integration.sh v0.4.3-beta
```

### Update Failed - Rollback

```bash
# Find your backup
ls -la /root/config/custom_components/ | grep backup

# Restore from most recent backup (replace timestamp)
rm -rf /root/config/custom_components/home_agent
mv /root/config/custom_components/home_agent_backup_YYYYMMDD_HHMMSS /root/config/custom_components/home_agent

# Restart Home Assistant
ha core restart
```

## Best Practices

1. **Always Dry-Run First**
   ```bash
   /root/home-agent/scripts/update_integration.sh --dry-run --latest
   ```

2. **Check Release Notes**
   - Visit GitHub releases page
   - Review breaking changes
   - Note new features or requirements

3. **Backup Before Major Updates**
   - Script creates automatic backups
   - Keep a few versions for safety
   - Test new version before removing old backups

4. **Test After Update**
   - Restart Home Assistant
   - Check integration loads correctly
   - Test basic conversation functionality
   - Review logs for errors

5. **Clean Old Backups Periodically**
   ```bash
   # List backups
   ls -la /root/config/custom_components/ | grep backup

   # Remove old backups (be careful!)
   rm -rf /root/config/custom_components/home_agent_backup_20241001_*
   ```

## Automation (Advanced)

### Scheduled Updates

You can create an automation to check for updates:

```yaml
# configuration.yaml
shell_command:
  update_home_agent: "/root/home-agent/scripts/update_integration.sh --latest && ha core restart"

automation:
  - alias: "Weekly Home Agent Update Check"
    trigger:
      - platform: time
        at: "03:00:00"
    condition:
      - condition: time
        weekday:
          - sun
    action:
      - service: notify.persistent_notification
        data:
          message: "Home Agent update available. Run update_home_agent shell command to update."
          title: "Home Agent Update"
```

**Warning:** Automatic updates are not recommended for production. Always review changes first.

## Version Management

### Check Current Version

```bash
# From git repository
cd /root/home-agent
git describe --tags --abbrev=0

# From installed integration
grep '"version"' /root/config/custom_components/home_agent/manifest.json
```

### List Available Versions

```bash
cd /root/home-agent
git fetch --tags
git tag -l
```

### Compare Versions

```bash
cd /root/home-agent

# See changes between two versions
git log v0.4.3-beta..v0.4.4-beta --oneline

# See file changes
git diff v0.4.3-beta v0.4.4-beta
```

## Support

If you encounter issues:

1. Check the [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
2. Review Home Assistant logs: Settings > System > Logs
3. Create an issue on GitHub with:
   - Script output (dry-run and actual)
   - Home Assistant version
   - Current integration version
   - Target version
   - Error messages

## Quick Reference

```bash
# Dry-run latest
/root/home-agent/scripts/update_integration.sh --dry-run --latest

# Update to latest
/root/home-agent/scripts/update_integration.sh --latest && ha core restart

# Update to specific version
/root/home-agent/scripts/update_integration.sh v0.4.4-beta && ha core restart

# Check current version
grep '"version"' /root/config/custom_components/home_agent/manifest.json

# List available versions
cd /root/home-agent && git fetch --tags && git tag -l

# Rollback to backup
rm -rf /root/config/custom_components/home_agent
mv /root/config/custom_components/home_agent_backup_YYYYMMDD_HHMMSS /root/config/custom_components/home_agent
ha core restart
```

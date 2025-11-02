# Installing Home Agent from Private GitHub Repository

This guide explains how to install Home Agent via HACS from a private GitHub repository using a Personal Access Token.

## Prerequisites

- Home Assistant with HACS installed
- Private GitHub repository access

## Step 1: Generate GitHub Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Direct link: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Configure the token:
   - **Note**: `HACS Home Agent Access` (or any descriptive name)
   - **Expiration**: Choose your preferred expiration (90 days, 1 year, or no expiration)
   - **Scopes**: Select `repo` (Full control of private repositories)
4. Click "Generate token"
5. **Important**: Copy the token immediately - you won't be able to see it again!

## Step 2: Add Repository to HACS

### Method 1: Configuration File (RECOMMENDED - Most Reliable)

**The token-in-URL method does NOT work with HACS. Use this method instead:**

1. **Add HACS configuration to `configuration.yaml`**:

```yaml
hacs:
  token: !secret github_token
```

2. **Add your token to `secrets.yaml`**:

```yaml
github_token: ghp_YOUR_PERSONAL_ACCESS_TOKEN_HERE
```

3. **Restart Home Assistant**

4. **Add repository in HACS UI** (now without token in URL):
   - Go to HACS → Integrations
   - Click three-dot menu (⋮) → Custom repositories
   - Repository: `aradlein/home-agent` (just username/repo, no https://, no token!)
   - Category: Integration
   - Click Add

### Method 2: Manual Installation (If HACS Issues Persist)

If HACS still has issues accessing the private repository:

1. **Download latest release** from GitHub:
   - Visit: https://github.com/aradlein/home-agent/releases
   - Download `home-agent-0.4.3-beta.zip` (or latest version)

2. **Extract and install**:
   ```bash
   # Extract the downloaded ZIP
   unzip home-agent-0.4.3-beta.zip

   # Copy to Home Assistant
   cp -r home-agent-0.4.3-beta/custom_components/home_agent \
     /config/custom_components/
   ```

3. **Restart Home Assistant**

## Step 3: Install Home Agent

1. In HACS, search for "Home Agent"
2. Click on it
3. Click "Download" in the bottom right
4. Select the version you want to install (latest recommended)
5. Click "Download"
6. Restart Home Assistant

## Step 4: Configure the Integration

1. Go to **Settings** → **Devices & Services**
2. Click **Add Integration**
3. Search for "Home Agent"
4. Follow the configuration wizard to set up your LLM connection

## Updating Home Agent

HACS will automatically check for updates. When a new version is available:

1. Go to HACS → Integrations
2. Find Home Agent (it will show an update badge)
3. Click on it
4. Click "Update"
5. Restart Home Assistant

## Security Considerations

### Token Security
- **Never commit** your personal access token to any repository
- Store it securely (e.g., password manager)
- Use tokens with minimal required scope (only `repo`)
- Set an expiration date and rotate tokens regularly
- Revoke tokens when no longer needed

### Alternative: Use HACS with Private Repositories

If you don't want to include the token in configuration:

1. Add the repository using the HACS UI with token
2. HACS will store the token encrypted
3. The token is only used for GitHub API access, never exposed

## Troubleshooting

### "Repository not found" Error
- Verify your token has `repo` scope
- Check that the repository URL is correct
- Ensure the token hasn't expired

### "Authentication failed" Error
- Generate a new token
- Make sure you copied the entire token
- Verify the token in GitHub settings

### Can't See Home Agent in HACS
- Ensure the repository was added as type "Integration"
- Try clearing HACS cache: HACS → Three dots → Reload data
- Check HACS logs for errors

### Installation Fails
- Check Home Assistant logs: Settings → System → Logs
- Verify your Home Assistant version meets requirements (2024.1.0+)
- Ensure all dependencies can be installed

## Getting Help

For issues or questions:
- Check the documentation: [README.md](README.md)
- Review detailed guides: [docs/](docs/)
- Open an issue (if you have repository access)

## Token Expiration

When your token expires:
1. Generate a new token following Step 1
2. Update the repository in HACS:
   - Go to HACS → Three dots → Custom repositories
   - Remove the old entry
   - Add it again with the new token
3. Or update your configuration file with the new token
4. Reload HACS data

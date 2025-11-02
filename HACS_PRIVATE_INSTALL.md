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

### Method 1: Using HACS UI (Recommended)

1. Open Home Assistant
2. Go to **HACS** → **Integrations**
3. Click the three-dot menu (⋮) in the top right
4. Select **Custom repositories**
5. Add the repository:
   - **Repository URL**:
     ```
     https://YOUR_TOKEN@github.com/YOUR_USERNAME/home-agent
     ```
     Replace `YOUR_TOKEN` with your personal access token and `YOUR_USERNAME` with your GitHub username
   - **Category**: Select "Integration"
6. Click **Add**

### Method 2: Manual Configuration File

Alternatively, you can add it directly to your HACS configuration:

1. Edit your Home Assistant configuration file
2. Add to your `configuration.yaml` or create a HACS packages file:

```yaml
# In configuration.yaml or packages/hacs.yaml
hacs:
  custom_repositories:
    - repository: YOUR_USERNAME/home-agent
      token: YOUR_TOKEN
      type: integration
```

3. Restart Home Assistant

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

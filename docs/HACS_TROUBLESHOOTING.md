# HACS Installation Troubleshooting

## Common Issues and Solutions

### Issue: "GitHub returned 404" Error

**Problem**: When adding the repository with token in the URL format, HACS returns a 404 error.

**Cause**: HACS doesn't support tokens directly in the repository URL for custom repositories.

**Solution**: Use one of the methods below:

---

## Method 1: HACS Configuration File (Recommended)

Add the repository to your Home Assistant configuration:

1. **Edit your `configuration.yaml`** or create a new file in `config/packages/hacs.yaml`:

```yaml
hacs:
  token: !secret github_token
  custom_repositories:
    aradlein/home-agent:
      type: integration
```

2. **Add your token to `secrets.yaml`**:

```yaml
github_token: ghp_YOUR_PERSONAL_ACCESS_TOKEN_HERE
```

3. **Restart Home Assistant**

4. **Install from HACS**:
   - Go to HACS → Integrations
   - Search for "Home Agent"
   - Click Install

---

## Method 2: Public Repository Fork

If you want to avoid token management:

1. **Fork the repository to your account** (keep it private)
2. **In HACS**, add as custom repository: `yourusername/home-agent`
3. Make sure your GitHub account has access to the private fork

---

## Method 3: Manual Installation

If HACS isn't working, install manually:

1. **Download the latest release**:
   ```bash
   wget https://github.com/aradlein/home-agent/archive/refs/tags/v0.4.3-beta.zip
   ```

2. **Extract to custom_components**:
   ```bash
   unzip v0.4.3-beta.zip
   cp -r home-agent-0.4.3-beta/custom_components/home_agent config/custom_components/
   ```

3. **Restart Home Assistant**

---

## Method 4: Git Clone with Token

For development/testing:

1. **Clone with token**:
   ```bash
   cd config/custom_components
   git clone https://YOUR_TOKEN@github.com/aradlein/home-agent.git home_agent
   ```

2. **Keep updated**:
   ```bash
   cd config/custom_components/home_agent
   git pull
   ```

3. **Restart Home Assistant after updates**

---

## Verification

After installation, verify it's working:

1. **Check logs** (Settings → System → Logs):
   ```
   Search for "home_agent"
   Should see: "Home Agent initialized successfully"
   ```

2. **Check integration** (Settings → Devices & Services):
   - Should see "Home Agent" in the list
   - Click "+ Add Integration" to configure

3. **Test basic functionality**:
   ```yaml
   service: home_agent.process
   data:
     text: "What's the status of my home?"
   ```

---

## GitHub Token Scopes

Ensure your Personal Access Token has these scopes:

- ✅ `repo` - Full control of private repositories
  - `repo:status` - Access commit status
  - `repo_deployment` - Access deployment status
  - `public_repo` - Access public repositories
  - `repo:invite` - Access repository invitations

**To check/update token scopes:**
1. Go to https://github.com/settings/tokens
2. Click on your token
3. Verify `repo` is checked
4. Update scopes if needed
5. Regenerate token if you made changes

---

## HACS Debug Mode

Enable HACS debug logging:

```yaml
logger:
  default: info
  logs:
    custom_components.hacs: debug
```

Check logs at: Settings → System → Logs

---

## Alternative: Direct Download Links

If all else fails, use direct download for each release:

**Latest Beta (v0.4.3-beta)**:
```
https://github.com/aradlein/home-agent/archive/refs/tags/v0.4.3-beta.zip
```

**Extract and install**:
1. Download ZIP
2. Extract `custom_components/home_agent` folder
3. Copy to `config/custom_components/home_agent`
4. Restart Home Assistant

---

## Getting Help

If you're still having issues:

1. **Check GitHub repository access**:
   - Verify you can access: https://github.com/aradlein/home-agent
   - Ensure you're logged in to GitHub

2. **Verify token is valid**:
   ```bash
   curl -H "Authorization: token YOUR_TOKEN" \
     https://api.github.com/repos/aradlein/home-agent
   ```
   Should return repository JSON (not 404)

3. **Check Home Assistant logs** for specific error messages

4. **Try manual installation** to confirm the integration works

---

## Quick Reference: Configuration File Method

**Minimal configuration** in `configuration.yaml`:

```yaml
hacs:
  token: YOUR_GITHUB_TOKEN_HERE
```

Then add repository in HACS UI:
- Repository: `aradlein/home-agent`
- Category: Integration

This is the most reliable method for private repositories.

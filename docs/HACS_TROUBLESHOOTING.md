# HACS Installation Troubleshooting

## Common Issues and Solutions

### Issue: Repository Not Found

**Problem**: HACS can't find the Home Agent repository.

**Solution**: Add as a custom repository:

1. In HACS, go to **Integrations** → **⋮** (menu) → **Custom repositories**
2. Add repository: `https://github.com/aradlein/hass-agent-llm`
3. Category: **Integration**
4. Click **Add**
5. Search for "Home Agent" and install

---

### Issue: Installation Fails

**Problem**: HACS download or installation fails.

**Solutions**:

1. **Check Home Assistant version**: Requires 2024.1.0 or later
2. **Restart Home Assistant** after adding the custom repository
3. **Clear HACS cache**: HACS → ⋮ → Clear cache, then restart

---

## Manual Installation

If HACS isn't working, install manually:

1. **Download the latest release**:
   ```bash
   wget https://github.com/aradlein/hass-agent-llm/archive/refs/tags/v0.8.2.zip
   ```

2. **Extract to custom_components**:
   ```bash
   unzip v0.8.2.zip
   mv hass-agent-llm-0.8.2/custom_components/home_agent config/custom_components/
   ```

3. **Restart Home Assistant**

---

## Verification

After installation, verify it's working:

1. **Check logs** (Settings → System → Logs):
   ```
   Search for "home_agent"
   Should see: "Home Agent initialized successfully"
   ```

2. **Check integration** (Settings → Devices & Services):
   - Click "+ Add Integration"
   - Search for "Home Agent"
   - Complete the configuration wizard

3. **Test basic functionality**:
   ```yaml
   service: home_agent.process
   data:
     text: "What's the status of my home?"
   ```

---

## HACS Debug Mode

Enable HACS debug logging to troubleshoot issues:

```yaml
logger:
  default: info
  logs:
    custom_components.hacs: debug
```

Check logs at: Settings → System → Logs

---

## Getting Help

If you're still having issues:

1. **Check repository access**: Verify you can view https://github.com/aradlein/hass-agent-llm
2. **Check Home Assistant logs** for specific error messages
3. **Open an issue**: [GitHub Issues](https://github.com/aradlein/hass-agent-llm/issues)

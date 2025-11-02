# Home Agent Installation - Quick Start

## Overview

Home Agent is a Home Assistant custom component that brings advanced conversational AI capabilities with tool execution, context injection, and intelligent automation management. Works with OpenAI, Ollama, LocalAI, and any OpenAI-compatible endpoint.

## Prerequisites

- Home Assistant 2024.1.0 or later
- Network access to your chosen LLM endpoint (cloud or local)
- Optional: ChromaDB server (for vector search and memory features)

## HACS Installation (Coming Soon)

1. Open Home Assistant and navigate to **HACS** in the sidebar
2. Click **Integrations** then the **+** button
3. Search for **Home Agent**
4. Click **Download**
5. Restart Home Assistant

Note: Currently in development. Use manual installation below.

## Manual Installation

1. **Clone or download** the repository:
   ```bash
   cd /config
   git clone https://github.com/yourusername/home-agent.git
   ```

2. **Copy to custom components**:
   ```bash
   cp -r home-agent/custom_components/home_agent /config/custom_components/
   ```

3. **Verify files are in place**:
   ```bash
   ls -la /config/custom_components/home_agent/
   ```

4. **Restart Home Assistant** via Settings > System > Restart

## Initial Configuration

### Add the Integration

1. Navigate to **Settings** > **Devices & Services**
2. Click **+ Add Integration** and search for **Home Agent**
3. Configure the primary LLM:

| Field | Example Value |
|-------|---------------|
| Name | `Home Agent` |
| LLM Base URL | `https://api.openai.com/v1` (OpenAI) or `http://localhost:11434/v1` (Ollama) |
| API Key | Your API key (or leave blank for local models) |
| Model | `gpt-4o-mini` (OpenAI) or `llama3.2:3b` (Ollama) |
| Temperature | `0.7` |
| Max Tokens | `500` |

### LLM Provider Examples

**OpenAI:**
- Base URL: `https://api.openai.com/v1`
- API Key: Your OpenAI API key (sk-...)
- Model: `gpt-4o-mini` or `gpt-4o`

**Ollama (Local):**
- Install Ollama: `https://ollama.ai/download`
- Pull model: `ollama pull llama3.2:3b`
- Base URL: `http://localhost:11434/v1`
- API Key: Leave blank
- Model: `llama3.2:3b`

Other providers (LocalAI, LM Studio) follow similar patterns - see reference docs for details.

## Basic Setup

### Configure Context Mode

1. Go to **Settings** > **Devices & Services** > **Home Agent** > **Configure**
2. Select **Context Settings**
3. Choose your mode:

**Direct Mode (Simple):**
- Specify entities to always include
- Enter comma-separated entity IDs:
  ```
  climate.*,sensor.temperature_*,light.bedroom
  ```
- Good for small setups or specific use cases

**Vector DB Mode (Advanced):**
- Requires ChromaDB server (see [VECTOR_DB_SETUP.md](VECTOR_DB_SETUP.md))
- Automatically finds relevant entities via semantic search
- Required for memory features
- Better for large setups

### Enable Conversation History

1. Navigate to **Configure** > **History Settings**
2. Configure:
   - **Enable History**: On
   - **Max Messages**: `10`
   - **Max Tokens**: `4000`

## Quick Test

Test your installation:

1. Open **Developer Tools** > **Services**
2. Select `home_agent.process`
3. Enter:
   ```yaml
   text: "What is the current temperature?"
   ```
4. Verify you receive a response

Test control:
```yaml
text: "Turn on the living room lights"
```

## Troubleshooting

**Connection Errors:**
- Verify base URL is correct and accessible
- For local models, ensure server is running (e.g., `ollama list`)
- Check firewall rules

**Authentication Errors:**
- Verify API key is correct
- Ensure billing is enabled (for paid services)
- For local models, try leaving API key blank

**Model Not Found:**
- For Ollama: `ollama list` to see available models
- Verify model name spelling

**Entity Not Found:**
- Check entity IDs in Home Assistant
- Expose entities via **Settings** > **Voice Assistants** > **Expose Entities**

## Next Steps

1. **Advanced Features:**
   - [Vector DB Setup](VECTOR_DB_SETUP.md) - Enable semantic entity search
   - [Memory System](MEMORY_SYSTEM.md) - Add long-term memory capabilities

2. **Voice Assistant Integration:**
   - Navigate to **Settings** > **Voice Assistants**
   - Select **Home Agent** as the conversation agent

3. **Use in Automations:**
   ```yaml
   automation:
     - alias: "Morning Briefing"
       trigger:
         - platform: time
           at: "07:00:00"
       action:
         - service: home_agent.process
           data:
             text: "Give me a morning briefing"
   ```

4. **Monitor Performance:**
   - Enable debug logging in **Configure** > **Debug Settings**
   - Check logs for detailed execution info
   - Monitor events in **Developer Tools** > **Events**

## Need More Details?

See the [Complete Installation Reference](reference/INSTALLATION.md) for:
- Detailed configuration options
- All LLM provider examples
- Advanced troubleshooting
- Custom tool setup
- Performance optimization

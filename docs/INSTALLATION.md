# Home Agent Installation Guide

## Overview

Home Agent is a highly customizable Home Assistant custom component that extends conversational AI capabilities with advanced tool execution, context injection, and intelligent automation management. This guide will help you install and configure Home Agent for the first time.

## What is Home Agent?

Home Agent integrates with Home Assistant's native conversation platform to provide:

- **OpenAI-Compatible LLM Integration** - Works with OpenAI, Ollama, LocalAI, LM Studio, and any OpenAI-compatible endpoint
- **Smart Context Injection** - Automatically provides relevant entity states to the LLM using direct or semantic vector search
- **Conversation History** - Maintains context across multiple interactions
- **Tool Calling** - Native Home Assistant control and query capabilities
- **Long-Term Memory** - Remembers facts, preferences, and events across conversations
- **Streaming Responses** - Low-latency TTS integration for real-time voice interaction

## Prerequisites

### Minimum Requirements

- **Home Assistant Version**: 2024.1.0 or later
- **Python Version**: 3.11+ (included with Home Assistant)
- **Network Access**: Connection to your chosen LLM endpoint (cloud or local)

### Required Dependencies

The following dependencies are automatically installed with Home Agent:

- `aiohttp >= 3.9.0` - HTTP client for LLM API calls
- `chromadb-client == 1.3.0` - Vector database client (optional, for semantic search)

### Optional Dependencies

For advanced features, you may want to set up:

- **ChromaDB Server** - For vector-based semantic entity search and memory storage
  - Enables dynamic context injection based on query relevance
  - Required for long-term memory system
  - Docker installation recommended
- **OpenAI API Key** - For embedding generation (if using ChromaDB with OpenAI embeddings)
  - Alternative: Use Ollama for local embeddings (no API key needed)

## Installation Methods

### HACS Installation (Recommended - Coming Soon)

HACS (Home Assistant Community Store) is the easiest way to install and update Home Agent.

**Note**: Home Agent is not yet available in the default HACS repository. Manual installation is currently required.

Once available in HACS:

1. Open Home Assistant
2. Navigate to **HACS** in the sidebar
3. Click **Integrations**
4. Click the **+** button in the bottom right
5. Search for "**Home Agent**"
6. Click **Download**
7. Restart Home Assistant
8. Continue to [Initial Configuration](#initial-configuration)

### Manual Installation

If HACS is not available or you prefer manual installation:

1. **Download the Integration**

   Download the latest release from GitHub or clone the repository:

   ```bash
   cd /config  # Your Home Assistant config directory
   git clone https://github.com/yourusername/home-agent.git
   ```

2. **Copy Files to Custom Components**

   Copy the `custom_components/home_agent` directory to your Home Assistant `custom_components` folder:

   ```bash
   cp -r home-agent/custom_components/home_agent /config/custom_components/
   ```

   Your directory structure should look like:

   ```
   /config/
   ├── custom_components/
   │   └── home_agent/
   │       ├── __init__.py
   │       ├── agent.py
   │       ├── config_flow.py
   │       ├── manifest.json
   │       └── ... (other files)
   ├── configuration.yaml
   └── ... (other files)
   ```

3. **Verify Installation**

   Check that the files are in place:

   ```bash
   ls -la /config/custom_components/home_agent/
   ```

   You should see the integration files listed.

4. **Restart Home Assistant**

   Restart Home Assistant to load the new integration:

   - Navigate to **Settings** > **System** > **Restart**
   - Or use the command line: `ha core restart`

## Initial Configuration

### Adding the Integration

1. **Open Home Assistant**

   Navigate to **Settings** > **Devices & Services**

2. **Add Integration**

   - Click the **+ Add Integration** button in the bottom right
   - Search for "**Home Agent**"
   - Click on the **Home Agent** integration

3. **Configure Primary LLM**

   You'll be presented with the initial configuration form:

   | Field | Description | Example |
   |-------|-------------|---------|
   | **Name** | Friendly name for this instance | `Home Agent` |
   | **LLM Base URL** | OpenAI-compatible API endpoint | See examples below |
   | **API Key** | Authentication key for the LLM service | Your API key or leave blank for local models |
   | **Model** | Model name to use | See examples below |
   | **Temperature** | Creativity level (0.0-2.0) | `0.7` (default) |
   | **Max Tokens** | Maximum response length | `500` (default) |

### LLM Provider Examples

#### OpenAI

For OpenAI's GPT models:

- **Base URL**: `https://api.openai.com/v1`
- **API Key**: Your OpenAI API key (e.g., `sk-...`)
- **Model**: `gpt-4o-mini` (cost-effective) or `gpt-4o` (more capable)
- **Temperature**: `0.7`
- **Max Tokens**: `500`

#### Ollama (Local)

For locally-hosted Ollama models:

- **Base URL**: `http://localhost:11434/v1`
- **API Key**: Leave blank or enter `ollama`
- **Model**: `llama3.2:3b` or `qwen2.5:7b` (any Ollama model)
- **Temperature**: `0.7`
- **Max Tokens**: `500`

**Setup Ollama**:
1. Install Ollama: `https://ollama.ai/download`
2. Pull a model: `ollama pull llama3.2:3b`
3. Ollama runs on port 11434 by default

#### LocalAI

For LocalAI installations:

- **Base URL**: `http://localhost:8080/v1`
- **API Key**: Leave blank or use configured key
- **Model**: Name of your loaded model
- **Temperature**: `0.7`
- **Max Tokens**: `500`

#### LM Studio

For LM Studio local server:

- **Base URL**: `http://localhost:1234/v1`
- **API Key**: Leave blank
- **Model**: Model name loaded in LM Studio
- **Temperature**: `0.7`
- **Max Tokens**: `500`

### API Key Setup

**For Cloud LLMs (OpenAI, etc.)**:
- Obtain an API key from your provider
- Keep it secure - it provides access to your account
- Add billing information if required

**For Local LLMs (Ollama, LocalAI)**:
- No API key needed in most cases
- Leave the field blank or enter a placeholder like `local`

### Model Selection

Choose a model based on your needs:

| Model Type | Examples | Best For | Cost |
|------------|----------|----------|------|
| **Fast & Efficient** | `gpt-4o-mini`, `llama3.2:3b` | Quick responses, simple tasks | Low |
| **Balanced** | `gpt-4o`, `qwen2.5:7b` | General use, good reasoning | Medium |
| **Advanced** | `gpt-4-turbo`, `llama3.1:70b` | Complex analysis, detailed responses | High |

## Basic Setup

### Choosing Context Mode

After initial setup, configure how Home Agent provides entity context to the LLM:

1. **Go to Integration Options**

   Navigate to **Settings** > **Devices & Services** > **Home Agent** > **Configure**

2. **Select Context Settings**

   Choose **Context Settings** from the menu

3. **Choose Context Mode**

   - **Direct Mode** (default): Specify entities to always include
     - Simple and reliable
     - Lower latency
     - Good for small setups or specific use cases

   - **Vector DB Mode** (advanced): Semantic search for relevant entities
     - Requires ChromaDB server (see [Vector DB Setup Guide](VECTOR_DB_SETUP.md))
     - More efficient for large setups
     - Automatically finds relevant entities based on query
     - Required for long-term memory features

4. **Configure Direct Entities** (if using Direct Mode)

   Enter entity IDs as a comma-separated list:

   ```
   sensor.living_room_temperature,light.living_room,climate.thermostat
   ```

   Supports wildcards:

   ```
   climate.*,sensor.temperature_*,light.bedroom
   ```

### Configuring Conversation History

Enable conversation history to maintain context across interactions:

1. **Go to History Settings**

   Navigate to **Configure** > **History Settings**

2. **Configure History**

   - **Enable History**: `On` (recommended)
   - **Max Messages**: `10` (number of conversation turns to keep)
   - **Max Tokens**: `4000` (token limit for history)

### Testing Basic Functionality

Test your installation with a simple query:

1. **Open Developer Tools**

   Navigate to **Developer Tools** > **Services**

2. **Call the Process Service**

   Select `home_agent.process` and enter:

   ```yaml
   text: "What is the current temperature?"
   ```

3. **Check the Response**

   You should see a response from the LLM in the service response.

4. **Verify Tool Execution**

   Try a control command:

   ```yaml
   text: "Turn on the living room lights"
   ```

   Verify that the lights actually turn on (if the entity exists).

### Troubleshooting Initial Setup

#### Connection Errors

**Symptom**: "Failed to connect to LLM" error

**Solutions**:
- Verify the base URL is correct and accessible
- Check network connectivity to the LLM endpoint
- For local models, ensure the server is running (e.g., `ollama list` for Ollama)
- Check firewall rules if running on a different machine

#### Authentication Errors

**Symptom**: "Invalid API key" or 401 errors

**Solutions**:
- Verify your API key is correct and active
- Check for extra spaces or newlines in the API key field
- Ensure your API key has billing enabled (for paid services)
- For local models, try leaving the API key blank

#### Model Not Found

**Symptom**: "Model not found" or 404 errors

**Solutions**:
- Verify the model name is spelled correctly
- For Ollama: Run `ollama list` to see available models
- For OpenAI: Check the [models documentation](https://platform.openai.com/docs/models)
- Ensure the model is pulled/downloaded on local installations

#### Entity Not Found

**Symptom**: "Entity not found" errors in tool calls

**Solutions**:
- Verify entity IDs are correct in Home Assistant
- Check that entities are exposed to voice assistants (required for agent access)
- Navigate to **Settings** > **Voice Assistants** > **Expose Entities** to configure

## Next Steps

Once Home Agent is installed and configured:

1. **Explore Advanced Features**

   - [Vector DB Setup Guide](VECTOR_DB_SETUP.md) - Enable semantic search for entities
   - [Memory System Guide](MEMORY_SYSTEM.md) - Set up long-term memory for facts and preferences
   - [Custom Tools Guide](../README.md#custom-tools-phase-3-) - Extend functionality with REST APIs and services

2. **Configure Voice Assistant Integration**

   - Navigate to **Settings** > **Voice Assistants**
   - Create a new assistant or edit an existing one
   - Select **Home Agent** as the conversation agent

3. **Set Up Automations**

   Use Home Agent in automations to create intelligent home behaviors:

   ```yaml
   automation:
     - alias: "Morning Briefing"
       trigger:
         - platform: time
           at: "07:00:00"
       action:
         - service: home_agent.process
           data:
             text: "Give me a morning briefing with weather and calendar events"
   ```

4. **Monitor Performance**

   - Enable **Debug Logging** in **Configure** > **Debug Settings** (temporarily)
   - Check Home Assistant logs for detailed execution information
   - Monitor events in **Developer Tools** > **Events** (filter for `home_agent.*`)

## Getting Help

If you encounter issues:

- **Documentation**: Check the [troubleshooting section](#troubleshooting-initial-setup) above
- **Logs**: Enable debug logging and check Home Assistant logs
- **Community**:
  - [GitHub Issues](https://github.com/yourusername/home-agent/issues)
  - [GitHub Discussions](https://github.com/yourusername/home-agent/discussions)
  - Home Assistant Community Forums

## Additional Resources

- [Project Specification](PROJECT_SPEC.md) - Technical details and architecture
- [Development Guide](DEVELOPMENT.md) - Contributing and development standards
- [README](../README.md) - Feature overview and examples

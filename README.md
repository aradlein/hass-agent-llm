# Home Agent

[![Version](https://img.shields.io/badge/version-0.6.4--beta-blue.svg)](https://github.com/aradlein/hass-agent-llm/releases)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1.0+-blue.svg)](https://www.home-assistant.io/)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz/)

A highly customizable Home Assistant custom component that provides intelligent conversational AI capabilities with advanced tool calling, context injection, and conversation history management.

## What's New in v0.6.4-beta

🔧 **Streaming Tool Execution Fix** - Fixed "No LLM API configured" error during streaming with tool calls
🎯 **Improved Reliability** - Tools now execute seamlessly when LLM returns tool calls in streaming mode
⚡ **Backend Selection** - Support for Ollama backend selection (llama.cpp, vLLM, Ollama GPU)
🧠 **Memory System Enhancements** - Optimized memory search and extraction quality
📝 **Context Optimization** - Parallel entity and memory context retrieval for reduced latency

[View Full Changelog](https://github.com/aradlein/hass-agent-llm/releases)

## Overview

Home Agent extends Home Assistant's native conversation platform to enable natural language control and monitoring of your smart home. It works with any OpenAI-compatible LLM provider, giving you flexibility to use cloud services or run models locally.

**Key Capabilities:**
- Natural language home control through any OpenAI-compatible LLM
- Automatic context injection - LLM knows your home's current state
- Persistent conversation memory across interactions
- Extensible tool system for custom integrations
- Streaming responses for voice assistants
- Long-term memory system for personalized experiences

## Features

### Core Features

- **LLM Integration** - Works with OpenAI, Ollama, LocalAI, LM Studio, or any OpenAI-compatible endpoint
- **Entity Context** - Automatically provides relevant entity states to the LLM
- **Conversation History** - Maintains context across multiple interactions with persistent storage
- **Native Tools** - Built-in `ha_control` and `ha_query` tools for home automation
- **Custom Tools** - Define REST API and Home Assistant service tools in configuration
- **Event System** - Rich events for automation triggers and monitoring
- **Streaming Responses** - Low-latency streaming for voice assistant integration (~10x faster)

### Advanced Features

- **Vector Database Integration** - Semantic entity search using ChromaDB for efficient context management
- **Multi-LLM Support** - Use a fast local model for control + powerful cloud model for analysis
- **Memory System** - Automatic extraction and recall of facts, preferences, and context
- **Context Optimization** - Intelligent compression to stay within token limits
- **Tool Progress Indicators** - Real-time feedback during tool execution

## Requirements

### Required
- **Home Assistant** - Version 2024.1.0 or later
- **Python Dependencies** - `aiohttp >= 3.9.0` (included with Home Assistant)

### Optional (Enable Advanced Features)
- **ChromaDB** - For vector database context mode
  - `chromadb-client >= 0.4.0`
  - Required for: Vector DB context injection and memory system
- **OpenAI** - For embeddings in vector DB mode
  - `openai >= 1.3.8`
  - Required for: Vector DB entity indexing (alternative: use Ollama)
- **Wyoming Protocol TTS** - For streaming responses
  - Required for: Low-latency voice assistant integration

## Installation

### HACS (Recommended)

**For private repository installation:**

1. Generate a GitHub Personal Access Token with `repo` scope
2. In HACS, go to **Integrations** → **⋮** → **Custom repositories**
3. Add repository: `https://YOUR_TOKEN@github.com/YOUR_USERNAME/hass-agent-llm`
4. Category: **Integration**
5. Click **Add**
6. Search for "Home Agent" in HACS
7. Click Install
8. Restart Home Assistant
9. Go to Settings > Devices & Services > Add Integration
10. Search for "Home Agent" and follow the setup wizard

**See [HACS Private Installation Guide](docs/HACS_PRIVATE_INSTALL.md) for detailed instructions.**

### Manual Installation

1. Download the latest release from GitHub
2. Copy the `custom_components/home_agent` directory to your Home Assistant `config/custom_components` folder
3. Restart Home Assistant
4. Go to Settings > Devices & Services > Add Integration
5. Search for "Home Agent" and complete the configuration

**For detailed installation instructions, see [Installation Guide](docs/INSTALLATION.md)**

## Quick Start

### 1. Add the Integration

Navigate to Settings > Devices & Services > Add Integration, search for "Home Agent", and configure:

- **Name**: Friendly name (e.g., "Home Agent")
- **LLM Base URL**: Your OpenAI-compatible endpoint
  - OpenAI: `https://api.openai.com/v1`
  - Ollama (local): `http://localhost:11434/v1`
  - LocalAI: Your LocalAI URL
- **API Key**: Your API key (if required)
- **Model**: Model name (e.g., `gpt-4o-mini`, `llama3.2`, etc.)
- **Temperature**: 0.7 (recommended for most use cases)
- **Max Tokens**: 500 (adjust based on your needs)

### 2. Test Basic Functionality

Call the conversation service:

```yaml
service: home_agent.process
data:
  text: "Turn on the living room lights"
```

### 3. Explore Advanced Configuration

Access Settings > Devices & Services > Home Agent > Configure to:
- Configure context injection mode (direct or vector DB)
- Enable conversation history
- Set up custom tools
- Configure external LLM for complex queries
- Enable memory system
- Enable streaming for voice assistants

**For detailed configuration options, see [Configuration Reference](docs/CONFIGURATION.md)**

## Documentation

### Quick Start Guides
- [Installation](docs/INSTALLATION.md) - Get up and running in minutes
- [Configuration](docs/CONFIGURATION.md) - Essential settings explained
- [FAQ](docs/FAQ.md) - Top 20 questions answered

### Feature Guides
- [Memory System](docs/MEMORY_SYSTEM.md) - Enable long-term memory
- [Vector DB Setup](docs/VECTOR_DB_SETUP.md) - Semantic entity search
- [Custom Tools](docs/CUSTOM_TOOLS.md) - Extend with REST APIs and services
- [External LLM](docs/EXTERNAL_LLM.md) - Multi-LLM workflows

### Reference
- [API Reference](docs/API_REFERENCE.md) - Services, events, and tools
- [Observability](observability/README.md) - Monitoring with Prometheus, InfluxDB, and Grafana
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Quick fixes for common issues
- [Examples](docs/EXAMPLES.md) - 10 ready-to-use examples
- [Migration Guide](docs/MIGRATION.md) - Moving from extended_openai_conversation

### Complete Reference
For comprehensive documentation, see [docs/reference/](docs/reference/) for detailed guides covering all configuration options, advanced scenarios, and troubleshooting.

### For Developers
- [Project Specification](.claude/docs/PROJECT_SPEC.md) - Technical specs and roadmap
- [Development Standards](.claude/docs/DEVELOPMENT.md) - Code quality and testing

## Usage Examples

### Voice Control

```yaml
# Use with Home Assistant voice assistant
# Just speak naturally to your voice assistant
"Turn on the kitchen lights to 50%"
"What's the temperature in the living room?"
"Is the front door locked?"
```

### Automation Integration

```yaml
automation:
  - alias: "Morning Briefing"
    trigger:
      - platform: time
        at: "07:00:00"
    action:
      - service: home_agent.process
        data:
          text: "Good morning! Please prepare for the day."
          conversation_id: "morning_routine"
```

### Custom Tool Example

```yaml
# configuration.yaml
home_agent:
  tools_custom:
    - name: check_weather
      description: "Get weather forecast"
      handler:
        type: rest
        url: "https://api.open-meteo.com/v1/forecast"
        method: GET
        query_params:
          latitude: "47.6062"
          longitude: "-122.3321"
          current: "temperature_2m,precipitation"
```

**For more examples, see [Examples Documentation](docs/EXAMPLES.md)**

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Follow the coding standards in [Development Guide](docs/DEVELOPMENT.md)
6. Submit a pull request

## Testing

Home Agent maintains >80% code coverage with comprehensive unit and integration tests:

```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt

# Run tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=custom_components.home_agent --cov-report=html
```

**Test Status**: 400+ passing tests across core functionality, vector DB, memory system, custom tools, and streaming.

## Support

- **Issues**: [GitHub Issues](https://github.com/aradlein/hass-agent-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aradlein/hass-agent-llm/discussions)
- **Documentation**: See [docs/](docs/) directory

## License

MIT License

## Credits

Built with inspiration from the extended_openai_conversation integration. Special thanks to the Home Assistant community.

## Changelog

### v0.6.4-beta (Latest)
- **Fix**: Streaming tool execution error - "No LLM API configured"
- **Enhancement**: Register agent as llm_api with ChatLog for seamless tool execution
- **Feature**: Ollama backend selection support (llama.cpp, vLLM, Ollama GPU)
- **Enhancement**: Preserve options on LLM settings update
- **Enhancement**: Optimize memory search based on context mode
- **Enhancement**: Parallelize entity and memory context retrieval

### v0.6.0-beta
- **Feature**: Memory extraction quality improvements with content filtering
- **Feature**: Support for additional ChromaDB collections
- **Enhancement**: Improved token usage tracking in streaming mode
- **Fix**: Comprehensive type checking and linting fixes

### v0.5.0-beta
- **Feature**: Streaming response support for voice assistants
- **Feature**: Tool progress indicators
- **Feature**: Automatic fallback to synchronous mode
- **Enhancement**: ~10x latency improvement for TTS integration

### v0.4.0-beta
- **Feature**: Long-term memory system with automatic extraction
- **Feature**: Memory management services
- **Feature**: Memory context provider for enhanced responses

### v0.3.0-beta
- **Feature**: Custom tool framework (REST + Service handlers)
- **Feature**: External LLM delegation tool
- **Enhancement**: Comprehensive error handling

### v0.2.0-beta
- **Feature**: Vector DB (ChromaDB) integration
- **Feature**: History persistence across restarts
- **Enhancement**: Context optimization and compression
- **Enhancement**: Enhanced event system

### v0.1.0-beta (MVP)
- Initial release
- OpenAI-compatible LLM integration
- Core tools (ha_control, ha_query)
- Conversation history
- UI configuration

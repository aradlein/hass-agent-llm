# Home Agent (hass-agent-llm)

A highly customizable Home Assistant custom component that extends the native conversation platform with intelligent conversational AI capabilities. Home Agent works with any OpenAI-compatible LLM provider, giving you the flexibility to use cloud services like OpenAI or run models locally with Ollama, enabling natural language control and monitoring of your smart home with advanced context awareness.

## Key Features

- **Multi-LLM Support** - Works with OpenAI, Ollama, LocalAI, LM Studio, or any OpenAI-compatible endpoint. Use fast local models for control and powerful cloud models for analysis simultaneously.

- **Long-Term Memory System** - Automatic extraction and recall of facts, preferences, and context. The integration learns from interactions and personalizes responses based on your home and preferences.

- **Streaming Responses** - Low-latency streaming support for voice assistant integration (~10x faster). Enable real-time voice feedback for natural conversational experiences.

- **Vector Database Integration** - Semantic entity search using ChromaDB for intelligent context management. Automatically provides relevant home state information to the LLM for smarter decisions.

- **Custom Tool Framework** - Easily extend functionality with REST API and Home Assistant service tools. Define custom integrations directly in configuration without code modifications.

## Additional Capabilities

- **Automatic Context Injection** - The LLM knows your home's current state with entity awareness
- **Persistent Conversation History** - Maintains context across multiple interactions
- **Tool Progress Indicators** - Real-time feedback during tool execution
- **Rich Event System** - Automation triggers and monitoring capabilities
- **Multi-Mode Context** - Choose between direct entity injection or vector DB semantic search

## Installation

For installation instructions, see the [Installation Guide](docs/INSTALLATION.md).

## Documentation

Complete documentation available at [docs/](docs/) including:
- [Configuration Reference](docs/CONFIGURATION.md) - Essential settings and advanced options
- [Memory System Guide](docs/MEMORY_SYSTEM.md) - Enable long-term memory
- [Vector DB Setup](docs/VECTOR_DB_SETUP.md) - Semantic entity search
- [Custom Tools](docs/CUSTOM_TOOLS.md) - Create REST API and service integrations
- [FAQ](docs/FAQ.md) - Common questions and solutions
- [Examples](docs/EXAMPLES.md) - Ready-to-use automation examples
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Quick fixes for common issues

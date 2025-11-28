"""Home Agent conversation agent package.

This package implements the core conversation agent functionality for the Home Agent
integration. It provides the main HomeAgent class that orchestrates LLM interactions,
tool execution, context management, and conversation history tracking.

Architecture:
    The agent is implemented using a mixin-based architecture to separate concerns:

    - core.py: Main HomeAgent class and orchestration logic
    - llm.py: LLM API communication for synchronous calls
    - streaming.py: Streaming LLM support for real-time responses
    - memory_extraction.py: Automatic memory extraction from conversations

Key Components:
    HomeAgent: Main conversation agent class that integrates with Home Assistant's
        conversation platform. Inherits from LLMMixin, StreamingMixin, and
        MemoryExtractionMixin to provide full functionality.

Usage:
    The HomeAgent class is typically instantiated by the integration's __init__.py
    during config entry setup:

    Example:
        from custom_components.home_agent.agent import HomeAgent

        agent = HomeAgent(
            hass=hass,
            config=entry.data,
            session_manager=session_manager
        )

        # Process a conversation
        result = await agent.async_process(user_input)

Integration Points:
    - Home Assistant conversation platform (AbstractConversationAgent)
    - Context manager for entity and memory context injection
    - Tool handler for executing Home Assistant actions
    - Conversation history manager for multi-turn conversations
    - Memory manager for long-term memory storage
    - Session manager for persistent voice conversations

For backward compatibility, the HomeAgent class is re-exported from this module,
allowing imports like:
    from custom_components.home_agent.agent import HomeAgent
"""

from .core import HomeAgent

__all__ = ["HomeAgent"]

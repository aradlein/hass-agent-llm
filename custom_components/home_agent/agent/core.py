"""Core conversation agent implementation for Home Agent.

This module implements the HomeAgent class, which serves as the central orchestrator
for all conversation-related functionality in the Home Agent integration. It brings
together LLM communication, tool execution, context management, and conversation
history to provide an intelligent conversational AI assistant for Home Assistant.

Architecture:
    The HomeAgent class uses a mixin-based architecture to separate concerns:

    - Inherits from LLMMixin for LLM API communication
    - Inherits from StreamingMixin for real-time streaming responses
    - Inherits from MemoryExtractionMixin for automatic memory extraction
    - Inherits from AbstractConversationAgent to integrate with Home Assistant

Key Classes:
    HomeAgent: Main conversation agent that integrates with Home Assistant's
        conversation platform. Manages the complete lifecycle of a conversation
        from user input to assistant response, including:

        - Entity context injection
        - Conversation history management
        - Tool calling and execution
        - Streaming and synchronous response modes
        - Memory extraction and storage
        - Event emission for observability

Core Responsibilities:
    - Process user inputs through Home Assistant's conversation platform
    - Build and manage conversation context (system prompts, entity states)
    - Execute multi-turn conversations with tool calling support
    - Coordinate between LLM, tools, and Home Assistant services
    - Track conversation history and metrics
    - Support both streaming and synchronous response modes

Usage Example:
    Basic conversation processing:
        agent = HomeAgent(hass, config, session_manager)
        result = await agent.async_process(user_input)

    Direct message processing:
        response = await agent.process_message(
            text="Turn on the kitchen lights",
            conversation_id="conv_123",
            user_id="user_456"
        )

    Tool execution for debugging:
        result = await agent.execute_tool_debug(
            tool_name="ha_control",
            parameters={"action": "turn_on", "entity_id": "light.kitchen"}
        )

Integration Points:
    - AbstractConversationAgent: Base class for Home Assistant conversation agents
    - ContextManager: Provides entity and memory context for conversations
    - ConversationHistoryManager: Manages multi-turn conversation history
    - ToolHandler: Executes tools (Home Assistant actions, external services)
    - MemoryManager: Stores and retrieves long-term memories
    - ConversationSessionManager: Manages persistent voice conversation sessions

Tool Calling Flow:
    1. User message received via async_process() or process_message()
    2. Context manager assembles relevant entity states and memories
    3. System prompt built with context and conversation history
    4. LLM called with messages and tool definitions
    5. If LLM requests tools, execute them via ToolHandler
    6. Results fed back to LLM for final response
    7. Response saved to conversation history
    8. Memories extracted asynchronously for future context

Configuration:
    The agent is configured through the config dictionary passed during
    initialization. Key configuration options include:

    - LLM settings (model, temperature, API credentials)
    - Context mode (entity filtering and injection)
    - History settings (max messages, token limits)
    - Tool settings (max calls per turn, timeouts)
    - Streaming and memory extraction toggles

Events:
    The agent emits events for observability:

    - conversation_started: When a conversation begins
    - conversation_finished: When a conversation completes with metrics
    - error: When errors occur during processing
    - streaming_error: When streaming fails (with fallback)
    - memory_extracted: When memories are extracted and stored
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..conversation_session import ConversationSessionManager

import aiohttp
from homeassistant.components import conversation as ha_conversation
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent, template

from ..const import (
    CONF_CONTEXT_ENTITIES,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_HISTORY_PERSIST,
    CONF_LLM_MODEL,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_CUSTOM,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
    DEFAULT_MEMORY_EXTRACTION_ENABLED,
    DEFAULT_STREAMING_ENABLED,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
    EVENT_CONVERSATION_STARTED,
    EVENT_ERROR,
    EVENT_STREAMING_ERROR,
    CONF_MEMORY_EXTRACTION_ENABLED,
    TOOL_QUERY_EXTERNAL_LLM,
)
from ..exceptions import (
    HomeAgentError,
    AuthenticationError,
    TokenLimitExceeded,
    RateLimitExceeded,
    PermissionDenied,
    EntityNotFoundError,
    ServiceUnavailableError,
    ContextInjectionError,
)
from ..context_manager import ContextManager
from ..conversation import ConversationHistoryManager
from ..helpers import strip_thinking_blocks
from ..tool_handler import ToolHandler
from ..tools import HomeAssistantControlTool, HomeAssistantQueryTool
from ..tools.custom import CustomToolHandler
from ..tools.external_llm import ExternalLLMTool

from .llm import LLMMixin
from .streaming import StreamingMixin
from .memory_extraction import MemoryExtractionMixin

_LOGGER = logging.getLogger(__name__)


class HomeAgent(
    LLMMixin,
    StreamingMixin,
    MemoryExtractionMixin,
    AbstractConversationAgent,
):
    """Main conversation agent that orchestrates all Home Agent components.

    This class integrates with Home Assistant's conversation platform and provides
    intelligent conversational AI capabilities with tool calling, context injection,
    and conversation history management.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any],
        session_manager: "ConversationSessionManager",
    ) -> None:
        """Initialize the Home Agent.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary containing LLM settings, context config, etc.
            session_manager: Conversation session manager for persistent voice conversations
        """
        self.hass = hass
        self.config = config
        self.session_manager = session_manager

        # Initialize components
        self.context_manager = ContextManager(hass, config)
        self.conversation_manager = ConversationHistoryManager(
            max_messages=config.get(CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES),
            max_tokens=config.get(CONF_HISTORY_MAX_TOKENS, DEFAULT_HISTORY_MAX_TOKENS),
            hass=hass,
            persist=config.get(CONF_HISTORY_PERSIST, True),
        )
        self.tool_handler = ToolHandler(
            hass,
            {
                "max_calls_per_turn": config.get(CONF_TOOLS_MAX_CALLS_PER_TURN),
                "timeout": config.get(CONF_TOOLS_TIMEOUT),
                "emit_events": config.get(CONF_EMIT_EVENTS, True),
            },
        )

        # Tools will be registered lazily on first use
        # This ensures the exposure system is fully initialized
        self._tools_registered = False

        # HTTP session for LLM API calls
        self._session: aiohttp.ClientSession | None = None

        # Memory manager reference (will be populated from hass.data if available)
        self._memory_manager = None

        _LOGGER.info("Home Agent initialized with model %s", config.get(CONF_LLM_MODEL))

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages.

        Returns the top 10 most popular languages used with Home Assistant:
        - en (English)
        - de (German)
        - es (Spanish)
        - fr (French)
        - nl (Dutch)
        - it (Italian)
        - pl (Polish)
        - pt (Portuguese)
        - ru (Russian)
        - zh (Chinese)

        The agent preserves the language throughout the conversation pipeline,
        using it in ConversationInput and IntentResponse objects.
        """
        return ["en", "de", "es", "fr", "nl", "it", "pl", "pt", "ru", "zh"]

    @property
    def memory_manager(self) -> Any:
        """Get memory manager from hass.data if available."""
        if self._memory_manager is None:
            # Try to get memory manager from hass.data
            domain_data = self.hass.data.get(DOMAIN, {})
            for entry_data in domain_data.values():
                if isinstance(entry_data, dict) and "memory_manager" in entry_data:
                    self._memory_manager = entry_data["memory_manager"]
                    break
        return self._memory_manager

    def _ensure_tools_registered(self) -> None:
        """Ensure tools are registered (lazy registration).

        This method is called before the first message is processed to ensure
        the exposure system has been fully initialized by Home Assistant.
        """
        if self._tools_registered:
            return

        self._register_tools()
        self._tools_registered = True

        # Set memory provider in context manager if memory manager is available
        if self.memory_manager is not None:
            self.context_manager.set_memory_provider(self.memory_manager)
            _LOGGER.debug("Memory context provider enabled")

    async def async_process(
        self, user_input: ha_conversation.ConversationInput
    ) -> ha_conversation.ConversationResult:
        """Process a conversation turn with optional streaming support.

        This method is required by AbstractConversationAgent. It processes user input
        and returns a conversation result. It automatically detects if streaming is
        available and uses the appropriate processing path.

        Args:
            user_input: Conversation input from Home Assistant

        Returns:
            ConversationResult with the agent's response
        """
        try:
            # Ensure tools are registered (lazy initialization)
            self._ensure_tools_registered()

            # Check if we can stream
            if self._can_stream():
                try:
                    return await self._async_process_streaming(user_input)
                except Exception as err:
                    # Fallback to synchronous on streaming errors
                    _LOGGER.warning(
                        "Streaming failed, falling back to synchronous mode: %s",
                        err,
                        exc_info=True,
                    )
                    self.hass.bus.async_fire(
                        EVENT_STREAMING_ERROR,
                        {
                            "error": str(err),
                            "error_type": type(err).__name__,
                            "fallback": True,
                        },
                    )
                    # Fall through to synchronous processing

            # Synchronous processing (existing code path)
            return await self._async_process_synchronous(user_input)

        except AuthenticationError as err:
            _LOGGER.error("Authentication error: %s", err, exc_info=True)
            message = "I'm having trouble connecting to the AI service. Please check your API key."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except ServiceUnavailableError as err:
            _LOGGER.error("Service unavailable: %s", err, exc_info=True)
            message = "The AI service is temporarily unavailable. Please try again later."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except TokenLimitExceeded as err:
            _LOGGER.error("Token limit exceeded: %s", err, exc_info=True)
            message = "Your request was too long. Please try a shorter message."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except RateLimitExceeded as err:
            _LOGGER.error("Rate limit exceeded: %s", err, exc_info=True)
            message = "Too many requests. Please wait a moment and try again."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except EntityNotFoundError as err:
            _LOGGER.error("Entity not found: %s", err, exc_info=True)
            message = "I couldn't find the device. Please check if it's available."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except PermissionDenied as err:
            _LOGGER.error("Permission denied: %s", err, exc_info=True)
            message = "I don't have permission to control that device."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except ContextInjectionError as err:
            _LOGGER.error("Context injection error: %s", err, exc_info=True)
            message = "I had trouble getting device information. Please try again."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

        except Exception as err:
            _LOGGER.error("Unexpected error: %s", err, exc_info=True)
            message = "Something unexpected went wrong. Please check the logs."

            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                message,
            )

            return ha_conversation.ConversationResult(
                response=intent_response,
                conversation_id=user_input.conversation_id,
            )

    def _register_tools(self) -> None:
        """Register core Home Assistant tools."""
        # Get exposed entities from voice assistant settings
        # Use async_should_expose to respect Home Assistant's exposure settings
        from homeassistant.components import conversation as ha_conversation
        from homeassistant.components.homeassistant.exposed_entities import async_should_expose

        exposed_entity_ids = {
            state.entity_id
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, ha_conversation.DOMAIN, state.entity_id)
        }

        _LOGGER.debug(
            "Found %d exposed entities for tools: %s",
            len(exposed_entity_ids),
            sorted(exposed_entity_ids),
        )

        # Register ha_control tool
        ha_control = HomeAssistantControlTool(self.hass, exposed_entity_ids)
        self.tool_handler.register_tool(ha_control)

        # Register ha_query tool
        ha_query = HomeAssistantQueryTool(self.hass, exposed_entity_ids)
        self.tool_handler.register_tool(ha_query)

        # Register external LLM tool if enabled
        if self.config.get(CONF_EXTERNAL_LLM_ENABLED, False):
            external_llm = ExternalLLMTool(self.hass, self.config)
            self.tool_handler.register_tool(external_llm)
            _LOGGER.info("External LLM tool registered")

        # Register custom tools from configuration
        custom_tools_config = self.config.get(CONF_TOOLS_CUSTOM, [])
        if custom_tools_config:
            self._register_custom_tools(custom_tools_config)

        # Register memory tools if memory manager is available
        if self.memory_manager is not None:
            from ..tools.memory_tools import RecallMemoryTool, StoreMemoryTool

            store_memory = StoreMemoryTool(
                self.hass,
                self.memory_manager,
                conversation_id=None,  # Will be set per-conversation
            )
            self.tool_handler.register_tool(store_memory)

            recall_memory = RecallMemoryTool(self.hass, self.memory_manager)
            self.tool_handler.register_tool(recall_memory)

            _LOGGER.info("Memory tools registered")

        _LOGGER.debug("Registered %d tools", len(self.tool_handler.get_registered_tools()))

    def _register_custom_tools(self, custom_tools_config: list[dict[str, Any]]) -> None:
        """Register custom tools from configuration.

        Args:
            custom_tools_config: List of custom tool configuration dictionaries
        """
        from ..exceptions import ValidationError

        registered_count = 0
        failed_count = 0

        for tool_config in custom_tools_config:
            try:
                # Create tool from configuration
                custom_tool = CustomToolHandler.create_tool_from_config(
                    self.hass,
                    tool_config,
                )

                # Register with tool handler
                self.tool_handler.register_tool(custom_tool)
                registered_count += 1

                _LOGGER.info(
                    "Registered custom tool '%s' (type: %s)",
                    custom_tool.name,
                    tool_config.get("handler", {}).get("type"),
                )

            except ValidationError as err:
                failed_count += 1
                _LOGGER.error(
                    "Failed to register custom tool (validation error): %s. "
                    "Integration will continue without this tool.",
                    err,
                )
            except Exception as err:
                failed_count += 1
                _LOGGER.error(
                    "Failed to register custom tool (unexpected error): %s. "
                    "Integration will continue without this tool.",
                    err,
                    exc_info=True,
                )

        if registered_count > 0:
            _LOGGER.info(
                "Successfully registered %d custom tool(s)",
                registered_count,
            )

        if failed_count > 0:
            _LOGGER.warning(
                "Failed to register %d custom tool(s). Check logs for details.",
                failed_count,
            )

    def _get_exposed_entities(self) -> list[str]:
        """Get list of entities exposed to the agent.

        Returns:
            List of entity IDs that the agent can access
        """
        # Get entities from context configuration
        context_entities = self.config.get(CONF_CONTEXT_ENTITIES, [])

        exposed = set()
        for entity_config in context_entities:
            if isinstance(entity_config, dict):
                entity_id = entity_config.get("entity_id", "")
            else:
                entity_id = str(entity_config)

            # Handle wildcards by expanding them
            if "*" in entity_id:
                # Get all matching entities
                import fnmatch

                all_entities = self.hass.states.async_entity_ids()
                matching = [e for e in all_entities if fnmatch.fnmatch(e, entity_id)]
                exposed.update(matching)
            else:
                exposed.add(entity_id)

        return list(exposed)

    def get_exposed_entities(self) -> list[dict[str, Any]]:
        """Get exposed entities as structured dictionaries for template rendering.

        Returns:
            List of entity dictionaries with entity_id, name, state, and aliases
        """
        # Get all states that should be exposed to conversation
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, ha_conversation.DOMAIN, state.entity_id)
        ]

        entity_registry = er.async_get(self.hass)
        exposed_entities = []

        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = list(entity.aliases)

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": state.state,
                    "aliases": aliases,
                }
            )

        return exposed_entities

    async def close(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _build_system_prompt(
        self,
        entity_context: str = "",
        conversation_id: str | None = None,
        device_id: str | None = None,
        user_message: str | None = None,
    ) -> str:
        """Build the system prompt for the LLM.

        Args:
            entity_context: Formatted entity context to inject into template
            conversation_id: Current conversation ID
            device_id: Device that triggered the conversation
            user_message: User's current message

        Returns:
            Complete system prompt string
        """
        # Template variables available to custom prompts
        template_vars = {
            "entity_context": entity_context,
            "exposed_entities": self.get_exposed_entities(),
            "ha_name": self.hass.config.location_name,
            "current_device_id": device_id,
            "conversation_id": conversation_id,
            "user_message": user_message,
        }

        if not self.config.get(CONF_PROMPT_USE_DEFAULT, True):
            # Use only custom prompt if default is disabled
            custom_prompt = self.config.get(CONF_PROMPT_CUSTOM_ADDITIONS, "")
            return self._render_template(custom_prompt, template_vars)

        # Start with default prompt and render with entity_context
        prompt = self._render_template(DEFAULT_SYSTEM_PROMPT, template_vars)

        # Add custom additions if provided
        custom_additions = self.config.get(CONF_PROMPT_CUSTOM_ADDITIONS, "")
        if custom_additions:
            rendered_additions = self._render_template(custom_additions, template_vars)
            prompt += f"\n\n## Additional Context\n\n{rendered_additions}"

        return prompt

    def _render_template(self, template_str: str, variables: dict[str, Any] | None = None) -> str:
        """Render a Jinja2 template string.

        Args:
            template_str: Template string to render
            variables: Optional variables to pass to template

        Returns:
            Rendered template string

        Note:
            Templates have access to Home Assistant state via the template context.
            Available variables: states, state_attr, is_state, etc.
            Plus any custom variables passed via the variables parameter.
        """
        if not template_str:
            return ""

        try:
            tpl = template.Template(template_str, self.hass)
            result = tpl.async_render(variables or {})
            # Template.async_render can return Any, so we need to ensure it's a string
            return str(result) if result is not None else ""
        except TemplateError as err:
            _LOGGER.warning("Template rendering failed: %s. Using raw template.", err)
            return template_str

    async def process_message(
        self,
        text: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
    ) -> str:
        """Process a user message and return the agent's response.

        This is the main entry point for conversation processing.

        Args:
            text: User's message text
            conversation_id: Optional conversation ID for history tracking
            user_id: Optional user ID for the conversation
            device_id: Optional device ID that triggered the conversation

        Returns:
            Agent's response text

        Raises:
            HomeAgentError: If processing fails
        """
        # Ensure tools are registered (lazy initialization)
        self._ensure_tools_registered()

        start_time = time.time()

        # Get or create persistent conversation ID for voice interactions
        if conversation_id is None:
            # Try to get existing conversation for this user/device
            conversation_id = self.session_manager.get_conversation_id(
                user_id=user_id,
                device_id=device_id,
            )

            if conversation_id:
                _LOGGER.debug(
                    "Reusing conversation %s for user=%s device=%s",
                    conversation_id,
                    user_id,
                    device_id,
                )
            else:
                # Generate new conversation ID using Home Assistant's ULID format
                from homeassistant.util.ulid import ulid_now

                conversation_id = ulid_now()

                _LOGGER.info(
                    "Created new conversation %s for user=%s device=%s",
                    conversation_id,
                    user_id,
                    device_id,
                )

                # Store the mapping
                await self.session_manager.set_conversation_id(
                    conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                )
        else:
            # Update activity for explicitly provided conversation_id
            await self.session_manager.update_activity(
                user_id=user_id,
                device_id=device_id,
            )

        # Initialize metrics tracking
        metrics = {
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0,
            },
            "performance": {
                "llm_latency_ms": 0,
                "tool_latency_ms": 0,
                "context_latency_ms": 0,
            },
            "context": {},
            "tool_calls": 0,
        }

        # Fire conversation started event
        if self.config.get(CONF_EMIT_EVENTS, True):
            self.hass.bus.async_fire(
                EVENT_CONVERSATION_STARTED,
                {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "device_id": device_id,
                    "timestamp": time.time(),
                    "context_mode": self.config.get(CONF_CONTEXT_MODE),
                },
            )

        try:
            response = await self._process_conversation(text, conversation_id, device_id, metrics)

            # Calculate total duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Get tool metrics
            tool_metrics = self.tool_handler.get_metrics()
            tool_breakdown = {}
            for tool_name in self.tool_handler.get_registered_tools():
                count = tool_metrics.get(f"{tool_name}_executions", 0)
                if count > 0:
                    tool_breakdown[tool_name] = count

            # Check if external LLM was used
            used_external_llm = (
                TOOL_QUERY_EXTERNAL_LLM in tool_breakdown
                and tool_breakdown[TOOL_QUERY_EXTERNAL_LLM] > 0
            )

            # Fire conversation finished event with enhanced metrics
            if self.config.get(CONF_EMIT_EVENTS, True):
                try:
                    event_data = {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "duration_ms": duration_ms,
                        "tokens": metrics["tokens"],
                        "performance": metrics["performance"],
                        "context": metrics.get("context", {}),
                        "tool_calls": metrics["tool_calls"],
                        "tool_breakdown": tool_breakdown,
                        "used_external_llm": used_external_llm,
                    }
                    self.hass.bus.async_fire(EVENT_CONVERSATION_FINISHED, event_data)
                except Exception as event_err:
                    _LOGGER.warning("Failed to fire conversation finished event: %s", event_err)

            # Update session activity to prevent expiration of active conversations
            await self.session_manager.update_activity(
                user_id=user_id,
                device_id=device_id,
            )

            return response

        except Exception as err:
            _LOGGER.error("Error processing message: %s", err, exc_info=True)

            # Fire error event
            if self.config.get(CONF_EMIT_EVENTS, True):
                try:
                    self.hass.bus.async_fire(
                        EVENT_ERROR,
                        {
                            "error_type": type(err).__name__,
                            "error_message": str(err),
                            "conversation_id": conversation_id,
                            "component": "agent",
                            "context": {
                                "text_length": len(text),
                            },
                        },
                    )
                except Exception as event_err:
                    _LOGGER.warning("Failed to fire error event: %s", event_err)

            raise

    async def _async_process_streaming(
        self, user_input: ha_conversation.ConversationInput
    ) -> ha_conversation.ConversationResult:
        """Process conversation with streaming support.

        Args:
            user_input: The conversation input

        Returns:
            ConversationResult with the response
        """
        from homeassistant.components import conversation
        from homeassistant.components.conversation.chat_log import current_chat_log
        from homeassistant.helpers import llm

        from ..streaming import OpenAIStreamingHandler

        chat_log = current_chat_log.get()
        if chat_log is None:
            raise RuntimeError("ChatLog not available in streaming mode")

        user_message = user_input.text
        device_id = user_input.device_id
        user_id = user_input.context.user_id if user_input.context else None

        # Get or create persistent conversation ID for voice interactions
        conversation_id = user_input.conversation_id
        if conversation_id is None:
            # Try to get existing conversation for this user/device
            conversation_id = self.session_manager.get_conversation_id(
                user_id=user_id,
                device_id=device_id,
            )

            if conversation_id:
                _LOGGER.debug(
                    "Reusing conversation %s for user=%s device=%s (streaming)",
                    conversation_id,
                    user_id,
                    device_id,
                )
            else:
                # Generate new conversation ID using Home Assistant's ULID format
                from homeassistant.util.ulid import ulid_now

                conversation_id = ulid_now()

                _LOGGER.info(
                    "Created new conversation %s for user=%s device=%s (streaming)",
                    conversation_id,
                    user_id,
                    device_id,
                )

                # Store the mapping
                await self.session_manager.set_conversation_id(
                    conversation_id,
                    user_id=user_id,
                    device_id=device_id,
                )
        else:
            # Update activity for explicitly provided conversation_id
            await self.session_manager.update_activity(
                user_id=user_id,
                device_id=device_id,
            )

        # Create a simple APIInstance adapter to handle tool calls
        # This allows ChatLog to execute tools via our tool_handler
        class ToolHandlerAPIInstance:
            """Adapter to make tool_handler compatible with llm.APIInstance interface."""

            def __init__(self, tool_handler, metrics, conv_id, max_calls_per_turn):
                self.tool_handler = tool_handler
                self.metrics = metrics
                self.conversation_id = conv_id
                self.max_calls_per_turn = max_calls_per_turn
                self.calls_this_turn = 0

            async def async_call_tool(self, tool_input: llm.ToolInput) -> dict:
                """Execute tool via tool_handler."""
                # Enforce max calls per turn limit
                if self.calls_this_turn >= self.max_calls_per_turn:
                    error_msg = (
                        f"Tool call limit reached ({self.max_calls_per_turn} calls per turn). "
                        f"Skipping tool '{tool_input.tool_name}'."
                    )
                    _LOGGER.warning(error_msg)
                    return {"error": error_msg}

                self.calls_this_turn += 1

                try:
                    result = await self.tool_handler.execute_tool(
                        tool_input.tool_name, tool_input.tool_args, self.conversation_id
                    )
                    self.metrics["tool_calls"] += 1
                    return result if isinstance(result, dict) else {"result": str(result)}
                except Exception as err:
                    _LOGGER.error("Tool execution failed: %s", err, exc_info=True)
                    return {"error": str(err)}

        # Initialize metrics that will be passed to the adapter
        metrics: dict[str, Any] = {
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "performance": {"llm_latency_ms": 0, "tool_latency_ms": 0, "context_latency_ms": 0},
            "context": {},
            "tool_calls": 0,
        }

        # Get max calls per turn for enforcement
        max_calls_per_turn = self.config.get(
            CONF_TOOLS_MAX_CALLS_PER_TURN, DEFAULT_TOOLS_MAX_CALLS_PER_TURN
        )

        # Set the llm_api so ChatLog can execute tools
        chat_log.llm_api = ToolHandlerAPIInstance(
            self.tool_handler, metrics, conversation_id, max_calls_per_turn
        )

        # Track start time for metrics
        start_time = time.time()

        context_start = time.time()
        context = await self.context_manager.get_formatted_context(
            user_message, conversation_id, metrics
        )
        context_latency_ms = int((time.time() - context_start) * 1000)
        metrics["performance"]["context_latency_ms"] = context_latency_ms

        # Build system prompt with full context
        system_prompt = self._build_system_prompt(
            entity_context=context,
            conversation_id=conversation_id,
            device_id=device_id,
            user_message=user_message,
        )

        # Build messages list
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Add conversation history if enabled
        if self.config.get(CONF_HISTORY_ENABLED, True):
            history = self.conversation_manager.get_history(
                conversation_id,
                max_messages=self.config.get(
                    CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES
                ),
            )
            messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Tool calling loop (max iterations to prevent infinite loops)
        max_iterations = self.config.get(
            CONF_TOOLS_MAX_CALLS_PER_TURN, DEFAULT_TOOLS_MAX_CALLS_PER_TURN
        )

        entry_id = None
        # Try to find entry_id from config or hass.data
        if "entry_id" in self.config:
            entry_id = self.config["entry_id"]
        else:
            # Try to find it from domain data
            domain_data = self.hass.data.get(DOMAIN, {})
            for config_entry_id, entry_data in domain_data.items():
                if isinstance(entry_data, dict) and entry_data.get("agent") is self:
                    entry_id = config_entry_id
                    break

        if entry_id is None:
            _LOGGER.warning("Could not find entry_id for streaming, using 'home_agent'")
            entry_id = "home_agent"

        for iteration in range(max_iterations):
            # Call LLM with streaming
            llm_start = time.time()
            stream = self._call_llm_streaming(messages)

            # Transform and send to chat log
            handler = OpenAIStreamingHandler()

            # This will:
            # 1. Transform OpenAI SSE to HA deltas
            # 2. Send deltas to assist pipeline (via chat_log.delta_listener)
            # 3. Execute tools automatically (chat_log.async_add_delta_content_stream does this)
            # 4. Return the content that was added
            new_content = []
            async for content in chat_log.async_add_delta_content_stream(
                entry_id,
                handler.transform_openai_stream(stream),
            ):
                new_content.append(content)

            # Track LLM latency
            llm_latency = int((time.time() - llm_start) * 1000)
            metrics["performance"]["llm_latency_ms"] += llm_latency

            # Track token usage from stream
            usage = handler.get_usage()
            if usage:
                metrics["tokens"]["prompt"] += usage.get("prompt_tokens", 0)
                metrics["tokens"]["completion"] += usage.get("completion_tokens", 0)
                metrics["tokens"]["total"] += usage.get("total_tokens", 0)
                _LOGGER.info("Received token usage from LLM stream: %s", usage)
            else:
                _LOGGER.info("No token usage data received from LLM stream")

            # Convert new content back to messages for next iteration
            for content_item in new_content:
                if isinstance(content_item, conversation.AssistantContent):
                    if content_item.content:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content_item.content,
                            }
                        )
                    if content_item.tool_calls:
                        # Track tool calls
                        metrics["tool_calls"] += len(content_item.tool_calls)

                        # Add tool calls to messages
                        tool_calls_msg = {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.tool_name,
                                        "arguments": json.dumps(tc.tool_args),
                                    },
                                }
                                for tc in content_item.tool_calls
                            ],
                        }
                        messages.append(tool_calls_msg)

                elif isinstance(content_item, conversation.ToolResultContent):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": content_item.tool_call_id,
                            "name": content_item.tool_name,
                            "content": json.dumps(content_item.tool_result),
                        }
                    )

            # Check if we need another iteration (are there unresponded tool results?)
            if not chat_log.unresponded_tool_results:
                break

        # Save to conversation history if enabled
        if self.config.get(CONF_HISTORY_ENABLED, True):
            # Extract final response from chat log
            final_response = ""
            for content_item in new_content:
                if isinstance(content_item, conversation.AssistantContent) and content_item.content:
                    final_response = content_item.content
                    break

            self.conversation_manager.add_message(conversation_id, "user", user_message)
            if final_response:
                self.conversation_manager.add_message(conversation_id, "assistant", final_response)

        # Extract and store memories if enabled (fire and forget)
        if self.config.get(CONF_MEMORY_EXTRACTION_ENABLED, DEFAULT_MEMORY_EXTRACTION_ENABLED):
            # Extract final response for memory extraction
            final_response = ""
            for content_item in new_content:
                if isinstance(content_item, conversation.AssistantContent) and content_item.content:
                    final_response = content_item.content
                    break

            if final_response:
                self.hass.async_create_task(
                    self._extract_and_store_memories(
                        conversation_id=conversation_id,
                        user_message=user_message,
                        assistant_response=final_response,
                        full_messages=messages,
                    )
                )

        # Calculate total duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Get tool metrics
        tool_metrics = self.tool_handler.get_metrics()
        tool_breakdown = {}
        for tool_name in self.tool_handler.get_registered_tools():
            count = tool_metrics.get(f"{tool_name}_executions", 0)
            if count > 0:
                tool_breakdown[tool_name] = count

        # Check if external LLM was used
        used_external_llm = (
            TOOL_QUERY_EXTERNAL_LLM in tool_breakdown
            and tool_breakdown[TOOL_QUERY_EXTERNAL_LLM] > 0
        )

        # Fire conversation finished event with enhanced metrics
        if self.config.get(CONF_EMIT_EVENTS, True):
            try:
                event_data = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "duration_ms": duration_ms,
                    "tokens": metrics["tokens"],
                    "performance": metrics["performance"],
                    "context": metrics.get("context", {}),
                    "tool_calls": metrics["tool_calls"],
                    "tool_breakdown": tool_breakdown,
                    "used_external_llm": used_external_llm,
                }
                self.hass.bus.async_fire(EVENT_CONVERSATION_FINISHED, event_data)
            except Exception as event_err:
                _LOGGER.warning("Failed to fire conversation finished event: %s", event_err)

        # Update session activity to prevent expiration of active conversations
        await self.session_manager.update_activity(
            user_id=user_id,
            device_id=device_id,
        )

        # Extract result from chat log
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _async_process_synchronous(
        self, user_input: ha_conversation.ConversationInput
    ) -> ha_conversation.ConversationResult:
        """Process conversation without streaming (backward compatible mode).

        This is the original processing logic, preserved for backward compatibility
        and as a fallback when streaming fails.

        Args:
            user_input: The conversation input

        Returns:
            ConversationResult with the complete response
        """
        # Use the existing process_message method
        response_text = await self.process_message(
            text=user_input.text,
            conversation_id=user_input.conversation_id,
            user_id=user_input.context.user_id if user_input.context else None,
            device_id=user_input.device_id,
        )

        # Create and return conversation result
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ha_conversation.ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
        )

    async def _process_conversation(
        self,
        user_message: str,
        conversation_id: str,
        device_id: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        """Process a conversation with tool calling loop.

        Args:
            user_message: User's message
            conversation_id: Conversation ID for history
            device_id: Device ID that triggered the conversation
            metrics: Optional metrics dictionary to populate

        Returns:
            Final response text
        """
        if metrics is None:
            metrics = {}

        # Get context from context manager with timing
        context_start = time.time()
        context = await self.context_manager.get_formatted_context(
            user_message, conversation_id, metrics
        )
        context_latency_ms = int((time.time() - context_start) * 1000)

        if "performance" in metrics:
            metrics["performance"]["context_latency_ms"] = context_latency_ms

        # Build system prompt with full context including device_id
        system_prompt = self._build_system_prompt(
            entity_context=context,
            conversation_id=conversation_id,
            device_id=device_id,
            user_message=user_message,
        )

        # Debug: Log context injection
        if context:
            _LOGGER.debug(
                "Entity context injected: %d chars, contains %d entities",
                len(context),
                context.count('"entity_id"') if isinstance(context, str) else 0,
            )

        # Build messages list
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Add conversation history if enabled
        if self.config.get(CONF_HISTORY_ENABLED, True):
            history = self.conversation_manager.get_history(
                conversation_id,
                max_messages=self.config.get(
                    CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES
                ),
            )
            messages.extend(history)

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Get tool definitions
        tool_definitions = self.tool_handler.get_tool_definitions()

        # Tool calling loop
        max_iterations = self.config.get(CONF_TOOLS_MAX_CALLS_PER_TURN, 5)
        iteration = 0
        total_llm_latency_ms = 0
        total_tool_latency_ms = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM with timing
            llm_start = time.time()
            llm_response = await self._call_llm(messages, tools=tool_definitions)
            llm_latency_ms = int((time.time() - llm_start) * 1000)
            total_llm_latency_ms += llm_latency_ms

            # Track token usage from LLM response
            usage = llm_response.get("usage", {})
            if usage and "tokens" in metrics:
                metrics["tokens"]["prompt"] += usage.get("prompt_tokens", 0)
                metrics["tokens"]["completion"] += usage.get("completion_tokens", 0)
                metrics["tokens"]["total"] += usage.get("total_tokens", 0)

            # Extract response message
            response_message = llm_response.get("choices", [{}])[0].get("message", {})

            # Log response for debugging
            _LOGGER.debug(
                "LLM response (iteration %d): content=%s, has_tool_calls=%s",
                iteration,
                bool(response_message.get("content")),
                bool(response_message.get("tool_calls")),
            )

            # Check if LLM wants to call tools
            tool_calls = response_message.get("tool_calls", [])

            # Always log tool call detection for debugging
            if tool_calls:
                _LOGGER.info("Detected %d tool call(s) from LLM", len(tool_calls))
            else:
                _LOGGER.info("No tool calls in LLM response")

            if not tool_calls:
                # No tool calls, we're done
                # Strip thinking blocks from reasoning models before returning
                raw_content = response_message.get("content") or ""
                final_content = strip_thinking_blocks(raw_content) or ""

                # Log if we got an empty response
                if not final_content:
                    _LOGGER.error(
                        "LLM returned empty content after iteration %d. Response message: %s",
                        iteration,
                        response_message,
                    )
                    # Provide an error message instead of misleading success message
                    final_content = (
                        "There was an error completing your request. "
                        "The assistant did not provide a response."
                    )

                # Update final performance metrics
                if "performance" in metrics:
                    metrics["performance"]["llm_latency_ms"] = total_llm_latency_ms
                    metrics["performance"]["tool_latency_ms"] = total_tool_latency_ms

                # Save to conversation history
                if self.config.get(CONF_HISTORY_ENABLED, True):
                    self.conversation_manager.add_message(conversation_id, "user", user_message)
                    self.conversation_manager.add_message(
                        conversation_id, "assistant", final_content
                    )

                # Extract and store memories if enabled (fire and forget)
                if self.config.get(
                    CONF_MEMORY_EXTRACTION_ENABLED, DEFAULT_MEMORY_EXTRACTION_ENABLED
                ):
                    self.hass.async_create_task(
                        self._extract_and_store_memories(
                            conversation_id=conversation_id,
                            user_message=user_message,
                            assistant_response=final_content,
                            full_messages=messages,
                        )
                    )

                return final_content

            # Execute tool calls
            _LOGGER.info("Executing %d tool call(s)", len(tool_calls))

            # Enforce tool call limit
            max_calls = self.config.get(
                CONF_TOOLS_MAX_CALLS_PER_TURN, DEFAULT_TOOLS_MAX_CALLS_PER_TURN
            )
            if len(tool_calls) > max_calls:
                _LOGGER.warning(
                    "LLM requested %d tool calls, but limit is %d. Only executing first %d.",
                    len(tool_calls),
                    max_calls,
                    max_calls,
                )
                tool_calls = tool_calls[:max_calls]

            metrics["tool_calls"] = metrics.get("tool_calls", 0) + len(tool_calls)

            # Add assistant message with tool calls to messages
            messages.append(response_message)

            # Execute each tool
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args_raw = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id", "")

                try:
                    # Parse tool arguments - handle both string (OpenAI) and dict (Ollama) formats
                    if isinstance(tool_args_raw, str):
                        tool_args = json.loads(tool_args_raw)
                        _LOGGER.info("Tool '%s': parsed arguments from JSON string", tool_name)
                    elif isinstance(tool_args_raw, dict):
                        tool_args = tool_args_raw
                        _LOGGER.info("Tool '%s': using dict arguments (Ollama format)", tool_name)
                    else:
                        _LOGGER.error(
                            "Invalid tool arguments type for '%s': %s",
                            tool_name,
                            type(tool_args_raw),
                        )
                        tool_args = {}

                    # Execute tool with timing
                    _LOGGER.info("Calling tool '%s' with args: %s", tool_name, tool_args)
                    tool_start = time.time()
                    result = await self.tool_handler.execute_tool(
                        tool_name, tool_args, conversation_id
                    )
                    _LOGGER.info(
                        "Tool '%s' completed successfully in %.2fms",
                        tool_name,
                        (time.time() - tool_start) * 1000,
                    )
                    tool_latency_ms = int((time.time() - tool_start) * 1000)
                    total_tool_latency_ms += tool_latency_ms

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": json.dumps(result),
                        }
                    )

                except Exception as err:
                    _LOGGER.error("Tool execution failed: %s", err)
                    # Add error result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": json.dumps(
                                {
                                    "success": False,
                                    "error": str(err),
                                }
                            ),
                        }
                    )

            # Continue loop to get LLM's response with tool results

        # Update final performance metrics
        if "performance" in metrics:
            metrics["performance"]["llm_latency_ms"] = total_llm_latency_ms
            metrics["performance"]["tool_latency_ms"] = total_tool_latency_ms

        # Max iterations reached
        _LOGGER.warning("Max tool calling iterations reached")
        return (
            "I apologize, but I couldn't complete your request after "
            "multiple attempts. Please try rephrasing your request."
        )

    async def clear_history(self, conversation_id: str | None = None) -> None:
        """Clear conversation history.

        Args:
            conversation_id: Specific conversation to clear, or None for all
        """
        if conversation_id:
            self.conversation_manager.clear_history(conversation_id)
            _LOGGER.info("Cleared history for conversation %s", conversation_id)
        else:
            self.conversation_manager.clear_all()
            _LOGGER.info("Cleared all conversation history")

    async def reload_context(self) -> None:
        """Reload entity context (useful after entity changes)."""
        # Context is fetched fresh each time, but we can clear cache
        _LOGGER.info("Context reload requested (context is fetched dynamically)")

    async def execute_tool_debug(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool directly for debugging/testing.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        _LOGGER.info("Debug tool execution: %s", tool_name)
        return await self.tool_handler.execute_tool(tool_name, parameters, "debug")

    async def update_config(self, config: dict[str, Any]) -> None:
        """Update agent configuration.

        Args:
            config: New configuration dictionary
        """
        self.config.update(config)

        # Update sub-components
        await self.context_manager.update_config(config)
        self.conversation_manager.update_limits(
            max_messages=config.get(CONF_HISTORY_MAX_MESSAGES),
            max_tokens=config.get(CONF_HISTORY_MAX_TOKENS),
        )

        _LOGGER.info("Agent configuration updated")

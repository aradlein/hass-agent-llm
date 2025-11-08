"""Main conversation agent for Home Agent.

This module implements the core HomeAgent class that orchestrates all components
to provide intelligent conversation capabilities integrated with Home Assistant.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator

import aiohttp
from homeassistant.components import conversation as ha_conversation
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import intent, template

from .const import (
    CONF_CONTEXT_ENTITIES,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_HISTORY_PERSIST,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_LLM_TOP_P,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_MEMORY_EXTRACTION_LLM,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_CUSTOM,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
    DEFAULT_MEMORY_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_LLM,
    DEFAULT_STREAMING_ENABLED,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
    EVENT_CONVERSATION_STARTED,
    EVENT_ERROR,
    EVENT_MEMORY_EXTRACTED,
    EVENT_STREAMING_ERROR,
    HTTP_TIMEOUT,
    TOOL_QUERY_EXTERNAL_LLM,
)
from .context_manager import ContextManager
from .conversation import ConversationHistoryManager
from .exceptions import AuthenticationError, HomeAgentError
from .helpers import redact_sensitive_data
from .tool_handler import ToolHandler
from .tools import HomeAssistantControlTool, HomeAssistantQueryTool
from .tools.custom import CustomToolHandler
from .tools.external_llm import ExternalLLMTool

_LOGGER = logging.getLogger(__name__)


class HomeAgent(AbstractConversationAgent):
    """Main conversation agent that orchestrates all Home Agent components.

    This class integrates with Home Assistant's conversation platform and provides
    intelligent conversational AI capabilities with tool calling, context injection,
    and conversation history management.
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the Home Agent.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary containing LLM settings, context config, etc.
        """
        self.hass = hass
        self.config = config

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
        """Return list of supported languages."""
        return ["en"]

    @property
    def memory_manager(self):
        """Get memory manager from hass.data if available."""
        if self._memory_manager is None:
            # Try to get memory manager from hass.data
            domain_data = self.hass.data.get(DOMAIN, {})
            for entry_data in domain_data.values():
                if isinstance(entry_data, dict) and "memory_manager" in entry_data:
                    self._memory_manager = entry_data["memory_manager"]
                    break
        return self._memory_manager

    def _can_stream(self) -> bool:
        """Check if streaming is supported in the current context.

        Returns:
            True if streaming is enabled and ChatLog with delta_listener is available
        """
        from homeassistant.components.conversation.chat_log import current_chat_log

        # Check if streaming is enabled in config
        if not self.config.get(CONF_STREAMING_ENABLED, DEFAULT_STREAMING_ENABLED):
            return False

        # Check if ChatLog with delta_listener exists (means assist pipeline supports streaming)
        chat_log = current_chat_log.get()
        if chat_log is None or chat_log.delta_listener is None:
            return False

        return True

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

        except Exception as err:
            _LOGGER.error("Error in async_process: %s", err, exc_info=True)

            # Return error response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I encountered an error: {err}",
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
            from .tools.memory_tools import RecallMemoryTool, StoreMemoryTool

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
        from .exceptions import ValidationError

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

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists.

        Returns:
            Active aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

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
            return tpl.async_render(variables or {})
        except TemplateError as err:
            _LOGGER.warning("Template rendering failed: %s. Using raw template.", err)
            return template_str

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call the LLM API.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions
            temperature: Optional temperature override (uses config if not provided)
            max_tokens: Optional max_tokens override (uses config if not provided)

        Returns:
            LLM response dictionary

        Raises:
            AuthenticationError: If API authentication fails
            HomeAgentError: If API call fails
        """
        session = await self._ensure_session()

        url = f"{self.config[CONF_LLM_BASE_URL]}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config[CONF_LLM_API_KEY]}",
        }

        payload: dict[str, Any] = {
            "model": self.config[CONF_LLM_MODEL],
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.config.get(CONF_LLM_TEMPERATURE, 0.7),
            "max_tokens": max_tokens
            if max_tokens is not None
            else self.config.get(CONF_LLM_MAX_TOKENS, 500),
            "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if self.config.get(CONF_DEBUG_LOGGING):
            _LOGGER.debug(
                "Calling LLM at %s with %d messages and %d tools",
                redact_sensitive_data(url, self.config[CONF_LLM_API_KEY]),
                len(messages),
                len(tools) if tools else 0,
            )

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 401:
                    raise AuthenticationError("LLM API authentication failed. Check your API key.")

                if response.status != 200:
                    error_text = await response.text()
                    raise HomeAgentError(f"LLM API returned status {response.status}: {error_text}")

                result = await response.json()
                return result

        except aiohttp.ClientError as err:
            raise HomeAgentError(f"Failed to connect to LLM API: {err}") from err

    async def _call_llm_streaming(
        self, messages: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Call LLM API with streaming enabled.

        Args:
            messages: Conversation messages

        Yields:
            SSE lines from the streaming response
        """
        session = await self._ensure_session()

        url = f"{self.config[CONF_LLM_BASE_URL]}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config[CONF_LLM_API_KEY]}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.config[CONF_LLM_MODEL],
            "messages": messages,
            "temperature": self.config.get(CONF_LLM_TEMPERATURE, 0.7),
            "max_tokens": self.config.get(CONF_LLM_MAX_TOKENS, 1000),
            "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
            "stream": True,  # Enable streaming!
        }

        # Add tools if available
        tool_definitions = self.tool_handler.get_tool_definitions()
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = "auto"

        # Request usage statistics in stream
        payload["stream_options"] = {"include_usage": True}

        if self.config.get(CONF_DEBUG_LOGGING):
            _LOGGER.debug(
                "Calling LLM (streaming) at %s with %d messages and %d tools",
                redact_sensitive_data(url, self.config[CONF_LLM_API_KEY]),
                len(messages),
                len(tool_definitions) if tool_definitions else 0,
            )

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 401:
                    raise AuthenticationError("LLM API authentication failed. Check your API key.")

                if response.status != 200:
                    error_text = await response.text()
                    raise HomeAgentError(f"LLM API returned status {response.status}: {error_text}")

                # Stream SSE lines
                async for line in response.content:
                    if line:
                        yield line.decode("utf-8")

        except aiohttp.ClientError as err:
            raise HomeAgentError(f"Failed to connect to LLM API: {err}") from err

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
        conversation_id = conversation_id or "default"

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

        from .streaming import OpenAIStreamingHandler

        chat_log = current_chat_log.get()
        if chat_log is None:
            raise RuntimeError("ChatLog not available in streaming mode")

        conversation_id = user_input.conversation_id or "default"
        user_message = user_input.text
        device_id = user_input.device_id
        user_id = user_input.context.user_id if user_input.context else None

        # Track start time for metrics
        start_time = time.time()

        # Get context from context manager
        metrics: dict[str, Any] = {
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "performance": {"llm_latency_ms": 0, "tool_latency_ms": 0, "context_latency_ms": 0},
            "context": {},
            "tool_calls": 0,
        }

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
                final_content = response_message.get("content") or ""

                # Log if we got an empty response
                if not final_content:
                    _LOGGER.warning(
                        "LLM returned empty content after iteration %d. Response message: %s",
                        iteration,
                        response_message,
                    )
                    # Provide a fallback message
                    final_content = "I've completed your request."

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

    # Memory Extraction Methods

    def _format_conversation_for_extraction(self, messages: list[dict[str, Any]]) -> str:
        """Format conversation history for memory extraction.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted conversation text
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")

            # Skip system messages and empty messages
            if role.lower() == "system" or not content:
                continue

            # Skip tool messages
            if role.lower() == "tool":
                continue

            formatted_parts.append(f"{role}: {content}")

        return "\n".join(formatted_parts)

    def _build_extraction_prompt(
        self,
        user_message: str,
        assistant_response: str,
        full_messages: list[dict[str, Any]],
    ) -> str:
        """Build prompt for memory extraction.

        Args:
            user_message: Current user message
            assistant_response: Assistant's response
            full_messages: Complete conversation history

        Returns:
            Extraction prompt
        """
        # Format conversation history (exclude current turn)
        history_messages = [
            msg for msg in full_messages if msg.get("role") not in ["system", "tool"]
        ]

        # Get previous turns (exclude the current user message we just added)
        if history_messages and history_messages[-1].get("content") == user_message:
            previous_turns = history_messages[:-1]
        else:
            previous_turns = history_messages

        conversation_text = ""
        if previous_turns:
            conversation_text = self._format_conversation_for_extraction(previous_turns)

        prompt = f"""You are a memory extraction assistant. Analyze this conversation \
and extract important information that should be remembered for future conversations.

Extract the following types of information:
1. **Facts**: Concrete information about the home, devices, or user
2. **Preferences**: User preferences for temperature, lighting, routines, etc.
3. **Context**: Background information useful for future interactions
4. **Events**: Significant events or actions that occurred

## Previous Conversation

{conversation_text if conversation_text else "(No previous conversation)"}

## Current Turn

User: {user_message}
Assistant: {assistant_response}

## Instructions

Extract memories as a JSON array. Each memory should have:
- "type": One of "fact", "preference", "context", "event"
- "content": Clear, concise description (1-2 sentences)
- "importance": Score from 0.0 to 1.0 (1.0 = very important)
- "entities": List of Home Assistant entity IDs mentioned (if any)
- "topics": List of topic tags (e.g., ["temperature", "bedroom"])

**Important:**
- Only extract genuinely useful information
- Be specific and concrete
- Avoid extracting temporary states (e.g., "light is on" - too transient)
- DO extract patterns and preferences (e.g., "user prefers bedroom at 68°F")
- If nothing worth remembering, return empty array: []

Return ONLY valid JSON, no other text:

```json
[
  {{
    "type": "preference",
    "content": "User prefers bedroom temperature at 68°F for sleeping",
    "importance": 0.8,
    "entities": ["climate.bedroom"],
    "topics": ["temperature", "bedroom", "sleep"]
  }}
]
```"""

        return prompt

    async def _call_primary_llm_for_extraction(self, extraction_prompt: str) -> dict[str, Any]:
        """Call primary/local LLM for memory extraction.

        Args:
            extraction_prompt: The extraction prompt

        Returns:
            Dictionary with success, result, and error fields
        """
        try:
            # Build simple message for extraction
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a memory extraction assistant. Extract important "
                        "information from conversations and return it as JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": extraction_prompt,
                },
            ]

            # Call LLM without tool definitions
            # Use lower temperature (0.3) for more consistent extraction
            response = await self._call_llm(
                messages,
                tools=None,
                temperature=0.3,
            )

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            return {
                "success": True,
                "result": content,
                "error": None,
            }

        except Exception as err:
            _LOGGER.error("Primary LLM extraction failed: %s", err)
            return {
                "success": False,
                "result": None,
                "error": str(err),
            }

    async def _parse_and_store_memories(
        self,
        extraction_result: str,
        conversation_id: str,
    ) -> int:
        """Parse LLM extraction result and store memories.

        Args:
            extraction_result: JSON string from LLM
            conversation_id: Conversation ID

        Returns:
            Number of memories stored
        """
        try:
            # Clean up the result - extract JSON if wrapped in markdown
            result = extraction_result.strip()
            if "```json" in result:
                # Extract JSON from markdown code block
                start = result.find("```json") + 7
                end = result.find("```", start)
                result = result[start:end].strip()
            elif "```" in result:
                # Extract from generic code block
                start = result.find("```") + 3
                end = result.find("```", start)
                result = result[start:end].strip()

            # Parse JSON response
            memories = json.loads(result)

            if not isinstance(memories, list):
                _LOGGER.error(
                    "Expected JSON array from memory extraction, got: %s",
                    type(memories).__name__,
                )
                return 0

            if not memories:
                _LOGGER.debug("No memories extracted from conversation")
                return 0

            # Store each memory
            stored_count = 0
            for memory_data in memories:
                try:
                    # Validate memory data
                    if not isinstance(memory_data, dict):
                        _LOGGER.warning("Skipping invalid memory data: %s", memory_data)
                        continue

                    if "content" not in memory_data:
                        _LOGGER.warning("Skipping memory without content: %s", memory_data)
                        continue

                    content = memory_data["content"]
                    memory_type = memory_data.get("type", "fact")

                    # Validate: reject transient state stored as facts
                    if memory_type == "fact" and self.memory_manager._is_transient_state(content):
                        _LOGGER.warning(
                            "Rejecting transient state stored as fact: %s. "
                            "Transient states should use 'event' type.",
                            content[:50],
                        )
                        continue

                    memory_id = await self.memory_manager.add_memory(
                        content=content,
                        memory_type=memory_type,
                        conversation_id=conversation_id,
                        importance=memory_data.get("importance", 0.5),
                        metadata={
                            "entities_involved": memory_data.get("entities", []),
                            "topics": memory_data.get("topics", []),
                            "extraction_method": "automatic",
                        },
                    )
                    stored_count += 1
                    _LOGGER.debug("Stored memory %s: %s", memory_id, content[:50])

                except Exception as err:
                    _LOGGER.error("Failed to store memory: %s", err)
                    continue

            if stored_count > 0:
                _LOGGER.info(
                    "Extracted and stored %d memories from conversation %s",
                    stored_count,
                    conversation_id,
                )

            return stored_count

        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to parse memory extraction JSON: %s", err)
            _LOGGER.debug("Raw extraction result: %s", extraction_result)
            return 0
        except Exception as err:
            _LOGGER.error("Error parsing and storing memories: %s", err)
            return 0

    async def _extract_and_store_memories(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        full_messages: list[dict[str, Any]],
    ) -> None:
        """Extract memories from completed conversation using configured LLM.

        This method:
        1. Determines which LLM to use (external or local)
        2. Builds extraction prompt
        3. Calls LLM to extract memories
        4. Parses JSON response
        5. Stores memories via MemoryManager

        Args:
            conversation_id: Conversation ID
            user_message: User's message
            assistant_response: Assistant's response
            full_messages: Complete conversation history
        """
        try:
            # Check if memory system is enabled
            if not self.config.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED):
                return

            # Check if memory manager is available
            if self.memory_manager is None:
                _LOGGER.debug("Memory manager not available, skipping extraction")
                return

            # Determine which LLM to use for extraction
            extraction_llm = self.config.get(
                CONF_MEMORY_EXTRACTION_LLM, DEFAULT_MEMORY_EXTRACTION_LLM
            )

            # Build extraction prompt
            extraction_prompt = self._build_extraction_prompt(
                user_message=user_message,
                assistant_response=assistant_response,
                full_messages=full_messages,
            )

            # Call appropriate LLM
            if extraction_llm == "external":
                # Check if external LLM is enabled
                if not self.config.get(CONF_EXTERNAL_LLM_ENABLED, False):
                    _LOGGER.warning(
                        "Memory extraction configured to use external LLM, "
                        "but external LLM is not enabled. Skipping extraction."
                    )
                    return

                # Use external LLM tool
                _LOGGER.debug("Using external LLM for memory extraction")
                result = await self.tool_handler.execute_tool(
                    tool_name="query_external_llm",
                    parameters={"prompt": extraction_prompt},
                    conversation_id=conversation_id,
                )

                if not result.get("success"):
                    _LOGGER.error(
                        "External LLM memory extraction failed: %s",
                        result.get("error"),
                    )
                    return

                extraction_result = result.get("result", "[]")

            else:
                # Use local/primary LLM
                _LOGGER.debug("Using local LLM for memory extraction")
                result = await self._call_primary_llm_for_extraction(extraction_prompt)

                if not result.get("success"):
                    _LOGGER.error("Local LLM memory extraction failed: %s", result.get("error"))
                    return

                extraction_result = result.get("result", "[]")

            # Parse and store memories
            stored_count = await self._parse_and_store_memories(
                extraction_result=extraction_result,
                conversation_id=conversation_id,
            )

            # Fire event if memories were extracted
            if stored_count > 0 and self.config.get(CONF_EMIT_EVENTS, True):
                from datetime import datetime

                self.hass.bus.async_fire(
                    EVENT_MEMORY_EXTRACTED,
                    {
                        "conversation_id": conversation_id,
                        "memories_extracted": stored_count,
                        "extraction_llm": extraction_llm,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except Exception as err:
            _LOGGER.exception("Error during memory extraction: %s", err)

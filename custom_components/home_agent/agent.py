"""Main conversation agent for Home Agent.

This module implements the core HomeAgent class that orchestrates all components
to provide intelligent conversation capabilities integrated with Home Assistant.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiohttp

from homeassistant.components import conversation as ha_conversation
from homeassistant.components.conversation.models import AbstractConversationAgent
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er, intent, template
from homeassistant.components.homeassistant.exposed_entities import async_should_expose

from .const import (
    CONF_CONTEXT_ENTITIES,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_LLM_TOP_P,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
    DEFAULT_SYSTEM_PROMPT,
    EVENT_CONVERSATION_FINISHED,
    EVENT_CONVERSATION_STARTED,
    EVENT_ERROR,
    HTTP_TIMEOUT,
)
from .context_manager import ContextManager
from .conversation import ConversationHistoryManager
from .exceptions import (
    AuthenticationError,
    HomeAgentError,
)
from .helpers import redact_sensitive_data
from .tool_handler import ToolHandler
from .tools import HomeAssistantControlTool, HomeAssistantQueryTool

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
            max_messages=config.get(
                CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES
            ),
            max_tokens=config.get(CONF_HISTORY_MAX_TOKENS, DEFAULT_HISTORY_MAX_TOKENS),
        )
        self.tool_handler = ToolHandler(
            hass,
            {
                "max_calls_per_turn": config.get(CONF_TOOLS_MAX_CALLS_PER_TURN),
                "timeout": config.get(CONF_TOOLS_TIMEOUT),
                "emit_events": config.get(CONF_EMIT_EVENTS, True),
            },
        )

        # Register core tools
        self._register_tools()

        # HTTP session for LLM API calls
        self._session: aiohttp.ClientSession | None = None

        _LOGGER.info("Home Agent initialized with model %s", config.get(CONF_LLM_MODEL))

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return ["en"]

    async def async_process(
        self, user_input: ha_conversation.ConversationInput
    ) -> ha_conversation.ConversationResult:
        """Process a conversation turn.

        This method is required by AbstractConversationAgent and bridges the
        Home Assistant conversation platform with Home Agent's processing logic.

        Args:
            user_input: Conversation input from Home Assistant

        Returns:
            ConversationResult with the agent's response
        """
        try:
            # Process the message using our internal method
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
        from homeassistant.components.homeassistant.exposed_entities import (
            async_should_expose,
        )

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

        _LOGGER.debug(
            "Registered %d tools", len(self.tool_handler.get_registered_tools())
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

    def _render_template(
        self, template_str: str, variables: dict[str, Any] | None = None
    ) -> str:
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
        except template.TemplateError as err:
            _LOGGER.warning("Template rendering failed: %s. Using raw template.", err)
            return template_str

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call the LLM API.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions

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
            "temperature": self.config.get(CONF_LLM_TEMPERATURE, 0.7),
            "max_tokens": self.config.get(CONF_LLM_MAX_TOKENS, 500),
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
                    raise AuthenticationError(
                        "LLM API authentication failed. Check your API key."
                    )

                if response.status != 200:
                    error_text = await response.text()
                    raise HomeAgentError(
                        f"LLM API returned status {response.status}: {error_text}"
                    )

                result = await response.json()
                return result

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
        start_time = time.time()
        conversation_id = conversation_id or "default"

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
            response = await self._process_conversation(
                text, conversation_id, device_id
            )

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Fire conversation finished event
            if self.config.get(CONF_EMIT_EVENTS, True):
                self.hass.bus.async_fire(
                    EVENT_CONVERSATION_FINISHED,
                    {
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "duration_ms": duration_ms,
                        "tool_calls": self.tool_handler.get_metrics().get(
                            "total_executions", 0
                        ),
                    },
                )

            return response

        except Exception as err:
            _LOGGER.error("Error processing message: %s", err, exc_info=True)

            # Fire error event
            if self.config.get(CONF_EMIT_EVENTS, True):
                self.hass.bus.async_fire(
                    EVENT_ERROR,
                    {
                        "error_type": type(err).__name__,
                        "error_message": str(err),
                        "conversation_id": conversation_id,
                        "component": "agent",
                    },
                )

            raise

    async def _process_conversation(
        self,
        user_message: str,
        conversation_id: str,
        device_id: str | None = None,
    ) -> str:
        """Process a conversation with tool calling loop.

        Args:
            user_message: User's message
            conversation_id: Conversation ID for history
            device_id: Device ID that triggered the conversation

        Returns:
            Final response text
        """
        # Get context from context manager
        context = await self.context_manager.get_formatted_context(
            user_message, conversation_id
        )

        # Build system prompt with full context including device_id
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

        # Get tool definitions
        tool_definitions = self.tool_handler.get_tool_definitions()

        # Tool calling loop
        max_iterations = self.config.get(CONF_TOOLS_MAX_CALLS_PER_TURN, 5)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM
            llm_response = await self._call_llm(messages, tools=tool_definitions)

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

                # Save to conversation history
                if self.config.get(CONF_HISTORY_ENABLED, True):
                    self.conversation_manager.add_message(
                        conversation_id, "user", user_message
                    )
                    self.conversation_manager.add_message(
                        conversation_id, "assistant", final_content
                    )

                return final_content

            # Execute tool calls
            _LOGGER.debug("LLM requested %d tool calls", len(tool_calls))

            # Add assistant message with tool calls to messages
            messages.append(response_message)

            # Execute each tool
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id", "")

                try:
                    # Parse tool arguments
                    tool_args = json.loads(tool_args_str)

                    # Execute tool
                    result = await self.tool_handler.execute_tool(
                        tool_name, tool_args, conversation_id
                    )

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

        # Max iterations reached
        _LOGGER.warning("Max tool calling iterations reached")
        return "I apologize, but I couldn't complete your request after multiple attempts. Please try rephrasing your request."

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

    def update_config(self, config: dict[str, Any]) -> None:
        """Update agent configuration.

        Args:
            config: New configuration dictionary
        """
        self.config.update(config)

        # Update sub-components
        self.context_manager.update_config(config)
        self.conversation_manager.update_limits(
            max_messages=config.get(CONF_HISTORY_MAX_MESSAGES),
            max_tokens=config.get(CONF_HISTORY_MAX_TOKENS),
        )

        _LOGGER.info("Agent configuration updated")

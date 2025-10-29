"""Main conversation agent for Home Agent.

This module implements the core HomeAgent class that orchestrates all components
to provide intelligent conversation capabilities integrated with Home Assistant.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp

from homeassistant.core import HomeAssistant

from .const import (
    CONF_CONTEXT_ENTITIES,
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    CONF_EXTERNAL_LLM_ENABLED,
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
    ToolExecutionError,
)
from .helpers import redact_sensitive_data
from .tool_handler import ToolHandler
from .tools import HomeAssistantControlTool, HomeAssistantQueryTool

_LOGGER = logging.getLogger(__name__)


class HomeAgent:
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
            max_messages=config.get(CONF_HISTORY_MAX_MESSAGES),
            max_tokens=config.get(CONF_HISTORY_MAX_TOKENS),
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

        # Register core tools
        self._register_tools()

        # HTTP session for LLM API calls
        self._session: aiohttp.ClientSession | None = None

        _LOGGER.info("Home Agent initialized with model %s", config.get(CONF_LLM_MODEL))

    def _register_tools(self) -> None:
        """Register core Home Assistant tools."""
        # Get exposed entities from config (for security)
        exposed_entities = set(self._get_exposed_entities())

        # Register ha_control tool
        ha_control = HomeAssistantControlTool(self.hass, exposed_entities)
        self.tool_handler.register_tool(ha_control)

        # Register ha_query tool
        ha_query = HomeAssistantQueryTool(self.hass, exposed_entities)
        self.tool_handler.register_tool(ha_query)

        _LOGGER.debug("Registered %d tools", len(self.tool_handler.get_registered_tools()))

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

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM.

        Returns:
            Complete system prompt string
        """
        if not self.config.get(CONF_PROMPT_USE_DEFAULT, True):
            # Use only custom prompt if default is disabled
            return self.config.get(CONF_PROMPT_CUSTOM_ADDITIONS, "")

        # Start with default prompt
        prompt = DEFAULT_SYSTEM_PROMPT

        # Add custom additions if provided
        custom_additions = self.config.get(CONF_PROMPT_CUSTOM_ADDITIONS, "")
        if custom_additions:
            prompt += f"\n\n## Additional Context\n\n{custom_additions}"

        return prompt

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
                    raise AuthenticationError("LLM API authentication failed. Check your API key.")

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
    ) -> str:
        """Process a user message and return the agent's response.

        This is the main entry point for conversation processing.

        Args:
            text: User's message text
            conversation_id: Optional conversation ID for history tracking
            user_id: Optional user ID for the conversation

        Returns:
            Agent's response text

        Raises:
            HomeAgentError: If processing fails
        """
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
                    "timestamp": time.time(),
                    "context_mode": self.config.get(CONF_CONTEXT_MODE),
                },
            )

        try:
            response = await self._process_conversation(text, conversation_id, metrics)

            # Calculate total duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Get tool metrics
            tool_metrics = self.tool_handler.get_metrics()
            tool_breakdown = {}
            for tool_name in self.tool_handler.get_registered_tools():
                count = tool_metrics.get(f"{tool_name}_executions", 0)
                if count > 0:
                    tool_breakdown[tool_name] = count

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

    async def _process_conversation(
        self,
        user_message: str,
        conversation_id: str,
        metrics: dict[str, Any] | None = None,
    ) -> str:
        """Process a conversation with tool calling loop.

        Args:
            user_message: User's message
            conversation_id: Conversation ID for history
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

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Replace placeholders in system prompt
        system_prompt = system_prompt.replace("{{entity_context}}", context)

        # Build messages list
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Add conversation history if enabled
        if self.config.get(CONF_HISTORY_ENABLED, True):
            history = self.conversation_manager.get_history(
                conversation_id,
                max_messages=self.config.get(CONF_HISTORY_MAX_MESSAGES),
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

            # Check if LLM wants to call tools
            tool_calls = response_message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls, we're done
                final_content = response_message.get("content", "")

                # Update final performance metrics
                if "performance" in metrics:
                    metrics["performance"]["llm_latency_ms"] = total_llm_latency_ms
                    metrics["performance"]["tool_latency_ms"] = total_tool_latency_ms

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
            metrics["tool_calls"] = metrics.get("tool_calls", 0) + len(tool_calls)

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

                    # Execute tool with timing
                    tool_start = time.time()
                    result = await self.tool_handler.execute_tool(
                        tool_name, tool_args, conversation_id
                    )
                    tool_latency_ms = int((time.time() - tool_start) * 1000)
                    total_tool_latency_ms += tool_latency_ms

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    })

                except Exception as err:
                    _LOGGER.error("Tool execution failed: %s", err)
                    # Add error result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps({
                            "success": False,
                            "error": str(err),
                        }),
                    })

            # Continue loop to get LLM's response with tool results

        # Update final performance metrics
        if "performance" in metrics:
            metrics["performance"]["llm_latency_ms"] = total_llm_latency_ms
            metrics["performance"]["tool_latency_ms"] = total_tool_latency_ms

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

"""Tool handler for managing and executing tools in the Home Agent component.

This module provides the ToolHandler class which orchestrates tool registration,
validation, and execution. It acts as the bridge between the LLM's tool calls
and the actual tool implementations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from homeassistant.core import HomeAssistant

from .const import (
    CONF_EMIT_EVENTS,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
    DEFAULT_TOOLS_TIMEOUT,
    EVENT_TOOL_EXECUTED,
)
from .exceptions import ToolExecutionError, ValidationError
from .helpers import redact_sensitive_data, truncate_text

_LOGGER = logging.getLogger(__name__)


class ToolHandler:
    """Handles tool registration, validation, and execution for Home Agent.

    This class manages the lifecycle of tools that can be called by the LLM,
    including native tools (ha_control, ha_query), custom tools, and the
    optional external LLM tool.

    Attributes:
        hass: Home Assistant instance
        config: Configuration dictionary
        tools: Dictionary mapping tool names to tool instances
        max_calls_per_turn: Maximum number of tool calls allowed per conversation turn
        timeout: Timeout in seconds for each tool execution
        emit_events: Whether to fire Home Assistant events for tool execution
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the tool handler.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary containing tool settings

        Example:
            >>> handler = ToolHandler(hass, {
            ...     "tools_max_calls_per_turn": 5,
            ...     "tools_timeout": 30,
            ...     "emit_events": True
            ... })
        """
        self.hass = hass
        self.config = config
        self.tools: dict[str, Any] = {}
        self.max_calls_per_turn = config.get(
            CONF_TOOLS_MAX_CALLS_PER_TURN, DEFAULT_TOOLS_MAX_CALLS_PER_TURN
        )
        self.timeout = config.get(CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT)
        self.emit_events = config.get(CONF_EMIT_EVENTS, True)

        # Track metrics for monitoring
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_duration_ms = 0.0

        _LOGGER.debug(
            "ToolHandler initialized with max_calls_per_turn=%d, timeout=%ds",
            self.max_calls_per_turn,
            self.timeout,
        )

    def register_tool(self, tool: Any) -> None:
        """Register a tool for use by the LLM.

        Adds a tool to the registry so it can be called during conversations.
        Each tool must have a unique name and implement the required interface.

        Args:
            tool: Tool instance to register. Must have 'name' attribute and
                'execute' method.

        Raises:
            ValidationError: If tool is invalid or name is already registered

        Example:
            >>> tool = HomeAssistantControlTool(hass)
            >>> handler.register_tool(tool)
        """
        # Validate tool has required attributes
        if not hasattr(tool, "name"):
            raise ValidationError("Tool must have a 'name' attribute")

        if not hasattr(tool, "execute"):
            raise ValidationError(
                f"Tool '{tool.name}' must have an 'execute' method"
            )

        if not hasattr(tool, "get_definition"):
            raise ValidationError(
                f"Tool '{tool.name}' must have a 'get_definition' method"
            )

        # Check for duplicate registration
        if tool.name in self.tools:
            _LOGGER.warning(
                "Tool '%s' is already registered. Overwriting previous registration.",
                tool.name,
            )

        self.tools[tool.name] = tool
        _LOGGER.info("Registered tool: %s", tool.name)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return list of tool definitions in OpenAI format for LLM.

        Collects all registered tool definitions and formats them for inclusion
        in the LLM prompt. The format follows OpenAI's function calling schema.

        Returns:
            List of tool definition dictionaries suitable for LLM consumption

        Example:
            >>> definitions = handler.get_tool_definitions()
            >>> definitions[0]
            {
                "name": "ha_control",
                "description": "Control Home Assistant devices...",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        """
        definitions = []

        for tool_name, tool in self.tools.items():
            try:
                definition = tool.to_openai_format()
                definitions.append(definition)
                _LOGGER.debug("Added definition for tool: %s", tool_name)
            except Exception as error:
                _LOGGER.error(
                    "Failed to get definition for tool '%s': %s",
                    tool_name,
                    error,
                    exc_info=True,
                )

        _LOGGER.debug("Returning %d tool definitions", len(definitions))
        return definitions

    def validate_tool_call(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> None:
        """Validate a tool call before execution.

        Checks that the tool exists, has required parameters, and that
        execution limits haven't been exceeded.

        Args:
            tool_name: Name of the tool to validate
            parameters: Parameters for the tool call

        Raises:
            ValidationError: If tool call is invalid
            ToolExecutionError: If tool doesn't exist

        Example:
            >>> handler.validate_tool_call("ha_control", {
            ...     "action": "turn_on",
            ...     "entity_id": "light.living_room"
            ... })
        """
        # Check if tool exists
        if tool_name not in self.tools:
            available_tools = ", ".join(self.tools.keys())
            raise ToolExecutionError(
                f"Tool '{tool_name}' not found. Available tools: {available_tools}"
            )

        # Validate parameters is a dictionary
        if not isinstance(parameters, dict):
            raise ValidationError(
                f"Tool parameters must be a dictionary, got {type(parameters).__name__}"
            )

        # Let the tool validate its own parameters if it has a validate method
        tool = self.tools[tool_name]
        if hasattr(tool, "validate_parameters"):
            try:
                tool.validate_parameters(parameters)
            except Exception as error:
                raise ValidationError(
                    f"Invalid parameters for tool '{tool_name}': {error}"
                ) from error

        _LOGGER.debug(
            "Validated tool call: %s with parameters: %s",
            tool_name,
            truncate_text(str(parameters), 200),
        )

    async def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool call from the LLM.

        Executes the specified tool with the given parameters, handling
        timeouts, errors, and event emission. Tracks execution metrics.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            conversation_id: Optional conversation ID for event tracking

        Returns:
            Dictionary containing execution result with keys:
                - success: Boolean indicating if execution succeeded
                - result: Tool execution result (if successful)
                - error: Error message (if failed)
                - duration_ms: Execution duration in milliseconds

        Raises:
            ToolExecutionError: If tool execution fails critically

        Example:
            >>> result = await handler.execute_tool(
            ...     "ha_query",
            ...     {"entity_id": "sensor.temperature"},
            ...     "conv_123"
            ... )
            >>> result
            {
                "success": True,
                "result": {"entity_id": "sensor.temperature", "state": "23.5"},
                "duration_ms": 45.2
            }
        """
        start_time = time.time()
        success = False
        result = None
        error_message = None

        try:
            # Validate the tool call
            self.validate_tool_call(tool_name, parameters)

            # Get the tool instance
            tool = self.tools[tool_name]

            # Execute with timeout
            _LOGGER.debug(
                "Executing tool '%s' with timeout %ds", tool_name, self.timeout
            )

            try:
                result = await asyncio.wait_for(
                    tool.execute(**parameters), timeout=self.timeout
                )
                success = True
                self._success_count += 1
                _LOGGER.info(
                    "Tool '%s' executed successfully in %.2fms",
                    tool_name,
                    (time.time() - start_time) * 1000,
                )

            except asyncio.TimeoutError as error:
                error_message = f"Tool execution timed out after {self.timeout}s"
                self._failure_count += 1
                _LOGGER.warning(
                    "Tool '%s' execution timed out after %ds",
                    tool_name,
                    self.timeout,
                )
                raise ToolExecutionError(error_message) from error

            except Exception as error:
                error_message = str(error)
                self._failure_count += 1
                _LOGGER.error(
                    "Tool '%s' execution failed: %s",
                    tool_name,
                    error,
                    exc_info=True,
                )
                raise ToolExecutionError(
                    f"Tool '{tool_name}' execution failed: {error}"
                ) from error

        finally:
            # Track execution metrics
            duration_ms = (time.time() - start_time) * 1000
            self._execution_count += 1
            self._total_duration_ms += duration_ms

            # Prepare execution result
            execution_result = {
                "success": success,
                "duration_ms": round(duration_ms, 2),
            }

            if success:
                execution_result["result"] = result
            else:
                execution_result["error"] = error_message

            # Fire event if enabled
            if self.emit_events:
                await self._fire_tool_executed_event(
                    tool_name=tool_name,
                    parameters=parameters,
                    result=execution_result,
                    conversation_id=conversation_id,
                )

        return execution_result

    async def execute_multiple_tools(
        self,
        tool_calls: list[dict[str, Any]],
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute multiple tool calls with limits enforcement.

        Executes a list of tool calls from the LLM, enforcing the max_calls_per_turn
        limit. Tool calls can be executed in parallel for better performance.

        Args:
            tool_calls: List of tool call dictionaries with 'name' and 'parameters'
            conversation_id: Optional conversation ID for tracking

        Returns:
            List of execution results, one per tool call

        Raises:
            ValidationError: If too many tool calls requested

        Example:
            >>> tool_calls = [
            ...     {"name": "ha_query", "parameters": {"entity_id": "light.living_room"}},
            ...     {"name": "ha_control", "parameters": {"action": "turn_on", ...}}
            ... ]
            >>> results = await handler.execute_multiple_tools(tool_calls)
        """
        if len(tool_calls) > self.max_calls_per_turn:
            raise ValidationError(
                f"Too many tool calls requested ({len(tool_calls)}). "
                f"Maximum allowed: {self.max_calls_per_turn}"
            )

        _LOGGER.debug(
            "Executing %d tool calls for conversation %s",
            len(tool_calls),
            conversation_id,
        )

        # Execute all tool calls in parallel
        tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            parameters = tool_call.get("parameters", {})

            if not tool_name:
                _LOGGER.warning("Tool call missing 'name' field, skipping")
                continue

            task = self.execute_tool(tool_name, parameters, conversation_id)
            tasks.append(task)

        # Wait for all executions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to error results
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(
                    {
                        "success": False,
                        "error": str(result),
                        "duration_ms": 0,
                    }
                )
                _LOGGER.error(
                    "Tool call %d failed with exception: %s", i, result, exc_info=result
                )
            else:
                formatted_results.append(result)

        return formatted_results

    async def _fire_tool_executed_event(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: dict[str, Any],
        conversation_id: str | None = None,
    ) -> None:
        """Fire a tool execution event.

        Args:
            tool_name: Name of the executed tool
            parameters: Tool parameters
            result: Execution result
            conversation_id: Optional conversation ID
        """
        # Truncate large results for event data
        truncated_result = result.copy()
        if "result" in truncated_result:
            result_str = str(truncated_result["result"])
            if len(result_str) > 1000:
                truncated_result["result"] = truncate_text(result_str, 1000)

        event_data = {
            "tool_name": tool_name,
            "parameters": parameters,
            "success": result["success"],
            "duration_ms": result["duration_ms"],
        }

        if conversation_id:
            event_data["conversation_id"] = conversation_id

        if result["success"]:
            event_data["result"] = truncated_result.get("result")
        else:
            event_data["error"] = result.get("error")

        self.hass.bus.async_fire(EVENT_TOOL_EXECUTED, event_data)
        _LOGGER.debug("Fired event: %s", EVENT_TOOL_EXECUTED)

    def get_metrics(self) -> dict[str, Any]:
        """Get tool execution metrics.

        Returns:
            Dictionary containing execution statistics:
                - total_executions: Total number of tool executions
                - successful_executions: Number of successful executions
                - failed_executions: Number of failed executions
                - success_rate: Percentage of successful executions
                - average_duration_ms: Average execution duration

        Example:
            >>> metrics = handler.get_metrics()
            >>> metrics
            {
                "total_executions": 150,
                "successful_executions": 145,
                "failed_executions": 5,
                "success_rate": 96.67,
                "average_duration_ms": 234.5
            }
        """
        success_rate = 0.0
        if self._execution_count > 0:
            success_rate = (self._success_count / self._execution_count) * 100

        avg_duration = 0.0
        if self._execution_count > 0:
            avg_duration = self._total_duration_ms / self._execution_count

        return {
            "total_executions": self._execution_count,
            "successful_executions": self._success_count,
            "failed_executions": self._failure_count,
            "success_rate": round(success_rate, 2),
            "average_duration_ms": round(avg_duration, 2),
        }

    def reset_metrics(self) -> None:
        """Reset execution metrics to zero.

        Useful for testing or when you want to start fresh metric tracking.
        """
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_duration_ms = 0.0
        _LOGGER.debug("Tool execution metrics reset")

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            List of tool names currently registered

        Example:
            >>> handler.get_registered_tools()
            ['ha_control', 'ha_query', 'query_external_llm']
        """
        return list(self.tools.keys())

    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was unregistered, False if tool wasn't registered

        Example:
            >>> handler.unregister_tool("custom_tool")
            True
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            _LOGGER.info("Unregistered tool: %s", tool_name)
            return True

        _LOGGER.warning("Attempted to unregister unknown tool: %s", tool_name)
        return False

    def clear_tools(self) -> None:
        """Clear all registered tools.

        Removes all tools from the registry. Useful for reconfiguration
        or cleanup scenarios.
        """
        tool_count = len(self.tools)
        self.tools.clear()
        _LOGGER.info("Cleared %d registered tools", tool_count)

"""Tool registry for managing and executing Home Agent tools.

This module provides the ToolRegistry class that manages tool registration,
formatting for LLM consumption, and parameter validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant

from ..exceptions import ToolExecutionError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseTool(ABC):
    """Base class for all Home Agent tools.

    All tools must inherit from this class and implement the execute method.
    Tools define their name, description, and parameter schema using OpenAI
    function calling format.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the tool.

        Args:
            hass: Home Assistant instance
        """
        self.hass = hass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name.

        The name should be concise and descriptive, using snake_case format.
        This name will be used by the LLM to call the tool.

        Returns:
            Tool name (e.g., "ha_control", "ha_query")
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description.

        The description should clearly explain what the tool does and when
        the LLM should use it. This is critical for proper tool selection.

        Returns:
            Detailed description of the tool's purpose and use cases
        """

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema in OpenAI function format.

        The schema defines the parameters the tool accepts, their types,
        descriptions, and which are required. Uses JSON Schema format.

        Returns:
            JSON Schema dict describing the tool's parameters

        Example:
            {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "The entity ID to control"
                    }
                },
                "required": ["entity_id"]
            }
        """

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the given parameters.

        This method performs the actual tool operation and returns the result
        in a structured format that the LLM can understand and use.

        Args:
            **kwargs: Tool parameters as defined in the schema

        Returns:
            Dict containing the execution result with at minimum:
                - success: bool indicating if execution succeeded
                - Additional keys depend on the specific tool

        Raises:
            ToolExecutionError: If execution fails
            ValidationError: If parameters are invalid
        """

    def get_definition(self) -> dict[str, Any]:
        """Get the tool definition in a format suitable for the LLM.

        This returns a simplified format with name, description, and parameters
        that the ToolHandler expects.

        Returns:
            Dict containing tool definition:
            {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {...}
            }
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert tool definition to OpenAI function format.

        This formats the tool for consumption by OpenAI-compatible LLM APIs
        that support function/tool calling.

        Returns:
            Dict in OpenAI function format:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "Tool description",
                    "parameters": {...}
                }
            }
        """
        return {
            "type": "function",
            "function": self.get_definition(),
        }


class ToolRegistry:
    """Registry for managing Home Agent tools.

    The ToolRegistry maintains a collection of available tools, formats them
    for LLM consumption, validates parameters, and orchestrates tool execution.

    Example:
        registry = ToolRegistry(hass)
        registry.register(HomeAssistantControlTool(hass))
        registry.register(HomeAssistantQueryTool(hass))

        # Get tools for LLM
        tools = registry.get_tools_for_llm()

        # Execute a tool call
        result = await registry.execute_tool("ha_control", {
            "action": "turn_on",
            "entity_id": "light.living_room"
        })
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the tool registry.

        Args:
            hass: Home Assistant instance
        """
        self.hass = hass
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValidationError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValidationError(
                f"Tool '{tool.name}' is already registered. "
                f"Each tool must have a unique name."
            )

        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister

        Raises:
            ValidationError: If the tool is not found
        """
        if tool_name not in self._tools:
            raise ValidationError(
                f"Tool '{tool_name}' is not registered and cannot be unregistered."
            )

        del self._tools[tool_name]

    def get_tool(self, tool_name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(tool_name)

    def get_all_tools(self) -> dict[str, BaseTool]:
        """Get all registered tools.

        Returns:
            Dict mapping tool names to tool instances
        """
        return self._tools.copy()

    def get_tools_for_llm(
        self,
        filter_fn: Callable[[BaseTool], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Format registered tools for LLM consumption.

        Converts all registered tools (or a filtered subset) into OpenAI
        function calling format that can be included in LLM API calls.

        Args:
            filter_fn: Optional function to filter tools. Should return True
                for tools to include. Example: lambda t: t.name != "excluded_tool"

        Returns:
            List of tool definitions in OpenAI format

        Example:
            # Get all tools
            tools = registry.get_tools_for_llm()

            # Get only control tools
            tools = registry.get_tools_for_llm(
                filter_fn=lambda t: "control" in t.name
            )
        """
        tools = self._tools.values()

        if filter_fn:
            tools = [tool for tool in tools if filter_fn(tool)]
        else:
            tools = list(tools)

        return [tool.to_openai_format() for tool in tools]

    async def execute_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool with the given parameters.

        This is the main entry point for executing tools. It validates the
        tool exists, passes parameters to the tool, and returns the result.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Dict containing the execution result

        Raises:
            ValidationError: If the tool is not found
            ToolExecutionError: If tool execution fails

        Example:
            result = await registry.execute_tool("ha_control", {
                "action": "turn_on",
                "entity_id": "light.living_room",
                "parameters": {"brightness_pct": 50}
            })

            if result["success"]:
                print(f"Entity is now {result['new_state']}")
        """
        tool = self.get_tool(tool_name)

        if tool is None:
            raise ValidationError(
                f"Tool '{tool_name}' not found. "
                f"Available tools: {', '.join(self._tools.keys())}"
            )

        try:
            return await tool.execute(**parameters)
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except ToolExecutionError:
            # Re-raise tool execution errors as-is
            raise
        except Exception as error:
            # Wrap unexpected exceptions
            raise ToolExecutionError(
                f"Unexpected error executing tool '{tool_name}': {error}"
            ) from error

    def validate_parameters(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> bool:
        """Validate parameters against a tool's schema.

        This performs basic validation that required parameters are present.
        More detailed validation is done by the tool's execute method.

        Args:
            tool_name: Name of the tool
            parameters: Parameters to validate

        Returns:
            True if parameters are valid

        Raises:
            ValidationError: If validation fails
        """
        tool = self.get_tool(tool_name)

        if tool is None:
            raise ValidationError(f"Tool '{tool_name}' not found")

        schema = tool.parameters
        required_params = schema.get("required", [])

        # Check for missing required parameters
        missing_params = [
            param for param in required_params if param not in parameters
        ]

        if missing_params:
            raise ValidationError(
                f"Missing required parameters for tool '{tool_name}': "
                f"{', '.join(missing_params)}"
            )

        return True

    def list_tool_names(self) -> list[str]:
        """Get a list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def count(self) -> int:
        """Get the number of registered tools.

        Returns:
            Number of tools in the registry
        """
        return len(self._tools)

    def clear(self) -> None:
        """Clear all registered tools from the registry.

        Warning: This removes all tools. Typically only used for testing
        or when completely rebuilding the tool registry.
        """
        self._tools.clear()

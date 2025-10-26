"""Unit tests for the ToolRegistry and BaseTool."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from custom_components.home_agent.tools.registry import BaseTool, ToolRegistry
from custom_components.home_agent.exceptions import ValidationError, ToolExecutionError


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, hass, tool_name="mock_tool"):
        """Initialize the mock tool."""
        super().__init__(hass)
        self._name = tool_name

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter"
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter"
                }
            },
            "required": ["param1"]
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool."""
        return {
            "success": True,
            "message": "Mock execution successful",
            "params": kwargs
        }


class FailingMockTool(BaseTool):
    """Mock tool that always fails for testing error handling."""

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "failing_tool"

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "A tool that fails"

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool (always fails)."""
        raise ToolExecutionError("Mock execution failed")


class TestBaseTool:
    """Test the BaseTool abstract base class."""

    def test_base_tool_initialization(self, mock_hass):
        """Test that BaseTool can be initialized with a tool instance."""
        tool = MockTool(mock_hass)
        assert tool.hass == mock_hass

    def test_base_tool_properties(self, mock_hass):
        """Test that BaseTool properties are accessible."""
        tool = MockTool(mock_hass)
        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing"
        assert isinstance(tool.parameters, dict)
        assert "type" in tool.parameters
        assert tool.parameters["type"] == "object"

    def test_to_openai_format(self, mock_hass):
        """Test conversion to OpenAI function format."""
        tool = MockTool(mock_hass)
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert "function" in openai_format
        assert openai_format["function"]["name"] == "mock_tool"
        assert openai_format["function"]["description"] == "A mock tool for testing"
        assert openai_format["function"]["parameters"] == tool.parameters

    def test_to_openai_format_structure(self, mock_hass):
        """Test that OpenAI format has correct structure."""
        tool = MockTool(mock_hass)
        openai_format = tool.to_openai_format()

        # Verify required keys
        assert "type" in openai_format
        assert "function" in openai_format

        # Verify function structure
        function = openai_format["function"]
        assert "name" in function
        assert "description" in function
        assert "parameters" in function

        # Verify parameters structure
        params = function["parameters"]
        assert "type" in params
        assert "properties" in params
        assert "required" in params

    @pytest.mark.asyncio
    async def test_execute_method(self, mock_hass):
        """Test that execute method works."""
        tool = MockTool(mock_hass)
        result = await tool.execute(param1="test", param2=42)

        assert result["success"] is True
        assert result["params"]["param1"] == "test"
        assert result["params"]["param2"] == 42


class TestToolRegistry:
    """Test the ToolRegistry class."""

    def test_registry_initialization(self, mock_hass):
        """Test that registry can be initialized."""
        registry = ToolRegistry(mock_hass)
        assert registry.hass == mock_hass
        assert registry.count() == 0
        assert registry.list_tool_names() == []

    def test_register_tool(self, mock_hass):
        """Test registering a tool."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)

        registry.register(tool)

        assert registry.count() == 1
        assert "mock_tool" in registry.list_tool_names()
        assert registry.get_tool("mock_tool") == tool

    def test_register_multiple_tools(self, mock_hass):
        """Test registering multiple tools."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "tool1")
        tool2 = MockTool(mock_hass, "tool2")
        tool3 = MockTool(mock_hass, "tool3")

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        assert registry.count() == 3
        assert set(registry.list_tool_names()) == {"tool1", "tool2", "tool3"}

    def test_register_duplicate_tool_raises_error(self, mock_hass):
        """Test that registering duplicate tool names raises ValidationError."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "duplicate")
        tool2 = MockTool(mock_hass, "duplicate")

        registry.register(tool1)

        with pytest.raises(ValidationError) as exc_info:
            registry.register(tool2)

        assert "already registered" in str(exc_info.value)
        assert "duplicate" in str(exc_info.value)

    def test_unregister_tool(self, mock_hass):
        """Test unregistering a tool."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)

        registry.register(tool)
        assert registry.count() == 1

        registry.unregister("mock_tool")
        assert registry.count() == 0
        assert registry.get_tool("mock_tool") is None

    def test_unregister_nonexistent_tool_raises_error(self, mock_hass):
        """Test that unregistering a nonexistent tool raises ValidationError."""
        registry = ToolRegistry(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            registry.unregister("nonexistent")

        assert "not registered" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_get_tool_returns_correct_tool(self, mock_hass):
        """Test that get_tool returns the correct tool."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "tool1")
        tool2 = MockTool(mock_hass, "tool2")

        registry.register(tool1)
        registry.register(tool2)

        assert registry.get_tool("tool1") == tool1
        assert registry.get_tool("tool2") == tool2

    def test_get_tool_returns_none_for_nonexistent(self, mock_hass):
        """Test that get_tool returns None for nonexistent tools."""
        registry = ToolRegistry(mock_hass)
        assert registry.get_tool("nonexistent") is None

    def test_get_all_tools(self, mock_hass):
        """Test getting all registered tools."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "tool1")
        tool2 = MockTool(mock_hass, "tool2")

        registry.register(tool1)
        registry.register(tool2)

        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2
        assert all_tools["tool1"] == tool1
        assert all_tools["tool2"] == tool2

    def test_get_all_tools_returns_copy(self, mock_hass):
        """Test that get_all_tools returns a copy, not the internal dict."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)
        registry.register(tool)

        all_tools = registry.get_all_tools()
        all_tools["new_tool"] = MockTool(mock_hass, "new")

        # Internal registry should not be modified
        assert registry.count() == 1
        assert "new_tool" not in registry.list_tool_names()

    def test_get_tools_for_llm_no_filter(self, mock_hass):
        """Test getting tools for LLM without filter."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "tool1")
        tool2 = MockTool(mock_hass, "tool2")

        registry.register(tool1)
        registry.register(tool2)

        llm_tools = registry.get_tools_for_llm()

        assert len(llm_tools) == 2
        assert all(tool["type"] == "function" for tool in llm_tools)
        assert {tool["function"]["name"] for tool in llm_tools} == {"tool1", "tool2"}

    def test_get_tools_for_llm_with_filter(self, mock_hass):
        """Test getting tools for LLM with filter function."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "include_this")
        tool2 = MockTool(mock_hass, "exclude_this")
        tool3 = MockTool(mock_hass, "include_also")

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        # Filter to only include tools with "include" in name
        llm_tools = registry.get_tools_for_llm(
            filter_fn=lambda t: "include" in t.name
        )

        assert len(llm_tools) == 2
        names = {tool["function"]["name"] for tool in llm_tools}
        assert names == {"include_this", "include_also"}

    def test_get_tools_for_llm_empty_registry(self, mock_hass):
        """Test getting tools for LLM from empty registry."""
        registry = ToolRegistry(mock_hass)
        llm_tools = registry.get_tools_for_llm()
        assert llm_tools == []

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_hass):
        """Test executing a tool successfully."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)
        registry.register(tool)

        result = await registry.execute_tool("mock_tool", {
            "param1": "test_value",
            "param2": 123
        })

        assert result["success"] is True
        assert result["params"]["param1"] == "test_value"
        assert result["params"]["param2"] == 123

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mock_hass):
        """Test executing a nonexistent tool raises ValidationError."""
        registry = ToolRegistry(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            await registry.execute_tool("nonexistent", {})

        assert "not found" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
        assert "Available tools" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_tool_execution_error(self, mock_hass):
        """Test that tool execution errors are propagated."""
        registry = ToolRegistry(mock_hass)
        tool = FailingMockTool(mock_hass)
        registry.register(tool)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute_tool("failing_tool", {})

        assert "Mock execution failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_tool_unexpected_error(self, mock_hass):
        """Test that unexpected errors are wrapped in ToolExecutionError."""
        registry = ToolRegistry(mock_hass)

        # Create a tool that raises an unexpected error
        class UnexpectedErrorTool(MockTool):
            async def execute(self, **kwargs):
                raise RuntimeError("Unexpected error")

        tool = UnexpectedErrorTool(mock_hass, "error_tool")
        registry.register(tool)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute_tool("error_tool", {})

        assert "Unexpected error executing tool" in str(exc_info.value)
        assert "error_tool" in str(exc_info.value)

    def test_validate_parameters_success(self, mock_hass):
        """Test validating parameters successfully."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)
        registry.register(tool)

        # Should not raise an error
        result = registry.validate_parameters("mock_tool", {
            "param1": "required_value",
            "param2": 42
        })

        assert result is True

    def test_validate_parameters_missing_required(self, mock_hass):
        """Test validating parameters with missing required parameter."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)
        registry.register(tool)

        with pytest.raises(ValidationError) as exc_info:
            registry.validate_parameters("mock_tool", {
                "param2": 42  # Missing required param1
            })

        assert "Missing required parameters" in str(exc_info.value)
        assert "param1" in str(exc_info.value)

    def test_validate_parameters_tool_not_found(self, mock_hass):
        """Test validating parameters for nonexistent tool."""
        registry = ToolRegistry(mock_hass)

        with pytest.raises(ValidationError) as exc_info:
            registry.validate_parameters("nonexistent", {})

        assert "not found" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_validate_parameters_optional_params_ok(self, mock_hass):
        """Test validating parameters with only required params provided."""
        registry = ToolRegistry(mock_hass)
        tool = MockTool(mock_hass)
        registry.register(tool)

        # Should pass with only required parameter
        result = registry.validate_parameters("mock_tool", {
            "param1": "required_value"
        })

        assert result is True

    def test_list_tool_names(self, mock_hass):
        """Test listing tool names."""
        registry = ToolRegistry(mock_hass)
        tool1 = MockTool(mock_hass, "tool1")
        tool2 = MockTool(mock_hass, "tool2")
        tool3 = MockTool(mock_hass, "tool3")

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        names = registry.list_tool_names()
        assert set(names) == {"tool1", "tool2", "tool3"}

    def test_count(self, mock_hass):
        """Test counting registered tools."""
        registry = ToolRegistry(mock_hass)
        assert registry.count() == 0

        registry.register(MockTool(mock_hass, "tool1"))
        assert registry.count() == 1

        registry.register(MockTool(mock_hass, "tool2"))
        assert registry.count() == 2

        registry.unregister("tool1")
        assert registry.count() == 1

    def test_clear(self, mock_hass):
        """Test clearing all tools from registry."""
        registry = ToolRegistry(mock_hass)
        registry.register(MockTool(mock_hass, "tool1"))
        registry.register(MockTool(mock_hass, "tool2"))
        registry.register(MockTool(mock_hass, "tool3"))

        assert registry.count() == 3

        registry.clear()

        assert registry.count() == 0
        assert registry.list_tool_names() == []
        assert registry.get_all_tools() == {}

    def test_clear_empty_registry(self, mock_hass):
        """Test clearing an already empty registry."""
        registry = ToolRegistry(mock_hass)
        registry.clear()
        assert registry.count() == 0

    @pytest.mark.asyncio
    async def test_validation_error_not_wrapped(self, mock_hass):
        """Test that ValidationError from tool execution is not wrapped."""
        registry = ToolRegistry(mock_hass)

        class ValidationErrorTool(MockTool):
            async def execute(self, **kwargs):
                raise ValidationError("Validation failed")

        tool = ValidationErrorTool(mock_hass, "validation_tool")
        registry.register(tool)

        with pytest.raises(ValidationError) as exc_info:
            await registry.execute_tool("validation_tool", {})

        # Should be the original error, not wrapped
        assert str(exc_info.value) == "Validation failed"

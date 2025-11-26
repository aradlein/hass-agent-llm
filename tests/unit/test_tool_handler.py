"""Unit tests for ToolHandler class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_agent.const import (
    CONF_EMIT_EVENTS,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    EVENT_TOOL_EXECUTED,
    EVENT_TOOL_PROGRESS,
)
from custom_components.home_agent.exceptions import ToolExecutionError, ValidationError
from custom_components.home_agent.tool_handler import ToolHandler


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        """Initialize mock tool."""
        self.name = name
        self.should_fail = should_fail
        self.execute_called = False
        self.execute_params = None

    async def execute(self, **kwargs):
        """Mock execute method."""
        self.execute_called = True
        self.execute_params = kwargs
        if self.should_fail:
            raise Exception(f"Tool {self.name} failed")
        return {"result": f"Success from {self.name}"}

    def get_definition(self):
        """Mock get_definition method."""
        return {
            "name": self.name,
            "description": f"Test tool {self.name}",
            "parameters": {
                "type": "object",
                "properties": {"test_param": {"type": "string"}},
                "required": ["test_param"],
            },
        }

    def to_openai_format(self):
        """Mock to_openai_format method."""
        return {
            "type": "function",
            "function": self.get_definition(),
        }

    def validate_parameters(self, parameters: dict):
        """Mock validate_parameters method."""
        if "test_param" not in parameters:
            raise ValueError("Missing test_param")


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = MagicMock()
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    return hass


@pytest.fixture
def default_config():
    """Create default configuration."""
    return {
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
        CONF_TOOLS_TIMEOUT: 30,
        CONF_EMIT_EVENTS: True,
    }


@pytest.fixture
def tool_handler(mock_hass, default_config):
    """Create ToolHandler instance."""
    return ToolHandler(mock_hass, default_config)


class TestToolHandlerInitialization:
    """Test ToolHandler initialization."""

    def test_init_with_config(self, mock_hass, default_config):
        """Test initialization with config."""
        handler = ToolHandler(mock_hass, default_config)

        assert handler.hass == mock_hass
        assert handler.config == default_config
        assert handler.max_calls_per_turn == 5
        assert handler.timeout == 30
        assert handler.emit_events is True
        assert handler.tools == {}
        assert handler._execution_count == 0
        assert handler._success_count == 0
        assert handler._failure_count == 0

    def test_init_with_defaults(self, mock_hass):
        """Test initialization with default values."""
        handler = ToolHandler(mock_hass, {})

        assert handler.max_calls_per_turn == 5  # DEFAULT_TOOLS_MAX_CALLS_PER_TURN
        assert handler.timeout == 30  # DEFAULT_TOOLS_TIMEOUT
        assert handler.emit_events is True  # Default value

    def test_init_with_custom_values(self, mock_hass):
        """Test initialization with custom values."""
        config = {
            CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
            CONF_TOOLS_TIMEOUT: 60,
            CONF_EMIT_EVENTS: False,
        }
        handler = ToolHandler(mock_hass, config)

        assert handler.max_calls_per_turn == 10
        assert handler.timeout == 60
        assert handler.emit_events is False


class TestToolRegistration:
    """Test tool registration and unregistration."""

    def test_register_tool_success(self, tool_handler):
        """Test successful tool registration."""
        tool = MockTool("test_tool")

        tool_handler.register_tool(tool)

        assert "test_tool" in tool_handler.tools
        assert tool_handler.tools["test_tool"] == tool

    def test_register_multiple_tools(self, tool_handler):
        """Test registering multiple tools."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")

        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        assert len(tool_handler.tools) == 2
        assert "tool1" in tool_handler.tools
        assert "tool2" in tool_handler.tools

    def test_register_tool_without_name(self, tool_handler):
        """Test registering tool without name attribute."""
        tool = MagicMock()
        del tool.name

        with pytest.raises(ValidationError, match="Tool must have a 'name' attribute"):
            tool_handler.register_tool(tool)

    def test_register_tool_without_execute(self, tool_handler):
        """Test registering tool without execute method."""
        tool = MagicMock()
        tool.name = "test_tool"
        del tool.execute

        with pytest.raises(ValidationError, match="must have an 'execute' method"):
            tool_handler.register_tool(tool)

    def test_register_tool_without_get_definition(self, tool_handler):
        """Test registering tool without get_definition method."""
        tool = MagicMock()
        tool.name = "test_tool"
        tool.execute = AsyncMock()
        del tool.get_definition

        with pytest.raises(ValidationError, match="must have a 'get_definition' method"):
            tool_handler.register_tool(tool)

    def test_register_duplicate_tool(self, tool_handler):
        """Test registering tool with duplicate name (should overwrite)."""
        tool1 = MockTool("test_tool")
        tool2 = MockTool("test_tool")

        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        assert tool_handler.tools["test_tool"] == tool2

    def test_unregister_tool_success(self, tool_handler):
        """Test successful tool unregistration."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        result = tool_handler.unregister_tool("test_tool")

        assert result is True
        assert "test_tool" not in tool_handler.tools

    def test_unregister_nonexistent_tool(self, tool_handler):
        """Test unregistering non-existent tool."""
        result = tool_handler.unregister_tool("nonexistent")

        assert result is False

    def test_clear_tools(self, tool_handler):
        """Test clearing all tools."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        tool_handler.clear_tools()

        assert len(tool_handler.tools) == 0

    def test_get_registered_tools(self, tool_handler):
        """Test getting list of registered tool names."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        tools = tool_handler.get_registered_tools()

        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools


class TestGetToolDefinitions:
    """Test get_tool_definitions method."""

    def test_get_tool_definitions_success(self, tool_handler):
        """Test getting tool definitions."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        definitions = tool_handler.get_tool_definitions()

        assert len(definitions) == 2
        assert all(isinstance(d, dict) for d in definitions)
        assert definitions[0]["type"] == "function"
        assert definitions[0]["function"]["name"] == "tool1"
        assert definitions[1]["type"] == "function"
        assert definitions[1]["function"]["name"] == "tool2"

    def test_get_tool_definitions_empty(self, tool_handler):
        """Test getting tool definitions when no tools registered."""
        definitions = tool_handler.get_tool_definitions()

        assert len(definitions) == 0
        assert isinstance(definitions, list)

    def test_get_tool_definitions_with_error(self, tool_handler):
        """Test getting tool definitions when a tool fails."""
        tool1 = MockTool("tool1")
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.to_openai_format = MagicMock(side_effect=Exception("Definition error"))

        tool_handler.register_tool(tool1)
        tool_handler.tools["tool2"] = tool2

        definitions = tool_handler.get_tool_definitions()

        # Should still return definition for tool1, skipping tool2
        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "tool1"


class TestValidateToolCall:
    """Test validate_tool_call method."""

    def test_validate_tool_call_success(self, tool_handler):
        """Test successful tool call validation."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Should not raise exception and return None on success
        result = tool_handler.validate_tool_call("test_tool", {"test_param": "value"})
        assert result is None

        # Verify the tool was found and parameters were validated
        assert "test_tool" in tool_handler.tools

    def test_validate_tool_call_nonexistent_tool(self, tool_handler):
        """Test validating call to non-existent tool."""
        with pytest.raises(ToolExecutionError, match="Tool 'nonexistent' not found"):
            tool_handler.validate_tool_call("nonexistent", {})

    def test_validate_tool_call_invalid_parameters_type(self, tool_handler):
        """Test validating tool call with non-dict parameters."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        with pytest.raises(ValidationError, match="must be a dictionary"):
            tool_handler.validate_tool_call("test_tool", "invalid")

    def test_validate_tool_call_with_validation_method(self, tool_handler):
        """Test validation using tool's validate_parameters method."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Missing required parameter
        with pytest.raises(ValidationError, match="Invalid parameters"):
            tool_handler.validate_tool_call("test_tool", {})

    def test_validate_tool_call_without_validation_method(self, tool_handler):
        """Test validation when tool doesn't have validate_parameters."""
        tool = MagicMock()
        tool.name = "test_tool"
        tool.execute = AsyncMock()
        tool.get_definition = MagicMock(return_value={})
        del tool.validate_parameters

        tool_handler.tools["test_tool"] = tool

        # Should not raise exception and return None when validation method missing
        result = tool_handler.validate_tool_call("test_tool", {})
        assert result is None

        # Verify tool was found
        assert "test_tool" in tool_handler.tools


@pytest.mark.asyncio
class TestExecuteTool:
    """Test execute_tool method."""

    async def test_execute_tool_success(self, tool_handler):
        """Test successful tool execution."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        result = await tool_handler.execute_tool("test_tool", {"test_param": "value"}, "conv_123")

        assert result["success"] is True
        assert "result" in result
        assert result["result"]["result"] == "Success from test_tool"
        assert "duration_ms" in result
        assert tool.execute_called
        assert tool.execute_params == {"test_param": "value"}
        assert tool_handler._execution_count == 1
        assert tool_handler._success_count == 1
        assert tool_handler._failure_count == 0

    async def test_execute_tool_failure(self, tool_handler):
        """Test tool execution failure."""
        tool = MockTool("test_tool", should_fail=True)
        tool_handler.register_tool(tool)

        with pytest.raises(ToolExecutionError, match="execution failed"):
            await tool_handler.execute_tool("test_tool", {"test_param": "value"})

        assert tool_handler._execution_count == 1
        assert tool_handler._success_count == 0
        assert tool_handler._failure_count == 1

    async def test_execute_tool_timeout(self, tool_handler):
        """Test tool execution timeout."""

        async def slow_execute(**kwargs):
            await asyncio.sleep(100)
            return {"result": "too slow"}

        tool = MockTool("test_tool")
        tool.execute = slow_execute
        tool_handler.register_tool(tool)
        tool_handler.timeout = 0.1

        with pytest.raises(ToolExecutionError, match="timed out"):
            await tool_handler.execute_tool("test_tool", {"test_param": "value"})

        assert tool_handler._failure_count == 1

    async def test_execute_tool_validation_error(self, tool_handler):
        """Test tool execution with validation error."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        with pytest.raises(ValidationError):
            await tool_handler.execute_tool("test_tool", {})  # Missing required param

    async def test_execute_tool_event_firing(self, tool_handler, mock_hass):
        """Test that tool execution fires events."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        await tool_handler.execute_tool("test_tool", {"test_param": "value"}, "conv_123")

        # Verify exact event count and types
        calls = mock_hass.bus.async_fire.call_args_list
        assert len(calls) == 3  # started, completed, executed

        # Verify EVENT_TOOL_PROGRESS events
        progress_events = [call for call in calls if call[0][0] == EVENT_TOOL_PROGRESS]
        assert len(progress_events) == 2
        assert progress_events[0][0][1]["status"] == "started"
        assert progress_events[0][0][1]["tool_name"] == "test_tool"
        assert progress_events[1][0][1]["status"] == "completed"
        assert progress_events[1][0][1]["success"] is True

        # Find the EVENT_TOOL_EXECUTED event
        executed_events = [
            call
            for call in mock_hass.bus.async_fire.call_args_list
            if call[0][0] == EVENT_TOOL_EXECUTED
        ]

        assert len(executed_events) == 1
        event_data = executed_events[0][0][1]
        assert event_data["tool_name"] == "test_tool"
        assert event_data["conversation_id"] == "conv_123"
        assert event_data["success"] is True
        assert "result" in event_data
        assert "duration_ms" in event_data

    async def test_execute_tool_no_event_firing(self, tool_handler, mock_hass):
        """Test tool execution without event firing."""
        tool_handler.emit_events = False
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        await tool_handler.execute_tool("test_tool", {"test_param": "value"})

        mock_hass.bus.async_fire.assert_not_called()

    async def test_execute_tool_large_result_truncation(self, tool_handler, mock_hass):
        """Test that large results are truncated in events."""

        async def large_result_execute(**kwargs):
            return {"result": "x" * 2000}

        tool = MockTool("test_tool")
        tool.execute = large_result_execute
        tool_handler.register_tool(tool)

        await tool_handler.execute_tool("test_tool", {"test_param": "value"})

        # Verify events were fired with appropriate data
        calls = mock_hass.bus.async_fire.call_args_list
        assert len(calls) == 3  # started, completed, executed

        # Find the EVENT_TOOL_EXECUTED event and verify result handling
        executed_events = [call for call in calls if call[0][0] == EVENT_TOOL_EXECUTED]

        assert len(executed_events) == 1
        event_data = executed_events[0][0][1]
        assert event_data["success"] is True
        assert "result" in event_data
        # Large results get converted to string and truncated
        assert isinstance(event_data["result"], str)
        assert len(event_data["result"]) <= 1024  # Verify truncation occurred
        assert "result" in event_data["result"]  # Should contain part of the original data


@pytest.mark.asyncio
class TestExecuteMultipleTools:
    """Test execute_multiple_tools method."""

    async def test_execute_multiple_tools_success(self, tool_handler):
        """Test executing multiple tools successfully."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        tool_calls = [
            {"name": "tool1", "parameters": {"test_param": "value1"}},
            {"name": "tool2", "parameters": {"test_param": "value2"}},
        ]

        results = await tool_handler.execute_multiple_tools(tool_calls, "conv_123")

        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert tool1.execute_called
        assert tool2.execute_called

    async def test_execute_multiple_tools_exceeds_limit(self, tool_handler):
        """Test executing too many tools."""
        tool_calls = [{"name": f"tool{i}", "parameters": {}} for i in range(10)]

        with pytest.raises(ValidationError, match="Too many tool calls"):
            await tool_handler.execute_multiple_tools(tool_calls)

    async def test_execute_multiple_tools_with_failures(self, tool_handler):
        """Test executing multiple tools with some failures."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2", should_fail=True)
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        tool_calls = [
            {"name": "tool1", "parameters": {"test_param": "value1"}},
            {"name": "tool2", "parameters": {"test_param": "value2"}},
        ]

        results = await tool_handler.execute_multiple_tools(tool_calls)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert "error" in results[1]

    async def test_execute_multiple_tools_missing_name(self, tool_handler):
        """Test executing tools with missing name field."""
        tool_calls = [
            {"parameters": {"test_param": "value"}},
        ]

        results = await tool_handler.execute_multiple_tools(tool_calls)

        # Should skip the call with missing name
        assert len(results) == 0

    async def test_execute_multiple_tools_parallel_execution(self, tool_handler):
        """Test that tools are executed in parallel."""
        execution_order = []

        async def tool1_execute(**kwargs):
            execution_order.append("tool1_start")
            await asyncio.sleep(0.1)
            execution_order.append("tool1_end")
            return {"result": "tool1"}

        async def tool2_execute(**kwargs):
            execution_order.append("tool2_start")
            await asyncio.sleep(0.1)
            execution_order.append("tool2_end")
            return {"result": "tool2"}

        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        tool1.execute = tool1_execute
        tool2.execute = tool2_execute
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        tool_calls = [
            {"name": "tool1", "parameters": {"test_param": "value1"}},
            {"name": "tool2", "parameters": {"test_param": "value2"}},
        ]

        await tool_handler.execute_multiple_tools(tool_calls)

        # Both should start before either finishes (parallel execution)
        assert execution_order.index("tool2_start") < execution_order.index("tool1_end")


class TestMetrics:
    """Test metrics tracking."""

    @pytest.mark.asyncio
    async def test_get_metrics_initial(self, tool_handler):
        """Test getting metrics initially."""
        metrics = tool_handler.get_metrics()

        assert metrics["total_executions"] == 0
        assert metrics["successful_executions"] == 0
        assert metrics["failed_executions"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["average_duration_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_after_executions(self, tool_handler):
        """Test getting metrics after some executions."""
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2", should_fail=True)
        tool_handler.register_tool(tool1)
        tool_handler.register_tool(tool2)

        # Execute successful tool
        await tool_handler.execute_tool("tool1", {"test_param": "value"})

        # Execute failing tool
        try:
            await tool_handler.execute_tool("tool2", {"test_param": "value"})
        except ToolExecutionError:
            pass

        metrics = tool_handler.get_metrics()

        assert metrics["total_executions"] == 2
        assert metrics["successful_executions"] == 1
        assert metrics["failed_executions"] == 1
        assert metrics["success_rate"] == 50.0
        assert metrics["average_duration_ms"] > 0

    def test_reset_metrics(self, tool_handler):
        """Test resetting metrics."""
        tool_handler._execution_count = 10
        tool_handler._success_count = 8
        tool_handler._failure_count = 2
        tool_handler._total_duration_ms = 1000.0

        tool_handler.reset_metrics()

        assert tool_handler._execution_count == 0
        assert tool_handler._success_count == 0
        assert tool_handler._failure_count == 0
        assert tool_handler._total_duration_ms == 0.0


class TestEventFiring:
    """Test event firing behavior."""

    @pytest.mark.asyncio
    async def test_fire_tool_executed_event_success(self, tool_handler, mock_hass):
        """Test firing event for successful execution."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        await tool_handler.execute_tool("test_tool", {"test_param": "value"}, "conv_123")

        # Verify exact event count
        calls = mock_hass.bus.async_fire.call_args_list
        assert len(calls) == 3  # started, completed, executed

        # Verify EVENT_TOOL_PROGRESS events
        progress_events = [call[0] for call in calls if call[0][0] == EVENT_TOOL_PROGRESS]
        assert len(progress_events) == 2
        assert progress_events[0][1]["status"] == "started"
        assert progress_events[1][1]["status"] == "completed"

        # Find the EVENT_TOOL_EXECUTED event
        executed_events = [
            call[0]
            for call in mock_hass.bus.async_fire.call_args_list
            if call[0][0] == EVENT_TOOL_EXECUTED
        ]

        assert len(executed_events) == 1
        event_name, event_data = executed_events[0]

        assert event_name == EVENT_TOOL_EXECUTED
        assert event_data["tool_name"] == "test_tool"
        assert event_data["conversation_id"] == "conv_123"
        assert event_data["success"] is True
        assert "result" in event_data
        assert event_data["result"]["result"] == "Success from test_tool"
        assert "duration_ms" in event_data
        assert isinstance(event_data["duration_ms"], (int, float))
        assert event_data["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_fire_tool_executed_event_failure(self, tool_handler, mock_hass):
        """Test firing event for failed execution."""
        tool = MockTool("test_tool", should_fail=True)
        tool_handler.register_tool(tool)

        try:
            await tool_handler.execute_tool("test_tool", {"test_param": "value"}, "conv_123")
        except ToolExecutionError:
            pass

        # Verify exact event count
        calls = mock_hass.bus.async_fire.call_args_list
        assert len(calls) == 3  # started, failed, executed

        # Verify EVENT_TOOL_PROGRESS events
        progress_events = [call[0] for call in calls if call[0][0] == EVENT_TOOL_PROGRESS]
        assert len(progress_events) == 2
        assert progress_events[0][1]["status"] == "started"
        assert progress_events[1][1]["status"] == "failed"
        assert progress_events[1][1]["success"] is False

        # Find the EVENT_TOOL_EXECUTED event
        executed_events = [
            call[0]
            for call in mock_hass.bus.async_fire.call_args_list
            if call[0][0] == EVENT_TOOL_EXECUTED
        ]

        assert len(executed_events) == 1
        event_name, event_data = executed_events[0]

        assert event_name == EVENT_TOOL_EXECUTED
        assert event_data["tool_name"] == "test_tool"
        assert event_data["conversation_id"] == "conv_123"
        assert event_data["success"] is False
        assert "error" in event_data
        assert "Tool test_tool failed" in event_data["error"]
        assert "duration_ms" in event_data

    @pytest.mark.asyncio
    async def test_fire_tool_executed_event_without_conversation_id(self, tool_handler, mock_hass):
        """Test firing event without conversation ID."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        await tool_handler.execute_tool("test_tool", {"test_param": "value"})

        event_data = mock_hass.bus.async_fire.call_args[0][1]
        assert "conversation_id" not in event_data


class TestToolProgressEvents:
    """Test tool progress event emission."""

    @pytest.mark.asyncio
    async def test_execute_tool_emits_started_event(self, tool_handler, mock_hass):
        """Test that tool execution emits started event."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Execute tool
        await tool_handler.execute_tool(
            "test_tool", {"test_param": "value"}, tool_call_id="call_123"
        )

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find started event
        started_events = [
            call
            for call in calls
            if call[0][0] == EVENT_TOOL_PROGRESS and call[0][1].get("status") == "started"
        ]

        assert len(started_events) == 1
        event_data = started_events[0][0][1]
        assert event_data["tool_name"] == "test_tool"
        assert event_data["tool_call_id"] == "call_123"
        assert event_data["status"] == "started"
        assert "timestamp" in event_data

    @pytest.mark.asyncio
    async def test_execute_tool_emits_completed_event(self, tool_handler, mock_hass):
        """Test that successful tool execution emits completed event."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Execute tool
        await tool_handler.execute_tool(
            "test_tool", {"test_param": "value"}, tool_call_id="call_123"
        )

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find completed event
        completed_events = [
            call
            for call in calls
            if call[0][0] == EVENT_TOOL_PROGRESS and call[0][1].get("status") == "completed"
        ]

        assert len(completed_events) == 1
        event_data = completed_events[0][0][1]
        assert event_data["tool_name"] == "test_tool"
        assert event_data["tool_call_id"] == "call_123"
        assert event_data["status"] == "completed"
        assert event_data["success"] is True
        assert "timestamp" in event_data

    @pytest.mark.asyncio
    async def test_execute_tool_emits_failed_event_on_error(self, tool_handler, mock_hass):
        """Test that failed tool execution emits failed event."""
        tool = MockTool("failing_tool", should_fail=True)
        tool_handler.register_tool(tool)

        # Execute tool (should raise)
        with pytest.raises(ToolExecutionError):
            await tool_handler.execute_tool(
                "failing_tool", {"test_param": "value"}, tool_call_id="call_123"
            )

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find failed event
        failed_events = [
            call
            for call in calls
            if call[0][0] == EVENT_TOOL_PROGRESS and call[0][1].get("status") == "failed"
        ]

        assert len(failed_events) == 1
        event_data = failed_events[0][0][1]
        assert event_data["tool_name"] == "failing_tool"
        assert event_data["tool_call_id"] == "call_123"
        assert event_data["status"] == "failed"
        assert event_data["success"] is False
        assert "error" in event_data
        assert "error_type" in event_data

    @pytest.mark.asyncio
    async def test_execute_tool_emits_failed_event_on_timeout(self, tool_handler, mock_hass):
        """Test that timeout emits failed event with TimeoutError."""

        async def slow_execute(**kwargs):
            await asyncio.sleep(100)
            return {"result": "too slow"}

        tool = MockTool("test_tool")
        tool.execute = slow_execute
        tool_handler.register_tool(tool)
        tool_handler.timeout = 0.1

        # Execute tool (should timeout)
        with pytest.raises(ToolExecutionError, match="timed out"):
            await tool_handler.execute_tool(
                "test_tool", {"test_param": "value"}, tool_call_id="call_123"
            )

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find failed event
        failed_events = [
            call
            for call in calls
            if call[0][0] == EVENT_TOOL_PROGRESS and call[0][1].get("status") == "failed"
        ]

        assert len(failed_events) == 1
        event_data = failed_events[0][0][1]
        assert event_data["error_type"] == "TimeoutError"
        assert "timed out" in event_data["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_event_includes_timestamp(self, tool_handler, mock_hass):
        """Test that progress events include timestamps."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Execute tool
        await tool_handler.execute_tool(
            "test_tool", {"test_param": "value"}, tool_call_id="call_123"
        )

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find all progress events
        progress_events = [call for call in calls if call[0][0] == EVENT_TOOL_PROGRESS]

        # Verify all events have timestamps
        assert len(progress_events) >= 2  # At least started and completed
        for call in progress_events:
            event_data = call[0][1]
            assert "timestamp" in event_data
            assert isinstance(event_data["timestamp"], (int, float))

    @pytest.mark.asyncio
    async def test_execute_tool_no_progress_events_when_disabled(self, tool_handler, mock_hass):
        """Test that progress events are not emitted when emit_events is False."""
        tool_handler.emit_events = False
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Execute tool
        await tool_handler.execute_tool(
            "test_tool", {"test_param": "value"}, tool_call_id="call_123"
        )

        # Verify no events were fired
        mock_hass.bus.async_fire.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_tool_progress_events_without_tool_call_id(self, tool_handler, mock_hass):
        """Test that progress events work without tool_call_id."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Execute tool without tool_call_id
        await tool_handler.execute_tool("test_tool", {"test_param": "value"})

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find progress events
        progress_events = [call for call in calls if call[0][0] == EVENT_TOOL_PROGRESS]

        # Verify events were emitted with None tool_call_id
        assert len(progress_events) >= 2
        for call in progress_events:
            event_data = call[0][1]
            assert event_data["tool_call_id"] is None

    @pytest.mark.asyncio
    async def test_execute_tool_event_order(self, tool_handler, mock_hass):
        """Test that events are emitted in correct order."""
        tool = MockTool("test_tool")
        tool_handler.register_tool(tool)

        # Execute tool
        await tool_handler.execute_tool(
            "test_tool", {"test_param": "value"}, tool_call_id="call_123"
        )

        # Get all async_fire calls
        calls = mock_hass.bus.async_fire.call_args_list

        # Find progress events in order
        progress_events = [call[0][1] for call in calls if call[0][0] == EVENT_TOOL_PROGRESS]

        # Should have started, then completed
        assert len(progress_events) >= 2
        assert progress_events[0]["status"] == "started"
        assert progress_events[1]["status"] == "completed"

        # Timestamps should be in order
        assert progress_events[0]["timestamp"] <= progress_events[1]["timestamp"]

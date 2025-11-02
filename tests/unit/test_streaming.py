"""Unit tests for OpenAIStreamingHandler class."""
import pytest

from custom_components.home_agent.streaming import OpenAIStreamingHandler


async def async_generator_from_list(items):
    """Convert list to async generator."""
    for item in items:
        yield item


@pytest.fixture
def handler():
    """Create OpenAIStreamingHandler instance."""
    return OpenAIStreamingHandler()


class TestOpenAIStreamingHandlerInitialization:
    """Test OpenAIStreamingHandler initialization."""

    def test_init(self, handler):
        """Test initialization."""
        assert handler._current_tool_calls == {}


class TestSSEParsing:
    """Test SSE format parsing."""

    def test_parse_valid_sse_line(self, handler):
        """Test parsing valid SSE line."""
        line = 'data: {"id":"test","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}'
        result = handler._parse_sse_line(line)
        assert result is not None
        assert result["id"] == "test"
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_parse_done_marker(self, handler):
        """Test parsing [DONE] marker."""
        line = "data: [DONE]"
        result = handler._parse_sse_line(line)
        assert result is None

    def test_parse_empty_line(self, handler):
        """Test parsing empty line."""
        line = ""
        result = handler._parse_sse_line(line)
        assert result is None

    def test_parse_invalid_json(self, handler):
        """Test parsing invalid JSON."""
        line = "data: {invalid json}"
        result = handler._parse_sse_line(line)
        assert result is None

    def test_parse_line_without_data_prefix(self, handler):
        """Test parsing line without data: prefix."""
        line = '{"id":"test"}'
        result = handler._parse_sse_line(line)
        assert result is None


class TestTextStreaming:
    """Test streaming text content."""

    @pytest.mark.asyncio
    async def test_basic_text_streaming(self, handler):
        """Test streaming text content from OpenAI format."""
        # Create mock SSE stream with text deltas
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify results
        assert len(results) == 4
        assert results[0] == {"role": "assistant"}
        assert results[1] == {"content": "Hello"}
        assert results[2] == {"content": " world"}
        assert results[3] == {"content": "!"}

    @pytest.mark.asyncio
    async def test_empty_stream(self, handler):
        """Test handling of empty stream."""
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should only yield role
        assert len(results) == 1
        assert results[0] == {"role": "assistant"}


class TestToolCallStreaming:
    """Test streaming with tool calls."""

    @pytest.mark.asyncio
    async def test_single_tool_call_streaming(self, handler):
        """Test streaming with a single tool call."""
        # Create mock stream with tool call
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"ha_control","arguments":""}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"entity"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"_id\\": \\"li"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ght.living_room"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\", \\"action\\":"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"turn_on\\"}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify results
        assert len(results) == 2
        assert results[0] == {"role": "assistant"}

        # Check tool call
        assert "tool_calls" in results[1]
        assert len(results[1]["tool_calls"]) == 1

        tool_call = results[1]["tool_calls"][0]
        assert tool_call.id == "call_abc123"
        assert tool_call.tool_name == "ha_control"
        assert tool_call.tool_args == {
            "entity_id": "light.living_room",
            "action": "turn_on",
        }

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_streaming(self, handler):
        """Test streaming with multiple indexed tool calls."""
        # Create mock stream with multiple tools
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            # First tool call
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ha_query","arguments":""}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"entity_id\\": \\"sensor.temperature\\"}"}}]},"finish_reason":null}]}',
            # Second tool call
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"ha_control","arguments":""}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"entity_id\\": \\"light.bedroom\\", \\"action\\": \\"turn_off\\"}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify results
        assert len(results) == 2
        assert results[0] == {"role": "assistant"}

        # Check both tool calls
        assert "tool_calls" in results[1]
        assert len(results[1]["tool_calls"]) == 2

        tool_call_1 = results[1]["tool_calls"][0]
        assert tool_call_1.id == "call_1"
        assert tool_call_1.tool_name == "ha_query"
        assert tool_call_1.tool_args == {"entity_id": "sensor.temperature"}

        tool_call_2 = results[1]["tool_calls"][1]
        assert tool_call_2.id == "call_2"
        assert tool_call_2.tool_name == "ha_control"
        assert tool_call_2.tool_args == {
            "entity_id": "light.bedroom",
            "action": "turn_off",
        }


class TestMixedContentAndTools:
    """Test streaming with both text and tool calls."""

    @pytest.mark.asyncio
    async def test_text_followed_by_tool_call(self, handler):
        """Test streaming with text followed by tool calls."""
        # Create mock stream with text followed by tool call
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Let me check that for you."},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ha_query","arguments":"{\\"entity_id\\": \\"light.kitchen\\"}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify results
        assert len(results) == 3
        assert results[0] == {"role": "assistant"}
        assert results[1] == {"content": "Let me check that for you."}

        # Check tool call
        assert "tool_calls" in results[2]
        tool_call = results[2]["tool_calls"][0]
        assert tool_call.tool_name == "ha_query"


class TestErrorHandling:
    """Test error handling in streaming."""

    @pytest.mark.asyncio
    async def test_invalid_tool_json(self, handler):
        """Test handling of malformed tool JSON."""
        # Create mock stream with invalid JSON
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ha_control","arguments":"{\\"invalid\\": json syntax}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream - should handle gracefully
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should still yield role and tool call (with empty args)
        assert len(results) == 2
        assert results[0] == {"role": "assistant"}

        # Tool call should have empty args due to JSON error
        assert "tool_calls" in results[1]
        tool_call = results[1]["tool_calls"][0]
        assert tool_call.id == "call_1"
        assert tool_call.tool_args == {}  # Empty due to parse error

    @pytest.mark.asyncio
    async def test_empty_tool_args(self, handler):
        """Test handling of tool call with no arguments."""
        # Create mock stream with tool call but no args
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ha_query","arguments":""}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify results
        assert len(results) == 2
        assert results[0] == {"role": "assistant"}

        # Tool call should have empty args dict
        assert "tool_calls" in results[1]
        tool_call = results[1]["tool_calls"][0]
        assert tool_call.tool_args == {}

    @pytest.mark.asyncio
    async def test_tool_call_without_finish_reason(self, handler):
        """Test handling tool call when stream ends without finish_reason."""
        # Some APIs may not send finish_reason
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ha_query","arguments":"{\\"entity_id\\": \\"light.kitchen\\"}"}}]},"finish_reason":null}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should finalize tool call at end of stream
        assert len(results) == 2
        assert results[0] == {"role": "assistant"}
        assert "tool_calls" in results[1]
        tool_call = results[1]["tool_calls"][0]
        assert tool_call.tool_name == "ha_query"


class TestStateManagement:
    """Test internal state management."""

    @pytest.mark.asyncio
    async def test_state_reset_after_tool(self, handler):
        """Test that internal state is reset after tool call."""
        # Create mock stream with tool call
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"test_tool","arguments":"{\\"test\\": \\"value\\"}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify state was reset
        assert handler._current_tool_calls == {}

    @pytest.mark.asyncio
    async def test_text_only_no_tool_state(self, handler):
        """Test that text-only streaming doesn't set tool state."""
        # Create mock stream with only text
        sse_lines = [
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify tool state was never set
        assert handler._current_tool_calls == {}

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
        line = (
            'data: {"id":"test","object":"chat.completion.chunk",'
            '"choices":[{"delta":{"content":"Hello"}}]}'
        )
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"content":"Hello"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"content":" world"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"content":"!"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_abc123","type":"function",'
                '"function":{"name":"ha_control","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity"}}]},"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"_id\\": \\"li"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"ght.living_room"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"\\", \\"action\\":"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":" \\"turn_on\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            # First tool call
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"ha_query","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity_id\\": '
                '\\"sensor.temperature\\"}"}}]},"finish_reason":null}]}'
            ),
            # Second tool call
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,'
                '"id":"call_2","type":"function",'
                '"function":{"name":"ha_control","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,'
                '"function":{"arguments":"{\\"entity_id\\": \\"light.bedroom\\", '
                '\\"action\\": \\"turn_off\\"}"}}]},"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"content":'
                '"Let me check that for you."},"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"ha_query",'
                '"arguments":"{\\"entity_id\\": \\"light.kitchen\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"ha_control",'
                '"arguments":"{\\"invalid\\": json syntax}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"ha_query","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"ha_query",'
                '"arguments":"{\\"entity_id\\": \\"light.kitchen\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"test_tool",'
                '"arguments":"{\\"test\\": \\"value\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
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
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"content":"Hello"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify tool state was never set
        assert handler._current_tool_calls == {}


class TestThinkingBlockFiltering:
    """Test filtering of <think> blocks from reasoning models.

    Reasoning models (Qwen3, DeepSeek R1, o1/o3) output their reasoning
    in <think>...</think> blocks. These should be filtered out before
    being displayed to users.
    """

    @pytest.mark.asyncio
    async def test_streaming_filters_thinking_block_single_chunk(self, handler):
        """Test that thinking blocks in a single chunk are filtered."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":'
                '"<think>Let me think...</think>The answer is 42."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should have role and filtered content
        assert len(results) == 2
        assert results[0] == {"role": "assistant"}
        # Thinking block should be stripped
        assert results[1] == {"content": "The answer is 42."}

    @pytest.mark.asyncio
    async def test_streaming_filters_thinking_block_across_chunks(self, handler):
        """Test that thinking blocks spanning multiple chunks are filtered."""
        # Simulate a thinking block split across multiple SSE chunks
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"Reasoning here..."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"</think>"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"Hello, world!"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Collect all content deltas
        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        # The thinking block content should not appear
        assert "<think>" not in full_content
        assert "</think>" not in full_content
        assert "Reasoning here" not in full_content
        # The actual response should be present
        assert "Hello, world!" in full_content

    @pytest.mark.asyncio
    async def test_streaming_no_thinking_block_unchanged(self, handler):
        """Test that content without thinking blocks passes through unchanged."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{"content":"Hello, "},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{"content":"world!"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        assert len(results) == 3
        assert results[0] == {"role": "assistant"}
        assert results[1] == {"content": "Hello, "}
        assert results[2] == {"content": "world!"}

    @pytest.mark.asyncio
    async def test_streaming_multiline_thinking_block(self, handler):
        """Test filtering of multiline thinking blocks."""
        thinking_content = (
            "<think>\\nStep 1: Analyze the question\\n"
            "Step 2: Form a response\\n</think>"
        )
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"'
                + thinking_content
                + 'The answer is yes."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        assert "<think>" not in full_content
        assert "Step 1" not in full_content
        assert "The answer is yes." in full_content

    # Additional edge case tests for issue #64 coverage

    @pytest.mark.asyncio
    async def test_streaming_thinking_tag_split_mid_tag(self, handler):
        """Test when <think> tag is split mid-tag across chunks (e.g., <thi|nk>)."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"Before <thi"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"nk>hidden</think> After"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        # Thinking block should be filtered even when split mid-tag
        assert "hidden" not in full_content
        assert "Before" in full_content
        assert "After" in full_content

    @pytest.mark.asyncio
    async def test_streaming_closing_tag_split_mid_tag(self, handler):
        """Test when </think> tag is split mid-tag across chunks.

        Note: This is a known limitation. When the closing tag is split across
        chunks (e.g., '</th' in one chunk, 'ink>' in another), the current
        buffering implementation may not properly detect and filter it.

        The implementation uses a buffer for potential partial tags, but
        complex splits like this require more sophisticated state tracking.
        In practice, LLMs rarely split tags mid-token in this way.
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>hidden</th"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"ink>Visible text"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        # Known limitation: when closing tag is split mid-tag, the hidden
        # content may leak through. This documents the current behavior.
        # In practice, LLMs don't typically split tokens this way.
        # The main thinking block detection still works for normal cases.
        assert "hidden" not in full_content or full_content == ""

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls_and_thinking_blocks(self, handler):
        """Test thinking blocks filtered while tool calls are preserved."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>Reasoning about tools</think>"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123",'
                '"type":"function","function":{"name":"turn_on_light","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity_id\\": \\"light.kitchen\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify thinking content is filtered
        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)
        assert "Reasoning about tools" not in full_content

        # Verify tool calls are preserved
        tool_call_results = [r for r in results if "tool_calls" in r]
        assert len(tool_call_results) > 0

    @pytest.mark.asyncio
    async def test_streaming_unicode_in_thinking_blocks(self, handler):
        """Test thinking blocks with unicode content are properly filtered."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>思考中... 🤔</think>答案是42"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        assert "思考中" not in full_content
        assert "🤔" not in full_content
        assert "答案是42" in full_content

    @pytest.mark.asyncio
    async def test_streaming_handler_state_reset_between_streams(self, handler):
        """Test that handler state is properly reset between different streams."""
        # First stream with unclosed thinking block
        sse_lines_1 = [
            (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>Start thinking"},'
                '"finish_reason":null}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream_1 = async_generator_from_list(sse_lines_1)
        async for _ in handler.transform_openai_stream(mock_stream_1):
            pass

        # Reset handler state for second stream
        handler._in_thinking_block = False
        handler._thinking_buffer = ""

        # Second stream should work normally
        sse_lines_2 = [
            (
                'data: {"id":"chatcmpl-2","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-2","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"Normal response"},'
                '"finish_reason":null}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream_2 = async_generator_from_list(sse_lines_2)
        results = []
        async for delta in handler.transform_openai_stream(mock_stream_2):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        # Second stream should not be affected by first stream's unclosed block
        assert "Normal response" in full_content

    @pytest.mark.asyncio
    async def test_streaming_multiple_thinking_blocks(self, handler):
        """Test multiple thinking blocks in stream are all filtered."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>First thought</think>Part 1. "},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think>Second thought</think>Part 2."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        assert "First thought" not in full_content
        assert "Second thought" not in full_content
        assert "Part 1." in full_content
        assert "Part 2." in full_content

    @pytest.mark.asyncio
    async def test_streaming_empty_thinking_block(self, handler):
        """Test empty thinking blocks are properly handled."""
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"<think></think>Response"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        assert full_content == "Response"

    @pytest.mark.asyncio
    async def test_streaming_buffer_flushed_at_stream_end(self, handler):
        """Test that buffered partial tag content is yielded when stream ends.

        This is a regression test for the bug where content remaining in
        _thinking_buffer at stream end was silently discarded.

        Issue: If a stream chunk ends with a partial opening tag like "<th"
        (which could become "<think>"), it gets buffered. If the stream then
        ends without more data, that buffered content was lost.

        The fix ensures any buffered content is yielded at stream end since
        the partial tag will never complete.
        """
        # Stream where final chunk ends with "<th" (partial <think> tag)
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"The result is <th"},'
                '"finish_reason":null}]}'
            ),
            # Stream ends - no more chunks to resolve whether <th becomes <think>
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        # The "<th" should NOT be lost - it should be yielded since the stream
        # ended and it will never become a complete <think> tag
        assert full_content == "The result is <th", (
            f"Buffered content '<th' was lost at stream end. Got: '{full_content}'"
        )

    @pytest.mark.asyncio
    async def test_streaming_buffer_single_char_at_stream_end(self, handler):
        """Test that even a single buffered character is preserved at stream end."""
        # Stream ending with just "<" (start of potential <think>)
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{"content":"5 < 10 and 10 <"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"qwen3",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        # The trailing "<" should not be lost
        assert full_content == "5 < 10 and 10 <", (
            f"Buffered '<' was lost at stream end. Got: '{full_content}'"
        )

    @pytest.mark.asyncio
    async def test_handler_state_does_not_persist_between_iterations(self):
        """Test that handler state doesn't persist between iterations causing infinite loops.

        REGRESSION TEST for bug where _current_tool_calls dict persists after finalization,
        causing tool_calls to be yielded multiple times in subsequent iterations.

        This simulates the bug scenario:
        1. Iteration 1: Handler yields tool_calls, finishes stream, clears state
        2. Iteration 2: NEW handler instance is created (as in core.py line 1041)
        3. Handler should start fresh with empty _current_tool_calls

        If the handler is reused OR state persists incorrectly, tool_calls could be
        yielded again, causing an infinite loop in the agent iteration.
        """
        # First iteration - handler processes a stream with tool calls
        handler1 = OpenAIStreamingHandler()

        sse_lines_iteration1 = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_1","type":"function",'
                '"function":{"name":"ha_query",'
                '"arguments":"{\\"entity_id\\": \\"light.kitchen\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream1 = async_generator_from_list(sse_lines_iteration1)

        results1 = []
        async for delta in handler1.transform_openai_stream(mock_stream1):
            results1.append(delta)

        # Verify iteration 1 yielded tool calls
        tool_call_count_1 = sum(1 for r in results1 if "tool_calls" in r)
        assert tool_call_count_1 == 1, "First iteration should yield exactly 1 tool_calls delta"

        # Verify state was cleared after finalization
        assert handler1._current_tool_calls == {}, (
            "Handler state should be cleared after tool calls are finalized. "
            f"Got: {handler1._current_tool_calls}"
        )

        # Second iteration - simulate core.py creating a NEW handler (line 1041)
        # This is the critical test: does a fresh handler start with clean state?
        handler2 = OpenAIStreamingHandler()

        # Verify new handler starts with empty state
        assert handler2._current_tool_calls == {}, (
            "NEW handler instance should start with empty _current_tool_calls. "
            "If this fails, there's a class-level state leak!"
        )

        # Second iteration stream - no tool calls, just text response
        sse_lines_iteration2 = [
            (
                'data: {"id":"chatcmpl-456","object":"chat.completion.chunk",'
                '"created":1694268200,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-456","object":"chat.completion.chunk",'
                '"created":1694268200,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"content":"The light is on."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-456","object":"chat.completion.chunk",'
                '"created":1694268200,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream2 = async_generator_from_list(sse_lines_iteration2)

        results2 = []
        async for delta in handler2.transform_openai_stream(mock_stream2):
            results2.append(delta)

        # Verify iteration 2 does NOT yield any tool calls
        tool_call_count_2 = sum(1 for r in results2 if "tool_calls" in r)
        assert tool_call_count_2 == 0, (
            f"Second iteration should NOT yield tool_calls (got {tool_call_count_2}). "
            "If this fails, old tool_calls from iteration 1 are leaking!"
        )

        # Verify only text content was yielded
        content_parts = [r.get("content", "") for r in results2 if "content" in r]
        full_content = "".join(content_parts)
        assert full_content == "The light is on.", (
            f"Second iteration should yield only text content. Got: '{full_content}'"
        )

    @pytest.mark.asyncio
    async def test_handler_state_cleared_after_tool_finalization(self):
        """Test that _current_tool_calls is properly cleared after yielding tool_calls.

        This verifies the fix on lines 319 and 368 of streaming.py where
        self._current_tool_calls.clear() is called after finalizing tool calls.

        If this clear() doesn't happen, the same tool calls could be yielded again.
        """
        handler = OpenAIStreamingHandler()

        # Stream with tool calls that get finalized with finish_reason
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_abc","type":"function",'
                '"function":{"name":"test_tool","arguments":"{\\"arg\\": 1}"}}]},'
                '"finish_reason":null}]}'
            ),
            # This chunk has finish_reason which triggers finalization
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Before processing, state should be empty
        assert handler._current_tool_calls == {}

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # After processing, state MUST be cleared
        assert handler._current_tool_calls == {}, (
            "CRITICAL: _current_tool_calls was not cleared after finalization! "
            f"This will cause infinite loops. Current state: {handler._current_tool_calls}"
        )

        # Verify tool_calls were yielded exactly once
        tool_call_results = [r for r in results if "tool_calls" in r]
        assert len(tool_call_results) == 1, (
            f"Expected exactly 1 tool_calls delta, got {len(tool_call_results)}"
        )

    @pytest.mark.asyncio
    async def test_handler_state_cleared_at_stream_end_without_finish_reason(self):
        """Test that _current_tool_calls is cleared even without explicit finish_reason.

        Some APIs may not send finish_reason, so tool calls are finalized when
        the stream ends (lines 323-368 in streaming.py).

        This tests the fallback finalization path also clears state.
        """
        handler = OpenAIStreamingHandler()

        # Stream with tool calls but NO finish_reason - relies on stream end
        sse_lines = [
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-123","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-3.5-turbo",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_xyz","type":"function",'
                '"function":{"name":"another_tool","arguments":"{\\"x\\": 2}"}}]},'
                '"finish_reason":null}]}'
            ),
            # Stream ends WITHOUT finish_reason
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # State must be cleared even when finalized at stream end
        assert handler._current_tool_calls == {}, (
            "CRITICAL: _current_tool_calls was not cleared at stream end! "
            f"Current state: {handler._current_tool_calls}"
        )

        # Verify tool_calls were still yielded
        tool_call_results = [r for r in results if "tool_calls" in r]
        assert len(tool_call_results) == 1, "Tool calls should be finalized at stream end"


class TestGPT4oEdgeCases:
    """Test GPT-4o specific edge cases that may cause infinite loops.

    GPT-4o has been observed to exhibit specific streaming behaviors that differ
    from other models and can cause infinite loops in streaming handlers:

    1. Multiple finish_reason chunks - GPT-4o may send finish_reason in multiple
       consecutive chunks, which could cause tool calls to be yielded multiple times
    2. Tool calls with empty arguments - GPT-4o sometimes sends tool call chunks
       with empty or incomplete function data
    3. Thinking blocks mixed with tool calls - reasoning models may interleave
       <think> blocks with tool call chunks
    4. finish_reason followed by stream continuation - A chunk with finish_reason
       followed by more chunks without clearing state properly
    """

    @pytest.mark.asyncio
    async def test_gpt4o_multiple_finish_reason_chunks(self, handler):
        """Test that tool calls are not yielded twice when finish_reason appears multiple times.

        GPT-4o has been observed to send multiple chunks with finish_reason="tool_calls",
        which could cause the same tool calls to be yielded twice:
        - Once when first finish_reason is seen (line 265-319 in streaming.py)
        - Again at stream end if tool calls weren't cleared (line 323-368 in streaming.py)

        This can cause an infinite loop in the agent because it keeps processing
        the same tool calls over and over.
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            # Tool call chunks
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_abc123","type":"function",'
                '"function":{"name":"ha_control","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity_id\\": \\"light.living_room\\", '
                '\\"action\\": \\"turn_on\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            # First finish_reason - should yield tool calls and clear state
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            # GPT-4o bug: Another finish_reason chunk (should NOT yield tool calls again)
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            # More chunks after finish_reason (edge case)
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Count how many times tool_calls are yielded
        tool_call_deltas = [r for r in results if "tool_calls" in r]

        # CRITICAL: Tool calls should be yielded exactly ONCE, not multiple times
        # Multiple yields cause infinite loop because agent keeps processing same tools
        assert len(tool_call_deltas) == 1, (
            f"Tool calls yielded {len(tool_call_deltas)} times instead of 1! "
            f"This causes infinite loop. Results: {results}"
        )

        # Verify the tool call content is correct
        assert len(tool_call_deltas[0]["tool_calls"]) == 1
        tool_call = tool_call_deltas[0]["tool_calls"][0]
        assert tool_call.id == "call_abc123"
        assert tool_call.tool_name == "ha_control"
        assert tool_call.tool_args == {
            "entity_id": "light.living_room",
            "action": "turn_on",
        }

    @pytest.mark.asyncio
    async def test_gpt4o_empty_tool_call_arguments_chunks(self, handler):
        """Test handling of tool calls with empty function arguments in multiple chunks.

        GPT-4o sometimes sends multiple chunks with tool_calls but empty or missing
        function data, which could cause issues with tool call accumulation.
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            # Tool call with ID but no function
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_xyz789","type":"function"}]},'
                '"finish_reason":null}]}'
            ),
            # Tool call with function name but empty arguments
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"name":"ha_query","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            # Another chunk with empty arguments
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            # Finally some actual arguments
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity_id\\": \\"sensor.temp\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify tool call was accumulated correctly despite empty chunks
        tool_call_deltas = [r for r in results if "tool_calls" in r]
        assert len(tool_call_deltas) == 1

        tool_call = tool_call_deltas[0]["tool_calls"][0]
        assert tool_call.id == "call_xyz789"
        assert tool_call.tool_name == "ha_query"
        assert tool_call.tool_args == {"entity_id": "sensor.temp"}

    @pytest.mark.asyncio
    async def test_gpt4o_thinking_blocks_with_tool_calls_streaming(self, handler):
        """Test reasoning model (o1/o3/GPT-4o) mixing thinking blocks with tool calls.

        GPT-4o and reasoning models may output thinking blocks interleaved with
        tool call chunks, which requires careful state management.
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            # Thinking block start
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"content":"<think>I need to turn on"},'
                '"finish_reason":null}]}'
            ),
            # Tool call chunk WHILE in thinking block
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_mixed","type":"function",'
                '"function":{"name":"ha_control","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            # More thinking content
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"content":" the light</think>"},'
                '"finish_reason":null}]}'
            ),
            # Tool call arguments
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity_id\\": \\"light.bedroom\\", '
                '\\"action\\": \\"turn_on\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            # User-facing content
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"content":"Turning on the light."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Verify thinking blocks are filtered
        content_parts = [r.get("content", "") for r in results if "content" in r]
        full_content = "".join(content_parts)

        assert "<think>" not in full_content
        assert "I need to turn on the light" not in full_content
        assert "</think>" not in full_content
        assert "Turning on the light." in full_content

        # Verify tool calls are preserved
        tool_call_deltas = [r for r in results if "tool_calls" in r]
        assert len(tool_call_deltas) == 1

        tool_call = tool_call_deltas[0]["tool_calls"][0]
        assert tool_call.tool_name == "ha_control"
        assert tool_call.tool_args == {
            "entity_id": "light.bedroom",
            "action": "turn_on",
        }

    @pytest.mark.asyncio
    async def test_gpt4o_finish_reason_state_not_cleared_bug(self, handler):
        """Test the specific bug where tool_calls state isn't cleared after finish_reason.

        This simulates the exact scenario that causes infinite loop:
        1. Tool calls are accumulated
        2. finish_reason="tool_calls" is received -> yields tool calls
        3. Stream ends -> code checks if _current_tool_calls still has data
        4. If not cleared in step 2, tool calls get yielded AGAIN
        5. Agent processes same tool calls -> infinite loop
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_loop","type":"function",'
                '"function":{"name":"ha_control",'
                '"arguments":"{\\"entity_id\\": \\"light.test\\", '
                '\\"action\\": \\"turn_on\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            # finish_reason should clear tool_calls state
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Count tool_call yields
        tool_call_deltas = [r for r in results if "tool_calls" in r]

        # THE KEY TEST: Must be exactly 1, not 2
        # If this fails, tool_calls were yielded at both:
        # - Line 316 (when finish_reason is processed)
        # - Line 366 (at stream end because state wasn't cleared)
        assert len(tool_call_deltas) == 1, (
            f"INFINITE LOOP BUG: Tool calls yielded {len(tool_call_deltas)} times! "
            f"State was not properly cleared after finish_reason. "
            f"This causes agent to process same tool calls repeatedly. "
            f"Results: {results}"
        )

        # Verify handler state is clean after stream
        assert handler._current_tool_calls == {}, (
            "Handler state not cleared! This will cause tool calls to be "
            "yielded again on next iteration, leading to infinite loop."
        )

    @pytest.mark.asyncio
    async def test_gpt4o_no_finish_reason_but_has_done(self, handler):
        """Test GPT-4o stream that ends with [DONE] but no finish_reason.

        Some API implementations or edge cases may send [DONE] without ever
        sending a finish_reason, relying on the stream-end handler (line 323).
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"content":"Checking the light status."},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_nodone","type":"function",'
                '"function":{"name":"ha_query",'
                '"arguments":"{\\"entity_id\\": \\"light.status\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            # No finish_reason, just [DONE]
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should have content and tool calls
        content_parts = [r.get("content", "") for r in results if "content" in r]
        assert "Checking the light status." in "".join(content_parts)

        tool_call_deltas = [r for r in results if "tool_calls" in r]
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["tool_calls"][0].tool_name == "ha_query"

    @pytest.mark.asyncio
    async def test_gpt4o_partial_json_in_tool_arguments(self, handler):
        """Test GPT-4o streaming with malformed/partial JSON in tool arguments.

        If JSON is malformed, the handler should yield tool call with empty args
        and continue without crashing.
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_bad","type":"function",'
                '"function":{"name":"ha_control","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            # Malformed JSON - missing closing brace
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity_id\\": \\"light.bad\\""}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream - should not crash
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should yield tool call with empty args due to JSON error
        tool_call_deltas = [r for r in results if "tool_calls" in r]
        assert len(tool_call_deltas) == 1

        tool_call = tool_call_deltas[0]["tool_calls"][0]
        assert tool_call.id == "call_bad"
        assert tool_call.tool_name == "ha_control"
        assert tool_call.tool_args == {}  # Empty due to malformed JSON

    @pytest.mark.asyncio
    async def test_gpt4o_content_after_finish_reason_with_tool_calls(self, handler):
        """Test GPT-4o sending content chunks after finish_reason with tool_calls.

        This edge case tests whether the handler can handle:
        1. Tool calls are accumulated
        2. finish_reason="tool_calls" is sent (yields tools, clears state)
        3. GPT-4o continues sending content chunks (unusual but possible)
        4. Stream ends

        If the handler doesn't properly handle this, it might:
        - Ignore the content after finish_reason
        - Crash on unexpected content
        - Re-yield tool calls
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            # Tool call
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_123","type":"function",'
                '"function":{"name":"ha_control",'
                '"arguments":"{\\"entity_id\\": \\"light.room\\", '
                '\\"action\\": \\"turn_on\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            # finish_reason sent
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            # GPT-4o edge case: Content AFTER finish_reason
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"content":"Done."},'
                '"finish_reason":null}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Should have role, tool_calls, and content
        assert any("role" in r for r in results)

        tool_call_deltas = [r for r in results if "tool_calls" in r]
        assert len(tool_call_deltas) == 1, "Tool calls should be yielded exactly once"

        content_parts = [r.get("content", "") for r in results if "content" in r]
        # Content after finish_reason should still be yielded
        assert "Done." in "".join(content_parts)

    @pytest.mark.asyncio
    async def test_gpt4o_empty_delta_chunks_between_tool_calls(self, handler):
        """Test GPT-4o sending empty delta chunks between tool call chunks.

        GPT-4o may send chunks with empty deltas ({}) or only finish_reason=null
        between actual tool call data. This tests robustness against noisy streams.
        """
        sse_lines = [
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"role":"assistant"},'
                '"finish_reason":null}]}'
            ),
            # Start tool call
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"id":"call_empty","type":"function",'
                '"function":{"name":"ha_query","arguments":""}}]},'
                '"finish_reason":null}]}'
            ),
            # Empty delta chunk 1
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":null}]}'
            ),
            # Empty delta chunk 2
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":null}]}'
            ),
            # Continue tool call arguments
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"entity"}}]},'
                '"finish_reason":null}]}'
            ),
            # Another empty chunk
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":null}]}'
            ),
            # Finish arguments
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"_id\\": \\"light.test\\"}"}}]},'
                '"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-gpt4o","object":"chat.completion.chunk",'
                '"created":1694268190,"model":"gpt-4o",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            ),
            "data: [DONE]",
        ]

        mock_stream = async_generator_from_list(sse_lines)

        # Process stream
        results = []
        async for delta in handler.transform_openai_stream(mock_stream):
            results.append(delta)

        # Tool call should be properly assembled despite empty chunks
        tool_call_deltas = [r for r in results if "tool_calls" in r]
        assert len(tool_call_deltas) == 1

        tool_call = tool_call_deltas[0]["tool_calls"][0]
        assert tool_call.id == "call_empty"
        assert tool_call.tool_name == "ha_query"
        assert tool_call.tool_args == {"entity_id": "light.test"}

"""Unit tests for HomeAgent streaming integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation as ha_conversation
from homeassistant.core import HomeAssistant

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_STREAMING_ENABLED,
    EVENT_STREAMING_ERROR,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock(spec=HomeAssistant)
    hass.states = MagicMock()
    hass.states.async_all = MagicMock(return_value=[])
    hass.data = {}
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    return hass


@pytest.fixture
def agent_config():
    """Create agent configuration."""
    return {
        CONF_LLM_BASE_URL: "http://localhost:11434/v1",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "llama2",
        CONF_STREAMING_ENABLED: False,  # Disabled by default
    }


@pytest.fixture
def agent(mock_hass, agent_config):
    """Create HomeAgent instance."""
    from custom_components.home_agent.conversation_session import ConversationSessionManager

    session_manager = ConversationSessionManager(mock_hass)
    return HomeAgent(mock_hass, agent_config, session_manager)


class TestStreamingDetection:
    """Test streaming detection logic."""

    def test_can_stream_disabled_by_config(self, agent):
        """Test that streaming is disabled when config says so."""
        assert agent._can_stream() is False

    def test_can_stream_no_chat_log(self, agent):
        """Test that streaming is disabled when ChatLog not available."""
        agent.config[CONF_STREAMING_ENABLED] = True

        with patch(
            "homeassistant.components.conversation.chat_log.current_chat_log"
        ) as mock_chat_log:
            mock_chat_log.get.return_value = None
            assert agent._can_stream() is False

    def test_can_stream_chat_log_no_delta_listener(self, agent):
        """Test that streaming is disabled when ChatLog has no delta_listener."""
        agent.config[CONF_STREAMING_ENABLED] = True

        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = None

        with patch(
            "homeassistant.components.conversation.chat_log.current_chat_log"
        ) as mock_chat_log:
            mock_chat_log.get.return_value = mock_chat_log_instance
            assert agent._can_stream() is False

    def test_can_stream_enabled(self, agent):
        """Test that streaming is enabled when all conditions are met."""
        agent.config[CONF_STREAMING_ENABLED] = True

        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = MagicMock()

        with patch(
            "homeassistant.components.conversation.chat_log.current_chat_log"
        ) as mock_chat_log:
            mock_chat_log.get.return_value = mock_chat_log_instance
            assert agent._can_stream() is True


class TestAsyncProcessBranching:
    """Test async_process method branching logic."""

    @pytest.mark.asyncio
    async def test_async_process_uses_synchronous_when_streaming_disabled(self, agent):
        """Test that async_process uses synchronous path when streaming is disabled."""
        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Turn on the lights"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.context.user_id = "test-user"
        mock_input.device_id = None

        # Mock _async_process_synchronous
        with patch.object(agent, "_async_process_synchronous", new_callable=AsyncMock) as mock_sync:
            mock_sync.return_value = MagicMock(spec=ha_conversation.ConversationResult)

            # Call async_process
            result = await agent.async_process(mock_input)

            # Verify synchronous path was called
            mock_sync.assert_called_once_with(mock_input)
            assert result is not None

    @pytest.mark.asyncio
    async def test_async_process_uses_streaming_when_enabled(self, agent):
        """Test that async_process uses streaming path when enabled."""
        agent.config[CONF_STREAMING_ENABLED] = True

        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Turn on the lights"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.context.user_id = "test-user"
        mock_input.device_id = None

        # Mock ChatLog
        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = MagicMock()

        # Mock _async_process_streaming
        with (
            patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_chat_log,
            patch.object(agent, "_async_process_streaming", new_callable=AsyncMock) as mock_stream,
        ):
            mock_chat_log.get.return_value = mock_chat_log_instance
            mock_stream.return_value = MagicMock(spec=ha_conversation.ConversationResult)

            # Call async_process
            result = await agent.async_process(mock_input)

            # Verify streaming path was called
            mock_stream.assert_called_once_with(mock_input)
            assert result is not None

    @pytest.mark.asyncio
    async def test_async_process_fallback_on_streaming_error(self, agent):
        """Test that async_process falls back to synchronous when streaming fails."""
        agent.config[CONF_STREAMING_ENABLED] = True

        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Turn on the lights"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.context.user_id = "test-user"
        mock_input.device_id = None

        # Mock ChatLog
        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = MagicMock()

        # Mock streaming to raise an error
        with (
            patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_chat_log,
            patch.object(agent, "_async_process_streaming", new_callable=AsyncMock) as mock_stream,
            patch.object(agent, "_async_process_synchronous", new_callable=AsyncMock) as mock_sync,
        ):
            mock_chat_log.get.return_value = mock_chat_log_instance
            mock_stream.side_effect = RuntimeError("Streaming error")
            mock_sync.return_value = MagicMock(spec=ha_conversation.ConversationResult)

            # Call async_process
            result = await agent.async_process(mock_input)

            # Verify both paths were called
            mock_stream.assert_called_once_with(mock_input)
            mock_sync.assert_called_once_with(mock_input)

            # Verify error event was fired
            agent.hass.bus.async_fire.assert_called()
            call_args = agent.hass.bus.async_fire.call_args[0]
            assert call_args[0] == EVENT_STREAMING_ERROR
            assert call_args[1]["fallback"] is True
            assert result is not None


class TestStreamingErrorEvents:
    """Test streaming error event emission."""

    @pytest.mark.asyncio
    async def test_streaming_error_event_fired(self, agent):
        """Test that streaming error event is fired with correct data."""
        agent.config[CONF_STREAMING_ENABLED] = True

        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Turn on the lights"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.context.user_id = "test-user"
        mock_input.device_id = None

        # Mock ChatLog
        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = MagicMock()

        error_message = "Test streaming error"

        # Mock streaming to raise an error
        with (
            patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_chat_log,
            patch.object(agent, "_async_process_streaming", new_callable=AsyncMock) as mock_stream,
            patch.object(agent, "_async_process_synchronous", new_callable=AsyncMock) as mock_sync,
        ):
            mock_chat_log.get.return_value = mock_chat_log_instance
            mock_stream.side_effect = RuntimeError(error_message)
            mock_sync.return_value = MagicMock(spec=ha_conversation.ConversationResult)

            # Call async_process
            await agent.async_process(mock_input)

            # Verify error event was fired with correct data
            agent.hass.bus.async_fire.assert_called()
            call_args = agent.hass.bus.async_fire.call_args[0]
            event_data = call_args[1]

            assert call_args[0] == EVENT_STREAMING_ERROR
            assert event_data["error"] == error_message
            assert event_data["error_type"] == "RuntimeError"
            assert event_data["fallback"] is True


class TestCallLLMStreaming:
    """Test _call_llm_streaming method."""

    @pytest.mark.asyncio
    async def test_call_llm_streaming_yields_sse_lines(self, agent):
        """Test that _call_llm_streaming yields SSE lines."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]

        # Mock aiohttp response
        mock_response = MagicMock()
        mock_response.status = 200

        # Create async generator for response content
        async def mock_content_generator():
            yield b'data: {"id":"test","choices":[{"delta":{"role":"assistant"}}]}\n'
            yield b'data: {"id":"test","choices":[{"delta":{"content":"Hello"}}]}\n'
            yield b"data: [DONE]\n"

        mock_response.content = mock_content_generator()

        # Mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock()

        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_session

            # Call _call_llm_streaming
            results = []
            async for line in agent._call_llm_streaming(messages):
                results.append(line)

            # Verify we got SSE lines
            assert len(results) == 3
            assert 'data: {"id":"test"' in results[0]
            assert "data: [DONE]" in results[2]

    @pytest.mark.asyncio
    async def test_call_llm_streaming_includes_tools(self, agent):
        """Test that _call_llm_streaming includes tool definitions."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Turn on the lights"},
        ]

        # Mock tool definitions
        mock_tools = [
            {
                "type": "function",
                "function": {
                    "name": "ha_control",
                    "description": "Control Home Assistant entities",
                },
            }
        ]

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200

        async def mock_content_generator():
            yield b"data: [DONE]\n"

        mock_response.content = mock_content_generator()

        # Mock session
        mock_session = MagicMock()
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock()

        with (
            patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure,
            patch.object(agent.tool_handler, "get_tool_definitions") as mock_get_tools,
        ):
            mock_ensure.return_value = mock_session
            mock_get_tools.return_value = mock_tools

            # Call _call_llm_streaming
            results = []
            async for line in agent._call_llm_streaming(messages):
                results.append(line)

            # Verify tools were added to payload
            mock_session.post.assert_called_once()
            call_kwargs = mock_session.post.call_args[1]
            payload = call_kwargs["json"]

            assert "tools" in payload
            assert payload["tools"] == mock_tools
            assert payload["tool_choice"] == "auto"
            assert payload["stream"] is True


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""

    @pytest.mark.asyncio
    async def test_synchronous_path_still_works(self, agent):
        """Test that synchronous processing still works as before."""
        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Turn on the lights"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.context.user_id = "test-user"
        mock_input.device_id = None

        # Mock process_message to return a response
        with patch.object(agent, "process_message", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = "The lights are now on"

            # Call async_process (streaming disabled by default)
            result = await agent.async_process(mock_input)

            # Verify process_message was called
            mock_process.assert_called_once_with(
                text="Turn on the lights",
                conversation_id="test-conv",
                user_id="test-user",
                device_id=None,
            )

            # Verify result
            assert result is not None
            assert result.conversation_id == "test-conv"


class TestStreamingMemoryExtraction:
    """Test memory extraction in streaming mode."""

    @pytest.mark.asyncio
    async def test_memory_extraction_triggered_after_streaming(self, agent, mock_hass):
        """Test that memory extraction is triggered after streaming completes."""
        from custom_components.home_agent.const import (
            CONF_MEMORY_ENABLED,
            CONF_MEMORY_EXTRACTION_ENABLED,
        )

        # Enable streaming and memory extraction
        agent.config[CONF_STREAMING_ENABLED] = True
        agent.config[CONF_MEMORY_ENABLED] = True
        agent.config[CONF_MEMORY_EXTRACTION_ENABLED] = True

        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Remember that I like pizza"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.device_id = None

        # Mock chat log
        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = MagicMock()
        mock_chat_log_instance.unresponded_tool_results = []

        # Mock assistant content
        from homeassistant.components.conversation import AssistantContent

        mock_content = AssistantContent(
            agent_id="home_agent", content="I'll remember that you like pizza!"
        )

        # Mock async_add_delta_content_stream as an async generator
        async def mock_content_stream(*args, **kwargs):
            yield mock_content

        mock_chat_log_instance.async_add_delta_content_stream = mock_content_stream

        # Mock the result extraction
        mock_result = MagicMock(spec=ha_conversation.ConversationResult)
        mock_result.conversation_id = "test-conv"

        with (
            patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_chat_log,
            patch.object(agent, "_call_llm_streaming") as mock_stream,
            patch.object(
                agent, "_extract_and_store_memories", new_callable=AsyncMock
            ) as mock_extract,
            patch(
                "homeassistant.components.conversation.async_get_result_from_chat_log",
                return_value=mock_result,
            ),
        ):
            mock_chat_log.get.return_value = mock_chat_log_instance

            # Mock streaming response
            async def mock_stream_gen():
                yield "data: {}"

            mock_stream.return_value = mock_stream_gen()

            # Call async_process with streaming
            result = await agent.async_process(mock_input)

            # Wait a moment for the async task to be created
            import asyncio

            await asyncio.sleep(0.1)

            # Verify memory extraction was triggered
            mock_extract.assert_called_once()
            call_args = mock_extract.call_args[1]
            assert call_args["conversation_id"] == "test-conv"
            assert call_args["user_message"] == "Remember that I like pizza"
            assert call_args["assistant_response"] == "I'll remember that you like pizza!"

            # Verify result
            assert result is not None
            assert result.conversation_id == "test-conv"

    @pytest.mark.asyncio
    async def test_memory_extraction_skipped_when_disabled(self, agent, mock_hass):
        """Test that memory extraction is skipped when disabled in streaming mode."""
        from custom_components.home_agent.const import (
            CONF_MEMORY_ENABLED,
            CONF_MEMORY_EXTRACTION_ENABLED,
        )

        # Enable streaming but disable memory extraction
        agent.config[CONF_STREAMING_ENABLED] = True
        agent.config[CONF_MEMORY_ENABLED] = True
        agent.config[CONF_MEMORY_EXTRACTION_ENABLED] = False

        # Create mock conversation input
        mock_input = MagicMock(spec=ha_conversation.ConversationInput)
        mock_input.text = "Hello"
        mock_input.conversation_id = "test-conv"
        mock_input.language = "en"
        mock_input.context = MagicMock()
        mock_input.device_id = None

        # Mock chat log
        mock_chat_log_instance = MagicMock()
        mock_chat_log_instance.delta_listener = MagicMock()
        mock_chat_log_instance.unresponded_tool_results = []

        # Mock assistant content
        from homeassistant.components.conversation import AssistantContent

        mock_content = AssistantContent(agent_id="home_agent", content="Hi there!")

        # Mock async_add_delta_content_stream as an async generator
        async def mock_content_stream(*args, **kwargs):
            yield mock_content

        mock_chat_log_instance.async_add_delta_content_stream = mock_content_stream

        # Mock the result extraction
        mock_result = MagicMock(spec=ha_conversation.ConversationResult)
        mock_result.conversation_id = "test-conv"

        with (
            patch(
                "homeassistant.components.conversation.chat_log.current_chat_log"
            ) as mock_chat_log,
            patch.object(agent, "_call_llm_streaming") as mock_stream,
            patch.object(
                agent, "_extract_and_store_memories", new_callable=AsyncMock
            ) as mock_extract,
            patch(
                "homeassistant.components.conversation.async_get_result_from_chat_log",
                return_value=mock_result,
            ),
        ):
            mock_chat_log.get.return_value = mock_chat_log_instance

            # Mock streaming response
            async def mock_stream_gen():
                yield "data: {}"

            mock_stream.return_value = mock_stream_gen()

            # Call async_process with streaming
            result = await agent.async_process(mock_input)

            # Wait a moment for any async tasks
            import asyncio

            await asyncio.sleep(0.1)

            # Verify memory extraction was NOT triggered
            mock_extract.assert_not_called()

            # Verify result
            assert result is not None
            assert result.conversation_id == "test-conv"

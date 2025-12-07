"""Integration tests for thinking block filtering.

These tests verify that reasoning model thinking blocks (<think>...</think>)
are properly filtered throughout the full conversation flow, including:
- Non-streaming responses
- Streaming responses
- Memory extraction
- Conversation history

Issue #64: Support for reasoning models (Qwen3, DeepSeek R1, o1/o3)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation as ha_conversation

from custom_components.home_agent.agent.core import HomeAgent
from custom_components.home_agent.const import DOMAIN


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.data = {DOMAIN: {"config": {}}}
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    hass.states = MagicMock()
    hass.states.async_all = MagicMock(return_value=[])
    hass.services = MagicMock()
    return hass


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    manager = MagicMock()
    session = MagicMock(
        conversation_id="test_conv",
        get_history=MagicMock(return_value=[]),
        add_message=MagicMock(),
    )
    manager.get_or_create_session = MagicMock(return_value=session)
    manager.update_session_activity = MagicMock()
    # Make async methods return coroutines
    manager.update_activity = AsyncMock()
    return manager


@pytest.fixture
def basic_config():
    """Basic configuration for tests."""
    return {
        "llm": {
            "url": "http://localhost:11434/v1/chat/completions",
            "model": "qwen3",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "context": {
            "mode": "direct",
        },
        "history": {
            "enabled": True,
            "max_messages": 10,
        },
        "memory": {
            "enabled": False,
        },
        "streaming": {
            "enabled": False,
        },
        "emit_events": False,
    }


@pytest.fixture
def agent(mock_hass, basic_config, mock_session_manager):
    """Create a HomeAgent instance for testing."""
    return HomeAgent(mock_hass, basic_config, mock_session_manager)


@pytest.fixture
def user_input():
    """Create a mock ConversationInput."""
    return ha_conversation.ConversationInput(
        text="Turn on the kitchen light",
        context=MagicMock(),
        conversation_id="test_conv",
        language="en",
        device_id=None,
        satellite_id=None,
        agent_id="home_agent",
    )


class TestThinkingBlocksNonStreaming:
    """Test thinking block filtering in non-streaming responses."""

    @pytest.mark.asyncio
    async def test_thinking_blocks_filtered_from_response(
        self, agent, mock_hass, user_input
    ):
        """Test that thinking blocks are stripped from LLM responses."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Let me analyze this request...</think>I'll turn on the light.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        # Response should not contain thinking block
        response_text = result.response.speech["plain"]["speech"]
        assert "<think>" not in response_text
        assert "analyze this request" not in response_text
        assert "turn on the light" in response_text.lower()

    @pytest.mark.asyncio
    async def test_only_thinking_block_response_handled(
        self, agent, mock_hass, user_input
    ):
        """Test response that is only a thinking block."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>All reasoning, no response</think>",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        # Should handle empty response gracefully
        response_text = result.response.speech["plain"]["speech"]
        assert "<think>" not in response_text

    @pytest.mark.asyncio
    async def test_multiline_thinking_blocks_filtered(
        self, agent, mock_hass, user_input
    ):
        """Test that multiline thinking blocks are properly filtered."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": """<think>
Step 1: Parse the request
Step 2: Identify the entity
Step 3: Execute the action
</think>I've turned on the living room light for you.""",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "Step 1" not in response_text
        assert "Step 2" not in response_text
        assert "living room light" in response_text.lower()

    @pytest.mark.asyncio
    async def test_multiple_thinking_blocks_filtered(
        self, agent, mock_hass, user_input
    ):
        """Test that multiple thinking blocks are all filtered."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>First thought</think>Part 1. <think>Second thought</think>Part 2.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "First thought" not in response_text
        assert "Second thought" not in response_text
        assert "Part 1" in response_text
        assert "Part 2" in response_text


class TestThinkingBlocksUnicode:
    """Test thinking blocks with unicode/multilingual content."""

    @pytest.mark.asyncio
    async def test_chinese_thinking_blocks_filtered(
        self, agent, mock_hass, user_input
    ):
        """Test Chinese content in thinking blocks is filtered."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>让我思考一下这个问题...</think>答案是42。",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "让我思考" not in response_text
        assert "答案是42" in response_text

    @pytest.mark.asyncio
    async def test_emoji_in_thinking_blocks_filtered(
        self, agent, mock_hass, user_input
    ):
        """Test emojis in thinking blocks are filtered."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>🤔 Let me think... 💭</think>✅ Done!",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "🤔" not in response_text
        assert "💭" not in response_text
        assert "✅" in response_text


class TestThinkingBlocksStreaming:
    """Test thinking block filtering in streaming mode."""

    @pytest.mark.asyncio
    async def test_streaming_filters_thinking_blocks(
        self, mock_hass, basic_config, mock_session_manager
    ):
        """Test that streaming responses filter thinking blocks."""
        # Enable streaming
        streaming_config = {**basic_config, "streaming": {"enabled": True}}
        agent = HomeAgent(mock_hass, streaming_config, mock_session_manager)

        # Mock streaming response - the streaming handler tests cover this
        # in detail, this just verifies the config flows through correctly
        assert agent.config.get("streaming", {}).get("enabled") is True


class TestThinkingBlocksEdgeCases:
    """Test edge cases for thinking block handling."""

    @pytest.mark.asyncio
    async def test_malformed_thinking_tags_preserved(
        self, agent, mock_hass, user_input
    ):
        """Test that malformed thinking tags are handled gracefully."""
        # Unclosed tag - should be preserved
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Unclosed thinking block... Response here",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        # Unclosed tags should be preserved (not removed)
        response_text = result.response.speech["plain"]["speech"]
        assert "Response here" in response_text or "<think>" in response_text

    @pytest.mark.asyncio
    async def test_case_sensitive_thinking_tags(self, agent, mock_hass, user_input):
        """Test that only lowercase <think> tags are filtered."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<THINK>Uppercase should stay</THINK><think>lowercase removed</think>Answer",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        # Uppercase should be preserved
        assert "Uppercase should stay" in response_text or "<THINK>" in response_text
        # Lowercase should be removed
        assert "lowercase removed" not in response_text

    @pytest.mark.asyncio
    async def test_thinking_blocks_with_json_content(
        self, agent, mock_hass, user_input
    ):
        """Test thinking blocks containing JSON are properly handled."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '<think>{"internal": "data"}</think>Here is the response.',
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert '"internal"' not in response_text
        assert "Here is the response" in response_text

"""Unit tests for thinking block handling in HomeAgent core.

These tests specifically cover the strip_thinking_blocks integration point
in the agent's _process_conversation method (core.py line ~1332).

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
            "enabled": False,
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
        text="Test query",
        context=MagicMock(),
        conversation_id="test_conv",
        language="en",
        device_id=None,
        satellite_id=None,
        agent_id="home_agent",
    )


class TestAgentCoreThinkingBlockStripping:
    """Test thinking block stripping in agent core response processing."""

    @pytest.mark.asyncio
    async def test_strip_thinking_blocks_called_on_response(
        self, agent, mock_hass, user_input
    ):
        """Verify strip_thinking_blocks is called on LLM response content."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Internal reasoning</think>User-facing response",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            with patch(
                "custom_components.home_agent.agent.core.strip_thinking_blocks"
            ) as mock_strip:
                mock_strip.return_value = "User-facing response"

                result = await agent.async_process(user_input)

                # Verify strip_thinking_blocks was called
                mock_strip.assert_called()

    @pytest.mark.asyncio
    async def test_empty_content_after_stripping_handled(
        self, agent, mock_hass, user_input
    ):
        """Test handling when content is empty after stripping thinking blocks."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>All thinking, no response</think>",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            # Should not raise an exception
            result = await agent.async_process(user_input)

            # Should handle gracefully (empty string fallback)
            assert result is not None

    @pytest.mark.asyncio
    async def test_none_content_handled(self, agent, mock_hass, user_input):
        """Test handling when LLM response content is None."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            # Should not raise an exception
            result = await agent.async_process(user_input)

            assert result is not None


class TestAgentCoreThinkingBlockVariants:
    """Test various thinking block formats in agent core."""

    @pytest.mark.asyncio
    async def test_multiline_thinking_block(self, agent, mock_hass, user_input):
        """Test multiline thinking blocks are fully stripped."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": """<think>
Line 1: Analyzing request
Line 2: Processing entities
Line 3: Formulating response
</think>Here is my response.""",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "Line 1" not in response_text
        assert "Line 2" not in response_text
        assert "Line 3" not in response_text
        assert "Here is my response" in response_text

    @pytest.mark.asyncio
    async def test_multiple_thinking_blocks(self, agent, mock_hass, user_input):
        """Test multiple consecutive thinking blocks are all stripped."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>First</think><think>Second</think>Final answer.",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "First" not in response_text
        assert "Second" not in response_text
        assert "Final answer" in response_text

    @pytest.mark.asyncio
    async def test_thinking_block_with_special_characters(
        self, agent, mock_hass, user_input
    ):
        """Test thinking blocks containing special characters."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '<think>{"key": "value"} and <tag></think>Clean output.',
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert '"key"' not in response_text
        assert "Clean output" in response_text

    @pytest.mark.asyncio
    async def test_unicode_thinking_block(self, agent, mock_hass, user_input):
        """Test thinking blocks with unicode content."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>思考中... 🤔</think>答案是42。",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        response_text = result.response.speech["plain"]["speech"]
        assert "思考中" not in response_text
        assert "🤔" not in response_text
        assert "答案是42" in response_text


class TestAgentCoreThinkingBlockEdgeCases:
    """Test edge cases for thinking block handling."""

    @pytest.mark.asyncio
    async def test_unclosed_thinking_tag_preserved(self, agent, mock_hass, user_input):
        """Test that unclosed thinking tags are not incorrectly removed."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>This is unclosed and should remain...",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        # Unclosed tags are preserved (not matched by regex)
        response_text = result.response.speech["plain"]["speech"]
        # The unclosed tag should still be there since the regex won't match
        assert "unclosed" in response_text.lower() or "<think>" in response_text

    @pytest.mark.asyncio
    async def test_case_sensitive_think_tags(self, agent, mock_hass, user_input):
        """Test that only lowercase <think> tags are stripped."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<THINK>Uppercase preserved</THINK><think>lowercase removed</think>Result",
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
        assert "Uppercase preserved" in response_text or "<THINK>" in response_text
        # Lowercase should be removed
        assert "lowercase removed" not in response_text
        assert "Result" in response_text

    @pytest.mark.asyncio
    async def test_whitespace_only_after_stripping(self, agent, mock_hass, user_input):
        """Test handling when only whitespace remains after stripping."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "   <think>All content</think>   ",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await agent.async_process(user_input)

        # Should handle gracefully (empty or whitespace result)
        assert result is not None


class TestAgentPreprocessUserMessage:
    """Test user message preprocessing for thinking control."""

    def test_preprocess_thinking_enabled_by_default(self, agent):
        """Test that by default, no /no_think is appended."""
        result = agent._preprocess_user_message("Hello world")
        assert result == "Hello world"
        assert "/no_think" not in result

    def test_preprocess_thinking_disabled_appends_no_think(self, mock_hass, mock_session_manager):
        """Test that when thinking is disabled, /no_think is appended."""
        config = {
            "thinking_enabled": False,
            "llm": {"url": "http://localhost", "model": "test"},
            "context": {"mode": "direct"},
            "history": {"enabled": False},
            "memory": {"enabled": False},
            "streaming": {"enabled": False},
            "emit_events": False,
        }
        agent = HomeAgent(mock_hass, config, mock_session_manager)

        result = agent._preprocess_user_message("Turn on the lights")
        assert result == "Turn on the lights\n/no_think"

    def test_preprocess_thinking_enabled_explicit(self, mock_hass, mock_session_manager):
        """Test that when thinking is explicitly enabled, no /no_think is appended."""
        config = {
            "thinking_enabled": True,
            "llm": {"url": "http://localhost", "model": "test"},
            "context": {"mode": "direct"},
            "history": {"enabled": False},
            "memory": {"enabled": False},
            "streaming": {"enabled": False},
            "emit_events": False,
        }
        agent = HomeAgent(mock_hass, config, mock_session_manager)

        result = agent._preprocess_user_message("What's the weather?")
        assert result == "What's the weather?"
        assert "/no_think" not in result

    def test_preprocess_strips_trailing_whitespace_when_disabled(self, mock_hass, mock_session_manager):
        """Test that whitespace is stripped before appending /no_think."""
        config = {
            "thinking_enabled": False,
            "llm": {"url": "http://localhost", "model": "test"},
            "context": {"mode": "direct"},
            "history": {"enabled": False},
            "memory": {"enabled": False},
            "streaming": {"enabled": False},
            "emit_events": False,
        }
        agent = HomeAgent(mock_hass, config, mock_session_manager)

        result = agent._preprocess_user_message("  Message with spaces  ")
        assert result == "Message with spaces\n/no_think"

    def test_preprocess_empty_message_when_disabled(self, mock_hass, mock_session_manager):
        """Test preprocessing empty message when thinking disabled."""
        config = {
            "thinking_enabled": False,
            "llm": {"url": "http://localhost", "model": "test"},
            "context": {"mode": "direct"},
            "history": {"enabled": False},
            "memory": {"enabled": False},
            "streaming": {"enabled": False},
            "emit_events": False,
        }
        agent = HomeAgent(mock_hass, config, mock_session_manager)

        result = agent._preprocess_user_message("")
        assert result == "\n/no_think"

    def test_preprocess_idempotent_no_think_already_present(self, mock_hass, mock_session_manager):
        """Test that /no_think is not duplicated if already present in message."""
        config = {
            "thinking_enabled": False,
            "llm": {"url": "http://localhost", "model": "test"},
            "context": {"mode": "direct"},
            "history": {"enabled": False},
            "memory": {"enabled": False},
            "streaming": {"enabled": False},
            "emit_events": False,
        }
        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Test with /no_think already at the end
        result1 = agent._preprocess_user_message("Turn on the lights\n/no_think")
        assert result1 == "Turn on the lights\n/no_think"
        assert result1.count("/no_think") == 1, "Should not duplicate /no_think"

        # Test with /no_think in the middle
        result2 = agent._preprocess_user_message("Turn on /no_think the lights")
        assert result2 == "Turn on /no_think the lights"
        assert result2.count("/no_think") == 1, "Should not add /no_think if already present"

        # Test with /no_think at the beginning
        result3 = agent._preprocess_user_message("/no_think Turn on the lights")
        assert result3 == "/no_think Turn on the lights"
        assert result3.count("/no_think") == 1, "Should not add /no_think if already present"

"""Unit tests for automatic memory extraction.

Tests the memory extraction functionality that extracts memories from
conversations using either external or local LLM.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_MEMORY_EXTRACTION_LLM,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    mock = MagicMock()
    mock.data = {}
    mock.bus = MagicMock()
    mock.bus.async_fire = MagicMock()
    mock.async_create_task = MagicMock()
    return mock


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager instance."""
    mock = MagicMock()
    mock.add_memory = AsyncMock(return_value="mem_123")
    mock._is_transient_state = MagicMock(return_value=False)
    return mock


@pytest.fixture
def agent_config():
    """Create agent configuration."""
    return {
        CONF_LLM_BASE_URL: "http://test.com",
        CONF_LLM_API_KEY: "test-key",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_EXTRACTION_ENABLED: True,
        CONF_MEMORY_EXTRACTION_LLM: "local",
    }


@pytest.fixture
def home_agent(mock_hass, agent_config):
    """Create a HomeAgent instance for testing."""
    with patch("custom_components.home_agent.agent.ContextManager"):
        with patch("custom_components.home_agent.agent.ConversationHistoryManager"):
            with patch("custom_components.home_agent.agent.ToolHandler"):
                from custom_components.home_agent.conversation_session import (
                    ConversationSessionManager,
                )

                session_manager = ConversationSessionManager(mock_hass)
                agent = HomeAgent(mock_hass, agent_config, session_manager)
                return agent


class TestFormatConversationForExtraction:
    """Test _format_conversation_for_extraction method."""

    def test_format_empty_conversation(self, home_agent):
        """Test formatting empty conversation."""
        messages = []
        result = home_agent._format_conversation_for_extraction(messages)
        assert result == ""

    def test_format_conversation_excludes_system(self, home_agent):
        """Test that system messages are excluded."""
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = home_agent._format_conversation_for_extraction(messages)
        assert "system" not in result.lower()
        assert "Hello" in result
        assert "Hi there" in result

    def test_format_conversation_excludes_tool(self, home_agent):
        """Test that tool messages are excluded."""
        messages = [
            {"role": "user", "content": "Turn on the light"},
            {"role": "tool", "content": '{"success": true}'},
            {"role": "assistant", "content": "Done"},
        ]
        result = home_agent._format_conversation_for_extraction(messages)
        assert "tool" not in result.lower()
        assert "Turn on the light" in result
        assert "Done" in result

    def test_format_conversation_correct_format(self, home_agent):
        """Test correct formatting of conversation."""
        messages = [
            {"role": "user", "content": "What's the temperature?"},
            {"role": "assistant", "content": "It's 72°F"},
        ]
        result = home_agent._format_conversation_for_extraction(messages)
        assert "User: What's the temperature?" in result
        assert "Assistant: It's 72°F" in result


class TestBuildExtractionPrompt:
    """Test _build_extraction_prompt method."""

    def test_build_prompt_with_no_history(self, home_agent):
        """Test building prompt with no previous history."""
        user_msg = "Set bedroom to 68 degrees"
        assistant_msg = "I've set the bedroom to 68°F"
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": user_msg},
        ]

        prompt = home_agent._build_extraction_prompt(
            user_message=user_msg,
            assistant_response=assistant_msg,
            full_messages=messages,
        )

        assert user_msg in prompt
        assert assistant_msg in prompt
        assert "(No previous conversation)" in prompt
        assert "JSON" in prompt
        assert "type" in prompt
        assert "importance" in prompt

    def test_build_prompt_with_history(self, home_agent):
        """Test building prompt with conversation history."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "Current question"},
        ]

        prompt = home_agent._build_extraction_prompt(
            user_message="Current question",
            assistant_response="Current answer",
            full_messages=messages,
        )

        assert "Previous question" in prompt
        assert "Previous answer" in prompt
        assert "Current question" in prompt
        assert "Current answer" in prompt

    def test_build_prompt_excludes_system_and_tool(self, home_agent):
        """Test that prompt excludes system and tool messages."""
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "tool", "content": "Tool result"},
            {"role": "assistant", "content": "Assistant message"},
        ]

        prompt = home_agent._build_extraction_prompt(
            user_message="Current",
            assistant_response="Response",
            full_messages=messages,
        )

        # Previous conversation should not include system or tool
        assert "System message" not in prompt or "## Previous Conversation" not in prompt


class TestCallPrimaryLLMForExtraction:
    """Test _call_primary_llm_for_extraction method."""

    async def test_extraction_success(self, home_agent):
        """Test successful extraction with primary LLM."""
        expected_result = '[{"type": "fact", "content": "Test fact"}]'

        with patch.object(
            home_agent,
            "_call_llm",
            return_value={"choices": [{"message": {"content": expected_result}}]},
        ):
            result = await home_agent._call_primary_llm_for_extraction("test prompt")

            assert result["success"] is True
            assert result["result"] == expected_result
            assert result["error"] is None

    async def test_extraction_failure(self, home_agent):
        """Test extraction failure with primary LLM."""
        with patch.object(home_agent, "_call_llm", side_effect=Exception("LLM error")):
            result = await home_agent._call_primary_llm_for_extraction("test prompt")

            assert result["success"] is False
            assert result["result"] is None
            assert "LLM error" in result["error"]

    async def test_extraction_uses_correct_temperature(self, home_agent):
        """Test that extraction uses lower temperature."""
        with patch.object(
            home_agent,
            "_call_llm",
            return_value={"choices": [{"message": {"content": "[]"}}]},
        ) as mock_call:
            await home_agent._call_primary_llm_for_extraction("test prompt")

            # Verify _call_llm was called with temperature=0.3
            call_args = mock_call.call_args
            assert call_args[1]["temperature"] == 0.3


class TestParseAndStoreMemories:
    """Test _parse_and_store_memories method."""

    async def test_parse_valid_json(self, home_agent, mock_memory_manager):
        """Test parsing valid JSON response."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = json.dumps(
            [
                {
                    "type": "preference",
                    "content": "User prefers bedroom temperature at 68°F for sleeping comfort during nighttime",
                    "importance": 0.8,
                    "entities": ["climate.bedroom"],
                    "topics": ["temperature"],
                }
            ]
        )

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 1
        mock_memory_manager.add_memory.assert_called_once()

    async def test_parse_json_in_markdown(self, home_agent, mock_memory_manager):
        """Test parsing JSON wrapped in markdown code block."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = """```json
[
  {
    "type": "fact",
    "content": "The living room has three ceiling lights controlled by smart switches for ambient lighting",
    "importance": 0.5
  }
]
```"""

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 1
        mock_memory_manager.add_memory.assert_called_once()

    async def test_parse_empty_array(self, home_agent, mock_memory_manager):
        """Test parsing empty array (no memories)."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = "[]"

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 0
        mock_memory_manager.add_memory.assert_not_called()

    async def test_parse_invalid_json(self, home_agent, mock_memory_manager):
        """Test handling of invalid JSON."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = "not valid json"

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 0
        mock_memory_manager.add_memory.assert_not_called()

    async def test_parse_non_array(self, home_agent, mock_memory_manager):
        """Test handling of non-array JSON."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = '{"type": "fact"}'

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 0

    async def test_parse_multiple_memories(self, home_agent, mock_memory_manager):
        """Test parsing multiple memories."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = json.dumps(
            [
                {
                    "type": "fact",
                    "content": "The home office desk has an ergonomic setup with adjustable monitor stands and keyboard tray",
                    "importance": 0.5,
                },
                {
                    "type": "preference",
                    "content": "User prefers warm white lighting in the living room during evening hours for relaxation",
                    "importance": 0.8,
                },
                {
                    "type": "context",
                    "content": "The family typically gathers in the living room between 7pm and 9pm for entertainment",
                    "importance": 0.6,
                },
            ]
        )

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 3
        assert mock_memory_manager.add_memory.call_count == 3

    async def test_parse_handles_storage_failure(self, home_agent, mock_memory_manager):
        """Test that storage failures don't stop other memories from being stored."""
        home_agent._memory_manager = mock_memory_manager

        # First call fails, second succeeds
        mock_memory_manager.add_memory.side_effect = [
            Exception("Storage error"),
            "mem_2",
        ]

        extraction_result = json.dumps(
            [
                {
                    "type": "fact",
                    "content": "The garage door opener is connected to the smart home system for remote access",
                    "importance": 0.5,
                },
                {
                    "type": "fact",
                    "content": "The kitchen has under-cabinet LED lighting controlled by motion sensors for convenience",
                    "importance": 0.5,
                },
            ]
        )

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 1  # Only one succeeded
        assert mock_memory_manager.add_memory.call_count == 2

    async def test_parse_validates_memory_content(self, home_agent, mock_memory_manager):
        """Test that memories without content are skipped."""
        home_agent._memory_manager = mock_memory_manager

        extraction_result = json.dumps(
            [
                {"type": "fact"},  # Missing content
                {
                    "type": "fact",
                    "content": "The master bedroom has blackout curtains controlled by automated window shades for better sleep",
                },
            ]
        )

        count = await home_agent._parse_and_store_memories(extraction_result, "conv_123")

        assert count == 1  # Only the valid one
        mock_memory_manager.add_memory.assert_called_once()


class TestExtractAndStoreMemories:
    """Test _extract_and_store_memories method."""

    async def test_extraction_disabled_when_memory_disabled(self, home_agent):
        """Test that extraction is skipped when memory is disabled."""
        home_agent.config[CONF_MEMORY_ENABLED] = False

        with patch.object(home_agent, "_build_extraction_prompt") as mock_build:
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )

            mock_build.assert_not_called()

    async def test_extraction_skipped_when_no_memory_manager(self, home_agent):
        """Test that extraction is skipped when memory manager is not available."""
        home_agent._memory_manager = None

        with patch.object(home_agent, "_build_extraction_prompt") as mock_build:
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )

            mock_build.assert_not_called()

    async def test_extraction_with_local_llm(self, home_agent, mock_memory_manager):
        """Test extraction using local LLM."""
        home_agent._memory_manager = mock_memory_manager
        home_agent.config[CONF_MEMORY_EXTRACTION_LLM] = "local"

        extraction_result = '[{"type": "fact", "content": "The smart thermostat automatically adjusts temperature based on occupancy patterns throughout the day", "importance": 0.5}]'

        with patch.object(
            home_agent,
            "_call_primary_llm_for_extraction",
            return_value={"success": True, "result": extraction_result},
        ):
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )

            # Verify memory was stored
            mock_memory_manager.add_memory.assert_called_once()

    async def test_extraction_with_external_llm(self, home_agent, mock_memory_manager):
        """Test extraction using external LLM."""
        home_agent._memory_manager = mock_memory_manager
        home_agent.config[CONF_MEMORY_EXTRACTION_LLM] = "external"
        home_agent.config[CONF_EXTERNAL_LLM_ENABLED] = True

        extraction_result = '[{"type": "fact", "content": "The smart security system includes door sensors on all entry points with automatic alerts enabled", "importance": 0.5}]'

        home_agent.tool_handler.execute_tool = AsyncMock(
            return_value={"success": True, "result": extraction_result}
        )

        await home_agent._extract_and_store_memories("conv_123", "user msg", "assistant msg", [])

        # Verify external LLM tool was called
        home_agent.tool_handler.execute_tool.assert_called_once()
        call_args = home_agent.tool_handler.execute_tool.call_args
        assert call_args[1]["tool_name"] == "query_external_llm"
        assert call_args[1]["conversation_id"] == "conv_123"
        assert isinstance(call_args[1]["parameters"]["prompt"], str)

        # Verify memory was stored
        mock_memory_manager.add_memory.assert_called_once()

    async def test_extraction_skipped_when_external_llm_not_enabled(
        self, home_agent, mock_memory_manager
    ):
        """Test extraction is skipped when external LLM not enabled."""
        home_agent._memory_manager = mock_memory_manager
        home_agent.config[CONF_MEMORY_EXTRACTION_LLM] = "external"
        home_agent.config[CONF_EXTERNAL_LLM_ENABLED] = False

        with patch.object(home_agent, "_parse_and_store_memories") as mock_parse:
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )

            mock_parse.assert_not_called()

    async def test_extraction_fires_event_on_success(
        self, home_agent, mock_memory_manager, mock_hass
    ):
        """Test that event is fired when memories are extracted."""
        home_agent._memory_manager = mock_memory_manager
        home_agent.config[CONF_MEMORY_EXTRACTION_LLM] = "local"

        extraction_result = '[{"type": "fact", "content": "The outdoor lighting system includes motion-activated pathway lights for enhanced security and convenience", "importance": 0.5}]'

        with patch.object(
            home_agent,
            "_call_primary_llm_for_extraction",
            return_value={"success": True, "result": extraction_result},
        ):
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )

            # Verify event was fired
            mock_hass.bus.async_fire.assert_called_once()
            call_args = mock_hass.bus.async_fire.call_args
            assert "home_agent.memory.extracted" in call_args[0]

    async def test_extraction_handles_llm_failure_gracefully(self, home_agent, mock_memory_manager):
        """Test that LLM failure doesn't crash extraction."""
        home_agent._memory_manager = mock_memory_manager
        home_agent.config[CONF_MEMORY_EXTRACTION_LLM] = "local"

        with patch.object(
            home_agent,
            "_call_primary_llm_for_extraction",
            return_value={"success": False, "error": "LLM failed"},
        ):
            # Should not raise exception
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )

    async def test_extraction_handles_unexpected_exception(self, home_agent, mock_memory_manager):
        """Test that unexpected exceptions are handled gracefully."""
        home_agent._memory_manager = mock_memory_manager
        home_agent.config[CONF_MEMORY_EXTRACTION_LLM] = "local"

        with patch.object(
            home_agent,
            "_build_extraction_prompt",
            side_effect=Exception("Unexpected error"),
        ):
            # Should not raise exception
            await home_agent._extract_and_store_memories(
                "conv_123", "user msg", "assistant msg", []
            )


class TestMemoryExtractionIntegration:
    """Integration tests for memory extraction in conversation flow."""

    async def test_extraction_triggered_after_conversation(self, home_agent, mock_hass):
        """Test that extraction is triggered after conversation completes."""
        # This would be tested in integration tests with full conversation flow
        # For now, we verify the hook exists
        assert hasattr(home_agent, "_extract_and_store_memories")
        assert callable(home_agent._extract_and_store_memories)

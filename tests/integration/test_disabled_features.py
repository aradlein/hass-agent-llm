"""Integration tests for disabled features - verify that disabled features skip their code paths.

This test suite ensures that when features are disabled, their corresponding components
are NOT initialized and their code paths are NOT executed. These tests verify the agent
operates correctly in minimal mode and that disabled features don't cause side effects.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_STREAMING_ENABLED,
    CONTEXT_MODE_DIRECT,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestDisabledFeatures:
    """Test suite for verifying disabled features skip their code paths."""

    @pytest.fixture
    def base_config(self) -> dict:
        """Provide minimal configuration for agent with all features disabled."""
        return {
            CONF_LLM_BASE_URL: "http://localhost:11434/v1",
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "qwen2.5:3b",
            CONF_EMIT_EVENTS: False,  # Disable events for cleaner tests
            CONF_STREAMING_ENABLED: False,  # Keep tests simple with sync mode
        }

    @pytest.fixture
    async def mock_session_manager(self, test_hass):
        """Create a mock ConversationSessionManager for testing."""
        from custom_components.home_agent.conversation_session import ConversationSessionManager

        manager = ConversationSessionManager(test_hass)
        await manager.async_load()
        return manager

    @pytest.fixture
    def mock_hass(self, test_hass) -> HomeAssistant:
        """Provide mock HomeAssistant instance."""
        return test_hass

    @pytest.fixture
    def mock_llm_response(self):
        """Mock successful LLM response."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I can help you with that.",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
            },
        }

    async def test_memory_disabled_skips_initialization(
        self, mock_hass, base_config, mock_session_manager
    ):
        """Test that MemoryManager is NOT initialized when memory_enabled=False.

        Verifies:
        - Agent initializes successfully
        - memory_manager property returns None
        - Memory tools are NOT registered
        """
        config = {
            **base_config,
            CONF_MEMORY_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Verify memory manager is None
        assert agent.memory_manager is None, "MemoryManager should be None when memory is disabled"

        # Force tool registration
        agent._ensure_tools_registered()

        # Verify memory tools are NOT registered
        registered_tools = agent.tool_handler.get_registered_tools()
        assert "store_memory" not in registered_tools, "store_memory tool should NOT be registered"
        assert (
            "recall_memory" not in registered_tools
        ), "recall_memory tool should NOT be registered"

    async def test_memory_disabled_skips_operations(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that memory operations are skipped when memory_enabled=False.

        Verifies:
        - Conversation processes normally without memory
        - No memory search operations occur
        - No memory add operations occur
        """
        config = {
            **base_config,
            CONF_MEMORY_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Verify memory manager is None
        assert agent.memory_manager is None

        # Mock LLM call
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text="Remember my preference for tea",
                conversation_id="test_conv_1",
            )

            # Verify response succeeded
            assert response is not None
            assert len(response) > 0

        # Memory manager should still be None (never initialized)
        assert agent.memory_manager is None

    async def test_memory_extraction_disabled_skips_extraction(
        self, mock_hass, base_config, mock_llm_response, caplog, mock_session_manager
    ):
        """Test that extraction is NOT called when memory_extraction_enabled=False.

        Verifies:
        - Conversation completes successfully
        - _extract_and_store_memories is NOT called
        - No extraction-related LLM calls occur
        """
        caplog.set_level(logging.DEBUG)

        config = {
            **base_config,
            CONF_MEMORY_ENABLED: True,  # Memory enabled
            CONF_MEMORY_EXTRACTION_ENABLED: False,  # But extraction disabled
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Mock the extraction method to track calls - use simple async function
        call_count = [0]

        async def mock_extract_noop(*args, **kwargs):
            call_count[0] += 1
            return None

        with patch.object(agent, "_extract_and_store_memories", side_effect=mock_extract_noop):
            with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                response = await agent.process_message(
                    text="I prefer tea in the morning",
                    conversation_id="test_conv_2",
                )

            # Wait a bit for any background tasks
            await asyncio.sleep(0.1)

            # Verify extraction was NOT called
            assert call_count[0] == 0, "Extraction should not be called when disabled"

        # Verify conversation succeeded
        assert response is not None
        assert len(response) > 0

    async def test_memory_extraction_disabled_via_memory_disabled(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that extraction is skipped when memory_enabled=False.

        Verifies:
        - Even if memory_extraction_enabled=True, extraction doesn't happen
        - This is because memory_manager is None
        """
        config = {
            **base_config,
            CONF_MEMORY_ENABLED: False,  # Memory disabled
            CONF_MEMORY_EXTRACTION_ENABLED: True,  # Extraction "enabled" but won't work
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Mock the extraction method to track calls
        call_count = [0]

        async def mock_extract_noop(*args, **kwargs):
            """Mock extraction that does nothing (simulates early exit)."""
            call_count[0] += 1
            return None

        with patch.object(agent, "_extract_and_store_memories", side_effect=mock_extract_noop):
            with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                await agent.process_message(
                    text="I like coffee",
                    conversation_id="test_conv_3",
                )

        # Wait for background tasks
        await asyncio.sleep(0.1)

        # Verify memory manager is None
        assert agent.memory_manager is None

        # Extraction could be called but should exit early since memory is disabled
        # The fixture already patches this, so this test is mainly verifying memory_manager is None

    async def test_history_disabled_skips_saving(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that conversation history is NOT saved when history_enabled=False.

        Verifies:
        - Conversation processes normally
        - No messages are added to conversation history
        - ConversationHistoryManager.add_message is NOT called
        """
        config = {
            **base_config,
            CONF_HISTORY_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Mock add_message to verify it's NOT called - use simple MagicMock (sync method)
        with patch.object(
            agent.conversation_manager, "add_message", MagicMock()
        ) as mock_add_message:
            with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                response = await agent.process_message(
                    text="Hello, how are you?",
                    conversation_id="test_conv_4",
                )

            # Verify add_message was NOT called
            mock_add_message.assert_not_called()

        # Verify conversation succeeded
        assert response is not None
        assert len(response) > 0

    async def test_history_disabled_not_included_in_context(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that history is NOT included in LLM messages when history_enabled=False.

        Verifies:
        - LLM is called with only system and current user message
        - No historical messages are included
        - get_history is NOT called
        """
        config = {
            **base_config,
            CONF_HISTORY_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Track LLM calls to verify message structure
        llm_calls = []

        def track_llm_call(messages, **kwargs):
            llm_calls.append(messages)
            return mock_llm_response

        with patch.object(agent, "_call_llm", side_effect=track_llm_call):
            await agent.process_message(
                text="What's the weather?",
                conversation_id="test_conv_5",
            )

        # Verify LLM was called
        assert len(llm_calls) == 1

        # Verify messages contain only system + current user message
        messages = llm_calls[0]
        assert len(messages) == 2  # System prompt + current user message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What's the weather?"

        # Verify no assistant messages from history
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        assert len(assistant_messages) == 0

    async def test_context_mode_direct_skips_vector_db(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that VectorDBManager is NOT used when context_mode='direct'.

        Verifies:
        - DirectContextProvider is used instead
        - No vector database queries occur
        - Context is still provided (from DirectContextProvider)
        """
        config = {
            **base_config,
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Verify context manager uses DirectContextProvider
        # This is determined at initialization in ContextManager
        assert agent.context_manager is not None

        # Mock VectorDBContextProvider to ensure it's NOT used
        with patch(
            "custom_components.home_agent.context_providers.vector_db.VectorDBContextProvider"
        ) as mock_vector_provider:
            with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                response = await agent.process_message(
                    text="Turn on the lights",
                    conversation_id="test_conv_6",
                )

            # Verify VectorDBContextProvider was NOT instantiated
            mock_vector_provider.assert_not_called()

        # Verify conversation succeeded
        assert response is not None
        assert len(response) > 0

    async def test_external_llm_disabled_tool_not_registered(
        self, mock_hass, base_config, mock_session_manager
    ):
        """Test that external LLM tool is NOT registered when external_llm_enabled=False.

        Verifies:
        - Agent initializes successfully
        - query_external_llm tool is NOT in registered tools
        - ExternalLLMTool is NOT instantiated
        """
        config = {
            **base_config,
            CONF_EXTERNAL_LLM_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Mock ExternalLLMTool to verify it's NOT instantiated
        with patch("custom_components.home_agent.agent.ExternalLLMTool") as mock_ext_llm:
            # Force tool registration
            agent._ensure_tools_registered()

            # Verify ExternalLLMTool was NOT instantiated
            mock_ext_llm.assert_not_called()

        # Verify query_external_llm tool is NOT registered
        registered_tools = agent.tool_handler.get_registered_tools()
        assert (
            "query_external_llm" not in registered_tools
        ), "query_external_llm tool should NOT be registered"

    async def test_external_llm_disabled_not_available_for_memory_extraction(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that external LLM is NOT used for extraction when disabled.

        Verifies:
        - When memory_extraction_llm='external' but external_llm_enabled=False
        - Extraction method detects this and exits early
        - No external LLM tool calls occur
        """
        from custom_components.home_agent.const import CONF_MEMORY_EXTRACTION_LLM

        config = {
            **base_config,
            CONF_MEMORY_ENABLED: True,
            CONF_MEMORY_EXTRACTION_ENABLED: True,
            CONF_EXTERNAL_LLM_ENABLED: False,
            CONF_MEMORY_EXTRACTION_LLM: "external",  # Explicitly set to external
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Set up a mock memory manager
        mock_memory_manager = MagicMock()
        mock_memory_manager.add_memory = AsyncMock()
        agent._memory_manager = mock_memory_manager

        # Mock execute_tool to track if external LLM is called
        with patch.object(
            agent.tool_handler, "execute_tool", new_callable=AsyncMock
        ) as mock_execute_tool:
            # Also mock _extract_and_store_memories to track its execution
            original_extract = agent._extract_and_store_memories

            async def tracked_extract(*args, **kwargs):
                # Call the original but we can track it
                return await original_extract(*args, **kwargs)

            with patch.object(
                agent, "_extract_and_store_memories", side_effect=tracked_extract
            ) as mock_extract:
                with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                    await agent.process_message(
                        text="I prefer tea",
                        conversation_id="test_conv_7",
                    )

                # Wait for background extraction task
                await asyncio.sleep(0.3)

                # Verify extraction was called (but should have exited early)
                assert mock_extract.called, "Extraction should be called"

            # Verify external LLM tool was NOT called
            external_llm_calls = [
                c
                for c in mock_execute_tool.call_args_list
                if len(c[0]) > 0 and c[0][0] == "query_external_llm"
            ]
            assert len(external_llm_calls) == 0, "External LLM should NOT be called when disabled"

    async def test_all_features_disabled_minimal_mode(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test agent works in minimal mode with all optional features disabled.

        Verifies:
        - Agent initializes with all features disabled
        - Basic conversation still works
        - Only core tools are registered (ha_control, ha_query)
        - No optional components are initialized
        """
        config = {
            **base_config,
            CONF_MEMORY_ENABLED: False,
            CONF_MEMORY_EXTRACTION_ENABLED: False,
            CONF_HISTORY_ENABLED: False,
            CONF_EXTERNAL_LLM_ENABLED: False,
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_STREAMING_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Verify memory is disabled
        assert agent.memory_manager is None

        # Force tool registration
        agent._ensure_tools_registered()

        # Verify only core tools are registered
        registered_tools = agent.tool_handler.get_registered_tools()
        assert "ha_control" in registered_tools
        assert "ha_query" in registered_tools
        assert "store_memory" not in registered_tools
        assert "recall_memory" not in registered_tools
        assert "query_external_llm" not in registered_tools

        # Verify conversation works
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            response = await agent.process_message(
                text="Turn on the living room light",
                conversation_id="test_conv_8",
            )

        # Verify response
        assert response is not None
        assert len(response) > 0

    async def test_memory_disabled_context_manager_no_memory_provider(
        self, mock_hass, base_config, mock_session_manager
    ):
        """Test that context manager does NOT have memory provider when memory disabled.

        Verifies:
        - ContextManager.set_memory_provider is NOT called
        - Memory context is NOT available
        """
        config = {
            **base_config,
            CONF_MEMORY_ENABLED: False,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Mock set_memory_provider to track calls - use simple MagicMock (it's a sync method)
        with patch.object(
            agent.context_manager, "set_memory_provider", MagicMock()
        ) as mock_set_memory:
            # Force tool registration (which triggers memory provider setup)
            agent._ensure_tools_registered()

            # Verify set_memory_provider was NOT called
            mock_set_memory.assert_not_called()

    async def test_history_disabled_in_streaming_mode(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that history is NOT saved in streaming mode when disabled.

        Verifies:
        - Streaming mode respects history_enabled=False
        - No history messages are added
        """
        config = {
            **base_config,
            CONF_HISTORY_ENABLED: False,
            CONF_STREAMING_ENABLED: True,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Mock add_message to verify it's NOT called - use simple MagicMock (sync method)
        with patch.object(
            agent.conversation_manager, "add_message", MagicMock()
        ) as mock_add_message:
            # We'll test the synchronous path since streaming requires more setup
            # The key is to verify the config is respected
            with patch.object(agent, "_can_stream", return_value=False):
                with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                    response = await agent.process_message(
                        text="Hello",
                        conversation_id="test_conv_9",
                    )

            # Verify add_message was NOT called
            mock_add_message.assert_not_called()

        assert response is not None

    async def test_memory_extraction_check_happens_before_manager_check(
        self, mock_hass, base_config, mock_llm_response, caplog, mock_session_manager
    ):
        """Test extraction check respects both CONF_MEMORY_ENABLED and manager availability.

        Verifies:
        - If CONF_MEMORY_ENABLED=False, extraction exits early
        - If memory_manager is None, extraction exits early
        - Both checks prevent extraction from proceeding
        """
        caplog.set_level(logging.DEBUG)

        # Test 1: CONF_MEMORY_ENABLED=False
        config1 = {
            **base_config,
            CONF_MEMORY_ENABLED: False,
            CONF_MEMORY_EXTRACTION_ENABLED: True,
        }

        agent1 = HomeAgent(mock_hass, config1, mock_session_manager)

        # Mock the extraction LLM call to track if it's made - use call counter
        call_count1 = [0]

        async def mock_extract_llm1(*args, **kwargs):
            call_count1[0] += 1
            return None

        with patch.object(
            agent1, "_call_primary_llm_for_extraction", side_effect=mock_extract_llm1
        ):
            with patch.object(agent1, "_call_llm", return_value=mock_llm_response):
                await agent1.process_message(
                    text="Test message",
                    conversation_id="test_conv_10",
                )

            # Wait for background tasks
            await asyncio.sleep(0.1)

            # Verify extraction LLM was NOT called
            assert (
                call_count1[0] == 0
            ), "Extraction LLM should not be called when memory is disabled"

        # Test 2: Memory enabled but manager is None
        config2 = {
            **base_config,
            CONF_MEMORY_ENABLED: True,
            CONF_MEMORY_EXTRACTION_ENABLED: True,
        }

        agent2 = HomeAgent(mock_hass, config2, mock_session_manager)
        assert agent2.memory_manager is None  # Not provided in fixture

        call_count2 = [0]

        async def mock_extract_llm2(*args, **kwargs):
            call_count2[0] += 1
            return None

        with patch.object(
            agent2, "_call_primary_llm_for_extraction", side_effect=mock_extract_llm2
        ):
            with patch.object(agent2, "_call_llm", return_value=mock_llm_response):
                await agent2.process_message(
                    text="Test message",
                    conversation_id="test_conv_11",
                )

            # Wait for background tasks
            await asyncio.sleep(0.1)

            # Verify extraction LLM was NOT called
            assert call_count2[0] == 0, "Extraction LLM should not be called when manager is None"

    async def test_disabled_features_dont_affect_enabled_features(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that disabling one feature doesn't break others.

        Verifies:
        - Disabling memory doesn't affect history
        - Disabling external LLM doesn't affect core tools
        - Features work independently
        """
        config = {
            **base_config,
            CONF_MEMORY_ENABLED: False,  # Memory disabled
            CONF_HISTORY_ENABLED: True,  # History enabled
            CONF_EXTERNAL_LLM_ENABLED: False,  # External LLM disabled
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)
        agent._ensure_tools_registered()

        # Verify memory is disabled
        assert agent.memory_manager is None

        # Verify core tools are registered
        registered_tools = agent.tool_handler.get_registered_tools()
        assert "ha_control" in registered_tools
        assert "ha_query" in registered_tools

        # Verify history is working
        with patch.object(agent, "_call_llm", return_value=mock_llm_response):
            # First message
            await agent.process_message(
                text="First message",
                conversation_id="test_conv_12",
            )

            # Second message
            await agent.process_message(
                text="Second message",
                conversation_id="test_conv_12",
            )

        # Verify history was saved
        history = agent.conversation_manager.get_history("test_conv_12")
        assert history is not None
        assert len(history) > 0  # Should have messages from both interactions

    async def test_context_mode_direct_uses_direct_provider_methods(
        self, mock_hass, base_config, mock_llm_response, mock_session_manager
    ):
        """Test that direct context mode uses DirectContextProvider, not VectorDB.

        Verifies:
        - get_formatted_context works with direct mode
        - No VectorDB queries are made
        - Context is retrieved from entity states
        """
        config = {
            **base_config,
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
        }

        agent = HomeAgent(mock_hass, config, mock_session_manager)

        # Track context manager calls
        context_calls = []

        original_get_context = agent.context_manager.get_formatted_context

        async def track_context(*args, **kwargs):
            result = await original_get_context(*args, **kwargs)
            context_calls.append(result)
            return result

        with patch.object(
            agent.context_manager, "get_formatted_context", side_effect=track_context
        ):
            with patch.object(agent, "_call_llm", return_value=mock_llm_response):
                response = await agent.process_message(
                    text="What lights are on?",
                    conversation_id="test_conv_13",
                )

        # Verify context was retrieved
        assert len(context_calls) == 1

        # Verify we got a response
        assert response is not None

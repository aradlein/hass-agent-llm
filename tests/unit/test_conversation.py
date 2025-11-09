"""Unit tests for conversation history management."""

import pytest

from custom_components.home_agent.conversation import ConversationHistoryManager


class TestConversationHistoryManager:
    """Test ConversationHistoryManager class."""

    def test_initialization_defaults(self):
        """Test manager initializes with default values."""
        manager = ConversationHistoryManager()
        assert manager._max_messages == 10
        assert manager._max_tokens is None
        assert len(manager._histories) == 0

    def test_initialization_custom_limits(self):
        """Test manager initializes with custom limits."""
        manager = ConversationHistoryManager(max_messages=5, max_tokens=1000)
        assert manager._max_messages == 5
        assert manager._max_tokens == 1000

    def test_add_message(self):
        """Test adding messages to conversation history."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")

        assert "conv_123" in manager._histories
        assert len(manager._histories["conv_123"]) == 1
        assert manager._histories["conv_123"][0]["role"] == "user"
        assert manager._histories["conv_123"][0]["content"] == "Hello"
        assert "timestamp" in manager._histories["conv_123"][0]

    def test_add_multiple_messages(self):
        """Test adding multiple messages to same conversation."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi there!")
        manager.add_message("conv_123", "user", "How are you?")

        assert len(manager._histories["conv_123"]) == 3
        assert manager._histories["conv_123"][1]["role"] == "assistant"
        assert manager._histories["conv_123"][2]["content"] == "How are you?"

    def test_add_message_empty_conversation_id(self):
        """Test adding message with empty conversation_id is ignored."""
        manager = ConversationHistoryManager()
        manager.add_message("", "user", "Hello")

        assert len(manager._histories) == 0

    def test_add_message_empty_content(self):
        """Test adding message with empty content is ignored."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "")

        assert len(manager._histories) == 0

    def test_add_messages_to_different_conversations(self):
        """Test messages are stored separately per conversation."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello from 123")
        manager.add_message("conv_456", "user", "Hello from 456")

        assert len(manager._histories) == 2
        assert len(manager._histories["conv_123"]) == 1
        assert len(manager._histories["conv_456"]) == 1
        assert manager._histories["conv_123"][0]["content"] == "Hello from 123"
        assert manager._histories["conv_456"][0]["content"] == "Hello from 456"

    def test_get_history_empty(self):
        """Test getting history for non-existent conversation."""
        manager = ConversationHistoryManager()
        history = manager.get_history("conv_123")

        assert history == []

    def test_get_history_basic(self):
        """Test retrieving conversation history."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi!")

        history = manager.get_history("conv_123")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_get_history_with_message_limit(self):
        """Test history respects max_messages limit."""
        manager = ConversationHistoryManager(max_messages=2)

        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_123", role, f"Message {i}")

        history = manager.get_history("conv_123")

        # Should only get the last 2 messages
        assert len(history) == 2
        assert history[0]["content"] == "Message 3"
        assert history[1]["content"] == "Message 4"

    def test_get_history_override_message_limit(self):
        """Test overriding message limit in get_history."""
        manager = ConversationHistoryManager(max_messages=10)

        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_123", role, f"Message {i}")

        # Override to get only last 2
        history = manager.get_history("conv_123", max_messages=2)

        assert len(history) == 2
        assert history[0]["content"] == "Message 3"
        assert history[1]["content"] == "Message 4"

    def test_get_history_with_token_limit(self):
        """Test history respects max_tokens limit."""
        manager = ConversationHistoryManager(max_tokens=100)

        # Add messages with varying lengths that fit within limit
        manager.add_message("conv_123", "user", "Short")
        manager.add_message("conv_123", "assistant", "Medium length message")
        manager.add_message("conv_123", "user", "Another short one")

        history = manager.get_history("conv_123")

        # Should keep all messages as they fit within limit
        estimated_tokens = manager.estimate_tokens(history)
        assert estimated_tokens <= 100
        assert len(history) == 3

    def test_get_history_override_token_limit(self):
        """Test overriding token limit in get_history."""
        manager = ConversationHistoryManager()

        # Add long messages
        for i in range(3):
            manager.add_message("conv_123", "user", "A" * 200)

        # Set a tight token limit
        history = manager.get_history("conv_123", max_tokens=100)

        estimated_tokens = manager.estimate_tokens(history)
        assert estimated_tokens <= 100

    def test_clear_history(self):
        """Test clearing specific conversation history."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_456", "user", "Hi")

        manager.clear_history("conv_123")

        assert "conv_123" not in manager._histories
        assert "conv_456" in manager._histories
        assert len(manager._histories["conv_456"]) == 1

    def test_clear_nonexistent_history(self):
        """Test clearing non-existent conversation doesn't error."""
        manager = ConversationHistoryManager()
        # Should not raise an error
        manager.clear_history("conv_nonexistent")

    def test_clear_all(self):
        """Test clearing all conversation histories."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_456", "user", "Hi")
        manager.add_message("conv_789", "user", "Hey")

        manager.clear_all()

        assert len(manager._histories) == 0
        assert manager.get_all_conversation_ids() == []

    def test_get_all_conversation_ids(self):
        """Test retrieving all conversation IDs."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_456", "user", "Hi")

        ids = manager.get_all_conversation_ids()

        assert len(ids) == 2
        assert "conv_123" in ids
        assert "conv_456" in ids

    def test_get_all_conversation_ids_empty(self):
        """Test getting conversation IDs when none exist."""
        manager = ConversationHistoryManager()
        ids = manager.get_all_conversation_ids()

        assert ids == []

    def test_get_message_count(self):
        """Test getting message count for a conversation."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi")
        manager.add_message("conv_123", "user", "How are you?")

        count = manager.get_message_count("conv_123")

        assert count == 3

    def test_get_message_count_empty(self):
        """Test getting message count for non-existent conversation."""
        manager = ConversationHistoryManager()
        count = manager.get_message_count("conv_123")

        assert count == 0

    def test_estimate_tokens_single_message(self):
        """Test token estimation for single message."""
        manager = ConversationHistoryManager()
        messages = [{"role": "user", "content": "Hello"}]

        tokens = manager.estimate_tokens(messages)

        # "user" (4) + "Hello" (5) + overhead (20) = 29 chars / 4 = ~7 tokens
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_multiple_messages(self):
        """Test token estimation for multiple messages."""
        manager = ConversationHistoryManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        tokens = manager.estimate_tokens(messages)

        assert tokens > 0
        # Should be more than single message
        single_tokens = manager.estimate_tokens([messages[0]])
        assert tokens > single_tokens

    def test_estimate_tokens_empty_messages(self):
        """Test token estimation for empty message list."""
        manager = ConversationHistoryManager()
        tokens = manager.estimate_tokens([])

        assert tokens == 0

    def test_estimate_tokens_long_content(self):
        """Test token estimation scales with content length."""
        manager = ConversationHistoryManager()
        short_message = [{"role": "user", "content": "Hi"}]
        long_message = [{"role": "user", "content": "A" * 1000}]

        short_tokens = manager.estimate_tokens(short_message)
        long_tokens = manager.estimate_tokens(long_message)

        assert long_tokens > short_tokens

    def test_truncate_by_tokens_no_truncation_needed(self):
        """Test truncate by tokens when content fits."""
        manager = ConversationHistoryManager()
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        truncated = manager._truncate_by_tokens(history, 1000)

        assert len(truncated) == 2
        assert truncated == history

    def test_truncate_by_tokens_removes_oldest(self):
        """Test truncate by tokens removes oldest messages first."""
        manager = ConversationHistoryManager()
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]

        # Very tight limit - should only keep most recent
        truncated = manager._truncate_by_tokens(history, 50)

        # Should keep at least the most recent message
        assert len(truncated) >= 1
        assert truncated[-1]["content"] == "Response 2"

    def test_truncate_by_tokens_keeps_at_least_one(self):
        """Test truncate by tokens keeps at least one message."""
        manager = ConversationHistoryManager()
        history = [{"role": "user", "content": "A" * 1000}]

        # Even with very small limit, should keep at least one message
        truncated = manager._truncate_by_tokens(history, 10)

        assert len(truncated) == 1
        assert truncated[0]["content"] == "A" * 1000

    def test_truncate_by_tokens_empty_history(self):
        """Test truncate by tokens with empty history."""
        manager = ConversationHistoryManager()
        truncated = manager._truncate_by_tokens([], 100)

        assert truncated == []

    def test_update_limits_max_messages(self):
        """Test updating max_messages limit."""
        manager = ConversationHistoryManager(max_messages=5)

        manager.update_limits(max_messages=20)

        assert manager._max_messages == 20

    def test_update_limits_max_tokens(self):
        """Test updating max_tokens limit."""
        manager = ConversationHistoryManager(max_tokens=1000)

        manager.update_limits(max_tokens=2000)

        assert manager._max_tokens == 2000

    def test_update_limits_both(self):
        """Test updating both limits."""
        manager = ConversationHistoryManager(max_messages=5, max_tokens=1000)

        manager.update_limits(max_messages=10, max_tokens=2000)

        assert manager._max_messages == 10
        assert manager._max_tokens == 2000

    def test_update_limits_none_no_change(self):
        """Test update_limits with None doesn't change values."""
        manager = ConversationHistoryManager(max_messages=5, max_tokens=1000)

        manager.update_limits(max_messages=None, max_tokens=None)

        assert manager._max_messages == 5
        assert manager._max_tokens == 1000

    def test_history_order_preserved(self):
        """Test that message order is preserved chronologically."""
        manager = ConversationHistoryManager()

        messages = [
            ("user", "First"),
            ("assistant", "Second"),
            ("user", "Third"),
            ("assistant", "Fourth"),
        ]

        for role, content in messages:
            manager.add_message("conv_123", role, content)

        history = manager.get_history("conv_123")

        assert len(history) == 4
        for i, (expected_role, expected_content) in enumerate(messages):
            assert history[i]["role"] == expected_role
            assert history[i]["content"] == expected_content

    def test_message_format_openai_compatible(self):
        """Test that message format matches OpenAI format."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")

        history = manager.get_history("conv_123")

        # Should have exactly 'role' and 'content' keys
        assert len(history[0]) == 2
        assert "role" in history[0]
        assert "content" in history[0]
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_concurrent_conversations_isolated(self):
        """Test that multiple conversations remain isolated."""
        manager = ConversationHistoryManager()

        # Add messages to different conversations
        manager.add_message("conv_1", "user", "Hello from 1")
        manager.add_message("conv_2", "user", "Hello from 2")
        manager.add_message("conv_1", "assistant", "Response to 1")
        manager.add_message("conv_2", "assistant", "Response to 2")

        history_1 = manager.get_history("conv_1")
        history_2 = manager.get_history("conv_2")

        # Each should have their own messages
        assert len(history_1) == 2
        assert len(history_2) == 2
        assert history_1[0]["content"] == "Hello from 1"
        assert history_2[0]["content"] == "Hello from 2"

    def test_large_conversation_handling(self):
        """Test handling of large conversations with many messages."""
        manager = ConversationHistoryManager(max_messages=100)

        # Add 500 messages
        for i in range(500):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_large", role, f"Message {i}")

        history = manager.get_history("conv_large")

        # Should only return last 100 due to limit
        assert len(history) == 100
        assert history[-1]["content"] == "Message 499"
        assert history[0]["content"] == "Message 400"

    def test_special_characters_in_content(self):
        """Test handling of special characters in message content."""
        manager = ConversationHistoryManager()

        special_content = "Hello! 你好 مرحبا 🎉 \n\t\\\"'"
        manager.add_message("conv_123", "user", special_content)

        history = manager.get_history("conv_123")

        assert history[0]["content"] == special_content

    def test_very_long_single_message(self):
        """Test handling of very long single messages."""
        manager = ConversationHistoryManager()

        # Create a very long message (100KB)
        long_content = "A" * 100000
        manager.add_message("conv_123", "user", long_content)

        history = manager.get_history("conv_123")

        assert len(history) == 1
        assert history[0]["content"] == long_content

    @pytest.mark.parametrize(
        "max_messages,expected_count",
        [
            (1, 1),
            (5, 5),
            (10, 10),
            (100, 20),  # Added 20 messages, limit is 100, so all 20 returned
        ],
    )
    def test_various_message_limits(self, max_messages, expected_count):
        """Test various max_messages limits."""
        manager = ConversationHistoryManager(max_messages=max_messages)

        # Add 20 messages
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_123", role, f"Message {i}")

        history = manager.get_history("conv_123")

        assert len(history) == min(expected_count, 20)

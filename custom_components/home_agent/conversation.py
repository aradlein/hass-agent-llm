"""Conversation history management for Home Agent.

This module provides conversation history storage and retrieval for maintaining
context across multiple turns in a conversation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

_LOGGER = logging.getLogger(__name__)

# Token estimation: rough estimate of ~4 characters per token
# This is a conservative estimate that works across most models
CHARS_PER_TOKEN = 4


class ConversationHistoryManager:
    """Manage conversation history with token and message limits.

    Stores conversation history per conversation_id and provides methods to:
    - Add messages to conversation history
    - Retrieve recent history with limits
    - Clear specific or all conversations
    - Estimate and manage token usage

    History is stored in memory for Phase 1 (persistence will be added later).
    """

    def __init__(self, max_messages: int = 10, max_tokens: int | None = None) -> None:
        """Initialize the conversation history manager.

        Args:
            max_messages: Maximum number of messages to retain per conversation
            max_tokens: Maximum token count for conversation history (None = no limit)
        """
        self._histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._max_messages = max_messages
        self._max_tokens = max_tokens

        _LOGGER.debug(
            "Initialized ConversationHistoryManager with max_messages=%d, max_tokens=%s",
            max_messages,
            max_tokens,
        )

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            conversation_id: Unique identifier for the conversation
            role: Message role (typically "user" or "assistant")
            content: Message content

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Turn on the lights")
            >>> manager.add_message("conv_123", "assistant", "I've turned on the lights")
        """
        if not conversation_id:
            _LOGGER.warning("Attempted to add message with empty conversation_id")
            return

        if not content:
            _LOGGER.warning("Attempted to add empty message to conversation %s", conversation_id)
            return

        message = {"role": role, "content": content}

        self._histories[conversation_id].append(message)

        _LOGGER.debug(
            "Added %s message to conversation %s (now %d messages)",
            role,
            conversation_id,
            len(self._histories[conversation_id]),
        )

    def get_history(
        self, conversation_id: str, max_messages: int | None = None, max_tokens: int | None = None
    ) -> list[dict[str, str]]:
        """Get conversation history with optional limits.

        Retrieves recent conversation history, applying message and token limits.
        If both limits are specified, the more restrictive one is applied.

        Args:
            conversation_id: Unique identifier for the conversation
            max_messages: Override default max messages limit (None = use default)
            max_tokens: Override default max tokens limit (None = use default)

        Returns:
            List of message dictionaries with 'role' and 'content' keys,
            in chronological order (oldest first)

        Example:
            >>> manager = ConversationHistoryManager(max_messages=10)
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_123", "assistant", "Hi!")
            >>> history = manager.get_history("conv_123")
            >>> len(history)
            2
        """
        if conversation_id not in self._histories:
            _LOGGER.debug("No history found for conversation %s", conversation_id)
            return []

        history = self._histories[conversation_id]

        # Apply message limit
        effective_max_messages = max_messages if max_messages is not None else self._max_messages
        if len(history) > effective_max_messages:
            history = history[-effective_max_messages:]
            _LOGGER.debug(
                "Truncated conversation %s to %d messages (from %d)",
                conversation_id,
                effective_max_messages,
                len(self._histories[conversation_id]),
            )

        # Apply token limit
        effective_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        if effective_max_tokens is not None:
            history = self._truncate_by_tokens(history, effective_max_tokens)

        return history

    def clear_history(self, conversation_id: str) -> None:
        """Clear history for a specific conversation.

        Args:
            conversation_id: Unique identifier for the conversation to clear

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.clear_history("conv_123")
            >>> manager.get_history("conv_123")
            []
        """
        if conversation_id in self._histories:
            message_count = len(self._histories[conversation_id])
            del self._histories[conversation_id]
            _LOGGER.info("Cleared conversation %s (%d messages)", conversation_id, message_count)
        else:
            _LOGGER.debug("Attempted to clear non-existent conversation %s", conversation_id)

    def clear_all(self) -> None:
        """Clear all conversation histories.

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_456", "user", "Hi")
            >>> manager.clear_all()
            >>> len(manager.get_all_conversation_ids())
            0
        """
        conversation_count = len(self._histories)
        total_messages = sum(len(history) for history in self._histories.values())

        self._histories.clear()

        _LOGGER.info(
            "Cleared all conversation histories (%d conversations, %d total messages)",
            conversation_count,
            total_messages,
        )

    def get_all_conversation_ids(self) -> list[str]:
        """Get list of all conversation IDs with stored history.

        Returns:
            List of conversation IDs

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_456", "user", "Hi")
            >>> sorted(manager.get_all_conversation_ids())
            ['conv_123', 'conv_456']
        """
        return list(self._histories.keys())

    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            Number of messages in the conversation

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_123", "assistant", "Hi!")
            >>> manager.get_message_count("conv_123")
            2
        """
        return len(self._histories.get(conversation_id, []))

    def estimate_tokens(self, messages: list[dict[str, str]]) -> int:
        """Estimate token count for a list of messages.

        Uses a conservative estimate of ~4 characters per token, which works
        across most models. For more accurate token counting, consider using
        a model-specific tokenizer (e.g., tiktoken for OpenAI models).

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Estimated token count

        Example:
            >>> manager = ConversationHistoryManager()
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> manager.estimate_tokens(messages) > 0
            True
        """
        total_chars = 0
        for message in messages:
            # Count role characters
            total_chars += len(message.get("role", ""))
            # Count content characters
            total_chars += len(message.get("content", ""))
            # Add overhead for message structure (role/content keys, etc.)
            total_chars += 20

        estimated_tokens = total_chars // CHARS_PER_TOKEN

        _LOGGER.debug(
            "Estimated %d tokens for %d messages (%d chars)",
            estimated_tokens,
            len(messages),
            total_chars,
        )

        return estimated_tokens

    def _truncate_by_tokens(
        self, history: list[dict[str, str]], max_tokens: int
    ) -> list[dict[str, str]]:
        """Truncate history to fit within token limit.

        Removes oldest messages until the history fits within max_tokens.
        Always keeps at least the most recent message pair (user + assistant).

        Args:
            history: Full conversation history
            max_tokens: Maximum token count

        Returns:
            Truncated history that fits within token limit
        """
        if not history:
            return []

        # Start from the end and work backwards
        truncated: list[dict[str, str]] = []
        current_tokens = 0

        for message in reversed(history):
            message_tokens = self.estimate_tokens([message])

            if current_tokens + message_tokens <= max_tokens:
                truncated.insert(0, message)
                current_tokens += message_tokens
            else:
                # Stop adding messages if we've exceeded the limit
                # but ensure we keep at least one message
                if not truncated:
                    truncated.insert(0, message)
                break

        if len(truncated) < len(history):
            _LOGGER.debug(
                "Truncated history from %d to %d messages to fit %d token limit (estimated %d tokens)",
                len(history),
                len(truncated),
                max_tokens,
                current_tokens,
            )

        return truncated

    def update_limits(self, max_messages: int | None = None, max_tokens: int | None = None) -> None:
        """Update the default limits for conversation history.

        Args:
            max_messages: New max messages limit (None = don't change)
            max_tokens: New max tokens limit (None = don't change)

        Example:
            >>> manager = ConversationHistoryManager(max_messages=5)
            >>> manager.update_limits(max_messages=10, max_tokens=2000)
        """
        if max_messages is not None:
            old_max = self._max_messages
            self._max_messages = max_messages
            _LOGGER.info("Updated max_messages from %d to %d", old_max, max_messages)

        if max_tokens is not None:
            old_max_tokens = self._max_tokens
            self._max_tokens = max_tokens
            _LOGGER.info("Updated max_tokens from %s to %d", old_max_tokens, max_tokens)

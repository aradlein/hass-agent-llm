"""Metrics collection for E2E tests.

This module provides classes for collecting and analyzing metrics from
E2E test scenario executions.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation.

    Attributes:
        conversation_id: Unique identifier for the conversation
        total_turns: Number of conversation turns (user + assistant pairs)
        total_duration_ms: Total duration of all turns in milliseconds
        total_tokens: Total tokens used across all turns
        tool_calls: Dictionary mapping tool names to call counts
    """

    conversation_id: str
    total_turns: int = 0
    total_duration_ms: int = 0
    total_tokens: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)

    def add_turn(self, duration_ms: int, tokens: int, tools: list[str]) -> None:
        """Add metrics from a conversation turn.

        Args:
            duration_ms: Duration of the turn in milliseconds
            tokens: Number of tokens used in the turn
            tools: List of tool names called during the turn
        """
        self.total_turns += 1
        self.total_duration_ms += duration_ms
        self.total_tokens += tokens

        for tool in tools:
            self.tool_calls[tool] = self.tool_calls.get(tool, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary representation of metrics
        """
        return {
            "conversation_id": self.conversation_id,
            "total_turns": self.total_turns,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "tool_calls": dict(self.tool_calls),
            "avg_duration_per_turn_ms": (
                self.total_duration_ms / self.total_turns if self.total_turns > 0 else 0
            ),
            "avg_tokens_per_turn": (
                self.total_tokens / self.total_turns if self.total_turns > 0 else 0
            ),
        }


class MetricsCollector:
    """Collects and aggregates metrics across multiple conversations.

    This class provides methods to track conversation metrics, record turns,
    and generate summary statistics.
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._conversations: dict[str, ConversationMetrics] = {}
        self._turn_start_times: dict[str, float] = {}

    def start_conversation(self, conversation_id: str) -> None:
        """Start tracking a new conversation.

        Args:
            conversation_id: Unique identifier for the conversation
        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = ConversationMetrics(
                conversation_id=conversation_id
            )

    def start_turn(self, conversation_id: str) -> None:
        """Mark the start of a conversation turn.

        Args:
            conversation_id: Unique identifier for the conversation
        """
        self._turn_start_times[conversation_id] = time.time()

    def record_turn(
        self, conversation_id: str, duration_ms: int | None, tokens: int, tools: list[str]
    ) -> None:
        """Record metrics for a conversation turn.

        Args:
            conversation_id: Unique identifier for the conversation
            duration_ms: Duration of the turn in milliseconds (if None, calculated from start_turn)
            tokens: Number of tokens used in the turn
            tools: List of tool names called during the turn
        """
        # Ensure conversation exists
        if conversation_id not in self._conversations:
            self.start_conversation(conversation_id)

        # Calculate duration if not provided
        if duration_ms is None:
            if conversation_id in self._turn_start_times:
                start_time = self._turn_start_times.pop(conversation_id)
                duration_ms = int((time.time() - start_time) * 1000)
            else:
                duration_ms = 0

        # Record the turn
        self._conversations[conversation_id].add_turn(duration_ms, tokens, tools)

    def get_conversation_metrics(self, conversation_id: str) -> ConversationMetrics | None:
        """Get metrics for a specific conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            ConversationMetrics object or None if not found
        """
        return self._conversations.get(conversation_id)

    def get_summary(self) -> dict[str, Any]:
        """Generate summary statistics across all conversations.

        Returns:
            Dictionary containing aggregated metrics
        """
        if not self._conversations:
            return {
                "total_conversations": 0,
                "total_turns": 0,
                "total_tokens": 0,
                "total_duration_ms": 0,
                "tool_usage": {},
                "conversations": [],
            }

        total_turns = sum(c.total_turns for c in self._conversations.values())
        total_tokens = sum(c.total_tokens for c in self._conversations.values())
        total_duration_ms = sum(c.total_duration_ms for c in self._conversations.values())

        # Aggregate tool usage across all conversations
        tool_usage: dict[str, int] = defaultdict(int)
        for conv in self._conversations.values():
            for tool, count in conv.tool_calls.items():
                tool_usage[tool] += count

        return {
            "total_conversations": len(self._conversations),
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "total_duration_ms": total_duration_ms,
            "avg_turns_per_conversation": (
                total_turns / len(self._conversations) if self._conversations else 0
            ),
            "avg_tokens_per_conversation": (
                total_tokens / len(self._conversations) if self._conversations else 0
            ),
            "avg_duration_per_conversation_ms": (
                total_duration_ms / len(self._conversations) if self._conversations else 0
            ),
            "tool_usage": dict(tool_usage),
            "conversations": [c.to_dict() for c in self._conversations.values()],
        }

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._conversations.clear()
        self._turn_start_times.clear()

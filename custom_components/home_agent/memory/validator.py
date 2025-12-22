"""Memory validation functionality for Home Agent.

This module provides the MemoryValidator class that validates memory quality
before storage. It consolidates validation logic that was previously embedded
in the memory extraction mixin.

Architecture:
    MemoryValidator uses a multi-layer validation approach with declarative rules:
    1. Word count validation (minimum meaningful words)
    2. Low-value prefix detection
    3. Importance threshold checking
    4. Pattern-based filtering (transient states, meta-info)

Key Classes:
    MemoryValidator: Validates memory quality with:
        - Configurable thresholds for word count and importance
        - Declarative pattern lists for rejection rules
        - Context-aware transient state detection
        - Batch validation support
        - Validation statistics

Usage Example:
    from custom_components.home_agent.memory.validator import MemoryValidator

    # Create validator with default thresholds
    validator = MemoryValidator()

    # Or with custom thresholds
    validator = MemoryValidator(min_word_count=15, min_importance=0.6)

    # Validate a single memory
    is_valid, reason = validator.validate({
        "content": "User prefers bedroom temperature at 68F for sleeping",
        "type": "preference",
        "importance": 0.8
    })

    if is_valid:
        # Store memory
        pass
    else:
        logger.debug(f"Rejected: {reason}")

    # Validate batch
    results = validator.validate_batch(memories)

    # Get statistics
    stats = validator.get_validation_stats(memories)
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


class MemoryValidator:
    """Validates memory quality before storage.

    Multi-layer validation:
    1. Word count validation (minimum meaningful words)
    2. Low-value prefix detection
    3. Importance threshold
    4. Pattern-based filtering (transient states, meta-info)

    Attributes:
        min_word_count: Minimum meaningful words required (default: 10)
        min_importance: Minimum importance score (default: 0.4)
    """

    # Patterns that indicate low-value content (prefix matching)
    LOW_VALUE_PREFIXES: list[str] = [
        "there is no",
        "there are no",
    ]

    # Patterns that indicate low-value content (contains matching)
    LOW_VALUE_PATTERNS: list[str] = [
        # Negative existence statements
        "no specific",
        "does not have",
        "doesn't have",
        "do not have",
        "don't have",
        # Conversation meta-information
        "the conversation occurred",
        "conversation occurred",
        "conversation took place",
        "we discussed",
        "we talked about",
        "user asked about",
        "i mentioned",
        "during the conversation",
        "at the time",
        "in the conversation",
        "during our conversation",
        "we were discussing",
    ]

    # Patterns that indicate transient device state (organized by category)
    # These patterns indicate ephemeral information that shouldn't be stored as memories
    TRANSIENT_STATE_PATTERNS: list[str] = [
        # === Device State Patterns ===
        "is on",
        "is off",
        "are on",
        "are off",
        "was on",
        "was off",
        "were on",
        "were off",
        "is open",
        "is closed",
        "is locked",
        "is unlocked",
        "lights are",
        "light is",
        "is playing",
        "is paused",
        "is stopped",
        "is running",
        "status is",
        "state is",
        # === Temporal State Patterns ===
        "is currently",
        "are currently",
        "is now",
        "are now",
        "right now",
        "at the moment",
        "at this time",
        # === Time/Clock Patterns ===
        "current time is",
        "the time is",
        "time is currently",
        "it is currently",
        "it's currently",
        # === Weather Patterns ===
        "weather is",
        "it's raining",
        "it is raining",
        "it's snowing",
        "it is snowing",
        "it's sunny",
        "it is sunny",
        "it's cloudy",
        "it is cloudy",
        "forecast is",
        "forecast shows",
        "weather forecast",
        "temperature outside is",
        "outside temperature is",
        "humidity outside is",
        "wind speed is",
        "wind is",
        # === Indoor Environment (current readings) ===
        "temperature is",
        "humidity is",
        "indoor temperature is",
        "room temperature is",
        # === Current Date/Day Patterns ===
        "today is",
        "it's monday",
        "it's tuesday",
        "it's wednesday",
        "it's thursday",
        "it's friday",
        "it's saturday",
        "it's sunday",
        "it is monday",
        "it is tuesday",
        "it is wednesday",
        "it is thursday",
        "it is friday",
        "it is saturday",
        "it is sunday",
        "this week",
        "this month",
        "this year",
        # === Location/Presence Patterns ===
        "user is at",
        "user is home",
        "user is away",
        "user is not home",
        "nobody is home",
        "someone is home",
        "is at home",
        "is away",
        "just arrived",
        "just left",
    ]

    # Words that indicate permanent temporal facts (not transient state)
    # When these words precede patterns like "is on", they indicate
    # permanent facts (e.g., "birthday is on May 4th") not device states
    TEMPORAL_CONTEXT_WORDS: list[str] = [
        "birthday",
        "event",
        "date",
        "day",
        "anniversary",
        "holiday",
        "vacation",
        "appointment",
        "meeting",
        "schedule",
    ]

    def __init__(
        self,
        min_word_count: int = 10,
        min_importance: float = 0.4,
    ) -> None:
        """Initialize the memory validator.

        Args:
            min_word_count: Minimum meaningful words required (default: 10)
            min_importance: Minimum importance score (default: 0.4)
        """
        self.min_word_count = min_word_count
        self.min_importance = min_importance

    def validate(self, memory: dict[str, Any]) -> tuple[bool, str]:
        """Validate a memory against quality criteria.

        Args:
            memory: Memory dictionary with at least "content" key,
                   optionally "importance" and "type"

        Returns:
            Tuple of (is_valid, rejection_reason).
            rejection_reason is empty string if valid.
        """
        if not isinstance(memory, dict):
            return False, "invalid_format"

        content = memory.get("content", "")
        if not content:
            return False, "missing_content"

        importance = memory.get("importance", 0.5)

        # 1. Word count validation
        is_valid, reason = self._validate_word_count(content)
        if not is_valid:
            return False, reason

        # 2. Low-value prefix detection
        is_valid, reason = self._validate_low_value_prefix(content)
        if not is_valid:
            return False, reason

        # 3. Low-value pattern detection
        is_valid, reason = self._validate_low_value_patterns(content)
        if not is_valid:
            return False, reason

        # 4. Importance threshold
        is_valid, reason = self._validate_importance(importance)
        if not is_valid:
            return False, reason

        # 5. Transient state detection
        is_valid, reason = self._validate_transient_state(content)
        if not is_valid:
            return False, reason

        return True, ""

    def validate_batch(
        self, memories: list[dict[str, Any]]
    ) -> list[tuple[bool, str]]:
        """Validate a batch of memories.

        Args:
            memories: List of memory dictionaries

        Returns:
            List of (is_valid, rejection_reason) tuples
        """
        return [self.validate(memory) for memory in memories]

    def get_validation_stats(
        self, memories: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Get validation statistics for a batch of memories.

        Args:
            memories: List of memory dictionaries

        Returns:
            Dictionary with validation statistics including:
            - total: Total memories processed
            - valid: Number of valid memories
            - invalid: Number of invalid memories
            - rejection_reasons: Count by rejection reason
        """
        results = self.validate_batch(memories)

        rejection_reasons: dict[str, int] = {}
        valid_count = 0
        invalid_count = 0

        for is_valid, reason in results:
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        return {
            "total": len(memories),
            "valid": valid_count,
            "invalid": invalid_count,
            "rejection_reasons": rejection_reasons,
        }

    def _validate_word_count(self, content: str) -> tuple[bool, str]:
        """Validate minimum word count.

        Counts only meaningful words (>2 chars) to filter filler words.

        Args:
            content: Memory content

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        words = content.split()
        meaningful_words = [w for w in words if len(w) > 2]
        meaningful_word_count = len(meaningful_words)

        if meaningful_word_count < self.min_word_count:
            return False, f"too_short:{meaningful_word_count}"

        return True, ""

    def _validate_low_value_prefix(self, content: str) -> tuple[bool, str]:
        """Validate against low-value prefixes.

        Args:
            content: Memory content

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        content_lower = content.lower()

        for prefix in self.LOW_VALUE_PREFIXES:
            if content_lower.startswith(prefix):
                return False, f"low_value_prefix:{prefix}"

        # Special case: temporal "at" references with timestamps
        if content_lower.startswith("at ") and ":" in content:
            return False, "low_value_prefix:temporal_at"

        return True, ""

    def _validate_low_value_patterns(self, content: str) -> tuple[bool, str]:
        """Validate against low-value content patterns.

        Args:
            content: Memory content

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        content_lower = content.lower()

        for pattern in self.LOW_VALUE_PATTERNS:
            if pattern in content_lower:
                return False, f"low_value_pattern:{pattern}"

        return True, ""

    def _validate_importance(self, importance: float) -> tuple[bool, str]:
        """Validate importance score.

        Args:
            importance: Importance score (0.0 to 1.0)

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        if importance < self.min_importance:
            return False, f"low_importance:{importance:.2f}"

        return True, ""

    def _validate_transient_state(self, content: str) -> tuple[bool, str]:
        """Validate against transient state patterns.

        Uses context awareness to avoid false positives like "birthday is on".

        Args:
            content: Memory content

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        content_lower = content.lower()

        for pattern in self.TRANSIENT_STATE_PATTERNS:
            if pattern in content_lower:
                # Check for context-aware exceptions
                idx = content_lower.find(pattern)
                if idx > 0:
                    # Get the word before the pattern
                    before = content_lower[:idx].strip().split()
                    if before:
                        last_word = before[-1]
                        # Temporal context words are not transient states
                        if last_word in self.TEMPORAL_CONTEXT_WORDS:
                            continue

                return False, f"transient_state:{pattern}"

        return True, ""

    def is_transient_state(self, content: str) -> bool:
        """Check if content represents a transient state or low-quality content.

        This is a convenience method that combines all pattern-based checks
        (low-value patterns and transient states) but not word count or importance.

        This method provides backward compatibility with the existing
        MemoryManager._is_transient_state() API.

        Args:
            content: Memory content to check

        Returns:
            True if content appears to be transient state or low-quality
        """
        content_lower = content.lower()

        # Check low-value prefixes
        for prefix in self.LOW_VALUE_PREFIXES:
            if content_lower.startswith(prefix):
                return True

        # Check low-value patterns
        for pattern in self.LOW_VALUE_PATTERNS:
            if pattern in content_lower:
                return True

        # Check transient state patterns with context awareness
        for pattern in self.TRANSIENT_STATE_PATTERNS:
            if pattern in content_lower:
                idx = content_lower.find(pattern)
                if idx > 0:
                    before = content_lower[:idx].strip().split()
                    if before:
                        last_word = before[-1]
                        if last_word in self.TEMPORAL_CONTEXT_WORDS:
                            continue
                return True

        return False

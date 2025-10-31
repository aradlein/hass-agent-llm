"""Context optimization and compression for Home Agent.

This module provides intelligent context compression to stay within token limits
while preserving the most important information.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .helpers import estimate_tokens

_LOGGER = logging.getLogger(__name__)


class ContextOptimizer:
    """Optimizes and compresses context to fit within token limits."""

    def __init__(
        self,
        compression_level: str = "medium",
        preserve_recent_messages: int = 3,
    ) -> None:
        """Initialize the context optimizer.

        Args:
            compression_level: Compression level (none, low, medium, high)
            preserve_recent_messages: Number of recent messages to keep uncompressed
        """
        self.compression_level = compression_level
        self.preserve_recent_messages = preserve_recent_messages

        _LOGGER.debug(
            "Context optimizer initialized (level=%s, preserve=%d)",
            compression_level,
            preserve_recent_messages,
        )

    def compress_entity_context(
        self, context: str, target_tokens: int
    ) -> tuple[str, dict[str, Any]]:
        """Compress entity context to target token count.

        Args:
            context: Original entity context string
            target_tokens: Target token count

        Returns:
            Tuple of (compressed_context, metrics)
        """
        original_tokens = estimate_tokens(context)

        if original_tokens <= target_tokens:
            return context, {
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "compression_ratio": 1.0,
                "was_compressed": False,
            }

        _LOGGER.debug(
            "Compressing entity context: %d -> %d tokens",
            original_tokens,
            target_tokens,
        )

        compressed = context

        if self.compression_level in ["medium", "high"]:
            # Remove extra whitespace
            compressed = re.sub(r"\s+", " ", compressed)
            compressed = re.sub(r"\n\s*\n", "\n", compressed)

        if self.compression_level == "high":
            # More aggressive: remove attribute descriptions
            compressed = re.sub(r"\s*\([^)]+\)", "", compressed)

        # If still too large, truncate
        compressed_tokens = estimate_tokens(compressed)
        if compressed_tokens > target_tokens:
            # Simple character-based truncation
            ratio = target_tokens / compressed_tokens
            target_length = int(len(compressed) * ratio * 0.9)  # 90% to be safe
            compressed = compressed[:target_length] + "..."

        final_tokens = estimate_tokens(compressed)

        return compressed, {
            "original_tokens": original_tokens,
            "compressed_tokens": final_tokens,
            "compression_ratio": final_tokens / original_tokens
            if original_tokens > 0
            else 1.0,
            "was_compressed": True,
        }

    def compress_conversation_history(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compress conversation history to target token count.

        Args:
            messages: List of message dictionaries
            target_tokens: Target token count

        Returns:
            Tuple of (compressed_messages, metrics)
        """
        if not messages:
            return messages, {"original_tokens": 0, "compressed_tokens": 0}

        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        if original_tokens <= target_tokens:
            return messages, {
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "compression_ratio": 1.0,
                "was_compressed": False,
            }

        _LOGGER.debug(
            "Compressing history: %d messages, %d -> %d tokens",
            len(messages),
            original_tokens,
            target_tokens,
        )

        # Keep recent messages, compress or remove old ones
        preserved = messages[-self.preserve_recent_messages :]
        older = messages[: -self.preserve_recent_messages]

        # Start with recent messages
        result = preserved.copy()
        current_tokens = sum(estimate_tokens(m.get("content", "")) for m in result)

        # Add older messages if space allows
        for msg in reversed(older):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= target_tokens:
                result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        final_tokens = sum(estimate_tokens(m.get("content", "")) for m in result)

        return result, {
            "original_tokens": original_tokens,
            "compressed_tokens": final_tokens,
            "compression_ratio": final_tokens / original_tokens
            if original_tokens > 0
            else 1.0,
            "was_compressed": True,
            "messages_removed": len(messages) - len(result),
        }

    def smart_truncate(
        self,
        text: str,
        max_tokens: int,
        preserve_patterns: list[str] | None = None,
    ) -> str:
        """Intelligently truncate text while preserving important patterns.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            preserve_patterns: Regex patterns to preserve if possible

        Returns:
            Truncated text
        """
        current_tokens = estimate_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Calculate target character length
        ratio = max_tokens / current_tokens
        target_length = int(len(text) * ratio * 0.9)  # 90% to be safe

        if not preserve_patterns:
            return text[:target_length] + "..."

        # Try to preserve important patterns
        truncated = text[:target_length]

        # Check if we cut off any preserve patterns
        for pattern in preserve_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                last_match = matches[-1]
                if last_match.start() > target_length:
                    # Pattern was cut off, try to include it
                    new_length = min(last_match.end(), len(text))
                    if estimate_tokens(text[:new_length]) <= max_tokens:
                        truncated = text[:new_length]

        return truncated + "..."

    def optimize_for_model(
        self, context: dict[str, Any], model_name: str
    ) -> dict[str, Any]:
        """Optimize context for specific model characteristics.

        Args:
            context: Context dictionary
            model_name: Model name (e.g., "gpt-4", "llama2")

        Returns:
            Optimized context dictionary
        """
        # Model-specific token limits
        model_limits = {
            "gpt-4": 8000,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4000,
            "llama2": 4096,
            "mistral": 8192,
        }

        max_tokens = model_limits.get(model_name, 4000)

        # Adjust compression based on model
        optimized = context.copy()

        for key, value in context.items():
            if isinstance(value, str):
                tokens = estimate_tokens(value)
                if tokens > max_tokens * 0.5:  # Use max 50% per field
                    target = int(max_tokens * 0.5)
                    optimized[key] = self.smart_truncate(value, target)

        return optimized

    def prioritize_entities(
        self,
        entities: list[dict[str, Any]],
        user_query: str,
        max_entities: int | None = None,
    ) -> list[dict[str, Any]]:
        """Prioritize entities based on relevance to user query.

        Args:
            entities: List of entity dictionaries
            user_query: User's query text
            max_entities: Maximum entities to return (None for all)

        Returns:
            Prioritized list of entities
        """
        if not entities:
            return entities

        # Score entities based on relevance
        scored_entities = []

        for entity in entities:
            score = 0

            # Check if entity ID mentioned in query
            entity_id = entity.get("entity_id", "")
            if entity_id.lower() in user_query.lower():
                score += 10

            # Check friendly name
            friendly_name = entity.get("attributes", {}).get("friendly_name", "")
            if friendly_name and friendly_name.lower() in user_query.lower():
                score += 8

            # Check domain relevance (simple keyword matching)
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain in user_query.lower():
                score += 5

            scored_entities.append((score, entity))

        # Sort by score (descending)
        scored_entities.sort(key=lambda x: x[0], reverse=True)

        # Return top entities
        result = [entity for score, entity in scored_entities]

        if max_entities is not None:
            result = result[:max_entities]

        _LOGGER.debug(
            "Prioritized %d entities (from %d total)",
            len(result),
            len(entities),
        )

        return result

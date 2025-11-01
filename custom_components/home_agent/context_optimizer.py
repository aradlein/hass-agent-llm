"""Context optimization and compression for Home Agent.

This module provides intelligent context compression to stay within token limits
while preserving the most important information.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .helpers import estimate_tokens

_LOGGER = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Metrics for compression operations."""

    original_tokens: int
    compressed_tokens: int
    entities_before: int = 0
    entities_after: int = 0

    @property
    def reduction_percent(self) -> float:
        """Calculate reduction percentage."""
        if self.original_tokens == 0:
            return 0.0
        return ((self.original_tokens - self.compressed_tokens) / self.original_tokens) * 100


@dataclass
class EntityPriority:
    """Priority information for an entity."""

    entity_id: str
    score: float
    reasons: list[str] = field(default_factory=list)


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
        self._access_counts: dict[str, int] = {}
        self._last_metrics: CompressionMetrics | None = None

        _LOGGER.debug(
            "Context optimizer initialized (level=%s, preserve=%d)",
            compression_level,
            preserve_recent_messages,
        )

    def remove_redundant_attributes(
        self, entities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove redundant attributes from entities.

        Args:
            entities: List of entity dictionaries

        Returns:
            List of entities with redundant attributes removed
        """
        result = []

        # Attributes to remove
        bloat_attrs = {
            "supported_features",
            "icon",
            "entity_picture",
            "context_id",
            "entity_picture_local",
        }

        for entity in entities:
            cleaned = {
                "entity_id": entity.get("entity_id", ""),
                "state": entity.get("state", ""),
            }

            if "attributes" in entity and isinstance(entity["attributes"], dict):
                cleaned_attrs = {}
                for key, value in entity["attributes"].items():
                    # Skip internal attributes (starting with _)
                    if key.startswith("_"):
                        continue
                    # Skip bloat attributes
                    if key in bloat_attrs:
                        continue
                    # Truncate very long values
                    if isinstance(value, str) and len(value) > 1000:
                        value = value[:1000] + "..."
                    cleaned_attrs[key] = value

                if cleaned_attrs:
                    cleaned["attributes"] = cleaned_attrs

            result.append(cleaned)

        return result

    def compress_entity_context(
        self,
        entities: list[dict[str, Any]],
        target_tokens: int,
        user_query: str | None = None,
    ) -> list[dict[str, Any]]:
        """Compress entity context to target token count.

        Args:
            entities: List of entity dictionaries
            target_tokens: Target token count
            user_query: Optional user query for prioritization

        Returns:
            Compressed list of entities
        """
        if not entities:
            self._last_metrics = CompressionMetrics(
                original_tokens=0,
                compressed_tokens=0,
                entities_before=0,
                entities_after=0,
            )
            return []

        # Calculate original tokens
        original_json = json.dumps(entities)
        original_tokens = estimate_tokens(original_json)
        entities_before = len(entities)

        # If under target, return as is
        if original_tokens <= target_tokens:
            self._last_metrics = CompressionMetrics(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                entities_before=entities_before,
                entities_after=entities_before,
            )
            return entities

        _LOGGER.debug(
            "Compressing entity context: %d entities, %d -> %d tokens",
            len(entities),
            original_tokens,
            target_tokens,
        )

        # Apply compression level
        compressed = self._apply_compression_level(entities, self.compression_level)

        # Prioritize if user query provided
        if user_query:
            priorities = self.prioritize_entities(compressed, user_query)
            # Sort by priority
            entity_map = {p.entity_id: p.score for p in priorities}
            compressed = sorted(
                compressed,
                key=lambda e: entity_map.get(e.get("entity_id", ""), 0),
                reverse=True,
            )

        # Remove entities until we fit
        result = compressed
        while result:
            result_json = json.dumps(result)
            current_tokens = estimate_tokens(result_json)
            if current_tokens <= target_tokens:
                break
            # Remove lowest priority entity
            result = result[:-1]

        # Track metrics
        final_json = json.dumps(result)
        final_tokens = estimate_tokens(final_json)
        self._last_metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=final_tokens,
            entities_before=entities_before,
            entities_after=len(result),
        )

        return result

    def compress_conversation_history(
        self,
        messages: list[dict[str, Any]],
        target_tokens: int,
    ) -> list[dict[str, Any]]:
        """Compress conversation history to target token count.

        Args:
            messages: List of message dictionaries
            target_tokens: Target token count

        Returns:
            Compressed list of messages
        """
        if not messages:
            return []

        original_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        if original_tokens <= target_tokens:
            return messages

        _LOGGER.debug(
            "Compressing history: %d messages, %d -> %d tokens",
            len(messages),
            original_tokens,
            target_tokens,
        )

        # Keep recent messages, compress or remove old ones
        # preserve_recent_messages is in pairs (user + assistant)
        preserve_count = self.preserve_recent_messages * 2
        preserved = messages[-preserve_count:] if len(messages) > preserve_count else messages
        older = messages[:-preserve_count] if len(messages) > preserve_count else []

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

        return result

    def smart_truncate(
        self,
        text: str,
        max_tokens: int,
        preserve: list[str] | None = None,
    ) -> str:
        """Intelligently truncate text while preserving important patterns.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            preserve: Terms to preserve if possible

        Returns:
            Truncated text
        """
        if not text:
            return text

        current_tokens = estimate_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Calculate target character length
        ratio = max_tokens / current_tokens
        target_length = int(len(text) * ratio * 0.9)  # 90% to be safe

        if not preserve:
            # Try to end at sentence boundary
            truncated = text[:target_length]
            # Find last sentence boundary
            last_period = truncated.rfind(". ")
            if last_period > target_length * 0.7:  # At least 70% of target
                truncated = truncated[:last_period + 1]
            return truncated + "..."

        # Try to preserve important terms
        truncated = text[:target_length]

        # Check if we cut off any preserve terms
        for term in preserve:
            if term.lower() in text.lower():
                # Find the term
                term_pos = text.lower().find(term.lower())
                if term_pos > target_length:
                    # Term was cut off, try to include it
                    new_length = min(term_pos + len(term) + 50, len(text))
                    if estimate_tokens(text[:new_length]) <= max_tokens:
                        truncated = text[:new_length]

        # Try to end at sentence boundary
        last_period = truncated.rfind(". ")
        if last_period > len(truncated) * 0.7:
            truncated = truncated[:last_period + 1]

        if not truncated.endswith("."):
            truncated += "..."

        return truncated

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
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4": 8000,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 4000,
            "claude-3-opus": 200000,
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
            elif isinstance(value, list):
                # Handle entity lists and conversation history
                if key == "entity_context":
                    value_json = json.dumps(value)
                    tokens = estimate_tokens(value_json)
                    if tokens > max_tokens * 0.4:  # Use max 40% for entities
                        target = int(max_tokens * 0.4)
                        optimized[key] = self.compress_entity_context(value, target)
                elif key == "conversation_history":
                    tokens = sum(estimate_tokens(m.get("content", "")) for m in value if isinstance(m, dict))
                    if tokens > max_tokens * 0.3:  # Use max 30% for history
                        target = int(max_tokens * 0.3)
                        optimized[key] = self.compress_conversation_history(value, target)

        return optimized

    def prioritize_entities(
        self,
        entities: list[dict[str, Any]],
        user_query: str,
    ) -> list[EntityPriority]:
        """Prioritize entities based on relevance to user query.

        Args:
            entities: List of entity dictionaries
            user_query: User's query text

        Returns:
            List of EntityPriority objects sorted by score (descending)
        """
        if not entities:
            return []

        # Score entities based on relevance
        priorities: list[EntityPriority] = []

        for entity in entities:
            entity_id = entity.get("entity_id", "")
            score = 0.0
            reasons = []

            # Track access
            self._access_counts[entity_id] = self._access_counts.get(entity_id, 0) + 1

            # Check if entity ID mentioned in query
            if entity_id.lower() in user_query.lower():
                score += 10
                reasons.append("mentioned_in_query")

            # Check friendly name
            friendly_name = entity.get("attributes", {}).get("friendly_name", "")
            if friendly_name and friendly_name.lower() in user_query.lower():
                score += 8
                reasons.append("name_mentioned")

            # Check domain relevance (simple keyword matching)
            domain = entity_id.split(".")[0] if "." in entity_id else ""
            if domain in user_query.lower():
                score += 5
                reasons.append("domain_relevant")

            # Check for related domain keywords
            domain_keywords = {
                "light": ["light", "lights", "lamp", "illuminate"],
                "climate": ["temperature", "thermostat", "heat", "cool", "climate"],
                "sensor": ["temperature", "sensor", "reading"],
                "switch": ["switch", "turn", "power"],
            }

            if domain in domain_keywords:
                for keyword in domain_keywords[domain]:
                    if keyword in user_query.lower():
                        score += 3
                        if "keyword_match" not in reasons:
                            reasons.append("keyword_match")
                        break

            # Base score to ensure ordering
            if not reasons:
                score = 1.0
                reasons.append("available")

            priorities.append(
                EntityPriority(entity_id=entity_id, score=score, reasons=reasons)
            )

        # Sort by score (descending)
        priorities.sort(key=lambda x: x.score, reverse=True)

        _LOGGER.debug(
            "Prioritized %d entities",
            len(priorities),
        )

        return priorities

    def estimate_context_tokens(self, context: dict[str, Any]) -> dict[str, int]:
        """Estimate token counts for each context component.

        Args:
            context: Context dictionary with components

        Returns:
            Dictionary with token estimates for each component
        """
        estimates = {}

        # System prompt
        system_prompt = context.get("system_prompt", "")
        if isinstance(system_prompt, str):
            estimates["system_prompt"] = estimate_tokens(system_prompt)
        else:
            estimates["system_prompt"] = 0

        # Entity context
        entity_context = context.get("entity_context", "")
        if isinstance(entity_context, str):
            estimates["entity_context"] = estimate_tokens(entity_context)
        elif isinstance(entity_context, list):
            entity_json = json.dumps(entity_context)
            estimates["entity_context"] = estimate_tokens(entity_json)
        else:
            estimates["entity_context"] = 0

        # Conversation history
        conversation_history = context.get("conversation_history", [])
        if isinstance(conversation_history, list):
            total = sum(
                estimate_tokens(m.get("content", ""))
                for m in conversation_history
                if isinstance(m, dict)
            )
            estimates["conversation_history"] = total
        else:
            estimates["conversation_history"] = 0

        # User message
        user_message = context.get("user_message", "")
        if isinstance(user_message, str):
            estimates["user_message"] = estimate_tokens(user_message)
        else:
            estimates["user_message"] = 0

        # Total
        estimates["total"] = sum(estimates.values())

        return estimates

    def get_metrics(self) -> CompressionMetrics | None:
        """Get the last compression metrics.

        Returns:
            Last compression metrics or None if no compression has occurred
        """
        return self._last_metrics

    def reset_metrics(self) -> None:
        """Reset all metrics and tracking data."""
        self._last_metrics = None
        self._access_counts = {}

    def _apply_compression_level(
        self, entities: list[dict[str, Any]], level: str
    ) -> list[dict[str, Any]]:
        """Apply compression level to entities.

        Args:
            entities: List of entity dictionaries
            level: Compression level (none, low, medium, high)

        Returns:
            Compressed entities
        """
        if level == "none":
            return entities

        result = []

        for entity in entities:
            compressed = {
                "entity_id": entity.get("entity_id", ""),
                "state": entity.get("state", ""),
            }

            if "attributes" not in entity or not isinstance(entity["attributes"], dict):
                result.append(compressed)
                continue

            attrs = entity["attributes"]
            compressed_attrs = {}

            if level == "low":
                # Keep most attributes except bloat
                bloat = {"supported_features", "icon", "entity_picture"}
                for key, value in attrs.items():
                    if not key.startswith("_") and key not in bloat:
                        compressed_attrs[key] = value

            elif level == "medium":
                # Keep only essential attributes
                essential = {"friendly_name", "unit_of_measurement", "device_class"}
                for key, value in attrs.items():
                    if key in essential:
                        compressed_attrs[key] = value

            elif level == "high":
                # Keep only friendly_name
                if "friendly_name" in attrs:
                    compressed_attrs["friendly_name"] = attrs["friendly_name"]

            if compressed_attrs:
                compressed["attributes"] = compressed_attrs

            result.append(compressed)

        return result

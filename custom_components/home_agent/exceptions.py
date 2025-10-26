"""Custom exceptions for the Home Agent component.

This module defines all custom exceptions used throughout the Home Agent
integration for error handling and flow control.
"""

from __future__ import annotations


class HomeAgentError(Exception):
    """Base exception for all Home Agent errors.

    This is the parent exception class that all other Home Agent exceptions
    inherit from. Use this for catching any Home Agent-related error.

    Example:
        try:
            # Some Home Agent operation
            pass
        except HomeAgentError as error:
            _LOGGER.error("Home Agent error: %s", error)
    """


class ContextInjectionError(HomeAgentError):
    """Exception raised when context injection fails.

    This exception is raised when the component fails to inject entity context
    into the LLM conversation, either through direct entity access or vector
    database retrieval.

    Common causes:
        - Entity does not exist or is not accessible
        - Vector database connection failure
        - Context size exceeds limits
        - Invalid entity ID format
        - Missing required attributes

    Example:
        raise ContextInjectionError(
            f"Failed to inject context for entity {entity_id}: {error}"
        )
    """


class ToolExecutionError(HomeAgentError):
    """Exception raised when a tool execution fails.

    This exception is raised when a tool (ha_control, ha_query, custom tools,
    or query_external_llm) fails to execute properly.

    Common causes:
        - Invalid tool parameters
        - Entity not found or not accessible
        - Service call failure
        - External API timeout or error
        - Permission denied for requested action

    Example:
        raise ToolExecutionError(
            f"Tool '{tool_name}' failed: {error}. "
            f"Check entity_id '{entity_id}' exists and is accessible."
        )
    """


class AuthenticationError(HomeAgentError):
    """Exception raised when LLM API authentication fails.

    This exception is raised when authentication with an OpenAI-compatible
    LLM API endpoint fails, either for the primary LLM or external LLM.

    Common causes:
        - Invalid or expired API key
        - Missing API key
        - Incorrect authentication header format
        - API endpoint requires authentication but none provided

    Example:
        raise AuthenticationError(
            f"Failed to authenticate with LLM at {base_url}: Invalid API key"
        )
    """


class TokenLimitExceeded(HomeAgentError):
    """Exception raised when token limits are exceeded.

    This exception is raised when the combined context (system prompt,
    conversation history, entity context, and user message) exceeds the
    model's token limit or the configured maximum.

    Common causes:
        - Too many entities in context
        - Conversation history too long
        - Very long user message
        - Accumulated context from multiple tool calls

    Example:
        raise TokenLimitExceeded(
            f"Context size {token_count} exceeds limit {max_tokens}. "
            f"Consider reducing history or entity count."
        )
    """


class RateLimitExceeded(HomeAgentError):
    """Exception raised when API rate limits are exceeded.

    This exception is raised when the LLM API provider's rate limit is hit,
    either for requests per minute, tokens per minute, or other quotas.

    Common causes:
        - Too many requests in short time period
        - High token consumption rate
        - Shared API key hitting account limits
        - Free tier limits reached

    Example:
        raise RateLimitExceeded(
            f"Rate limit exceeded for {provider}. "
            f"Retry after {retry_after} seconds."
        )
    """


class PermissionDenied(HomeAgentError):
    """Exception raised when permission is denied for an operation.

    This exception is raised when the component attempts to access or control
    an entity that the user does not have permission to interact with, or
    when a service call is not allowed.

    Common causes:
        - Entity not exposed to voice assistants
        - Entity not in allowed entity list
        - Service call not permitted
        - User lacks required permissions
        - Domain-level restrictions

    Example:
        raise PermissionDenied(
            f"Entity {entity_id} is not accessible. "
            f"Ensure it is exposed in the integration configuration."
        )
    """


class ValidationError(HomeAgentError):
    """Exception raised when validation fails.

    This exception is raised when input validation fails for configuration
    values, tool parameters, entity IDs, or other user-provided data.

    Common causes:
        - Invalid entity ID format
        - Missing required parameters
        - Parameter value out of valid range
        - Incompatible configuration options
        - Malformed JSON or YAML

    Example:
        raise ValidationError(
            f"Invalid entity_id format: {entity_id}. "
            f"Expected format: domain.entity_name"
        )
    """

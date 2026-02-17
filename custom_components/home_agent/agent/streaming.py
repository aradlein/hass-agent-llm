"""Streaming functionality mixin for HomeAgent.

This module provides the StreamingMixin class that enables real-time streaming
responses from the LLM. It integrates with Home Assistant's assist pipeline to
deliver incremental responses as they are generated, providing a more responsive
user experience for voice and text conversations.

Architecture:
    StreamingMixin is designed as a mixin class to be inherited by HomeAgent. It
    extends the base LLM functionality with streaming capabilities, allowing the
    agent to send partial responses to the user as the LLM generates them.

Key Classes:
    StreamingMixin: Mixin providing streaming functionality with support for:
        - Detection of streaming support in the current context
        - Server-Sent Events (SSE) streaming from LLM APIs
        - Integration with Home Assistant's ChatLog and delta listeners
        - Real-time tool execution during streaming
        - Graceful fallback to synchronous mode on errors

Core Responsibilities:
    - Detect if streaming is available (ChatLog with delta_listener exists)
    - Make streaming HTTP requests to LLM APIs
    - Parse Server-Sent Events (SSE) from the response stream
    - Coordinate with OpenAIStreamingHandler for delta transformation
    - Support tool calling during streaming conversations
    - Provide automatic fallback to synchronous processing

Streaming Flow:
    1. Check if streaming is supported via _can_stream()
    2. If supported, call _call_llm_streaming() with conversation messages
    3. LLM API returns SSE stream with incremental deltas
    4. OpenAIStreamingHandler transforms deltas to Home Assistant format
    5. Deltas sent to ChatLog which forwards to assist pipeline
    6. Tool calls executed automatically by ChatLog infrastructure
    7. Process continues until LLM completes response

Usage Example:
    The mixin is used through inheritance in HomeAgent:

        class HomeAgent(LLMMixin, StreamingMixin, MemoryExtractionMixin):
            async def async_process(self, user_input):
                # Check if streaming is available
                if self._can_stream():
                    # Use streaming path
                    return await self._async_process_streaming(user_input)
                else:
                    # Fall back to synchronous
                    return await self._async_process_synchronous(user_input)

Expected Host Class Attributes:
    The mixin expects the host class to provide:

    - config: dict[str, Any]
        Configuration dictionary containing streaming settings:
        - CONF_STREAMING_ENABLED: Enable/disable streaming (default: True)
        - CONF_LLM_BASE_URL: LLM API base URL
        - CONF_LLM_API_KEY: API authentication key
        - CONF_LLM_MODEL: Model name
        - CONF_LLM_TEMPERATURE: Temperature parameter
        - CONF_LLM_MAX_TOKENS: Maximum tokens
        - CONF_LLM_PROXY_HEADERS: Custom HTTP headers for routing (dict)
        - CONF_DEBUG_LOGGING: Enable debug output

    - _session: aiohttp.ClientSession | None
        HTTP session managed by LLMMixin._ensure_session()

    - tool_handler: ToolHandler
        Tool handler instance for executing tools during streaming

Integration Points:
    - Home Assistant ChatLog: Receives streaming deltas
    - Home Assistant assist pipeline: Delivers real-time responses
    - OpenAIStreamingHandler: Transforms SSE to HA delta format
    - ToolHandler: Executes tools called by LLM during streaming
    - LLMMixin: Provides _ensure_session() for HTTP management

Streaming API Format:
    The module expects OpenAI-compatible Server-Sent Events:

    Request:
        POST {base_url}/chat/completions
        Body: {
            "model": "gpt-4",
            "messages": [...],
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": [...]  // Optional
        }

    Response Stream:
        data: {"choices":[{"delta":{"content":"Hello"}}]}
        data: {"choices":[{"delta":{"content":" there"}}]}
        data: {"choices":[{"delta":{"tool_calls":[...]}}]}
        data: {"usage":{"total_tokens":42}}
        data: [DONE]

Stream Detection:
    Streaming is only enabled when ALL conditions are met:

    1. CONF_STREAMING_ENABLED is True in config
    2. Home Assistant's ChatLog is available in context
    3. ChatLog has a delta_listener configured
    4. Conversation is happening through assist pipeline

    If any condition fails, the agent automatically falls back to
    synchronous processing without user-visible errors.

Error Handling:
    - AuthenticationError: Raised on 401 API status
    - HomeAgentError: Raised on API errors or connection failures
    - Automatic fallback: On streaming errors, logs warning and falls
      back to synchronous mode

Benefits of Streaming:
    - Lower perceived latency for users
    - Real-time feedback during long responses
    - Better user experience for voice conversations
    - Supports interruption and cancellation
    - Token usage reporting in final chunk

Configuration Example:
    config = {
        CONF_STREAMING_ENABLED: True,
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "sk-...",
        CONF_LLM_MODEL: "gpt-4",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_DEBUG_LOGGING: False
    }
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

import aiohttp

from ..const import (
    CONF_AZURE_API_VERSION,
    CONF_DEBUG_LOGGING,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_ALIVE,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_PROXY_HEADERS,
    CONF_LLM_TEMPERATURE,
    CONF_LLM_TOP_P,
    CONF_STREAMING_ENABLED,
    DEFAULT_LLM_KEEP_ALIVE,
    DEFAULT_STREAMING_ENABLED,
)
from ..exceptions import AuthenticationError, HomeAgentError
from ..helpers import build_api_url, build_auth_headers, is_ollama_backend, redact_sensitive_data, render_template_value

if TYPE_CHECKING:
    from ..tool_handler import ToolHandler

_LOGGER = logging.getLogger(__name__)


class StreamingMixin:
    """Mixin providing streaming functionality.

    This mixin expects the following attributes from the host class:
    - config: dict[str, Any] - Configuration dictionary
    - _session: aiohttp.ClientSession | None - HTTP session
    - tool_handler: ToolHandler - Tool handler instance
    """

    config: dict[str, Any]
    hass: Any
    _session: aiohttp.ClientSession | None
    tool_handler: "ToolHandler"

    # This method is provided by LLMMixin, but we need to declare it for type checking
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists (provided by LLMMixin)."""
        ...

    def _can_stream(self) -> bool:
        """Check if streaming is supported in the current context.

        Returns:
            True if streaming is enabled and ChatLog with delta_listener is available
        """
        from homeassistant.components.conversation.chat_log import current_chat_log

        # Check if streaming is enabled in config
        if not self.config.get(CONF_STREAMING_ENABLED, DEFAULT_STREAMING_ENABLED):
            return False

        # Check if ChatLog with delta_listener exists (means assist pipeline supports streaming)
        chat_log = current_chat_log.get()
        if chat_log is None or chat_log.delta_listener is None:
            return False

        return True

    async def _call_llm_streaming(
        self, messages: list[dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Call LLM API with streaming enabled.

        Args:
            messages: Conversation messages

        Yields:
            SSE lines from the streaming response
        """
        session = await self._ensure_session()

        base_url = self.config[CONF_LLM_BASE_URL]
        url = build_api_url(
            base_url,
            self.config[CONF_LLM_MODEL],
            self.config.get(CONF_AZURE_API_VERSION),
        )
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }

        # Add authentication headers (Azure uses api-key, others use Bearer token)
        api_key = render_template_value(self.hass, self.config.get(CONF_LLM_API_KEY, ""))
        headers.update(build_auth_headers(base_url, api_key))

        # Add custom proxy headers if configured
        proxy_headers = self.config.get(CONF_LLM_PROXY_HEADERS, {})
        if proxy_headers:
            headers.update(proxy_headers)

        payload: dict[str, Any] = {
            "model": self.config[CONF_LLM_MODEL],
            "messages": messages,
            "temperature": self.config.get(CONF_LLM_TEMPERATURE, 0.7),
            "max_tokens": self.config.get(CONF_LLM_MAX_TOKENS, 1000),
            "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
            "stream": True,  # Enable streaming!
        }

        # Only include keep_alive for Ollama backends (not supported by OpenAI, etc.)
        # See: https://github.com/aradlein/home-agent/issues/65
        if is_ollama_backend(self.config[CONF_LLM_BASE_URL]):
            payload["keep_alive"] = self.config.get(CONF_LLM_KEEP_ALIVE, DEFAULT_LLM_KEEP_ALIVE)

        # Add tools if available
        tool_definitions = self.tool_handler.get_tool_definitions()
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = "auto"

        # Request usage statistics in stream
        payload["stream_options"] = {"include_usage": True}

        if self.config.get(CONF_DEBUG_LOGGING):
            _LOGGER.debug(
                "Calling LLM (streaming) at %s with %d messages and %d tools",
                redact_sensitive_data(url, [self.config[CONF_LLM_API_KEY]]),
                len(messages),
                len(tool_definitions) if tool_definitions else 0,
            )

        try:
            async with session.post(url, headers=headers, json=payload, allow_redirects=False) as response:
                if response.status == 401:
                    raise AuthenticationError("LLM API authentication failed. Check your API key.")

                if response.status != 200:
                    error_text = await response.text()
                    raise HomeAgentError(f"LLM API returned status {response.status}: {error_text}")

                # Stream SSE lines
                async for line in response.content:
                    if line:
                        yield line.decode("utf-8")

        except aiohttp.ClientError as err:
            raise HomeAgentError(f"Failed to connect to LLM API: {err}") from err

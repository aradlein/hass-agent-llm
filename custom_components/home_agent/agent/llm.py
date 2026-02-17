"""LLM API communication mixin for HomeAgent.

This module provides the LLMMixin class that handles all synchronous LLM API
communication for the HomeAgent. It manages HTTP sessions, constructs API requests
in OpenAI-compatible format, and processes responses.

Architecture:
    LLMMixin is designed as a mixin class to be inherited by HomeAgent. It provides
    low-level LLM communication functionality without concerning itself with
    conversation logic, tool execution, or context management.

Key Classes:
    LLMMixin: Mixin providing LLM API call functionality with support for:
        - OpenAI-compatible API communication
        - Tool calling (function calling) support
        - Configurable parameters (temperature, max_tokens, etc.)
        - HTTP session management and connection pooling
        - Authentication and error handling

Core Responsibilities:
    - Establish and manage HTTP sessions with the LLM API
    - Construct API requests in OpenAI-compatible format
    - Handle authentication (Bearer token)
    - Support custom backends via X-Ollama-Backend header
    - Parse and return LLM responses
    - Handle API errors and translate them to appropriate exceptions

Usage Example:
    The mixin is used through inheritance in HomeAgent:

        class HomeAgent(LLMMixin, StreamingMixin, MemoryExtractionMixin):
            def __init__(self, hass, config, session_manager):
                self.config = config
                self._session = None

            async def process_conversation(self, messages):
                # Call LLM via mixin method
                response = await self._call_llm(
                    messages=messages,
                    tools=tool_definitions,
                    temperature=0.7,
                    max_tokens=1000
                )
                return response

Expected Host Class Attributes:
    The mixin expects the host class to provide:

    - config: dict[str, Any]
        Configuration dictionary containing LLM settings:
        - CONF_LLM_BASE_URL: Base URL for the LLM API
        - CONF_LLM_API_KEY: API authentication key
        - CONF_LLM_MODEL: Model name to use
        - CONF_LLM_TEMPERATURE: Sampling temperature (default: 0.7)
        - CONF_LLM_MAX_TOKENS: Maximum response tokens (default: 500)
        - CONF_LLM_TOP_P: Nucleus sampling parameter (default: 1.0)
        - CONF_LLM_KEEP_ALIVE: Model keep-alive duration for Ollama
        - CONF_LLM_PROXY_HEADERS: Custom HTTP headers for routing (dict)

    - _session: aiohttp.ClientSession | None
        HTTP session for API calls, managed by _ensure_session()

API Compatibility:
    This module uses the OpenAI-compatible chat completions API format:

    Request Format:
        POST {base_url}/chat/completions
        Headers:
            - Authorization: Bearer {api_key}
            - Content-Type: application/json
            - Custom proxy headers (optional, from CONF_LLM_PROXY_HEADERS)

        Body:
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "max_tokens": 500,
                "tools": [...],  // Optional
                "tool_choice": "auto"  // When tools provided
            }

    Response Format:
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                    "tool_calls": [...]  // Optional
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

Integration Points:
    - HomeAgent core: Main consumer of LLM functionality
    - MemoryExtractionMixin: Uses _call_llm() for memory extraction
    - aiohttp: HTTP client library for async API communication
    - Custom exception types: AuthenticationError, HomeAgentError

Error Handling:
    The mixin translates API errors into domain-specific exceptions:

    - AuthenticationError: Raised on 401 status (invalid API key)
    - HomeAgentError: Raised on other API errors or network failures
    - aiohttp.ClientError: Network-level failures

Configuration Example:
    config = {
        CONF_LLM_BASE_URL: "https://api.openai.com/v1",
        CONF_LLM_API_KEY: "sk-...",
        CONF_LLM_MODEL: "gpt-4",
        CONF_LLM_TEMPERATURE: 0.7,
        CONF_LLM_MAX_TOKENS: 1000,
        CONF_LLM_TOP_P: 1.0,
        CONF_LLM_PROXY_HEADERS: {"X-Ollama-Backend": "llama-cpp"},  # optional
        CONF_DEBUG_LOGGING: True
    }
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    DEFAULT_LLM_KEEP_ALIVE,
    DEFAULT_RETRY_BACKOFF_FACTOR,
    DEFAULT_RETRY_INITIAL_DELAY,
    DEFAULT_RETRY_JITTER,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_RETRY_MAX_DELAY,
    HTTP_TIMEOUT,
)
from ..exceptions import AuthenticationError, HomeAgentError
from ..helpers import build_api_url, build_auth_headers, is_ollama_backend, redact_sensitive_data, render_template_value, retry_async

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class LLMMixin:
    """Mixin providing LLM API call functionality.

    This mixin expects the following attributes from the host class:
    - config: dict[str, Any] - Configuration dictionary
    - _session: aiohttp.ClientSession | None - HTTP session
    """

    config: dict[str, Any]
    hass: Any
    _session: aiohttp.ClientSession | None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists.

        Returns:
            Active aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call the LLM API.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions
            temperature: Optional temperature override (uses config if not provided)
            max_tokens: Optional max_tokens override (uses config if not provided)

        Returns:
            LLM response dictionary

        Raises:
            AuthenticationError: If API authentication fails
            HomeAgentError: If API call fails
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
        # Render template if value contains {{ }} (e.g., from input_text helper)
        api_key = render_template_value(self.hass, self.config.get(CONF_LLM_API_KEY, ""))
        headers.update(build_auth_headers(base_url, api_key))

        # Add custom proxy headers if configured
        proxy_headers = self.config.get(CONF_LLM_PROXY_HEADERS, {})
        if proxy_headers:
            headers.update(proxy_headers)

        payload: dict[str, Any] = {
            "model": self.config[CONF_LLM_MODEL],
            "messages": messages,
            "temperature": (
                temperature
                if temperature is not None
                else self.config.get(CONF_LLM_TEMPERATURE, 0.7)
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else self.config.get(CONF_LLM_MAX_TOKENS, 500)
            ),
            "top_p": self.config.get(CONF_LLM_TOP_P, 1.0),
        }

        # Only include keep_alive for Ollama backends (not supported by OpenAI, etc.)
        # See: https://github.com/aradlein/home-agent/issues/65
        if is_ollama_backend(self.config[CONF_LLM_BASE_URL]):
            payload["keep_alive"] = self.config.get(CONF_LLM_KEEP_ALIVE, DEFAULT_LLM_KEEP_ALIVE)

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if self.config.get(CONF_DEBUG_LOGGING):
            _LOGGER.debug(
                "Calling LLM at %s with %d messages and %d tools",
                redact_sensitive_data(url, [self.config[CONF_LLM_API_KEY]]),
                len(messages),
                len(tools) if tools else 0,
            )

        async def make_llm_request() -> dict[str, Any]:
            """Make the LLM API request."""
            try:
                async with session.post(url, headers=headers, json=payload, allow_redirects=False) as response:
                    if response.status == 401:
                        raise AuthenticationError(
                            "LLM API authentication failed. Check your API key "
                            "configuration. If using a Jinja2 template for the API key, "
                            "verify it renders correctly. If using a proxy or gateway, "
                            "check that the Authorization header is being forwarded."
                        )

                    if response.status != 200:
                        error_text = await response.text()
                        error_lower = error_text.lower()
                        if "authorization" in error_lower or "authentication" in error_lower or "auth" in error_lower:
                            raise HomeAgentError(
                                f"LLM API returned status {response.status}: {error_text}. "
                                "This may indicate an API key configuration issue. "
                                "Check that your API key is set correctly. If using a "
                                "Jinja2 template, verify it renders to a valid key. "
                                "If using a proxy or gateway (e.g., Cloudflare AI Gateway), "
                                "ensure the Authorization header is being forwarded."
                            )
                        raise HomeAgentError(f"LLM API returned status {response.status}: {error_text}")

                    result: dict[str, Any] = await response.json()
                    return result

            except aiohttp.ClientError as err:
                raise HomeAgentError(f"Failed to connect to LLM API: {err}") from err

        return await retry_async(
            make_llm_request,
            max_retries=DEFAULT_RETRY_MAX_ATTEMPTS,
            retryable_exceptions=(aiohttp.ClientError,),
            non_retryable_exceptions=(AuthenticationError,),
            initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
            backoff_factor=DEFAULT_RETRY_BACKOFF_FACTOR,
            max_delay=DEFAULT_RETRY_MAX_DELAY,
            jitter=DEFAULT_RETRY_JITTER,
        )

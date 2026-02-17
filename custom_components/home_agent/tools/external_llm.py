"""External LLM tool for the Home Agent integration.

This module provides the ExternalLLMTool that enables the primary LLM to
delegate complex queries to a more powerful external LLM API, implementing
a dual-LLM strategy where a fast local model handles tool execution but can
delegate to powerful cloud models for complex analysis.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import aiohttp
from homeassistant.core import HomeAssistant

from ..helpers import render_template_value
from ..const import (
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_KEEP_ALIVE,
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
    CONF_TOOLS_TIMEOUT,
    DEFAULT_EXTERNAL_LLM_KEEP_ALIVE,
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_MODEL,
    DEFAULT_EXTERNAL_LLM_TEMPERATURE,
    DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
    DEFAULT_TOOLS_TIMEOUT,
    TOOL_QUERY_EXTERNAL_LLM,
)
from ..exceptions import ToolExecutionError, ValidationError
from ..helpers import build_api_url, build_auth_headers, is_ollama_backend
from .registry import BaseTool

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class ExternalLLMTool(BaseTool):
    """Tool for querying an external LLM for complex analysis.

    This tool allows the primary LLM to delegate complex queries to a more
    powerful external LLM API. The primary LLM decides when to use this tool
    and must explicitly provide context via the prompt and context parameters.

    Example tool calls:
        # Analyze energy usage data
        {
            "prompt": "Analyze this energy usage and suggest optimizations",
            "context": {
                "energy_data": {
                    "sensor.energy_usage": [{"time": "...", "value": 150}, ...]
                }
            }
        }

        # Get complex recommendations
        {
            "prompt": "How can I improve my home automation setup?",
            "context": {
                "current_automations": [...],
                "devices": [...]
            }
        }

        # Simple query without additional context
        {
            "prompt": "Explain the benefits of automating lighting"
        }
    """

    def __init__(
        self,
        hass: HomeAssistant,
        config: dict[str, Any],
    ) -> None:
        """Initialize the external LLM tool.

        Args:
            hass: Home Assistant instance
            config: Configuration dictionary containing external LLM settings
        """
        super().__init__(hass)
        self._config = config
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        """Return the tool name."""
        return TOOL_QUERY_EXTERNAL_LLM

    @property
    def description(self) -> str:
        """Return the tool description."""
        return str(
            self._config.get(
                CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
            )
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Question or prompt for the external LLM. "
                        "Be specific and include all necessary context in this prompt "
                        "since conversation history is not automatically shared."
                    ),
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Additional explicit context to provide to the external LLM. "
                        "This can include sensor data, tool results, or other relevant "
                        "information. The conversation history is NOT automatically "
                        "included, so include any necessary context here."
                    ),
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute a query to the external LLM.

        Args:
            prompt: Question or prompt for the external LLM (required)
            context: Additional explicit context (optional)

        Returns:
            Dict containing:
                - success: bool indicating if execution succeeded
                - result: External LLM response text (if successful)
                - error: Error message (if failed)

        Raises:
            ValidationError: If parameters are invalid
        """
        prompt = kwargs.get("prompt")
        context = kwargs.get("context", {})

        # Validate required parameters
        if not prompt:
            raise ValidationError("Parameter 'prompt' is required")

        if not isinstance(prompt, str):
            raise ValidationError(
                f"Parameter 'prompt' must be a string, got {type(prompt).__name__}"
            )

        if context and not isinstance(context, dict):
            raise ValidationError(
                f"Parameter 'context' must be an object/dict, got {type(context).__name__}"
            )

        # Build the message to send to external LLM
        message_content = prompt
        if context:
            # Format context as a readable string to include with the prompt
            context_str = self._format_context(context)
            message_content = f"{prompt}\n\nAdditional Context:\n{context_str}"

        _LOGGER.info(
            "Querying external LLM with prompt length: %d chars, context keys: %s",
            len(prompt),
            list(context.keys()) if context else [],
        )

        try:
            # Call external LLM with timeout
            timeout = self._config.get(CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT)
            response_text = await asyncio.wait_for(
                self._call_external_llm(message_content),
                timeout=timeout,
            )

            _LOGGER.info(
                "External LLM query successful, response length: %d chars",
                len(response_text),
            )

            return {
                "success": True,
                "result": response_text,
                "error": None,
            }

        except asyncio.TimeoutError:
            error_msg = f"External LLM request timed out after {timeout} seconds"
            _LOGGER.warning(error_msg)
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

        except aiohttp.ClientResponseError as err:
            if err.status == 401:
                error_msg = "External LLM authentication failed. Check your API key."
                _LOGGER.warning(error_msg)
            elif err.status == 429:
                error_msg = "External LLM rate limit exceeded. Please try again later."
                _LOGGER.warning(error_msg)
            else:
                error_msg = f"External LLM API error (status {err.status}): {err.message}"
                _LOGGER.warning("External LLM API error: %s", err, exc_info=True)

            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

        except aiohttp.ClientError as err:
            error_msg = f"Failed to connect to external LLM: {err}"
            _LOGGER.warning("External LLM connection error: %s", err, exc_info=True)
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

        except Exception as err:
            error_msg = f"Unexpected error querying external LLM: {err}"
            _LOGGER.error("Unexpected external LLM error: %s", err, exc_info=True)
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context dictionary as a readable string.

        Args:
            context: Context dictionary to format

        Returns:
            Formatted context string
        """
        import json

        try:
            # Pretty print JSON for readability
            return json.dumps(context, indent=2, default=str)
        except Exception as err:
            _LOGGER.warning("Failed to format context as JSON: %s", err)
            # Fallback to simple string representation
            return str(context)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists.

        Returns:
            Active aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self._config.get(CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT)
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _call_external_llm(self, message: str) -> str:
        """Call the external LLM API.

        Args:
            message: Message content to send to the LLM

        Returns:
            LLM response text

        Raises:
            aiohttp.ClientError: If API call fails
        """
        session = await self._ensure_session()

        # Get configuration with defaults
        base_url = self._config.get(CONF_EXTERNAL_LLM_BASE_URL)
        api_key = render_template_value(
            self.hass, self._config.get(CONF_EXTERNAL_LLM_API_KEY, "")
        )
        model = self._config.get(CONF_EXTERNAL_LLM_MODEL, DEFAULT_EXTERNAL_LLM_MODEL)
        temperature = self._config.get(
            CONF_EXTERNAL_LLM_TEMPERATURE, DEFAULT_EXTERNAL_LLM_TEMPERATURE
        )
        max_tokens = self._config.get(CONF_EXTERNAL_LLM_MAX_TOKENS, DEFAULT_EXTERNAL_LLM_MAX_TOKENS)

        # Validate required configuration
        if not base_url:
            raise ValidationError(
                "External LLM base URL is not configured. "
                "Please configure it in the integration settings."
            )

        if not api_key:
            raise ValidationError(
                "External LLM API key is not configured. "
                "Please configure it in the integration settings."
            )

        # Build request (Azure OpenAI uses different URL structure and auth)
        url = build_api_url(base_url, model)
        headers = {
            "Content-Type": "application/json",
        }
        headers.update(build_auth_headers(base_url, api_key))

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Only include keep_alive for Ollama backends (not supported by OpenAI, etc.)
        # See: https://github.com/aradlein/home-agent/issues/65
        if is_ollama_backend(base_url):
            payload["keep_alive"] = self._config.get(
                CONF_EXTERNAL_LLM_KEEP_ALIVE, DEFAULT_EXTERNAL_LLM_KEEP_ALIVE
            )

        _LOGGER.debug(
            "Calling external LLM at %s with model %s",
            url,
            model,
        )

        # Make API call
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            result = await response.json()

            # Extract response text from OpenAI-compatible format
            choices = result.get("choices", [])
            if not choices:
                raise ToolExecutionError("External LLM returned empty response (no choices)")

            message_obj = choices[0].get("message", {})
            content = message_obj.get("content", "")

            if not content:
                raise ToolExecutionError("External LLM returned empty content in response")

            return str(content)

    async def close(self) -> None:
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()

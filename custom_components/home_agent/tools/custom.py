"""Custom tool framework for the Home Agent integration.

This module provides the CustomToolHandler factory and custom tool implementations
that allow users to define custom tools via configuration.yaml, including REST API
handlers and Home Assistant service handlers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.template import Template

from ..const import (
    CONF_TOOLS_TIMEOUT,
    CUSTOM_TOOL_HANDLER_REST,
    CUSTOM_TOOL_HANDLER_SERVICE,
    DEFAULT_TOOLS_TIMEOUT,
)
from ..exceptions import ValidationError
from .registry import BaseTool

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class CustomToolHandler:
    """Factory for creating custom tools from configuration.

    This class provides a factory method to create custom tool instances
    based on the handler type specified in the configuration.

    Supported handler types:
        - rest: REST API calls with configurable HTTP methods, headers, and body
        - service: Home Assistant service calls (future implementation)

    Example:
        config = {
            "name": "check_weather",
            "description": "Get weather forecast",
            "parameters": {...},
            "handler": {
                "type": "rest",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET"
            }
        }
        tool = CustomToolHandler.create_tool_from_config(hass, config)
    """

    @staticmethod
    def create_tool_from_config(
        hass: HomeAssistant,
        config: dict[str, Any],
    ) -> BaseTool:
        """Create a custom tool from configuration.

        Args:
            hass: Home Assistant instance
            config: Tool configuration dictionary containing:
                - name: Tool name
                - description: Tool description
                - parameters: JSON Schema for tool parameters
                - handler: Handler configuration with type and settings

        Returns:
            BaseTool instance configured according to the handler type

        Raises:
            ValidationError: If configuration is invalid or handler type unsupported
        """
        # Validate required configuration keys
        required_keys = ["name", "description", "handler"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(
                f"Custom tool configuration missing required keys: {', '.join(missing_keys)}"
            )

        # If parameters not provided, use empty object schema
        if "parameters" not in config:
            config["parameters"] = {
                "type": "object",
                "properties": {},
                "required": []
            }

        handler_config = config["handler"]
        if "type" not in handler_config:
            raise ValidationError("Handler configuration must include 'type' field")

        handler_type = handler_config["type"]

        # Create tool based on handler type
        if handler_type == CUSTOM_TOOL_HANDLER_REST:
            return RestCustomTool(hass, config)
        elif handler_type == CUSTOM_TOOL_HANDLER_SERVICE:
            # Future implementation for Phase 3
            raise ValidationError(
                f"Handler type '{handler_type}' is not yet implemented. "
                f"Currently supported: {CUSTOM_TOOL_HANDLER_REST}"
            )
        else:
            raise ValidationError(
                f"Unknown handler type: '{handler_type}'. "
                f"Supported types: {CUSTOM_TOOL_HANDLER_REST}, {CUSTOM_TOOL_HANDLER_SERVICE}"
            )


class RestCustomTool(BaseTool):
    """Custom tool that calls external REST APIs.

    This tool enables calling external HTTP APIs with configurable methods,
    headers, query parameters, and request bodies. It supports template
    rendering for dynamic values including secrets.

    Configuration example:
        {
            "name": "check_weather",
            "description": "Get weather forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            },
            "handler": {
                "type": "rest",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET",
                "headers": {
                    "Authorization": "Bearer {{ secrets.weather_api_key }}"
                },
                "query_params": {
                    "location": "{{ location }}"
                }
            }
        }

    All custom tools return standardized format:
        {
            "success": true,
            "result": {...},  # Parsed response
            "error": null
        }
    """

    def __init__(self, hass: HomeAssistant, config: dict[str, Any]) -> None:
        """Initialize the REST custom tool.

        Args:
            hass: Home Assistant instance
            config: Tool configuration dictionary
        """
        super().__init__(hass)
        self._config = config
        self._handler_config = config["handler"]
        self._session: aiohttp.ClientSession | None = None

        # Validate REST-specific configuration
        self._validate_rest_config()

    def _validate_rest_config(self) -> None:
        """Validate REST handler configuration.

        Raises:
            ValidationError: If REST configuration is invalid
        """
        required_rest_keys = ["url", "method"]
        missing_keys = [
            key for key in required_rest_keys if key not in self._handler_config
        ]
        if missing_keys:
            raise ValidationError(
                f"REST handler configuration missing required keys: {', '.join(missing_keys)}"
            )

        # Validate HTTP method
        valid_methods = ["GET", "POST", "PUT", "DELETE"]
        method = self._handler_config["method"]
        if method not in valid_methods:
            raise ValidationError(
                f"Invalid HTTP method '{method}'. "
                f"Supported methods: {', '.join(valid_methods)}"
            )

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self._config["name"]

    @property
    def description(self) -> str:
        """Return the tool description."""
        return self._config["description"]

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema."""
        return self._config["parameters"]

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the REST API call with the given parameters.

        Args:
            **kwargs: Tool parameters as defined in the schema

        Returns:
            Dict containing:
                - success: bool indicating if execution succeeded
                - result: Parsed response data (if successful)
                - error: Error message (if failed)
        """
        try:
            # Render templates with provided parameters
            url = await self._render_template(
                self._handler_config["url"],
                kwargs,
            )
            method = self._handler_config["method"]

            # Render headers if present
            headers = {}
            if "headers" in self._handler_config:
                for key, value in self._handler_config["headers"].items():
                    headers[key] = await self._render_template(value, kwargs)

            # Render query parameters if present
            params = {}
            if "query_params" in self._handler_config:
                for key, value in self._handler_config["query_params"].items():
                    params[key] = await self._render_template(value, kwargs)

            # Render request body if present (for POST/PUT)
            json_data = None
            if "body" in self._handler_config and self._handler_config["body"]:
                # Render each value in the body dict
                body = {}
                for key, value in self._handler_config["body"].items():
                    if isinstance(value, str):
                        body[key] = await self._render_template(value, kwargs)
                    else:
                        body[key] = value
                json_data = body

            _LOGGER.info(
                "Executing REST custom tool '%s': %s %s",
                self.name,
                method,
                url,
            )

            # Get timeout from configuration
            timeout_seconds = self._config.get(
                CONF_TOOLS_TIMEOUT,
                DEFAULT_TOOLS_TIMEOUT,
            )

            # Make HTTP request with timeout
            response_data = await asyncio.wait_for(
                self._make_request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json_data=json_data,
                ),
                timeout=timeout_seconds,
            )

            _LOGGER.info(
                "REST custom tool '%s' executed successfully",
                self.name,
            )

            return {
                "success": True,
                "result": response_data,
                "error": None,
            }

        except asyncio.TimeoutError:
            timeout = self._config.get(CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT)
            error_msg = f"Request timed out after {timeout} seconds"
            _LOGGER.warning(
                "REST custom tool '%s' timed out: %s",
                self.name,
                error_msg,
            )
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

        except aiohttp.ClientResponseError as err:
            error_msg = f"HTTP {err.status}: {err.message}"
            _LOGGER.warning(
                "REST custom tool '%s' HTTP error: %s",
                self.name,
                error_msg,
                exc_info=True,
            )
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

        except aiohttp.ClientError as err:
            error_msg = f"Network error: {err}"
            _LOGGER.warning(
                "REST custom tool '%s' network error: %s",
                self.name,
                error_msg,
                exc_info=True,
            )
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

        except Exception as err:
            error_msg = f"Unexpected error: {err}"
            _LOGGER.error(
                "REST custom tool '%s' unexpected error: %s",
                self.name,
                error_msg,
                exc_info=True,
            )
            return {
                "success": False,
                "result": None,
                "error": error_msg,
            }

    async def _render_template(
        self,
        template_str: str,
        variables: dict[str, Any],
    ) -> str:
        """Render a template string with variables.

        Supports Home Assistant template syntax including secrets.

        Args:
            template_str: Template string to render
            variables: Variables to make available in the template

        Returns:
            Rendered string

        Raises:
            ValidationError: If template rendering fails
        """
        # If not a template (no {{ }}), return as-is
        if "{{" not in template_str:
            return template_str

        try:
            # Create Home Assistant template
            template = Template(template_str, self.hass)

            # Render with provided variables
            rendered = template.async_render(variables)

            return str(rendered)

        except Exception as err:
            raise ValidationError(
                f"Failed to render template '{template_str}': {err}"
            ) from err

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session exists.

        Returns:
            Active aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            # Use Home Assistant's shared client session for better connection pooling
            self._session = async_get_clientsession(self.hass)
        return self._session

    async def _make_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> Any:
        """Make HTTP request to external API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Request URL
            headers: Optional request headers
            params: Optional query parameters
            json_data: Optional JSON request body

        Returns:
            Parsed response data (JSON or text)

        Raises:
            aiohttp.ClientError: If request fails
        """
        session = await self._ensure_session()

        _LOGGER.debug(
            "Making %s request to %s with params=%s",
            method,
            url,
            params,
        )

        async with session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_data,
        ) as response:
            # Raise exception for HTTP errors
            response.raise_for_status()

            # Try to parse as JSON first
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return await response.json()
            else:
                # Return as text for non-JSON responses
                text = await response.text()
                _LOGGER.debug(
                    "Non-JSON response (Content-Type: %s), returning as text",
                    content_type,
                )
                return text

    async def close(self) -> None:
        """Clean up resources.

        Note: We use Home Assistant's shared session, so we don't close it here.
        """
        # Don't close the shared session
        pass

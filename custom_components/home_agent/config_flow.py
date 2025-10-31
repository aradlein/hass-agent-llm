"""Config flow for Home Agent integration.

This module implements the configuration UI for the Home Agent custom component,
providing multi-step configuration flows for initial setup and options management.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_DIRECT_ENTITIES,
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    CONTEXT_FORMAT_HYBRID,
    CONTEXT_FORMAT_JSON,
    CONTEXT_FORMAT_NATURAL_LANGUAGE,
    CONTEXT_MODE_DIRECT,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_CONTEXT_FORMAT,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_DEBUG_LOGGING,
    DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    DEFAULT_EXTERNAL_LLM_ENABLED,
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_MODEL,
    DEFAULT_EXTERNAL_LLM_TEMPERATURE,
    DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
    DEFAULT_HISTORY_ENABLED,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_PROMPT_USE_DEFAULT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
    DEFAULT_TOOLS_TIMEOUT,
    DOMAIN,
)
from .exceptions import AuthenticationError, ValidationError

_LOGGER = logging.getLogger(__name__)

# OpenAI default base URL
OPENAI_BASE_URL = "https://api.openai.com/v1"


class HomeAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Home Agent.

    This config flow implements multi-step configuration for the Home Agent
    integration, including initial LLM setup and validation.
    """

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._data: dict[str, Any] = {}
        self._test_connection_passed = False

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - LLM configuration.

        This step collects basic LLM configuration including:
        - Integration name
        - LLM base URL (OpenAI-compatible endpoint)
        - API key
        - Model name
        - Temperature
        - Max tokens

        Args:
            user_input: User-provided configuration data

        Returns:
            FlowResult indicating next step or completion
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                # Validate the configuration
                await self._validate_llm_config(user_input)

                # Store the configuration
                self._data.update(user_input)

                # Create the config entry with basic configuration
                # Options flow will handle advanced settings
                return self.async_create_entry(
                    title=user_input.get("name", DEFAULT_NAME),
                    data=self._data,
                )

            except ValidationError as err:
                _LOGGER.error("Validation error: %s", err)
                errors["base"] = "invalid_config"
            except AuthenticationError as err:
                _LOGGER.error("Authentication error: %s", err)
                errors["base"] = "invalid_auth"
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error during config flow: %s", err)
                errors["base"] = "unknown"

        # Show the configuration form
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required("name", default=DEFAULT_NAME): str,
                    vol.Required(
                        CONF_LLM_BASE_URL,
                        default=OPENAI_BASE_URL,
                    ): str,
                    vol.Required(CONF_LLM_API_KEY): str,
                    vol.Required(
                        CONF_LLM_MODEL,
                        default=DEFAULT_LLM_MODEL,
                    ): str,
                    vol.Optional(
                        CONF_LLM_TEMPERATURE,
                        default=DEFAULT_TEMPERATURE,
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                    vol.Optional(
                        CONF_LLM_MAX_TOKENS,
                        default=DEFAULT_MAX_TOKENS,
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
                }
            ),
            errors=errors,
            description_placeholders={
                "openai_url": OPENAI_BASE_URL,
                "ollama_url": "http://localhost:11434/v1",
                "default_model": DEFAULT_LLM_MODEL,
            },
        )

    async def _validate_llm_config(self, config: dict[str, Any]) -> None:
        """Validate LLM configuration.

        Validates:
        - URL format is correct
        - API key is provided and not empty
        - Temperature is within valid range
        - Max tokens is reasonable

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValidationError: If configuration is invalid
            AuthenticationError: If API key is invalid (optional test)
        """
        # Validate URL format
        base_url = config.get(CONF_LLM_BASE_URL, "")
        if not base_url:
            raise ValidationError("LLM base URL cannot be empty")

        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError(
                f"Invalid URL format: {base_url}. "
                "Expected format: https://api.example.com/v1"
            )

        if parsed.scheme not in ("http", "https"):
            raise ValidationError(
                f"URL scheme must be http or https, got: {parsed.scheme}"
            )

        # Validate API key
        api_key = config.get(CONF_LLM_API_KEY, "")
        if not api_key or not api_key.strip():
            raise ValidationError("API key cannot be empty")

        # Validate model name
        model = config.get(CONF_LLM_MODEL, "")
        if not model or not model.strip():
            raise ValidationError("Model name cannot be empty")

        # Temperature and max_tokens are validated by voluptuous schema
        # but we can add additional checks here if needed
        temperature = config.get(CONF_LLM_TEMPERATURE, DEFAULT_TEMPERATURE)
        if not 0.0 <= temperature <= 2.0:
            raise ValidationError(
                f"Temperature must be between 0.0 and 2.0, got: {temperature}"
            )

        max_tokens = config.get(CONF_LLM_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        if max_tokens < 1:
            raise ValidationError(f"Max tokens must be at least 1, got: {max_tokens}")

    async def _test_llm_connection(self, config: dict[str, Any]) -> bool:
        """Test connection to LLM API.

        Optional validation step to verify the LLM configuration works
        by attempting a minimal API call.

        Args:
            config: LLM configuration to test

        Returns:
            True if connection successful, False otherwise

        Note:
            This is an optional enhancement. Currently not called in the main flow
            but can be enabled if desired.
        """
        base_url = config[CONF_LLM_BASE_URL]
        api_key = config[CONF_LLM_API_KEY]
        model = config[CONF_LLM_MODEL]

        # Construct the chat completions endpoint
        url = f"{base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Minimal test payload
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 5,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key")
                    if response.status == 404:
                        raise ValidationError(
                            f"Endpoint not found. Verify the base URL: {base_url}"
                        )
                    if response.status >= 400:
                        error_text = await response.text()
                        raise ValidationError(
                            f"API returned error {response.status}: {error_text}"
                        )

                    # Success
                    _LOGGER.debug("LLM connection test successful")
                    return True

        except aiohttp.ClientError as err:
            _LOGGER.error("Connection error during LLM test: %s", err)
            raise ValidationError(
                f"Failed to connect to LLM at {base_url}: {err}"
            ) from err

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> HomeAgentOptionsFlow:
        """Get the options flow for this handler.

        Args:
            config_entry: The config entry to create options flow for

        Returns:
            HomeAgentOptionsFlow instance
        """
        return HomeAgentOptionsFlow(config_entry)


class HomeAgentOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Home Agent.

    This options flow provides advanced configuration settings including:
    - Context injection mode and settings
    - Conversation history configuration
    - System prompt customization
    - Tool settings
    - External LLM tool configuration
    - Debug logging
    """

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize the options flow.

        Args:
            config_entry: The config entry to manage options for
        """
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options - main menu.

        Presents a menu of configuration categories:
        - Context settings
        - History settings
        - Prompt settings
        - Tool settings
        - External LLM settings
        - Debug settings

        Args:
            user_input: User selection

        Returns:
            FlowResult indicating next step
        """
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "llm_settings",
                "context_settings",
                "history_settings",
                "prompt_settings",
                "tool_settings",
                "external_llm_settings",
                "debug_settings",
            ],
        )

    async def async_step_llm_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure LLM settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate LLM configuration
            try:
                # Merge user input with entry data (not options)
                # LLM settings should update the entry.data
                test_config = dict(self._config_entry.data) | user_input

                # Create a temporary config flow instance to validate
                temp_flow = HomeAgentConfigFlow()
                temp_flow.hass = self.hass
                await temp_flow._validate_llm_config(test_config)

                # Update the config entry data with new LLM settings
                self.hass.config_entries.async_update_entry(
                    self._config_entry, data={**self._config_entry.data, **user_input}
                )

                return self.async_create_entry(title="", data={})

            except ValidationError as err:
                _LOGGER.error("LLM validation error: %s", err)
                errors["base"] = "invalid_config"
            except AuthenticationError as err:
                _LOGGER.error("LLM authentication error: %s", err)
                errors["base"] = "invalid_auth"
            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected error updating LLM settings: %s", err)
                errors["base"] = "unknown"

        # Get current values from entry.data (not options)
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="llm_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_LLM_BASE_URL,
                        default=current_data.get(CONF_LLM_BASE_URL, OPENAI_BASE_URL),
                    ): str,
                    vol.Required(
                        CONF_LLM_API_KEY,
                        default=current_data.get(CONF_LLM_API_KEY, ""),
                    ): str,
                    vol.Required(
                        CONF_LLM_MODEL,
                        default=current_data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
                    ): str,
                    vol.Optional(
                        CONF_LLM_TEMPERATURE,
                        default=current_data.get(
                            CONF_LLM_TEMPERATURE, DEFAULT_TEMPERATURE
                        ),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                    vol.Optional(
                        CONF_LLM_MAX_TOKENS,
                        default=current_data.get(
                            CONF_LLM_MAX_TOKENS, DEFAULT_MAX_TOKENS
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
                }
            ),
            errors=errors,
            description_placeholders={
                "openai_url": OPENAI_BASE_URL,
                "ollama_url": "http://localhost:11434/v1",
            },
        )

    async def async_step_context_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure context injection settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            # Update options with new context settings
            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="context_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CONTEXT_MODE,
                        default=current_options.get(
                            CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE
                        ),
                    ): vol.In([CONTEXT_MODE_DIRECT, CONTEXT_MODE_VECTOR_DB]),
                    vol.Optional(
                        CONF_CONTEXT_FORMAT,
                        default=current_options.get(
                            CONF_CONTEXT_FORMAT, DEFAULT_CONTEXT_FORMAT
                        ),
                    ): vol.In(
                        [
                            CONTEXT_FORMAT_JSON,
                            CONTEXT_FORMAT_NATURAL_LANGUAGE,
                            CONTEXT_FORMAT_HYBRID,
                        ]
                    ),
                    vol.Optional(
                        CONF_DIRECT_ENTITIES,
                        default=current_options.get(CONF_DIRECT_ENTITIES, ""),
                    ): str,
                }
            ),
            description_placeholders={
                "direct_mode": CONTEXT_MODE_DIRECT,
                "vector_db_mode": CONTEXT_MODE_VECTOR_DB,
                "entity_format": "sensor.temperature,light.living_room",
            },
        )

    async def async_step_history_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure conversation history settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="history_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_HISTORY_ENABLED,
                        default=current_options.get(
                            CONF_HISTORY_ENABLED, DEFAULT_HISTORY_ENABLED
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_HISTORY_MAX_MESSAGES,
                        default=current_options.get(
                            CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
                    vol.Optional(
                        CONF_HISTORY_MAX_TOKENS,
                        default=current_options.get(
                            CONF_HISTORY_MAX_TOKENS, DEFAULT_HISTORY_MAX_TOKENS
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=100, max=50000)),
                }
            ),
        )

    async def async_step_prompt_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure system prompt settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="prompt_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROMPT_USE_DEFAULT,
                        default=current_options.get(
                            CONF_PROMPT_USE_DEFAULT, DEFAULT_PROMPT_USE_DEFAULT
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_PROMPT_CUSTOM_ADDITIONS,
                        description={
                            "suggested_value": current_options.get(
                                CONF_PROMPT_CUSTOM_ADDITIONS, ""
                            )
                        },
                    ): selector.TemplateSelector(),
                }
            ),
            description_placeholders={
                "example_addition": (
                    "Additional context about my home:\n"
                    "- The thermostat prefers 68-72°F\n"
                    "- Keep doors locked after 10 PM"
                ),
            },
        )

    async def async_step_tool_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure tool execution settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="tool_settings",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TOOLS_MAX_CALLS_PER_TURN,
                        default=current_options.get(
                            CONF_TOOLS_MAX_CALLS_PER_TURN,
                            DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
                    vol.Optional(
                        CONF_TOOLS_TIMEOUT,
                        default=current_options.get(
                            CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                }
            ),
        )

    async def async_step_external_llm_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure external LLM tool settings.

        The external LLM tool allows the primary LLM to delegate complex
        queries to a more capable model.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            # Validate external LLM config if enabled
            if user_input.get(CONF_EXTERNAL_LLM_ENABLED, False):
                try:
                    await self._validate_external_llm_config(user_input)
                except ValidationError as err:
                    _LOGGER.error("External LLM validation error: %s", err)
                    return self.async_show_form(
                        step_id="external_llm_settings",
                        data_schema=self._get_external_llm_schema(user_input),
                        errors={"base": "invalid_external_llm"},
                    )

            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="external_llm_settings",
            data_schema=self._get_external_llm_schema(current_options),
            description_placeholders={
                "use_case": (
                    "Enable this to allow the primary LLM to delegate "
                    "complex queries to a more capable external model"
                ),
            },
        )

    def _get_external_llm_schema(self, current_options: dict[str, Any]) -> vol.Schema:
        """Get schema for external LLM settings.

        Args:
            current_options: Current option values

        Returns:
            Voluptuous schema for external LLM configuration
        """
        return vol.Schema(
            {
                vol.Required(
                    CONF_EXTERNAL_LLM_ENABLED,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_ENABLED, DEFAULT_EXTERNAL_LLM_ENABLED
                    ),
                ): bool,
                vol.Optional(
                    CONF_EXTERNAL_LLM_BASE_URL,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_BASE_URL, OPENAI_BASE_URL
                    ),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_API_KEY,
                    default=current_options.get(CONF_EXTERNAL_LLM_API_KEY, ""),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_MODEL,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_MODEL, DEFAULT_EXTERNAL_LLM_MODEL
                    ),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_TEMPERATURE,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_TEMPERATURE,
                        DEFAULT_EXTERNAL_LLM_TEMPERATURE,
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                vol.Optional(
                    CONF_EXTERNAL_LLM_MAX_TOKENS,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_MAX_TOKENS,
                        DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
                vol.Optional(
                    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                    description={
                        "suggested_value": current_options.get(
                            CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                            DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
                        )
                    },
                ): selector.TemplateSelector(),
                vol.Optional(
                    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                        DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                    ),
                ): bool,
            }
        )

    async def _validate_external_llm_config(self, config: dict[str, Any]) -> None:
        """Validate external LLM configuration.

        Args:
            config: External LLM configuration to validate

        Raises:
            ValidationError: If configuration is invalid
        """
        base_url = config.get(CONF_EXTERNAL_LLM_BASE_URL, "")
        if not base_url:
            raise ValidationError("External LLM base URL cannot be empty")

        parsed = urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError(f"Invalid external LLM URL format: {base_url}")

        api_key = config.get(CONF_EXTERNAL_LLM_API_KEY, "")
        if not api_key or not api_key.strip():
            raise ValidationError("External LLM API key cannot be empty")

        model = config.get(CONF_EXTERNAL_LLM_MODEL, "")
        if not model or not model.strip():
            raise ValidationError("External LLM model name cannot be empty")

    async def async_step_debug_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Configure debug and logging settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options

        return self.async_show_form(
            step_id="debug_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_DEBUG_LOGGING,
                        default=current_options.get(
                            CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING
                        ),
                    ): bool,
                }
            ),
            description_placeholders={
                "warning": (
                    "Debug logging may expose sensitive information "
                    "in Home Assistant logs. Use with caution."
                ),
            },
        )

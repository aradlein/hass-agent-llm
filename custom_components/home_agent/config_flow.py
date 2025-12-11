"""Config flow for Home Agent integration.

This module implements the configuration UI for the Home Agent custom component,
providing multi-step configuration flows for initial setup and options management.

NOTE: A refactored modular version of this code is available in the config/ package:
- config/flow.py - ConfigFlow and OptionsFlow classes
- config/schemas.py - Schema definitions
- config/validators.py - Validation logic
- config/utils.py - Helper functions

The refactored version can be adopted by replacing this file's imports when ready.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_ADDITIONAL_COLLECTIONS,
    CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
    CONF_ADDITIONAL_TOP_K,
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_DIRECT_ENTITIES,
    CONF_EMBEDDING_KEEP_ALIVE,
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_EXTERNAL_LLM_KEEP_ALIVE,
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
    CONF_HISTORY_ENABLED,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MAX_TOKENS,
    CONF_LLM_API_KEY,
    CONF_LLM_BACKEND,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_ALIVE,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_PROXY_HEADERS,
    CONF_LLM_TEMPERATURE,
    CONF_MEMORY_COLLECTION_NAME,
    CONF_MEMORY_CONTEXT_TOP_K,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_MEMORY_EXTRACTION_LLM,
    CONF_MEMORY_MAX_MEMORIES,
    CONF_MEMORY_MIN_IMPORTANCE,
    CONF_MEMORY_MIN_WORDS,
    CONF_OPENAI_API_KEY,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_SESSION_PERSISTENCE_ENABLED,
    CONF_SESSION_TIMEOUT,
    CONF_STREAMING_ENABLED,
    CONF_THINKING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    CONF_VECTOR_DB_COLLECTION,
    CONF_VECTOR_DB_EMBEDDING_BASE_URL,
    CONF_VECTOR_DB_EMBEDDING_MODEL,
    CONF_VECTOR_DB_EMBEDDING_PROVIDER,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
    CONF_VECTOR_DB_TOP_K,
    CONTEXT_FORMAT_HYBRID,
    CONTEXT_FORMAT_JSON,
    CONTEXT_FORMAT_NATURAL_LANGUAGE,
    CONTEXT_MODE_DIRECT,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_ADDITIONAL_COLLECTIONS,
    DEFAULT_ADDITIONAL_L2_DISTANCE_THRESHOLD,
    DEFAULT_ADDITIONAL_TOP_K,
    DEFAULT_CONTEXT_FORMAT,
    DEFAULT_CONTEXT_MODE,
    DEFAULT_DEBUG_LOGGING,
    DEFAULT_EMBEDDING_KEEP_ALIVE,
    DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
    DEFAULT_EXTERNAL_LLM_ENABLED,
    DEFAULT_EXTERNAL_LLM_KEEP_ALIVE,
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_MODEL,
    DEFAULT_EXTERNAL_LLM_TEMPERATURE,
    DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
    DEFAULT_HISTORY_ENABLED,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MAX_TOKENS,
    DEFAULT_LLM_KEEP_ALIVE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEMORY_COLLECTION_NAME,
    DEFAULT_MEMORY_CONTEXT_TOP_K,
    DEFAULT_MEMORY_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_LLM,
    DEFAULT_MEMORY_MAX_MEMORIES,
    DEFAULT_MEMORY_MIN_IMPORTANCE,
    DEFAULT_MEMORY_MIN_WORDS,
    DEFAULT_NAME,
    DEFAULT_PROMPT_USE_DEFAULT,
    DEFAULT_SESSION_PERSISTENCE_ENABLED,
    DEFAULT_SESSION_TIMEOUT,
    DEFAULT_STREAMING_ENABLED,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_ENABLED,
    DEFAULT_TOOLS_MAX_CALLS_PER_TURN,
    DEFAULT_TOOLS_TIMEOUT,
    DEFAULT_VECTOR_DB_COLLECTION,
    DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL,
    DEFAULT_VECTOR_DB_EMBEDDING_MODEL,
    DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER,
    DEFAULT_VECTOR_DB_HOST,
    DEFAULT_VECTOR_DB_PORT,
    DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_DB_TOP_K,
    DOMAIN,
    EMBEDDING_PROVIDER_OLLAMA,
    EMBEDDING_PROVIDER_OPENAI,
    LLM_BACKEND_DEFAULT,
    LLM_BACKEND_LLAMA_CPP,
    LLM_BACKEND_OLLAMA_GPU,
    LLM_BACKEND_VLLM,
)
from .exceptions import AuthenticationError, ValidationError

_LOGGER = logging.getLogger(__name__)

# OpenAI default base URL
OPENAI_BASE_URL = "https://api.openai.com/v1"


def _validate_proxy_headers(headers_input: str | dict[str, str] | None) -> dict[str, str]:
    """Validate proxy headers configuration.

    Args:
        headers_input: Either a JSON string or dict of headers

    Returns:
        Validated headers dictionary

    Raises:
        ValidationError: If headers are invalid
    """
    # Handle None or empty string
    if not headers_input:
        return {}

    # Parse JSON string if needed
    if isinstance(headers_input, str):
        headers_input = headers_input.strip()
        if not headers_input:
            return {}
        try:
            headers = json.loads(headers_input)
        except json.JSONDecodeError as err:
            raise ValidationError(f"Invalid JSON format for proxy headers: {err}") from err
    else:
        headers = headers_input

    # Validate it's a dictionary
    if not isinstance(headers, dict):
        raise ValidationError("Proxy headers must be a JSON object (dictionary)")

    # Validate each header
    for key, value in headers.items():
        # RFC 7230 header name validation
        # Header names must be tokens (alphanumeric and -_)
        if not re.match(r"^[a-zA-Z0-9\-_]+$", key):
            raise ValidationError(
                f"Invalid header name '{key}'. Header names must contain only "
                "alphanumeric characters, hyphens, and underscores (RFC 7230)"
            )

        # Ensure values are strings
        if not isinstance(value, str):
            raise ValidationError(
                f"Header value for '{key}' must be a string, got {type(value).__name__}"
            )

    return headers


def _migrate_legacy_backend(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy llm_backend setting to llm_proxy_headers.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration with migrated settings
    """
    # If proxy_headers already exist, don't migrate
    if CONF_LLM_PROXY_HEADERS in config:
        return config

    # Check for legacy backend setting
    backend = config.get(CONF_LLM_BACKEND)
    if backend and backend != LLM_BACKEND_DEFAULT:
        # Migrate to proxy headers
        config[CONF_LLM_PROXY_HEADERS] = {"X-Ollama-Backend": backend}
        _LOGGER.info("Migrated legacy llm_backend setting '%s' to llm_proxy_headers", backend)

    return config


class HomeAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):  # type: ignore[call-arg]
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
    ) -> config_entries.ConfigFlowResult:
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
                # Parse and validate proxy headers
                if CONF_LLM_PROXY_HEADERS in user_input:
                    validated_headers = _validate_proxy_headers(user_input[CONF_LLM_PROXY_HEADERS])
                    user_input[CONF_LLM_PROXY_HEADERS] = validated_headers

                # Validate the configuration
                await self._validate_llm_config(user_input)

                # Migrate legacy backend setting if needed
                user_input = _migrate_legacy_backend(user_input)

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
                    vol.Optional(
                        CONF_LLM_API_KEY,
                        description={"suggested_value": ""},
                    ): str,
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
                    vol.Optional(
                        CONF_LLM_KEEP_ALIVE,
                        default=DEFAULT_LLM_KEEP_ALIVE,
                    ): str,
                    vol.Optional(
                        CONF_LLM_PROXY_HEADERS,
                        description={"suggested_value": ""},
                    ): str,
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
                f"Invalid URL format: {base_url}. " "Expected format: https://api.example.com/v1"
            )

        if parsed.scheme not in ("http", "https"):
            raise ValidationError(f"URL scheme must be http or https, got: {parsed.scheme}")

        # API key is optional - some local LLMs don't require authentication

        # Validate model name
        model = config.get(CONF_LLM_MODEL, "")
        if not model or not model.strip():
            raise ValidationError("Model name cannot be empty")

        # Temperature and max_tokens are validated by voluptuous schema
        # but we can add additional checks here if needed
        temperature = config.get(CONF_LLM_TEMPERATURE, DEFAULT_TEMPERATURE)
        if not 0.0 <= temperature <= 2.0:
            raise ValidationError(f"Temperature must be between 0.0 and 2.0, got: {temperature}")

        max_tokens = config.get(CONF_LLM_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        if max_tokens < 1:
            raise ValidationError(f"Max tokens must be at least 1, got: {max_tokens}")

        # Validate proxy headers if provided
        if CONF_LLM_PROXY_HEADERS in config:
            _validate_proxy_headers(config[CONF_LLM_PROXY_HEADERS])

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
                        raise ValidationError(f"API returned error {response.status}: {error_text}")

                    # Success
                    _LOGGER.debug("LLM connection test successful")
                    return True

        except aiohttp.ClientError as err:
            _LOGGER.error("Connection error during LLM test: %s", err)
            raise ValidationError(f"Failed to connect to LLM at {base_url}: {err}") from err

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
    ) -> config_entries.ConfigFlowResult:
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
                "vector_db_settings",
                "history_settings",
                "memory_settings",
                "prompt_settings",
                "tool_settings",
                "external_llm_settings",
                "debug_settings",
            ],
        )

    async def async_step_llm_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Configure LLM settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            _LOGGER.debug("LLM settings form submitted with user_input: %s", user_input)
            # Validate LLM configuration
            try:
                # Parse and validate proxy headers
                if CONF_LLM_PROXY_HEADERS in user_input:
                    validated_headers = _validate_proxy_headers(user_input[CONF_LLM_PROXY_HEADERS])
                    user_input[CONF_LLM_PROXY_HEADERS] = validated_headers

                # Ensure API key is included even if empty (for local LLMs that don't need auth)
                # Home Assistant forms may not include empty optional fields in user_input
                if CONF_LLM_API_KEY not in user_input:
                    _LOGGER.debug("API key not in user_input, setting to empty string")
                    user_input[CONF_LLM_API_KEY] = ""

                # Merge user input with entry data (not options)
                # LLM settings should update the entry.data
                test_config = dict(self._config_entry.data) | user_input
                _LOGGER.debug("Merged test_config for validation: %s", test_config)

                # Create a temporary config flow instance to validate
                temp_flow = HomeAgentConfigFlow()
                temp_flow.hass = self.hass
                await temp_flow._validate_llm_config(test_config)
                _LOGGER.debug("LLM config validation passed")

                # Migrate legacy backend setting if needed
                user_input = _migrate_legacy_backend(user_input)

                # Update the config entry data with new LLM settings
                # Note: async_update_entry is synchronous despite the name in HA
                new_data = {**self._config_entry.data, **user_input}
                _LOGGER.debug("Updating config entry data to: %s", new_data)
                self.hass.config_entries.async_update_entry(
                    self._config_entry, data=new_data
                )
                _LOGGER.debug("Config entry data updated successfully")

                # Return current options to preserve them
                # The data update above modifies entry.data, this preserves entry.options
                return self.async_create_entry(title="", data=self._config_entry.options)

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

        # Convert proxy headers dict to JSON string for display
        proxy_headers = current_data.get(CONF_LLM_PROXY_HEADERS, {})
        proxy_headers_str = json.dumps(proxy_headers, indent=2) if proxy_headers else ""

        return self.async_show_form(
            step_id="llm_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_LLM_BASE_URL,
                        default=current_data.get(CONF_LLM_BASE_URL, OPENAI_BASE_URL),
                    ): str,
                    vol.Optional(
                        CONF_LLM_API_KEY,
                        description={
                            "suggested_value": current_data.get(CONF_LLM_API_KEY, "")
                        },
                    ): str,
                    vol.Required(
                        CONF_LLM_MODEL,
                        default=current_data.get(CONF_LLM_MODEL, DEFAULT_LLM_MODEL),
                    ): str,
                    vol.Optional(
                        CONF_LLM_TEMPERATURE,
                        default=current_data.get(CONF_LLM_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                    vol.Optional(
                        CONF_LLM_MAX_TOKENS,
                        default=current_data.get(CONF_LLM_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
                    vol.Optional(
                        CONF_LLM_KEEP_ALIVE,
                        default=current_data.get(CONF_LLM_KEEP_ALIVE, DEFAULT_LLM_KEEP_ALIVE),
                    ): str,
                    vol.Optional(
                        CONF_LLM_PROXY_HEADERS,
                        description={"suggested_value": proxy_headers_str},
                    ): str,
                    vol.Optional(
                        CONF_THINKING_ENABLED,
                        default=current_data.get(CONF_THINKING_ENABLED, DEFAULT_THINKING_ENABLED),
                    ): bool,
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
    ) -> config_entries.ConfigFlowResult:
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
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="context_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_CONTEXT_MODE,
                        default=current_options.get(
                            CONF_CONTEXT_MODE,
                            current_data.get(CONF_CONTEXT_MODE, DEFAULT_CONTEXT_MODE),
                        ),
                    ): vol.In([CONTEXT_MODE_DIRECT, CONTEXT_MODE_VECTOR_DB]),
                    vol.Optional(
                        CONF_CONTEXT_FORMAT,
                        default=current_options.get(
                            CONF_CONTEXT_FORMAT,
                            current_data.get(CONF_CONTEXT_FORMAT, DEFAULT_CONTEXT_FORMAT),
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
                        default=current_options.get(
                            CONF_DIRECT_ENTITIES, current_data.get(CONF_DIRECT_ENTITIES, "")
                        ),
                    ): str,
                }
            ),
            description_placeholders={
                "direct_mode": CONTEXT_MODE_DIRECT,
                "vector_db_mode": CONTEXT_MODE_VECTOR_DB,
                "entity_format": "sensor.temperature,light.living_room",
            },
        )

    async def async_step_vector_db_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Configure Vector DB (ChromaDB) settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            # Convert comma-separated collection names to list
            if CONF_ADDITIONAL_COLLECTIONS in user_input:
                collections_str = user_input[CONF_ADDITIONAL_COLLECTIONS]
                if isinstance(collections_str, str):
                    # Parse comma-separated string to list, removing whitespace
                    collections_list = [c.strip() for c in collections_str.split(",") if c.strip()]
                    user_input[CONF_ADDITIONAL_COLLECTIONS] = collections_list

            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options
        current_data = self._config_entry.data

        # Convert list to comma-separated string for display
        additional_collections = current_options.get(
            CONF_ADDITIONAL_COLLECTIONS, DEFAULT_ADDITIONAL_COLLECTIONS
        )
        if isinstance(additional_collections, list):
            additional_collections_str = ", ".join(additional_collections)
        else:
            additional_collections_str = ""

        return self.async_show_form(
            step_id="vector_db_settings",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_VECTOR_DB_HOST,
                        default=current_options.get(
                            CONF_VECTOR_DB_HOST,
                            current_data.get(CONF_VECTOR_DB_HOST, DEFAULT_VECTOR_DB_HOST),
                        ),
                    ): str,
                    vol.Optional(
                        CONF_VECTOR_DB_PORT,
                        default=current_options.get(
                            CONF_VECTOR_DB_PORT,
                            current_data.get(CONF_VECTOR_DB_PORT, DEFAULT_VECTOR_DB_PORT),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=65535)),
                    vol.Optional(
                        CONF_VECTOR_DB_COLLECTION,
                        default=current_options.get(
                            CONF_VECTOR_DB_COLLECTION,
                            current_data.get(
                                CONF_VECTOR_DB_COLLECTION, DEFAULT_VECTOR_DB_COLLECTION
                            ),
                        ),
                    ): str,
                    vol.Optional(
                        CONF_VECTOR_DB_EMBEDDING_PROVIDER,
                        default=current_options.get(
                            CONF_VECTOR_DB_EMBEDDING_PROVIDER,
                            current_data.get(
                                CONF_VECTOR_DB_EMBEDDING_PROVIDER,
                                DEFAULT_VECTOR_DB_EMBEDDING_PROVIDER,
                            ),
                        ),
                    ): selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=[
                                EMBEDDING_PROVIDER_OPENAI,
                                EMBEDDING_PROVIDER_OLLAMA,
                            ],
                            translation_key="embedding_provider",
                        )
                    ),
                    vol.Optional(
                        CONF_VECTOR_DB_EMBEDDING_BASE_URL,
                        default=current_options.get(
                            CONF_VECTOR_DB_EMBEDDING_BASE_URL,
                            current_data.get(
                                CONF_VECTOR_DB_EMBEDDING_BASE_URL,
                                DEFAULT_VECTOR_DB_EMBEDDING_BASE_URL,
                            ),
                        ),
                    ): str,
                    vol.Optional(
                        CONF_VECTOR_DB_EMBEDDING_MODEL,
                        default=current_options.get(
                            CONF_VECTOR_DB_EMBEDDING_MODEL,
                            current_data.get(
                                CONF_VECTOR_DB_EMBEDDING_MODEL,
                                DEFAULT_VECTOR_DB_EMBEDDING_MODEL,
                            ),
                        ),
                    ): str,
                    vol.Optional(
                        CONF_EMBEDDING_KEEP_ALIVE,
                        default=current_options.get(
                            CONF_EMBEDDING_KEEP_ALIVE,
                            current_data.get(
                                CONF_EMBEDDING_KEEP_ALIVE, DEFAULT_EMBEDDING_KEEP_ALIVE
                            ),
                        ),
                    ): str,
                    vol.Optional(
                        CONF_OPENAI_API_KEY,
                        default=current_options.get(
                            CONF_OPENAI_API_KEY, current_data.get(CONF_OPENAI_API_KEY, "")
                        ),
                    ): str,
                    vol.Optional(
                        CONF_VECTOR_DB_TOP_K,
                        default=current_options.get(
                            CONF_VECTOR_DB_TOP_K,
                            current_data.get(CONF_VECTOR_DB_TOP_K, DEFAULT_VECTOR_DB_TOP_K),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=50)),
                    vol.Optional(
                        CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
                        default=current_options.get(
                            CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
                            current_data.get(
                                CONF_VECTOR_DB_SIMILARITY_THRESHOLD,
                                DEFAULT_VECTOR_DB_SIMILARITY_THRESHOLD,
                            ),
                        ),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1000.0)),
                    vol.Optional(
                        CONF_ADDITIONAL_COLLECTIONS,
                        default=additional_collections_str,
                    ): str,
                    vol.Optional(
                        CONF_ADDITIONAL_TOP_K,
                        default=current_options.get(
                            CONF_ADDITIONAL_TOP_K, DEFAULT_ADDITIONAL_TOP_K
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=50)),
                    vol.Optional(
                        CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
                        default=current_options.get(
                            CONF_ADDITIONAL_L2_DISTANCE_THRESHOLD,
                            DEFAULT_ADDITIONAL_L2_DISTANCE_THRESHOLD,
                        ),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2000.0)),
                }
            ),
            description_placeholders={
                "default_host": DEFAULT_VECTOR_DB_HOST,
                "default_port": str(DEFAULT_VECTOR_DB_PORT),
                "default_collection": DEFAULT_VECTOR_DB_COLLECTION,
            },
        )

    async def async_step_history_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Configure conversation history settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        if user_input is not None:
            # Convert session_timeout from minutes to seconds for storage
            if CONF_SESSION_TIMEOUT in user_input:
                user_input[CONF_SESSION_TIMEOUT] = user_input[CONF_SESSION_TIMEOUT] * 60

            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="history_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_HISTORY_ENABLED,
                        default=current_options.get(
                            CONF_HISTORY_ENABLED,
                            current_data.get(CONF_HISTORY_ENABLED, DEFAULT_HISTORY_ENABLED),
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_HISTORY_MAX_MESSAGES,
                        default=current_options.get(
                            CONF_HISTORY_MAX_MESSAGES,
                            current_data.get(
                                CONF_HISTORY_MAX_MESSAGES, DEFAULT_HISTORY_MAX_MESSAGES
                            ),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
                    vol.Optional(
                        CONF_HISTORY_MAX_TOKENS,
                        default=current_options.get(
                            CONF_HISTORY_MAX_TOKENS,
                            current_data.get(CONF_HISTORY_MAX_TOKENS, DEFAULT_HISTORY_MAX_TOKENS),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=100, max=50000)),
                    vol.Required(
                        CONF_SESSION_PERSISTENCE_ENABLED,
                        default=current_options.get(
                            CONF_SESSION_PERSISTENCE_ENABLED,
                            current_data.get(
                                CONF_SESSION_PERSISTENCE_ENABLED,
                                DEFAULT_SESSION_PERSISTENCE_ENABLED,
                            ),
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_SESSION_TIMEOUT,
                        default=current_options.get(
                            CONF_SESSION_TIMEOUT,
                            current_data.get(CONF_SESSION_TIMEOUT, DEFAULT_SESSION_TIMEOUT),
                        )
                        // 60,  # Convert seconds to minutes for display
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=120)),
                }
            ),
        )

    async def async_step_prompt_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
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
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="prompt_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_PROMPT_USE_DEFAULT,
                        default=current_options.get(
                            CONF_PROMPT_USE_DEFAULT,
                            current_data.get(CONF_PROMPT_USE_DEFAULT, DEFAULT_PROMPT_USE_DEFAULT),
                        ),
                    ): bool,
                    vol.Optional(
                        CONF_PROMPT_CUSTOM_ADDITIONS,
                        description={
                            "suggested_value": current_options.get(
                                CONF_PROMPT_CUSTOM_ADDITIONS,
                                current_data.get(CONF_PROMPT_CUSTOM_ADDITIONS, ""),
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
    ) -> config_entries.ConfigFlowResult:
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
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="tool_settings",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_TOOLS_MAX_CALLS_PER_TURN,
                        default=current_options.get(
                            CONF_TOOLS_MAX_CALLS_PER_TURN,
                            current_data.get(
                                CONF_TOOLS_MAX_CALLS_PER_TURN, DEFAULT_TOOLS_MAX_CALLS_PER_TURN
                            ),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
                    vol.Optional(
                        CONF_TOOLS_TIMEOUT,
                        default=current_options.get(
                            CONF_TOOLS_TIMEOUT,
                            current_data.get(CONF_TOOLS_TIMEOUT, DEFAULT_TOOLS_TIMEOUT),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=300)),
                }
            ),
        )

    async def async_step_external_llm_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
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
                        data_schema=self._get_external_llm_schema(
                            dict(self._config_entry.options), dict(self._config_entry.data)
                        ),
                        errors={"base": "invalid_external_llm"},
                    )

            updated_options = {**self._config_entry.options, **user_input}
            return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="external_llm_settings",
            data_schema=self._get_external_llm_schema(dict(current_options), dict(current_data)),
            description_placeholders={
                "use_case": (
                    "Enable this to allow the primary LLM to delegate "
                    "complex queries to a more capable external model"
                ),
            },
        )

    def _get_external_llm_schema(
        self, current_options: dict[str, Any], current_data: dict[str, Any] | None = None
    ) -> vol.Schema:
        """Get schema for external LLM settings.

        Args:
            current_options: Current option values
            current_data: Current data values (fallback)

        Returns:
            Voluptuous schema for external LLM configuration
        """
        if current_data is None:
            current_data = {}

        return vol.Schema(
            {
                vol.Required(
                    CONF_EXTERNAL_LLM_ENABLED,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_ENABLED,
                        current_data.get(CONF_EXTERNAL_LLM_ENABLED, DEFAULT_EXTERNAL_LLM_ENABLED),
                    ),
                ): bool,
                vol.Optional(
                    CONF_EXTERNAL_LLM_BASE_URL,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_BASE_URL,
                        current_data.get(CONF_EXTERNAL_LLM_BASE_URL, OPENAI_BASE_URL),
                    ),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_API_KEY,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_API_KEY, current_data.get(CONF_EXTERNAL_LLM_API_KEY, "")
                    ),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_MODEL,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_MODEL,
                        current_data.get(CONF_EXTERNAL_LLM_MODEL, DEFAULT_EXTERNAL_LLM_MODEL),
                    ),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_TEMPERATURE,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_TEMPERATURE,
                        current_data.get(
                            CONF_EXTERNAL_LLM_TEMPERATURE, DEFAULT_EXTERNAL_LLM_TEMPERATURE
                        ),
                    ),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
                vol.Optional(
                    CONF_EXTERNAL_LLM_MAX_TOKENS,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_MAX_TOKENS,
                        current_data.get(
                            CONF_EXTERNAL_LLM_MAX_TOKENS, DEFAULT_EXTERNAL_LLM_MAX_TOKENS
                        ),
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100000)),
                vol.Optional(
                    CONF_EXTERNAL_LLM_KEEP_ALIVE,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_KEEP_ALIVE,
                        current_data.get(
                            CONF_EXTERNAL_LLM_KEEP_ALIVE, DEFAULT_EXTERNAL_LLM_KEEP_ALIVE
                        ),
                    ),
                ): str,
                vol.Optional(
                    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                    description={
                        "suggested_value": current_options.get(
                            CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                            current_data.get(
                                CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
                                DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
                            ),
                        )
                    },
                ): selector.TemplateSelector(),
                vol.Optional(
                    CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                    default=current_options.get(
                        CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                        current_data.get(
                            CONF_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                            DEFAULT_EXTERNAL_LLM_AUTO_INCLUDE_CONTEXT,
                        ),
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

    async def async_step_memory_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Configure long-term memory system settings.

        Args:
            user_input: User-provided configuration

        Returns:
            FlowResult indicating completion or next step
        """
        errors: dict[str, str] = {}

        if user_input is not None:
            # Validate configuration
            if user_input.get(CONF_MEMORY_EXTRACTION_LLM) == "external":
                # Check if external LLM is enabled
                current_data = self._config_entry.data
                current_options = self._config_entry.options
                external_llm_enabled = current_options.get(
                    CONF_EXTERNAL_LLM_ENABLED,
                    current_data.get(CONF_EXTERNAL_LLM_ENABLED, DEFAULT_EXTERNAL_LLM_ENABLED),
                )

                if not external_llm_enabled:
                    errors["base"] = "external_llm_required"

            if not errors:
                # Update config entry options
                updated_options = {**self._config_entry.options, **user_input}
                return self.async_create_entry(title="", data=updated_options)

        current_options = self._config_entry.options
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="memory_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_MEMORY_ENABLED,
                        default=current_options.get(
                            CONF_MEMORY_ENABLED,
                            current_data.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_MEMORY_EXTRACTION_ENABLED,
                        default=current_options.get(
                            CONF_MEMORY_EXTRACTION_ENABLED,
                            current_data.get(
                                CONF_MEMORY_EXTRACTION_ENABLED,
                                DEFAULT_MEMORY_EXTRACTION_ENABLED,
                            ),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_MEMORY_EXTRACTION_LLM,
                        default=current_options.get(
                            CONF_MEMORY_EXTRACTION_LLM,
                            current_data.get(
                                CONF_MEMORY_EXTRACTION_LLM,
                                DEFAULT_MEMORY_EXTRACTION_LLM,
                            ),
                        ),
                    ): vol.In(["external", "local"]),
                    vol.Optional(
                        CONF_MEMORY_MAX_MEMORIES,
                        default=current_options.get(
                            CONF_MEMORY_MAX_MEMORIES,
                            current_data.get(
                                CONF_MEMORY_MAX_MEMORIES,
                                DEFAULT_MEMORY_MAX_MEMORIES,
                            ),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=10, max=1000)),
                    vol.Optional(
                        CONF_MEMORY_MIN_IMPORTANCE,
                        default=current_options.get(
                            CONF_MEMORY_MIN_IMPORTANCE,
                            current_data.get(
                                CONF_MEMORY_MIN_IMPORTANCE,
                                DEFAULT_MEMORY_MIN_IMPORTANCE,
                            ),
                        ),
                    ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                    vol.Optional(
                        CONF_MEMORY_MIN_WORDS,
                        default=current_options.get(
                            CONF_MEMORY_MIN_WORDS,
                            current_data.get(
                                CONF_MEMORY_MIN_WORDS,
                                DEFAULT_MEMORY_MIN_WORDS,
                            ),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=50)),
                    vol.Optional(
                        CONF_MEMORY_CONTEXT_TOP_K,
                        default=current_options.get(
                            CONF_MEMORY_CONTEXT_TOP_K,
                            current_data.get(
                                CONF_MEMORY_CONTEXT_TOP_K,
                                DEFAULT_MEMORY_CONTEXT_TOP_K,
                            ),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
                    vol.Optional(
                        CONF_MEMORY_COLLECTION_NAME,
                        default=current_options.get(
                            CONF_MEMORY_COLLECTION_NAME,
                            current_data.get(
                                CONF_MEMORY_COLLECTION_NAME,
                                DEFAULT_MEMORY_COLLECTION_NAME,
                            ),
                        ),
                    ): str,
                }
            ),
            errors=errors,
            description_placeholders={
                "external_llm_note": (
                    "Memory extraction using external LLM requires the "
                    "external LLM tool to be enabled in the External LLM settings."
                ),
            },
        )

    async def async_step_debug_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
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
        current_data = self._config_entry.data

        return self.async_show_form(
            step_id="debug_settings",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_DEBUG_LOGGING,
                        default=current_options.get(
                            CONF_DEBUG_LOGGING,
                            current_data.get(CONF_DEBUG_LOGGING, DEFAULT_DEBUG_LOGGING),
                        ),
                    ): bool,
                    vol.Required(
                        CONF_STREAMING_ENABLED,
                        default=current_options.get(
                            CONF_STREAMING_ENABLED,
                            current_data.get(CONF_STREAMING_ENABLED, DEFAULT_STREAMING_ENABLED),
                        ),
                    ): bool,
                }
            ),
            description_placeholders={
                "warning": (
                    "Debug logging may expose sensitive information "
                    "in Home Assistant logs. Use with caution."
                ),
                "streaming_info": (
                    "Enable streaming responses for low-latency TTS integration. "
                    "Requires Assist Pipeline with Wyoming TTS. "
                    "When enabled, responses are sent incrementally for faster audio playback. "
                    "Automatically falls back to standard mode if streaming fails."
                ),
            },
        )

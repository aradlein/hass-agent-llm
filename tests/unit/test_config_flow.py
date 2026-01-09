"""Unit tests for Home Agent config flow."""

from unittest.mock import Mock, patch

import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.home_agent.config_flow import (
    HomeAgentConfigFlow,
    HomeAgentOptionsFlow,
    _migrate_legacy_backend,
    _validate_proxy_headers,
)
from custom_components.home_agent.const import (
    CONF_CONTEXT_FORMAT,
    CONF_CONTEXT_MODE,
    CONF_DEBUG_LOGGING,
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_LLM_BACKEND,
    CONF_LLM_PROXY_HEADERS,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_ENABLED,
    CONF_MEMORY_EXTRACTION_LLM,
    CONF_PROMPT_CUSTOM_ADDITIONS,
    CONF_PROMPT_INCLUDE_LABELS,
    CONF_PROMPT_USE_DEFAULT,
    CONF_SESSION_PERSISTENCE_ENABLED,
    CONF_SESSION_TIMEOUT,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    CONF_VECTOR_DB_HOST,
    CONF_VECTOR_DB_PORT,
    CONTEXT_MODE_DIRECT,
    CONTEXT_MODE_VECTOR_DB,
    DEFAULT_PROMPT_INCLUDE_LABELS,
    DEFAULT_SESSION_PERSISTENCE_ENABLED,
    DEFAULT_SESSION_TIMEOUT,
    DEFAULT_STREAMING_ENABLED,
    LLM_BACKEND_OLLAMA_GPU,
)
from custom_components.home_agent.exceptions import ValidationError


@pytest.fixture
def mock_config_entry():
    """Create mock config entry."""
    entry = Mock(spec=config_entries.ConfigEntry)
    entry.data = {
        "llm_base_url": "https://api.openai.com/v1",
        "llm_api_key": "test-key",
        "llm_model": "gpt-4o-mini",
    }
    entry.options = {}
    return entry


@pytest.fixture
def mock_hass():
    """Create mock Home Assistant instance."""
    hass = Mock()
    hass.config_entries = Mock()
    hass.config_entries.async_update_entry = Mock()
    return hass


class TestProxyHeadersValidation:
    """Test _validate_proxy_headers helper function."""

    def test_validate_proxy_headers_empty_string(self):
        """Test that empty string returns empty dict."""
        result = _validate_proxy_headers("")
        assert result == {}

    def test_validate_proxy_headers_none(self):
        """Test that None returns empty dict."""
        result = _validate_proxy_headers(None)
        assert result == {}

    def test_validate_proxy_headers_whitespace_only(self):
        """Test that whitespace-only string returns empty dict."""
        result = _validate_proxy_headers("   ")
        assert result == {}

    def test_validate_proxy_headers_valid_json_string(self):
        """Test that valid JSON string is parsed correctly."""
        headers_json = '{"X-Custom-Header": "value", "Authorization": "Bearer token"}'
        result = _validate_proxy_headers(headers_json)
        assert result == {"X-Custom-Header": "value", "Authorization": "Bearer token"}

    def test_validate_proxy_headers_valid_dict(self):
        """Test that valid dict is returned as-is."""
        headers_dict = {"X-Custom-Header": "value"}
        result = _validate_proxy_headers(headers_dict)
        assert result == headers_dict

    def test_validate_proxy_headers_invalid_json(self):
        """Test that invalid JSON raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid JSON format"):
            _validate_proxy_headers('{"invalid": json}')

    def test_validate_proxy_headers_not_dict(self):
        """Test that non-dict JSON raises ValidationError."""
        with pytest.raises(ValidationError, match="must be a JSON object"):
            _validate_proxy_headers('["array", "not", "dict"]')

    def test_validate_proxy_headers_invalid_header_name(self):
        """Test that invalid header names raise ValidationError."""
        # Header names with invalid characters
        with pytest.raises(ValidationError, match="Invalid header name"):
            _validate_proxy_headers({"Invalid Header": "value"})  # Space

        with pytest.raises(ValidationError, match="Invalid header name"):
            _validate_proxy_headers({"Header@Name": "value"})  # @

        with pytest.raises(ValidationError, match="Invalid header name"):
            _validate_proxy_headers({"Header:Name": "value"})  # :

    def test_validate_proxy_headers_non_string_value(self):
        """Test that non-string header values raise ValidationError."""
        with pytest.raises(ValidationError, match="must be a string"):
            _validate_proxy_headers({"X-Custom": 123})

        with pytest.raises(ValidationError, match="must be a string"):
            _validate_proxy_headers({"X-Custom": ["list", "value"]})

    def test_validate_proxy_headers_rfc7230_compliant_names(self):
        """Test that RFC 7230 compliant header names are accepted."""
        headers = {
            "X-Custom-Header": "value",
            "X_Underscore": "value",
            "X-Hyphen-123": "value",
            "Authorization": "Bearer token",
        }
        result = _validate_proxy_headers(headers)
        assert result == headers


class TestLegacyBackendMigration:
    """Test _migrate_legacy_backend helper function."""

    def test_migrate_legacy_backend_no_proxy_headers(self):
        """Test migration when no proxy headers exist."""
        config = {
            CONF_LLM_BACKEND: LLM_BACKEND_OLLAMA_GPU,
            "other_setting": "value",
        }
        result = _migrate_legacy_backend(config)
        assert CONF_LLM_PROXY_HEADERS in result
        assert result[CONF_LLM_PROXY_HEADERS] == {"X-Ollama-Backend": LLM_BACKEND_OLLAMA_GPU}

    def test_migrate_legacy_backend_already_has_proxy_headers(self):
        """Test that existing proxy headers are not overwritten."""
        existing_headers = {"X-Custom": "value"}
        config = {
            CONF_LLM_BACKEND: LLM_BACKEND_OLLAMA_GPU,
            CONF_LLM_PROXY_HEADERS: existing_headers,
        }
        result = _migrate_legacy_backend(config)
        assert result[CONF_LLM_PROXY_HEADERS] == existing_headers

    def test_migrate_legacy_backend_no_backend_setting(self):
        """Test that missing backend setting doesn't add proxy headers."""
        config = {"other_setting": "value"}
        result = _migrate_legacy_backend(config)
        assert CONF_LLM_PROXY_HEADERS not in result

    def test_migrate_legacy_backend_default_backend(self):
        """Test that default backend doesn't trigger migration."""
        config = {
            CONF_LLM_BACKEND: "default",
            "other_setting": "value",
        }
        result = _migrate_legacy_backend(config)
        assert CONF_LLM_PROXY_HEADERS not in result


class TestExternalLLMValidation:
    """Test _validate_external_llm_config helper function."""

    async def test_validate_external_llm_config_success(self):
        """Test successful external LLM config validation."""
        options_flow = HomeAgentOptionsFlow(Mock())

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        # Should not raise
        await options_flow._validate_external_llm_config(config)

    async def test_validate_external_llm_config_empty_url(self):
        """Test validation error for empty base URL."""
        options_flow = HomeAgentOptionsFlow(Mock())

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        with pytest.raises(ValidationError, match="base URL cannot be empty"):
            await options_flow._validate_external_llm_config(config)

    async def test_validate_external_llm_config_invalid_url(self):
        """Test validation error for invalid URL format."""
        options_flow = HomeAgentOptionsFlow(Mock())

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "not-a-url",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        with pytest.raises(ValidationError, match="Invalid external LLM URL format"):
            await options_flow._validate_external_llm_config(config)

    async def test_validate_external_llm_config_empty_api_key(self):
        """Test validation error for empty API key."""
        options_flow = HomeAgentOptionsFlow(Mock())

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        with pytest.raises(ValidationError, match="API key cannot be empty"):
            await options_flow._validate_external_llm_config(config)

    async def test_validate_external_llm_config_whitespace_api_key(self):
        """Test validation error for whitespace-only API key."""
        options_flow = HomeAgentOptionsFlow(Mock())

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "   ",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        with pytest.raises(ValidationError, match="API key cannot be empty"):
            await options_flow._validate_external_llm_config(config)

    async def test_validate_external_llm_config_empty_model(self):
        """Test validation error for empty model name."""
        options_flow = HomeAgentOptionsFlow(Mock())

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "",
        }

        with pytest.raises(ValidationError, match="model name cannot be empty"):
            await options_flow._validate_external_llm_config(config)


class TestHomeAgentOptionsFlow:
    """Test Home Agent options flow."""

    async def test_options_flow_includes_streaming_option(self, mock_config_entry, mock_hass):
        """Test that options flow includes streaming toggle in debug settings."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the debug settings form
        result = await options_flow.async_step_debug_settings()

        # Verify the form is shown
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "debug_settings"

        # Verify streaming option is in the schema
        schema_keys = list(result["data_schema"].schema.keys())
        streaming_key = None
        debug_key = None

        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_STREAMING_ENABLED:
                streaming_key = key
            if hasattr(key, "schema") and key.schema == CONF_DEBUG_LOGGING:
                debug_key = key

        assert streaming_key is not None, "Streaming option not found in schema"
        assert debug_key is not None, "Debug logging option not found in schema"

        # Verify description placeholders
        assert "streaming_info" in result["description_placeholders"]
        assert "Wyoming TTS" in result["description_placeholders"]["streaming_info"]

    async def test_streaming_defaults_to_disabled(self, mock_config_entry, mock_hass):
        """Test that streaming defaults to False."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the debug settings form
        result = await options_flow.async_step_debug_settings()

        # Find the streaming key and check its default
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_STREAMING_ENABLED:
                # The default should be False
                assert key.default() == DEFAULT_STREAMING_ENABLED
                assert DEFAULT_STREAMING_ENABLED is False

    async def test_streaming_option_can_be_enabled(self, mock_config_entry, mock_hass):
        """Test that user can enable streaming."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Submit form with streaming enabled
        user_input = {
            CONF_DEBUG_LOGGING: False,
            CONF_STREAMING_ENABLED: True,
        }

        result = await options_flow.async_step_debug_settings(user_input)

        # Verify the entry is created successfully
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_STREAMING_ENABLED] is True

    async def test_streaming_option_can_be_disabled(self, mock_config_entry, mock_hass):
        """Test that user can disable streaming."""
        # Set initial state with streaming enabled
        mock_config_entry.options = {CONF_STREAMING_ENABLED: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Submit form with streaming disabled
        user_input = {
            CONF_DEBUG_LOGGING: False,
            CONF_STREAMING_ENABLED: False,
        }

        result = await options_flow.async_step_debug_settings(user_input)

        # Verify the entry is updated with streaming disabled
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_STREAMING_ENABLED] is False

    async def test_streaming_option_persists(self, mock_config_entry, mock_hass):
        """Test that streaming option persists across reloads."""
        # Enable streaming
        mock_config_entry.options = {CONF_STREAMING_ENABLED: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the debug settings form
        result = await options_flow.async_step_debug_settings()

        # Find the streaming key and check its current value
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_STREAMING_ENABLED:
                # The default should now be True (from persisted options)
                assert key.default() is True

    async def test_streaming_backward_compatible_missing_option(self, mock_config_entry, mock_hass):
        """Test that missing streaming option defaults correctly for backward compatibility."""
        # Config entry without streaming option (simulating existing installation)
        mock_config_entry.options = {CONF_DEBUG_LOGGING: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the debug settings form
        result = await options_flow.async_step_debug_settings()

        # Verify streaming defaults to False when not present
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_STREAMING_ENABLED:
                assert key.default() == DEFAULT_STREAMING_ENABLED
                assert key.default() is False

    async def test_debug_settings_preserves_other_options(self, mock_config_entry, mock_hass):
        """Test that updating debug settings preserves other config options."""
        # Set up existing options
        mock_config_entry.options = {
            CONF_DEBUG_LOGGING: True,
            CONF_STREAMING_ENABLED: False,
            "history_enabled": True,
            "memory_enabled": False,
        }

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Update only streaming setting
        user_input = {
            CONF_DEBUG_LOGGING: True,
            CONF_STREAMING_ENABLED: True,
        }

        result = await options_flow.async_step_debug_settings(user_input)

        # Verify all options are preserved and merged
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_STREAMING_ENABLED] is True
        assert result["data"][CONF_DEBUG_LOGGING] is True
        assert result["data"]["history_enabled"] is True
        assert result["data"]["memory_enabled"] is False

    async def test_llm_settings_allows_clearing_api_key(self, mock_config_entry, mock_hass):
        """Test that user can clear API key to empty string for local LLMs.

        This tests the fix for the bug where clearing an API key wouldn't save
        because Home Assistant forms may not include empty optional fields.
        """
        # Set up existing config with an API key
        mock_config_entry.data = {
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key": "old-api-key-to-clear",
            "llm_model": "llama3",
        }

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Simulate user clearing the API key field
        # Note: HA forms may not include the field at all when cleared
        user_input = {
            "llm_base_url": "http://localhost:11434/v1",
            # llm_api_key intentionally omitted to simulate cleared field
            "llm_model": "llama3",
        }

        result = await options_flow.async_step_llm_settings(user_input)

        # Verify the entry was updated
        assert result["type"] == FlowResultType.CREATE_ENTRY

        # Verify async_update_entry was called with empty API key
        mock_hass.config_entries.async_update_entry.assert_called_once()
        call_args = mock_hass.config_entries.async_update_entry.call_args
        updated_data = call_args.kwargs.get("data") or call_args[1].get("data")

        assert updated_data["llm_api_key"] == "", "API key should be empty string when cleared"

    async def test_llm_settings_preserves_empty_api_key_when_provided(
        self, mock_config_entry, mock_hass
    ):
        """Test that explicitly providing empty API key is preserved."""
        mock_config_entry.data = {
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key": "old-api-key",
            "llm_model": "llama3",
        }

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Simulate user explicitly setting API key to empty string
        user_input = {
            "llm_base_url": "http://localhost:11434/v1",
            "llm_api_key": "",  # Explicitly empty
            "llm_model": "llama3",
        }

        result = await options_flow.async_step_llm_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        mock_hass.config_entries.async_update_entry.assert_called_once()
        call_args = mock_hass.config_entries.async_update_entry.call_args
        updated_data = call_args.kwargs.get("data") or call_args[1].get("data")

        assert updated_data["llm_api_key"] == "", "API key should be empty string"

    async def test_history_settings_includes_session_persistence_options(
        self, mock_config_entry, mock_hass
    ):
        """Test that history_settings step includes both session_persistence_enabled and session_timeout fields."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the history settings form
        result = await options_flow.async_step_history_settings()

        # Verify the form is shown
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "history_settings"

        # Verify session persistence options are in the schema
        schema_keys = list(result["data_schema"].schema.keys())
        session_persistence_key = None
        session_timeout_key = None

        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_SESSION_PERSISTENCE_ENABLED:
                session_persistence_key = key
            if hasattr(key, "schema") and key.schema == CONF_SESSION_TIMEOUT:
                session_timeout_key = key

        assert (
            session_persistence_key is not None
        ), "Session persistence enabled option not found in schema"
        assert session_timeout_key is not None, "Session timeout option not found in schema"

    async def test_session_persistence_defaults(self, mock_config_entry, mock_hass):
        """Test that session_persistence_enabled defaults to True and session_timeout defaults to 60 (minutes)."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the history settings form
        result = await options_flow.async_step_history_settings()

        # Find the session persistence keys and check their defaults
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_SESSION_PERSISTENCE_ENABLED:
                # The default should be True
                assert key.default() == DEFAULT_SESSION_PERSISTENCE_ENABLED
                assert DEFAULT_SESSION_PERSISTENCE_ENABLED is True
            if hasattr(key, "schema") and key.schema == CONF_SESSION_TIMEOUT:
                # The default should be 60 minutes (3600 seconds / 60)
                assert key.default() == DEFAULT_SESSION_TIMEOUT // 60
                assert DEFAULT_SESSION_TIMEOUT // 60 == 60

    async def test_session_timeout_converts_to_seconds(self, mock_config_entry, mock_hass):
        """Test that when user enters timeout in minutes (e.g., 30), it gets converted to seconds (1800) for storage."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Submit form with session timeout in minutes
        user_input = {
            "history_enabled": True,
            "history_max_messages": 10,
            "history_max_tokens": 1000,
            CONF_SESSION_PERSISTENCE_ENABLED: True,
            CONF_SESSION_TIMEOUT: 30,  # 30 minutes
        }

        result = await options_flow.async_step_history_settings(user_input)

        # Verify the entry is created successfully and timeout is converted to seconds
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_SESSION_TIMEOUT] == 1800  # 30 * 60 = 1800 seconds

    async def test_session_timeout_displayed_in_minutes(self, mock_config_entry, mock_hass):
        """Test that stored timeout in seconds is displayed as minutes in the form."""
        # Set initial state with timeout in seconds (7200 seconds = 120 minutes)
        mock_config_entry.options = {CONF_SESSION_TIMEOUT: 7200}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the history settings form
        result = await options_flow.async_step_history_settings()

        # Find the session timeout key and check its displayed value (in minutes)
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_SESSION_TIMEOUT:
                # The default should show 120 minutes (7200 seconds / 60)
                assert key.default() == 120

    async def test_context_settings_success(self, mock_config_entry, mock_hass):
        """Test successful context settings update."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_CONTEXT_MODE: CONTEXT_MODE_VECTOR_DB,
            CONF_CONTEXT_FORMAT: "json",
            "direct_entities": "sensor.temp,light.living",
        }

        result = await options_flow.async_step_context_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_CONTEXT_MODE] == CONTEXT_MODE_VECTOR_DB
        assert result["data"][CONF_CONTEXT_FORMAT] == "json"

    async def test_context_settings_shows_form(self, mock_config_entry, mock_hass):
        """Test that context settings shows form without user input."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        result = await options_flow.async_step_context_settings()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "context_settings"

    async def test_vector_db_settings_success(self, mock_config_entry, mock_hass):
        """Test successful vector DB settings update."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_VECTOR_DB_HOST: "localhost",
            CONF_VECTOR_DB_PORT: 8000,
            "vector_db_collection": "home_states",
        }

        result = await options_flow.async_step_vector_db_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_VECTOR_DB_HOST] == "localhost"
        assert result["data"][CONF_VECTOR_DB_PORT] == 8000

    async def test_vector_db_settings_additional_collections_parsing(
        self, mock_config_entry, mock_hass
    ):
        """Test that comma-separated collections are parsed to list."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_VECTOR_DB_HOST: "localhost",
            "additional_collections": "collection1, collection2, collection3",
        }

        result = await options_flow.async_step_vector_db_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"]["additional_collections"] == [
            "collection1",
            "collection2",
            "collection3",
        ]

    async def test_vector_db_settings_shows_form(self, mock_config_entry, mock_hass):
        """Test that vector DB settings shows form without user input."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        result = await options_flow.async_step_vector_db_settings()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "vector_db_settings"

    async def test_prompt_settings_success(self, mock_config_entry, mock_hass):
        """Test successful prompt settings update."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_PROMPT_USE_DEFAULT: True,
            CONF_PROMPT_CUSTOM_ADDITIONS: "Additional instructions here",
        }

        result = await options_flow.async_step_prompt_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_PROMPT_USE_DEFAULT] is True
        assert result["data"][CONF_PROMPT_CUSTOM_ADDITIONS] == "Additional instructions here"

    async def test_prompt_settings_shows_form(self, mock_config_entry, mock_hass):
        """Test that prompt settings shows form without user input."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        result = await options_flow.async_step_prompt_settings()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "prompt_settings"

    async def test_prompt_settings_includes_include_labels_option(
        self, mock_config_entry, mock_hass
    ):
        """Test that prompt settings form includes the include_labels option."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the prompt settings form
        result = await options_flow.async_step_prompt_settings()

        # Verify the form is shown
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "prompt_settings"

        # Verify include_labels option is in the schema
        schema_keys = list(result["data_schema"].schema.keys())
        include_labels_key = None

        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_PROMPT_INCLUDE_LABELS:
                include_labels_key = key
                break

        assert include_labels_key is not None, "Include labels option not found in schema"

    async def test_prompt_include_labels_defaults_to_false(self, mock_config_entry, mock_hass):
        """Test that prompt_include_labels defaults to False."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the prompt settings form
        result = await options_flow.async_step_prompt_settings()

        # Find the include_labels key and check its default
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_PROMPT_INCLUDE_LABELS:
                # The default should be False
                assert key.default() == DEFAULT_PROMPT_INCLUDE_LABELS
                assert DEFAULT_PROMPT_INCLUDE_LABELS is False

    async def test_prompt_include_labels_can_be_enabled(self, mock_config_entry, mock_hass):
        """Test that user can enable prompt_include_labels."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Submit form with include_labels enabled
        user_input = {
            CONF_PROMPT_USE_DEFAULT: True,
            CONF_PROMPT_INCLUDE_LABELS: True,
        }

        result = await options_flow.async_step_prompt_settings(user_input)

        # Verify the entry is created successfully
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_PROMPT_INCLUDE_LABELS] is True

    async def test_prompt_include_labels_can_be_disabled(self, mock_config_entry, mock_hass):
        """Test that user can disable prompt_include_labels."""
        # Set initial state with include_labels enabled
        mock_config_entry.options = {CONF_PROMPT_INCLUDE_LABELS: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Submit form with include_labels disabled
        user_input = {
            CONF_PROMPT_USE_DEFAULT: True,
            CONF_PROMPT_INCLUDE_LABELS: False,
        }

        result = await options_flow.async_step_prompt_settings(user_input)

        # Verify the entry is updated with include_labels disabled
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_PROMPT_INCLUDE_LABELS] is False

    async def test_prompt_include_labels_persists(self, mock_config_entry, mock_hass):
        """Test that prompt_include_labels option persists across reloads."""
        # Enable include_labels
        mock_config_entry.options = {CONF_PROMPT_INCLUDE_LABELS: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the prompt settings form
        result = await options_flow.async_step_prompt_settings()

        # Find the include_labels key and check its current value
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_PROMPT_INCLUDE_LABELS:
                # The default should now be True (from persisted options)
                assert key.default() is True

    async def test_prompt_include_labels_backward_compatible_missing_option(
        self, mock_config_entry, mock_hass
    ):
        """Test that missing prompt_include_labels option defaults correctly for backward compatibility."""
        # Config entry without include_labels option (simulating existing installation)
        mock_config_entry.options = {CONF_PROMPT_USE_DEFAULT: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Get the prompt settings form
        result = await options_flow.async_step_prompt_settings()

        # Verify include_labels defaults to False when not present
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == CONF_PROMPT_INCLUDE_LABELS:
                assert key.default() == DEFAULT_PROMPT_INCLUDE_LABELS
                assert key.default() is False

    async def test_prompt_settings_preserves_other_options(self, mock_config_entry, mock_hass):
        """Test that updating prompt settings preserves other config options."""
        # Set up existing options
        mock_config_entry.options = {
            CONF_PROMPT_USE_DEFAULT: True,
            CONF_PROMPT_INCLUDE_LABELS: False,
            "history_enabled": True,
            "memory_enabled": False,
        }

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Update only include_labels setting
        user_input = {
            CONF_PROMPT_USE_DEFAULT: True,
            CONF_PROMPT_INCLUDE_LABELS: True,
        }

        result = await options_flow.async_step_prompt_settings(user_input)

        # Verify all options are preserved and merged
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_PROMPT_INCLUDE_LABELS] is True
        assert result["data"][CONF_PROMPT_USE_DEFAULT] is True
        assert result["data"]["history_enabled"] is True
        assert result["data"]["memory_enabled"] is False

    async def test_tool_settings_success(self, mock_config_entry, mock_hass):
        """Test successful tool settings update."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_TOOLS_MAX_CALLS_PER_TURN: 10,
            CONF_TOOLS_TIMEOUT: 30,
        }

        result = await options_flow.async_step_tool_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_TOOLS_MAX_CALLS_PER_TURN] == 10
        assert result["data"][CONF_TOOLS_TIMEOUT] == 30

    async def test_tool_settings_shows_form(self, mock_config_entry, mock_hass):
        """Test that tool settings shows form without user input."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        result = await options_flow.async_step_tool_settings()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "tool_settings"

    async def test_external_llm_settings_success(self, mock_config_entry, mock_hass):
        """Test successful external LLM settings update."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_EXTERNAL_LLM_ENABLED: True,
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        result = await options_flow.async_step_external_llm_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_EXTERNAL_LLM_ENABLED] is True
        assert result["data"][CONF_EXTERNAL_LLM_MODEL] == "gpt-4"

    async def test_external_llm_settings_validation_error(self, mock_config_entry, mock_hass):
        """Test external LLM settings validation error."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        # Missing API key should trigger validation error
        user_input = {
            CONF_EXTERNAL_LLM_ENABLED: True,
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "",  # Empty API key
            CONF_EXTERNAL_LLM_MODEL: "gpt-4",
        }

        result = await options_flow.async_step_external_llm_settings(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "invalid_external_llm"

    async def test_external_llm_settings_shows_form(self, mock_config_entry, mock_hass):
        """Test that external LLM settings shows form without user input."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        result = await options_flow.async_step_external_llm_settings()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "external_llm_settings"

    async def test_memory_settings_success(self, mock_config_entry, mock_hass):
        """Test successful memory settings update."""
        # Enable external LLM first
        mock_config_entry.options = {CONF_EXTERNAL_LLM_ENABLED: True}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_MEMORY_ENABLED: True,
            CONF_MEMORY_EXTRACTION_ENABLED: True,
            CONF_MEMORY_EXTRACTION_LLM: "external",
        }

        result = await options_flow.async_step_memory_settings(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_MEMORY_ENABLED] is True
        assert result["data"][CONF_MEMORY_EXTRACTION_LLM] == "external"

    async def test_memory_settings_external_llm_not_enabled(self, mock_config_entry, mock_hass):
        """Test memory settings error when external LLM is not enabled."""
        # External LLM is disabled
        mock_config_entry.options = {CONF_EXTERNAL_LLM_ENABLED: False}

        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            CONF_MEMORY_ENABLED: True,
            CONF_MEMORY_EXTRACTION_ENABLED: True,
            CONF_MEMORY_EXTRACTION_LLM: "external",  # Requires external LLM
        }

        result = await options_flow.async_step_memory_settings(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "external_llm_required"

    async def test_memory_settings_shows_form(self, mock_config_entry, mock_hass):
        """Test that memory settings shows form without user input."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        result = await options_flow.async_step_memory_settings()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "memory_settings"

    async def test_llm_settings_validation_error_invalid_url(self, mock_config_entry, mock_hass):
        """Test LLM settings validation error for invalid URL."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            "llm_base_url": "not-a-valid-url",
            "llm_model": "gpt-4o-mini",
        }

        result = await options_flow.async_step_llm_settings(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "invalid_config"

    async def test_llm_settings_validation_error_invalid_proxy_headers(
        self, mock_config_entry, mock_hass
    ):
        """Test LLM settings validation error for invalid proxy headers."""
        options_flow = HomeAgentOptionsFlow(mock_config_entry)
        options_flow.hass = mock_hass

        user_input = {
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_PROXY_HEADERS: '{"invalid": json}',  # Invalid JSON
        }

        result = await options_flow.async_step_llm_settings(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "invalid_config"


class TestHomeAgentConfigFlow:
    """Test Home Agent config flow initialization."""

    async def test_config_flow_does_not_include_streaming_in_initial_setup(self):
        """Test that initial setup doesn't include streaming (options only)."""
        config_flow = HomeAgentConfigFlow()

        # Get the initial user step form
        result = await config_flow.async_step_user()

        # Verify streaming is not in the initial setup schema
        schema_keys = list(result["data_schema"].schema.keys())
        streaming_keys = [
            key
            for key in schema_keys
            if hasattr(key, "schema") and key.schema == CONF_STREAMING_ENABLED
        ]

        assert len(streaming_keys) == 0, "Streaming should not be in initial setup"

    async def test_config_flow_accepts_empty_api_key(self):
        """Test that config flow accepts empty API key for local LLMs."""
        config_flow = HomeAgentConfigFlow()

        # Get the initial user step form
        result = await config_flow.async_step_user()

        # Find the API key field and verify it's optional
        schema_keys = list(result["data_schema"].schema.keys())
        api_key_key = None
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == "llm_api_key":
                api_key_key = key
                break

        assert api_key_key is not None, "API key field not found in schema"

        # Verify it's optional (not required) by checking it's a vol.Optional
        import voluptuous as vol

        assert isinstance(
            api_key_key, vol.Optional
        ), "API key should be Optional, not Required"

    async def test_config_flow_api_key_uses_suggested_value(self):
        """Test that API key uses suggested_value pattern (not default) to allow clearing."""
        config_flow = HomeAgentConfigFlow()

        # Get the initial user step form
        result = await config_flow.async_step_user()

        # Find the API key field and verify it uses suggested_value pattern
        schema_keys = list(result["data_schema"].schema.keys())
        for key in schema_keys:
            if hasattr(key, "schema") and key.schema == "llm_api_key":
                # Should use description with suggested_value, not default
                # This allows users to clear the field (empty string is accepted)
                assert hasattr(key, "description"), "API key should have description"
                assert "suggested_value" in key.description, "API key should use suggested_value"
                assert key.description["suggested_value"] == "", "suggested_value should be empty"
                break

    async def test_config_flow_validation_error_invalid_url(self):
        """Test that invalid URL triggers validation error."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "not-a-url",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "invalid_config"

    async def test_config_flow_validation_error_empty_model(self):
        """Test that empty model name triggers validation error."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "",  # Empty model
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "invalid_config"

    async def test_config_flow_validation_error_invalid_proxy_headers(self):
        """Test that invalid proxy headers trigger validation error."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_PROXY_HEADERS: '{"invalid": json}',  # Invalid JSON
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.FORM
        assert "errors" in result
        assert result["errors"]["base"] == "invalid_config"

    async def test_config_flow_success_with_valid_config(self):
        """Test successful config entry creation."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            "llm_temperature": 0.7,
            "llm_max_tokens": 1000,
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "Test Agent"
        assert "data" in result

    async def test_config_flow_success_with_empty_api_key(self):
        """Test successful config entry creation with empty API key (local LLM)."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Local LLM",
            "llm_base_url": "http://localhost:11434/v1",
            # No API key provided (optional field)
            "llm_model": "llama3",
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "Local LLM"

    async def test_config_flow_success_with_valid_proxy_headers(self):
        """Test successful config entry with valid proxy headers."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "test-key",
            "llm_model": "gpt-4o-mini",
            CONF_LLM_PROXY_HEADERS: '{"X-Custom-Header": "value"}',
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_LLM_PROXY_HEADERS] == {"X-Custom-Header": "value"}

    async def test_config_flow_migrates_legacy_backend(self):
        """Test that legacy backend setting is migrated to proxy headers."""
        config_flow = HomeAgentConfigFlow()

        user_input = {
            "name": "Test Agent",
            "llm_base_url": "http://localhost:11434/v1",
            "llm_model": "llama3",
            CONF_LLM_BACKEND: LLM_BACKEND_OLLAMA_GPU,
        }

        result = await config_flow.async_step_user(user_input)

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert CONF_LLM_PROXY_HEADERS in result["data"]
        assert result["data"][CONF_LLM_PROXY_HEADERS] == {
            "X-Ollama-Backend": LLM_BACKEND_OLLAMA_GPU
        }

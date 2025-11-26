"""Unit tests for Home Agent config flow."""
from unittest.mock import Mock

import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.home_agent.config_flow import HomeAgentConfigFlow, HomeAgentOptionsFlow
from custom_components.home_agent.const import (
    CONF_DEBUG_LOGGING,
    CONF_SESSION_PERSISTENCE_ENABLED,
    CONF_SESSION_TIMEOUT,
    CONF_STREAMING_ENABLED,
    DEFAULT_SESSION_PERSISTENCE_ENABLED,
    DEFAULT_SESSION_TIMEOUT,
    DEFAULT_STREAMING_ENABLED,
)


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

    async def test_history_settings_includes_session_persistence_options(self, mock_config_entry, mock_hass):
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

        assert session_persistence_key is not None, "Session persistence enabled option not found in schema"
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

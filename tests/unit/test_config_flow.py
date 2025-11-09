"""Unit tests for Home Agent config flow."""
from unittest.mock import Mock

import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.home_agent.config_flow import HomeAgentConfigFlow, HomeAgentOptionsFlow
from custom_components.home_agent.const import (
    CONF_DEBUG_LOGGING,
    CONF_STREAMING_ENABLED,
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

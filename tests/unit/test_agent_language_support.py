"""Unit tests for Home Agent's language support.

This module tests the supported_languages property and ensures that the agent
correctly declares support for all languages via MATCH_ALL, since the underlying
LLM can handle any language.
"""

from unittest.mock import MagicMock

import pytest
from homeassistant.const import MATCH_ALL

from custom_components.home_agent.agent.core import HomeAgent


@pytest.fixture
def mock_home_agent(mock_hass, sample_config):
    """Create a HomeAgent instance for testing.

    Args:
        mock_hass: Mock Home Assistant instance from conftest.py
        sample_config: Sample configuration from conftest.py

    Returns:
        HomeAgent: A configured HomeAgent instance for testing
    """
    # Create a mock session manager
    session_manager = MagicMock()
    session_manager.get_session = MagicMock(return_value=None)

    # Create HomeAgent instance
    agent = HomeAgent(
        hass=mock_hass,
        config=sample_config,
        session_manager=session_manager,
    )

    return agent


class TestSupportedLanguages:
    """Tests for the supported_languages property."""

    def test_supported_languages_returns_match_all(self, mock_home_agent):
        """Test that supported_languages returns MATCH_ALL.

        Since Home Agent delegates to an LLM for natural language understanding,
        it should support all languages by returning MATCH_ALL ("*").
        """
        assert mock_home_agent.supported_languages == MATCH_ALL

    def test_supported_languages_is_wildcard_string(self, mock_home_agent):
        """Test that MATCH_ALL is the expected wildcard string.

        Home Assistant uses "*" as the wildcard to indicate all languages
        are supported.
        """
        supported = mock_home_agent.supported_languages
        assert supported == "*"
        assert isinstance(supported, str)

    def test_supported_languages_returns_same_value(self, mock_home_agent):
        """Test that the property is deterministic and returns consistent results.

        Ensures that multiple calls to supported_languages return the same
        value, maintaining consistency throughout the agent's lifecycle.
        """
        first_call = mock_home_agent.supported_languages
        second_call = mock_home_agent.supported_languages
        third_call = mock_home_agent.supported_languages

        assert first_call == second_call, (
            "supported_languages should return the same value on subsequent calls"
        )
        assert second_call == third_call, (
            "supported_languages should return the same value on subsequent calls"
        )

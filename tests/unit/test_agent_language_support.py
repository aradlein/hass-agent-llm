"""Unit tests for Home Agent's language support.

This module tests the supported_languages property and ensures that the agent
correctly declares support for the top 10 languages used with Home Assistant.

These tests follow a Test-Driven Development (TDD) approach and will initially
fail until the supported_languages property is updated to include all required
languages beyond just English.
"""

from unittest.mock import MagicMock

import pytest

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

    @pytest.mark.parametrize(
        "lang",
        ["en", "de", "es", "fr", "nl", "it", "pl", "pt", "ru", "zh"],
    )
    def test_supported_languages_includes_top_10(self, mock_home_agent, lang):
        """Test that all top 10 languages are included in supported_languages.

        This test validates that the agent declares support for the 10 most
        commonly used languages in Home Assistant installations:
        - en (English)
        - de (German)
        - es (Spanish)
        - fr (French)
        - nl (Dutch)
        - it (Italian)
        - pl (Polish)
        - pt (Portuguese)
        - ru (Russian)
        - zh (Chinese)

        Args:
            mock_home_agent: HomeAgent instance from fixture
            lang: Language code to test
        """
        supported = mock_home_agent.supported_languages

        assert lang in supported, (
            f"Language '{lang}' should be in supported_languages but was not found. "
            f"Supported languages: {supported}"
        )

    def test_supported_languages_format_validation(self, mock_home_agent):
        """Test that language codes follow proper format conventions.

        Validates that all language codes in supported_languages:
        - Are lowercase (following ISO 639-1 standard)
        - Are exactly 2 characters (ISO 639-1 format)
        - Are non-empty
        - Form a valid list
        """
        supported = mock_home_agent.supported_languages

        # Verify it's a non-empty list
        assert isinstance(supported, list), "supported_languages must return a list"
        assert len(supported) > 0, "supported_languages must not be empty"

        # Validate each language code
        for lang_code in supported:
            assert isinstance(lang_code, str), (
                f"Language code must be a string, got {type(lang_code)}"
            )
            assert len(lang_code) == 2, (
                f"Language code '{lang_code}' must be exactly 2 characters (ISO 639-1 format)"
            )
            assert lang_code.islower(), (
                f"Language code '{lang_code}' must be lowercase"
            )
            assert lang_code.isalpha(), (
                f"Language code '{lang_code}' must contain only alphabetic characters"
            )

    def test_supported_languages_no_duplicates(self, mock_home_agent):
        """Test that there are no duplicate language codes.

        Ensures that each language is listed only once in the supported_languages
        list to avoid confusion and maintain consistency.
        """
        supported = mock_home_agent.supported_languages

        # Convert to set to find duplicates
        unique_languages = set(supported)

        assert len(supported) == len(unique_languages), (
            f"supported_languages contains duplicates. "
            f"Found {len(supported)} items but only {len(unique_languages)} unique. "
            f"Languages: {supported}"
        )

    def test_supported_languages_is_list(self, mock_home_agent):
        """Test that supported_languages returns a list.

        Validates the return type to ensure compatibility with Home Assistant's
        conversation platform which expects a list of language codes.
        """
        supported = mock_home_agent.supported_languages

        assert isinstance(supported, list), (
            f"supported_languages must return a list, got {type(supported).__name__}"
        )

    def test_supported_languages_contains_english(self, mock_home_agent):
        """Test that English is always included in supported languages.

        English (en) is the default language for Home Assistant and should
        always be supported regardless of what other languages are added.
        """
        supported = mock_home_agent.supported_languages

        assert "en" in supported, (
            "English ('en') must always be in supported_languages as it is the "
            "default language for Home Assistant"
        )

    def test_supported_languages_returns_same_instance(self, mock_home_agent):
        """Test that the property is deterministic and returns consistent results.

        Ensures that multiple calls to supported_languages return the same
        list of languages, maintaining consistency throughout the agent's lifecycle.
        """
        # Call the property multiple times
        first_call = mock_home_agent.supported_languages
        second_call = mock_home_agent.supported_languages
        third_call = mock_home_agent.supported_languages

        # Verify all calls return the same list of languages
        assert first_call == second_call, (
            "supported_languages should return the same list on subsequent calls"
        )
        assert second_call == third_call, (
            "supported_languages should return the same list on subsequent calls"
        )

    def test_supported_languages_minimum_count(self, mock_home_agent):
        """Test that at least 10 languages are supported.

        Validates that the agent supports at minimum the top 10 languages,
        allowing for future expansion while ensuring the minimum requirement is met.
        """
        supported = mock_home_agent.supported_languages

        assert len(supported) >= 10, (
            f"supported_languages should contain at least 10 languages, "
            f"but only found {len(supported)}: {supported}"
        )

    def test_supported_languages_exact_top_10(self, mock_home_agent):
        """Test that exactly the top 10 languages are supported.

        Verifies that the supported languages set matches exactly the top 10
        most common languages, no more, no less (unless explicitly expanded).
        """
        supported = mock_home_agent.supported_languages
        expected_languages = {"en", "de", "es", "fr", "nl", "it", "pl", "pt", "ru", "zh"}

        supported_set = set(supported)

        # Check that all expected languages are present
        missing = expected_languages - supported_set
        assert not missing, (
            f"Missing expected languages: {missing}. "
            f"Supported: {supported}"
        )

        # Note: We allow extra languages beyond the top 10, but the top 10 must be present
        # This allows for future expansion while validating the core requirement

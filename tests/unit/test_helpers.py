"""Unit tests for helper utility functions.

This module tests all utility functions in the helpers module including
formatting, validation, security, and token estimation functions.
"""

import pytest
from unittest.mock import MagicMock, Mock

from homeassistant.const import ATTR_FRIENDLY_NAME, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State

from custom_components.home_agent.helpers import (
    format_entity_state,
    validate_entity_id,
    redact_sensitive_data,
    estimate_tokens,
    truncate_text,
    safe_get_state,
    format_duration,
    merge_dicts,
)
from custom_components.home_agent.exceptions import ValidationError


class TestFormatEntityState:
    """Tests for format_entity_state function."""

    def test_format_entity_state_json_with_all_attributes(self):
        """Test formatting entity state as JSON with all attributes."""
        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {
            "brightness": 128,
            "color_temp": 370,
            "friendly_name": "Living Room Light",
            "_internal": "hidden",
        }

        result = format_entity_state(state, format_type="json")

        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["attributes"]["brightness"] == 128
        assert result["attributes"]["color_temp"] == 370
        assert result["attributes"]["friendly_name"] == "Living Room Light"
        assert "_internal" not in result["attributes"]

    def test_format_entity_state_json_with_filtered_attributes(self):
        """Test formatting entity state as JSON with specific attributes."""
        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {
            "brightness": 128,
            "color_temp": 370,
            "friendly_name": "Living Room Light",
            "color_mode": "xy",
        }

        result = format_entity_state(
            state, attributes=["brightness"], format_type="json"
        )

        assert result["entity_id"] == "light.living_room"
        assert result["state"] == "on"
        assert result["attributes"]["brightness"] == 128
        assert result["attributes"]["friendly_name"] == "Living Room Light"
        assert "color_temp" not in result["attributes"]
        assert "color_mode" not in result["attributes"]

    def test_format_entity_state_natural_language(self):
        """Test formatting entity state as natural language."""
        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {
            "brightness": 128,
            "friendly_name": "Living Room Light",
        }

        result = format_entity_state(state, format_type="natural_language")

        assert isinstance(result, str)
        assert "Living Room Light" in result
        assert "light.living_room" in result
        assert "on" in result
        assert "brightness" in result

    def test_format_entity_state_natural_language_no_friendly_name(self):
        """Test natural language format without friendly_name attribute."""
        state = Mock(spec=State)
        state.entity_id = "sensor.temperature"
        state.state = "72"
        state.attributes = {
            "unit_of_measurement": "°F",
        }

        result = format_entity_state(state, format_type="natural_language")

        assert isinstance(result, str)
        assert "sensor.temperature" in result
        assert "72" in result

    def test_format_entity_state_invalid_format_type(self):
        """Test that invalid format_type raises ValidationError."""
        state = Mock(spec=State)
        state.entity_id = "light.living_room"
        state.state = "on"
        state.attributes = {}

        with pytest.raises(ValidationError) as exc_info:
            format_entity_state(state, format_type="invalid")

        assert "Invalid format_type" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_format_entity_state_empty_attributes(self):
        """Test formatting entity with no attributes."""
        state = Mock(spec=State)
        state.entity_id = "sensor.test"
        state.state = "123"
        state.attributes = {}

        result = format_entity_state(state, format_type="json")

        assert result["entity_id"] == "sensor.test"
        assert result["state"] == "123"
        assert result["attributes"] == {}

    def test_format_entity_state_natural_language_multiple_attributes(self):
        """Test natural language format with multiple attributes."""
        state = Mock(spec=State)
        state.entity_id = "climate.thermostat"
        state.state = "heat"
        state.attributes = {
            "friendly_name": "Thermostat",
            "temperature": 72,
            "humidity": 45,
        }

        result = format_entity_state(state, format_type="natural_language")

        assert "Thermostat" in result
        assert "heat" in result
        assert "temperature" in result or "humidity" in result


class TestValidateEntityId:
    """Tests for validate_entity_id function."""

    def test_validate_entity_id_valid(self):
        """Test validating a valid entity ID."""
        entity_id = "light.living_room"
        result = validate_entity_id(entity_id)
        assert result == entity_id

    def test_validate_entity_id_with_numbers(self):
        """Test validating entity ID with numbers."""
        entity_id = "sensor.temperature_sensor_1"
        result = validate_entity_id(entity_id)
        assert result == entity_id

    def test_validate_entity_id_with_underscores(self):
        """Test validating entity ID with underscores."""
        entity_id = "binary_sensor.front_door_sensor"
        result = validate_entity_id(entity_id)
        assert result == entity_id

    def test_validate_entity_id_no_dot(self):
        """Test that entity ID without dot raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id("invalid_entity")

        assert "Invalid entity_id format" in str(exc_info.value)
        assert "domain.entity_name" in str(exc_info.value)

    def test_validate_entity_id_empty_string(self):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id("")

        assert "Invalid entity_id" in str(exc_info.value)
        assert "non-empty string" in str(exc_info.value)

    def test_validate_entity_id_none(self):
        """Test that None raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id(None)

        assert "Invalid entity_id" in str(exc_info.value)

    def test_validate_entity_id_not_string(self):
        """Test that non-string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id(123)

        assert "Invalid entity_id" in str(exc_info.value)

    def test_validate_entity_id_invalid_domain_characters(self):
        """Test that invalid domain characters raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id("Light.living_room")

        assert "Invalid domain" in str(exc_info.value)
        assert "lowercase" in str(exc_info.value)

    def test_validate_entity_id_invalid_entity_characters(self):
        """Test that invalid entity name characters raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id("light.Living-Room")

        assert "Invalid entity name" in str(exc_info.value)

    def test_validate_entity_id_special_characters_in_domain(self):
        """Test that special characters in domain raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id("light!.living_room")

        assert "Invalid domain" in str(exc_info.value)

    def test_validate_entity_id_multiple_dots(self):
        """Test entity ID with multiple dots (only first split matters)."""
        # This should fail because the entity name has a dot
        with pytest.raises(ValidationError) as exc_info:
            validate_entity_id("light.living.room")

        assert "Invalid entity name" in str(exc_info.value)


class TestRedactSensitiveData:
    """Tests for redact_sensitive_data function."""

    def test_redact_sensitive_data_single_value(self):
        """Test redacting a single sensitive value."""
        text = "Error calling API with key sk-1234567890abcdef"
        sensitive = ["sk-1234567890abcdef"]

        result = redact_sensitive_data(text, sensitive)

        assert "sk-1234567890abcdef" not in result
        assert "***REDACTED***" in result

    def test_redact_sensitive_data_multiple_values(self):
        """Test redacting multiple sensitive values."""
        text = "Key: sk-12345 and Token: tkn-67890"
        sensitive = ["sk-12345", "tkn-67890"]

        result = redact_sensitive_data(text, sensitive)

        assert "sk-12345" not in result
        assert "tkn-67890" not in result
        assert result.count("***REDACTED***") == 2

    def test_redact_sensitive_data_no_sensitive_values(self):
        """Test when no sensitive values are in text."""
        text = "This is a safe message"
        sensitive = ["secret-key"]

        result = redact_sensitive_data(text, sensitive)

        assert result == text

    def test_redact_sensitive_data_empty_text(self):
        """Test with empty text."""
        result = redact_sensitive_data("", ["key"])
        assert result == ""

    def test_redact_sensitive_data_empty_sensitive_list(self):
        """Test with empty sensitive values list."""
        text = "Some text with key-123"
        result = redact_sensitive_data(text, [])
        assert result == text

    def test_redact_sensitive_data_none_text(self):
        """Test with None text."""
        result = redact_sensitive_data(None, ["key"])
        assert result is None

    def test_redact_sensitive_data_none_in_sensitive_list(self):
        """Test with None in sensitive values list."""
        text = "Text with key-123"
        result = redact_sensitive_data(text, [None, "key-123"])
        assert "key-123" not in result
        assert "***REDACTED***" in result

    def test_redact_sensitive_data_non_string_in_sensitive_list(self):
        """Test with non-string value in sensitive list."""
        text = "Text with 12345"
        result = redact_sensitive_data(text, [12345, "key"])
        assert result == text

    def test_redact_sensitive_data_multiple_occurrences(self):
        """Test redacting multiple occurrences of same value."""
        text = "Key: secret, again secret, and secret once more"
        sensitive = ["secret"]

        result = redact_sensitive_data(text, sensitive)

        assert "secret" not in result
        assert result.count("***REDACTED***") == 3


class TestEstimateTokens:
    """Tests for estimate_tokens function."""

    def test_estimate_tokens_short_text(self):
        """Test token estimation for short text."""
        text = "Hello, world!"
        result = estimate_tokens(text)
        assert result > 0
        assert result <= 5  # Should be around 3-4 tokens

    def test_estimate_tokens_medium_text(self):
        """Test token estimation for medium text."""
        text = "This is a longer sentence with more words to estimate."
        result = estimate_tokens(text)
        assert result > 5
        assert result <= 20

    def test_estimate_tokens_long_text(self):
        """Test token estimation for long text."""
        text = "word " * 100  # 100 words
        result = estimate_tokens(text)
        assert result > 50

    def test_estimate_tokens_empty_string(self):
        """Test token estimation for empty string."""
        result = estimate_tokens("")
        assert result == 0

    def test_estimate_tokens_none(self):
        """Test token estimation for None."""
        result = estimate_tokens(None)
        assert result == 0

    def test_estimate_tokens_single_character(self):
        """Test token estimation for single character."""
        result = estimate_tokens("a")
        assert result == 1  # Minimum is 1

    def test_estimate_tokens_whitespace_only(self):
        """Test token estimation for whitespace."""
        result = estimate_tokens("    ")
        assert result == 1  # Should return minimum of 1


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_truncate_text_short_text(self):
        """Test that short text is not truncated."""
        text = "Short text"
        result = truncate_text(text, 100)
        assert result == text

    def test_truncate_text_exact_length(self):
        """Test text at exact max length."""
        text = "12345"
        result = truncate_text(text, 5)
        assert result == text

    def test_truncate_text_needs_truncation(self):
        """Test that long text is truncated."""
        text = "This is a very long message that needs truncation"
        result = truncate_text(text, 15)

        assert len(result) == 15
        assert result.endswith("...")
        assert result.startswith("This is a ve")

    def test_truncate_text_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "Long text that will be truncated"
        result = truncate_text(text, 15, suffix="[...]")

        assert len(result) == 15
        assert result.endswith("[...]")

    def test_truncate_text_empty_string(self):
        """Test truncation with empty string."""
        result = truncate_text("", 10)
        assert result == ""

    def test_truncate_text_none(self):
        """Test truncation with None."""
        result = truncate_text(None, 10)
        assert result is None

    def test_truncate_text_max_length_smaller_than_suffix(self):
        """Test when max_length is smaller than suffix."""
        text = "Some text"
        result = truncate_text(text, 2, suffix="...")

        assert result == ".."  # Should return truncated suffix

    def test_truncate_text_max_length_zero(self):
        """Test with max_length of zero."""
        text = "Some text"
        result = truncate_text(text, 0, suffix="...")

        assert result == ""  # Should return empty suffix


class TestSafeGetState:
    """Tests for safe_get_state function."""

    def test_safe_get_state_valid_state(self):
        """Test getting state from valid State object."""
        state = Mock(spec=State)
        state.state = "on"

        result = safe_get_state(state)

        assert result == "on"

    def test_safe_get_state_none(self):
        """Test with None state object."""
        result = safe_get_state(None)
        assert result == STATE_UNAVAILABLE

    def test_safe_get_state_none_with_custom_default(self):
        """Test with None state and custom default."""
        result = safe_get_state(None, default="custom")
        assert result == "custom"

    def test_safe_get_state_unavailable(self):
        """Test with unavailable state."""
        state = Mock(spec=State)
        state.state = STATE_UNAVAILABLE

        result = safe_get_state(state)

        assert result == STATE_UNAVAILABLE

    def test_safe_get_state_unavailable_with_custom_default(self):
        """Test with unavailable state and custom default."""
        state = Mock(spec=State)
        state.state = STATE_UNAVAILABLE

        result = safe_get_state(state, default="offline")

        assert result == "offline"

    def test_safe_get_state_unknown(self):
        """Test with unknown state."""
        state = Mock(spec=State)
        state.state = STATE_UNKNOWN

        result = safe_get_state(state, default="missing")

        assert result == "missing"

    def test_safe_get_state_valid_numeric_state(self):
        """Test with numeric state value."""
        state = Mock(spec=State)
        state.state = "72.5"

        result = safe_get_state(state)

        assert result == "72.5"


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_duration_seconds_only(self):
        """Test formatting duration in seconds."""
        result = format_duration(45)
        assert result == "45s"

    def test_format_duration_minutes_and_seconds(self):
        """Test formatting duration with minutes and seconds."""
        result = format_duration(125)  # 2 minutes 5 seconds
        assert result == "2m 5s"

    def test_format_duration_hours_minutes_seconds(self):
        """Test formatting duration with hours, minutes, and seconds."""
        result = format_duration(3665)  # 1 hour, 1 minute, 5 seconds
        assert result == "1h 1m 5s"

    def test_format_duration_exact_hours(self):
        """Test formatting exact hours."""
        result = format_duration(7200)  # 2 hours
        assert result == "2h"

    def test_format_duration_exact_minutes(self):
        """Test formatting exact minutes."""
        result = format_duration(180)  # 3 minutes
        assert result == "3m"

    def test_format_duration_fractional_seconds(self):
        """Test formatting fractional seconds."""
        result = format_duration(0.5)
        assert result == "0.50s"

    def test_format_duration_very_small(self):
        """Test formatting very small duration."""
        result = format_duration(0.123)
        assert result == "0.12s"

    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        result = format_duration(0)
        assert result == "0s"

    def test_format_duration_large_value(self):
        """Test formatting large duration."""
        result = format_duration(86400)  # 24 hours
        assert "24h" in result

    def test_format_duration_hours_and_seconds_no_minutes(self):
        """Test formatting with hours and seconds but no minutes."""
        result = format_duration(3605)  # 1 hour and 5 seconds
        assert result == "1h 5s"


class TestMergeDicts:
    """Tests for merge_dicts function."""

    def test_merge_dicts_simple(self):
        """Test merging simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}
        # Ensure original dicts are not modified
        assert base == {"a": 1, "b": 2}
        assert override == {"b": 3, "c": 4}

    def test_merge_dicts_nested(self):
        """Test merging nested dictionaries."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 4}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": {"c": 10, "d": 3}, "e": 4}

    def test_merge_dicts_deep_nested(self):
        """Test merging deeply nested dictionaries."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 10}}}

        result = merge_dicts(base, override)

        assert result == {"a": {"b": {"c": 10, "d": 2}}}

    def test_merge_dicts_empty_base(self):
        """Test merging with empty base dictionary."""
        base = {}
        override = {"a": 1, "b": 2}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": 2}

    def test_merge_dicts_empty_override(self):
        """Test merging with empty override dictionary."""
        base = {"a": 1, "b": 2}
        override = {}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": 2}

    def test_merge_dicts_override_non_dict_with_dict(self):
        """Test overriding non-dict value with dict."""
        base = {"a": 1, "b": 2}
        override = {"b": {"c": 3}}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": {"c": 3}}

    def test_merge_dicts_override_dict_with_non_dict(self):
        """Test overriding dict value with non-dict."""
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": 3}

        result = merge_dicts(base, override)

        assert result == {"a": 1, "b": 3}

    def test_merge_dicts_with_lists(self):
        """Test merging dictionaries containing lists."""
        base = {"a": [1, 2, 3], "b": 2}
        override = {"a": [4, 5]}

        result = merge_dicts(base, override)

        # Lists are replaced, not merged
        assert result == {"a": [4, 5], "b": 2}

    def test_merge_dicts_complex_structure(self):
        """Test merging complex nested structures."""
        base = {
            "config": {
                "llm": {"model": "gpt-4", "temp": 0.7},
                "context": {"mode": "direct"}
            },
            "enabled": True
        }
        override = {
            "config": {
                "llm": {"model": "gpt-3.5"},
                "tools": {"enabled": True}
            }
        }

        result = merge_dicts(base, override)

        expected = {
            "config": {
                "llm": {"model": "gpt-3.5", "temp": 0.7},
                "context": {"mode": "direct"},
                "tools": {"enabled": True}
            },
            "enabled": True
        }
        assert result == expected

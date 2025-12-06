"""Unit tests for helper utility functions.

This module tests all utility functions in the helpers module including
formatting, validation, security, and token estimation functions.
"""

from unittest.mock import Mock

import pytest
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import State

from custom_components.home_agent.exceptions import ValidationError
from custom_components.home_agent.helpers import (
    estimate_tokens,
    format_duration,
    format_entity_state,
    merge_dicts,
    redact_sensitive_data,
    safe_get_state,
    strip_thinking_blocks,
    truncate_text,
    validate_entity_id,
)


class TestStripThinkingBlocks:
    """Tests for strip_thinking_blocks function.

    This function removes <think>...</think> blocks from reasoning model output
    (e.g., Qwen3, DeepSeek R1, o1/o3) to prevent them from appearing in user
    responses and breaking JSON parsing.
    """

    def test_strip_simple_thinking_block(self):
        """Test stripping a simple thinking block."""
        text = "<think>Let me think about this...</think>Hello, world!"
        result = strip_thinking_blocks(text)
        assert result == "Hello, world!"

    def test_strip_thinking_block_at_end(self):
        """Test stripping thinking block at end of text."""
        text = "Here is my answer.<think>I hope that was helpful.</think>"
        result = strip_thinking_blocks(text)
        assert result == "Here is my answer."

    def test_strip_thinking_block_in_middle(self):
        """Test stripping thinking block in the middle of text."""
        text = "First part.<think>Some reasoning here.</think>Second part."
        result = strip_thinking_blocks(text)
        assert result == "First part.Second part."

    def test_strip_multiple_thinking_blocks(self):
        """Test stripping multiple thinking blocks."""
        text = "<think>First thought</think>Hello<think>Second thought</think>World"
        result = strip_thinking_blocks(text)
        assert result == "HelloWorld"

    def test_strip_multiline_thinking_block(self):
        """Test stripping thinking block with newlines."""
        text = """<think>
Let me think step by step:
1. First, I need to understand the question.
2. Then, I need to formulate an answer.
3. Finally, I should respond clearly.
</think>The answer is 42."""
        result = strip_thinking_blocks(text)
        assert result == "The answer is 42."

    def test_strip_thinking_block_with_special_characters(self):
        """Test stripping thinking block containing special characters."""
        text = "<think>What about <entity> and {json: 'data'}?</think>Response here."
        result = strip_thinking_blocks(text)
        assert result == "Response here."

    def test_no_thinking_block(self):
        """Test text without thinking blocks is unchanged."""
        text = "Just a normal response without any thinking."
        result = strip_thinking_blocks(text)
        assert result == text

    def test_empty_string(self):
        """Test empty string returns empty string."""
        result = strip_thinking_blocks("")
        assert result == ""

    def test_none_input(self):
        """Test None input returns None."""
        result = strip_thinking_blocks(None)
        assert result is None

    def test_only_thinking_block(self):
        """Test text that is only a thinking block."""
        text = "<think>All thinking, no response.</think>"
        result = strip_thinking_blocks(text)
        assert result == ""

    def test_nested_angle_brackets_in_thinking(self):
        """Test thinking block containing angle brackets."""
        text = "<think>Consider if x < y and y > z, then...</think>The result is x < z."
        result = strip_thinking_blocks(text)
        assert result == "The result is x < z."

    def test_preserves_other_html_like_tags(self):
        """Test that other HTML-like tags are preserved."""
        text = "<b>Bold</b> and <think>reasoning</think> and <i>italic</i>"
        result = strip_thinking_blocks(text)
        assert result == "<b>Bold</b> and  and <i>italic</i>"

    def test_thinking_block_with_json_content(self):
        """Test thinking block containing JSON (common in reasoning models)."""
        text = '''<think>I should return this JSON:
{"type": "fact", "content": "User likes coffee", "importance": 0.8}
</think>[{"type": "fact", "content": "User likes coffee", "importance": 0.8}]'''
        result = strip_thinking_blocks(text)
        assert result == '[{"type": "fact", "content": "User likes coffee", "importance": 0.8}]'

    def test_whitespace_cleanup_after_stripping(self):
        """Test that result is stripped of leading/trailing whitespace."""
        text = "   <think>thinking</think>   Response   "
        result = strip_thinking_blocks(text)
        assert result == "Response"

    def test_consecutive_thinking_blocks(self):
        """Test stripping consecutive thinking blocks."""
        text = "<think>First</think><think>Second</think>Answer"
        result = strip_thinking_blocks(text)
        assert result == "Answer"

    def test_thinking_block_with_code(self):
        """Test thinking block containing code snippets."""
        text = """<think>
```python
def hello():
    return "world"
```
</think>Here's how to say hello."""
        result = strip_thinking_blocks(text)
        assert result == "Here's how to say hello."

    # Additional edge case tests for issue #64 coverage

    def test_unclosed_thinking_tag(self):
        """Test unclosed <think> tag is left unchanged.

        When a thinking block is not properly closed, we leave the text
        unchanged rather than risk removing valid content.
        """
        text = "<think>This thinking block never closes... Answer here"
        result = strip_thinking_blocks(text)
        # Unclosed tags should be left unchanged
        assert result == text

    def test_unopened_closing_tag(self):
        """Test orphaned </think> tag is preserved."""
        text = "Some text </think> more text"
        result = strip_thinking_blocks(text)
        assert result == "Some text </think> more text"

    def test_case_sensitive_tags(self):
        """Test that tag matching is case-sensitive (only lowercase matches)."""
        text = "<Think>Should not match</Think>Answer"
        result = strip_thinking_blocks(text)
        assert "<Think>" in result
        assert "Should not match" in result

    def test_uppercase_think_tags(self):
        """Test uppercase THINK tags are not removed."""
        text = "<THINK>uppercase thinking</THINK>Response"
        result = strip_thinking_blocks(text)
        assert "<THINK>" in result
        assert "uppercase thinking" in result

    def test_unicode_in_thinking_blocks(self):
        """Test thinking blocks with unicode content (multilingual)."""
        text = "<think>Pensando en español: ñ, é, ü</think>Respuesta: 你好"
        result = strip_thinking_blocks(text)
        assert result == "Respuesta: 你好"
        assert "español" not in result

    def test_emoji_in_thinking_blocks(self):
        """Test thinking blocks containing emojis."""
        text = "<think>🤔 Thinking... 💭</think>✅ Done!"
        result = strip_thinking_blocks(text)
        assert result == "✅ Done!"
        assert "🤔" not in result

    def test_cyrillic_in_thinking_blocks(self):
        """Test thinking blocks with Cyrillic characters."""
        text = "<think>Думаю о проблеме...</think>Ответ: да"
        result = strip_thinking_blocks(text)
        assert result == "Ответ: да"
        assert "Думаю" not in result

    def test_chinese_in_thinking_blocks(self):
        """Test thinking blocks with Chinese characters."""
        text = "<think>让我思考一下这个问题</think>答案是42"
        result = strip_thinking_blocks(text)
        assert result == "答案是42"

    def test_very_long_thinking_block(self):
        """Test with very large thinking block (performance check)."""
        long_thinking = "word " * 10000  # 10k words
        text = f"<think>{long_thinking}</think>Short answer"
        result = strip_thinking_blocks(text)
        assert result == "Short answer"
        assert len(result) < 20

    def test_many_thinking_blocks(self):
        """Test with many thinking blocks."""
        blocks = "".join(f"<think>thought {i}</think>text{i} " for i in range(100))
        result = strip_thinking_blocks(blocks)
        assert "<think>" not in result
        assert all(f"text{i}" in result for i in range(100))

    def test_thinking_block_empty_content(self):
        """Test thinking block with empty content."""
        text = "<think></think>Answer here"
        result = strip_thinking_blocks(text)
        assert result == "Answer here"

    def test_thinking_block_only_whitespace(self):
        """Test thinking block containing only whitespace."""
        text = "<think>   \n\t  </think>Answer"
        result = strip_thinking_blocks(text)
        assert result == "Answer"

    def test_mixed_valid_and_malformed_tags(self):
        """Test text with both valid and malformed thinking tags."""
        text = "<think>valid</think>Text<Think>invalid</Think><think>also valid</think>End"
        result = strip_thinking_blocks(text)
        assert result == "Text<Think>invalid</Think>End"

    def test_thinking_block_with_html_entities(self):
        """Test thinking block containing HTML entities."""
        text = "<think>&lt;div&gt; and &amp; symbols</think>Clean response"
        result = strip_thinking_blocks(text)
        assert result == "Clean response"

    def test_thinking_block_with_xml_like_content(self):
        """Test thinking block containing XML-like content."""
        text = "<think><internal><data>value</data></internal></think>Output"
        result = strip_thinking_blocks(text)
        assert result == "Output"


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

        result = format_entity_state(state, attributes=["brightness"], format_type="json")

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
            "config": {"llm": {"model": "gpt-4", "temp": 0.7}, "context": {"mode": "direct"}},
            "enabled": True,
        }
        override = {"config": {"llm": {"model": "gpt-3.5"}, "tools": {"enabled": True}}}

        result = merge_dicts(base, override)

        expected = {
            "config": {
                "llm": {"model": "gpt-3.5", "temp": 0.7},
                "context": {"mode": "direct"},
                "tools": {"enabled": True},
            },
            "enabled": True,
        }
        assert result == expected

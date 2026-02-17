"""Unit tests for template-based API key support.

This module tests the render_template_value helper function and verifies
backwards compatibility — plain string API keys must continue to work
unchanged, while Jinja template values (e.g., {{ states('input_text.api_key') }})
are rendered via Home Assistant's template engine.
"""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.home_agent.helpers import render_template_value

# Patch path for the lazy Template import inside render_template_value
_TEMPLATE_PATCH = "homeassistant.helpers.template.Template"


class TestRenderTemplateValue:
    """Tests for the render_template_value helper function."""

    def test_plain_string_passthrough(self, mock_hass):
        """Test that a plain API key string is returned unchanged."""
        result = render_template_value(mock_hass, "sk-1234567890abcdef")
        assert result == "sk-1234567890abcdef"

    def test_empty_string_passthrough(self, mock_hass):
        """Test that an empty string is returned unchanged."""
        result = render_template_value(mock_hass, "")
        assert result == ""

    def test_none_passthrough(self, mock_hass):
        """Test that None is returned as-is."""
        result = render_template_value(mock_hass, None)
        assert result is None

    def test_non_string_passthrough(self, mock_hass):
        """Test that non-string values are returned as-is."""
        result = render_template_value(mock_hass, 12345)
        assert result == 12345

    def test_single_brace_not_treated_as_template(self, mock_hass):
        """Test that a string with single { but not {{ is not treated as template."""
        result = render_template_value(mock_hass, "key-with-{brace}")
        assert result == "key-with-{brace}"

    def test_template_rendering(self, mock_hass):
        """Test that a value containing {{ }} is rendered via HA Template."""
        template_str = "{{ states('input_text.api_key') }}"

        with patch(_TEMPLATE_PATCH) as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                return_value="rendered-api-key"
            )
            mock_template_class.return_value = mock_template

            result = render_template_value(mock_hass, template_str)

            assert result == "rendered-api-key"
            mock_template_class.assert_called_once_with(template_str, mock_hass)
            mock_template.async_render.assert_called_once()

    def test_template_rendering_secrets(self, mock_hass):
        """Test that secrets template syntax triggers rendering."""
        template_str = "{{ secrets.llm_api_key }}"

        with patch(_TEMPLATE_PATCH) as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                return_value="secret-key-value"
            )
            mock_template_class.return_value = mock_template

            result = render_template_value(mock_hass, template_str)

            assert result == "secret-key-value"

    def test_template_rendering_failure_propagates(self, mock_hass):
        """Test that template rendering errors propagate (not silently swallowed)."""
        template_str = "{{ invalid_function() }}"

        with patch(_TEMPLATE_PATCH) as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                side_effect=Exception("Template rendering failed")
            )
            mock_template_class.return_value = mock_template

            with pytest.raises(Exception, match="Template rendering failed"):
                render_template_value(mock_hass, template_str)

    def test_rendered_value_converted_to_string(self, mock_hass):
        """Test that rendered template output is always converted to string."""
        template_str = "{{ states('sensor.count') }}"

        with patch(_TEMPLATE_PATCH) as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(return_value=42)
            mock_template_class.return_value = mock_template

            result = render_template_value(mock_hass, template_str)

            assert result == "42"
            assert isinstance(result, str)


class TestApiKeyBackwardsCompatibility:
    """Tests verifying that template rendering integrates correctly at usage points.

    These tests ensure that plain API key strings continue to work identically
    to the previous behavior — no template rendering is attempted.
    """

    def test_llm_mixin_plain_api_key_in_header(self, mock_hass):
        """Test that LLMMixin uses a plain API key directly in Authorization header."""
        api_key = "sk-test-key-12345"
        result = render_template_value(mock_hass, api_key)
        assert result == api_key
        assert f"Bearer {result}" == "Bearer sk-test-key-12345"

    def test_llm_mixin_empty_api_key_omits_header(self, mock_hass):
        """Test that an empty API key results in no Authorization header."""
        api_key = render_template_value(mock_hass, "")
        # Empty string is falsy, so Authorization header should be omitted
        assert not api_key

    def test_llm_mixin_template_api_key_renders(self, mock_hass):
        """Test that a template API key is rendered before use."""
        template_str = "{{ states('input_text.llm_key') }}"

        with patch(_TEMPLATE_PATCH) as mock_template_class:
            mock_template = MagicMock()
            mock_template.async_render = MagicMock(
                return_value="dynamic-key-from-helper"
            )
            mock_template_class.return_value = mock_template

            result = render_template_value(mock_hass, template_str)

            assert result == "dynamic-key-from-helper"
            assert f"Bearer {result}" == "Bearer dynamic-key-from-helper"


class TestSchemaAcceptsTemplates:
    """Tests that config schemas accept both plain strings and template strings.

    TemplateSelector validates within the event loop at schema call time,
    so we mock it to behave as a plain string passthrough for unit tests.
    """

    @pytest.fixture(autouse=True)
    def _mock_template_selector(self):
        """Mock TemplateSelector to accept strings without event loop."""
        with patch(
            "custom_components.home_agent.config.schemas.selector.TemplateSelector",
            return_value=str,
        ):
            yield

    def test_user_step_schema_accepts_plain_api_key(self):
        """Test that user step schema accepts a plain API key string."""
        from custom_components.home_agent.config.schemas import (
            get_user_step_schema,
        )

        schema = get_user_step_schema()
        data = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-plain-key-12345",
            "llm_model": "gpt-4",
        }
        result = schema(data)
        assert result["llm_api_key"] == "sk-plain-key-12345"

    def test_user_step_schema_accepts_template_api_key(self):
        """Test that user step schema accepts a Jinja template string."""
        from custom_components.home_agent.config.schemas import (
            get_user_step_schema,
        )

        schema = get_user_step_schema()
        data = {
            "name": "Test Agent",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "{{ states('input_text.api_key') }}",
            "llm_model": "gpt-4",
        }
        result = schema(data)
        assert result["llm_api_key"] == "{{ states('input_text.api_key') }}"

    def test_llm_settings_schema_accepts_plain_api_key(self):
        """Test that LLM settings schema accepts a plain API key."""
        from custom_components.home_agent.config.schemas import (
            get_llm_settings_schema,
        )

        schema = get_llm_settings_schema({"llm_api_key": ""})
        data = {
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "sk-key-from-options",
            "llm_model": "gpt-4",
        }
        result = schema(data)
        assert result["llm_api_key"] == "sk-key-from-options"

    def test_llm_settings_schema_accepts_template_api_key(self):
        """Test that LLM settings schema accepts a Jinja template."""
        from custom_components.home_agent.config.schemas import (
            get_llm_settings_schema,
        )

        schema = get_llm_settings_schema({"llm_api_key": ""})
        data = {
            "llm_base_url": "https://api.openai.com/v1",
            "llm_api_key": "{{ secrets.openai_key }}",
            "llm_model": "gpt-4",
        }
        result = schema(data)
        assert result["llm_api_key"] == "{{ secrets.openai_key }}"

    def test_external_llm_schema_accepts_template(self):
        """Test that external LLM schema accepts a template API key."""
        from custom_components.home_agent.config.schemas import (
            get_external_llm_settings_schema,
        )

        schema = get_external_llm_settings_schema({}, {})
        data = {
            "external_llm_enabled": True,
            "external_llm_api_key": "{{ states('input_text.ext_key') }}",
        }
        result = schema(data)
        assert (
            result["external_llm_api_key"]
            == "{{ states('input_text.ext_key') }}"
        )

    def test_vector_db_schema_accepts_template(self):
        """Test that vector DB schema accepts a template for OpenAI API key."""
        from custom_components.home_agent.config.schemas import (
            get_vector_db_settings_schema,
        )

        schema = get_vector_db_settings_schema({}, {}, "")
        data = {
            "openai_api_key": "{{ secrets.openai_embedding_key }}",
        }
        result = schema(data)
        assert (
            result["openai_api_key"]
            == "{{ secrets.openai_embedding_key }}"
        )

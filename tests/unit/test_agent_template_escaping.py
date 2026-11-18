"""Unit tests for literal template escaping in HomeAgent prompts."""

from unittest.mock import MagicMock

import pytest

from custom_components.home_agent.agent.core import HomeAgent


@pytest.fixture
def agent(mock_hass):
    """Create a HomeAgent instance for template escaping tests."""
    config = {
        "llm_model": "gpt-4o-mini",
        "context_mode": "direct",
        "history_enabled": False,
        "emit_events": False,
    }
    session_manager = MagicMock()
    return HomeAgent(mock_hass, config, session_manager)


def test_escape_literal_template_examples(agent):
    """Wrap literal template examples in raw blocks."""
    source = 'template="{{ now() }}"'
    escaped = agent._escape_literal_template_examples(source)

    assert escaped == 'template="{% raw %}{{ now() }}{% endraw %}"'


def test_escape_literal_template_examples_keeps_existing_raw(agent):
    """Keep already raw-wrapped template examples unchanged."""
    source = 'template="{% raw %}{{ now() }}{% endraw %}"'
    escaped = agent._escape_literal_template_examples(source)

    assert escaped == source


def test_escape_literal_template_examples_plain_text_unchanged(agent):
    """Leave strings without template syntax untouched."""
    source = "This string has no Jinja template example."
    escaped = agent._escape_literal_template_examples(source)

    assert escaped == source

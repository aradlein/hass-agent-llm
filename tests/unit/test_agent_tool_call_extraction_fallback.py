"""Unit tests for tool-call fallback extraction from assistant content."""

import json
from unittest.mock import MagicMock

import pytest

from custom_components.home_agent.agent.core import HomeAgent


@pytest.fixture
def agent(mock_hass):
    """Create a HomeAgent instance for fallback extraction tests."""
    config = {
        "llm_model": "gpt-4o-mini",
        "context_mode": "direct",
        "history_enabled": False,
        "emit_events": False,
    }
    session_manager = MagicMock()
    return HomeAgent(mock_hass, config, session_manager)


def _tool_name_and_args(tool_call):
    function = tool_call["function"]
    return function["name"], json.loads(function["arguments"])


def test_extract_tool_call_fallback_standard_format(agent):
    """Extract wrapped format: {tool, arguments}."""
    content = '{"tool": "ha_search", "arguments": {"query": "kitchen", "filters": {"domain": "light"}}}'

    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback(content)

    assert extraction_mode == "wrapped"
    assert len(tool_calls) == 1
    name, arguments = _tool_name_and_args(tool_calls[0])
    assert name == "ha_search"
    assert arguments == {"query": "kitchen", "filters": {"domain": "light"}}


def test_extract_tool_call_fallback_from_json_code_fence(agent):
    """Extract wrapped format from ```json fenced content."""
    content = '```json\n{"tool": "ha_search", "arguments": {"query": "office"}}\n```'

    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback(content)

    assert extraction_mode == "wrapped"
    assert len(tool_calls) == 1
    name, arguments = _tool_name_and_args(tool_calls[0])
    assert name == "ha_search"
    assert arguments == {"query": "office"}


def test_extract_tool_call_fallback_flat_registered_tool(agent):
    """Extract flat format when key matches a registered tool name."""
    agent.tool_handler.get_registered_tools = MagicMock(return_value=["ha_search", "ha_query"])
    content = '{"ha_search": {"query": "bedroom"}}'

    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback(content)

    assert extraction_mode == "flat"
    assert len(tool_calls) == 1
    name, arguments = _tool_name_and_args(tool_calls[0])
    assert name == "ha_search"
    assert arguments == {"query": "bedroom"}


def test_extract_tool_call_fallback_flat_unregistered_tool_returns_none(agent):
    """Prevent false positives for flat format with unknown tool names."""
    agent.tool_handler.get_registered_tools = MagicMock(return_value=["ha_query"])
    content = '{"ha_search": {"query": "garage"}}'

    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback(content)

    assert tool_calls == []
    assert extraction_mode is None


def test_extract_tool_call_fallback_invalid_json_returns_none(agent):
    """Return no tool call for invalid JSON content."""
    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback('{"tool": ')

    assert tool_calls == []
    assert extraction_mode is None


@pytest.mark.parametrize("content", [None, "", "   "])
def test_extract_tool_call_fallback_empty_or_none_returns_none(agent, content):
    """Return no tool call for None/empty content."""
    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback(content)

    assert tool_calls == []
    assert extraction_mode is None


@pytest.mark.parametrize("content", ['[]', '"string"', '123'])
def test_extract_tool_call_fallback_non_dict_json_returns_none(agent, content):
    """Return no tool call when parsed JSON is not an object."""
    tool_calls, extraction_mode = agent._extract_tool_call_from_content_fallback(content)

    assert tool_calls == []
    assert extraction_mode is None

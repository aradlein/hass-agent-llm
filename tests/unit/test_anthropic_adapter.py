"""Unit tests for the native Anthropic Messages adapter.

These tests pin the OpenAI <-> Anthropic wire translation, in particular the
``tool_use.input``-must-be-an-object contract that an OpenAI->Anthropic gateway
rewrite commonly gets wrong (forwarding the JSON-string ``arguments`` verbatim,
which Anthropic rejects with
``messages.N.content.0.tool_use.input: Input should be an object``).
"""

import json

from custom_components.home_agent.agent.anthropic_adapter import (
    from_anthropic_response,
    messages_endpoint,
    to_anthropic_request,
)
from custom_components.home_agent.helpers import is_anthropic_backend


class TestIsAnthropicBackend:
    """Detection of native Anthropic endpoints."""

    def test_official_host(self):
        assert is_anthropic_backend("https://api.anthropic.com") is True

    def test_gateway_anthropic_path(self):
        assert is_anthropic_backend("http://gateway:20128/anthropic") is True

    def test_openai_path_is_not_anthropic(self):
        assert is_anthropic_backend("http://gateway:20128/v1") is False

    def test_empty(self):
        assert is_anthropic_backend("") is False


class TestMessagesEndpoint:
    """Base-URL normalisation to the /v1/messages endpoint."""

    def test_anthropic_base(self):
        assert (
            messages_endpoint("http://gw:20128/anthropic")
            == "http://gw:20128/anthropic/v1/messages"
        )

    def test_anthropic_v1_base(self):
        assert (
            messages_endpoint("http://gw:20128/anthropic/v1")
            == "http://gw:20128/anthropic/v1/messages"
        )

    def test_trailing_slash(self):
        assert messages_endpoint("http://gw/anthropic/") == "http://gw/anthropic/v1/messages"


class TestToAnthropicRequest:
    """OpenAI request -> Anthropic request translation."""

    def _request(self):
        messages = [
            {"role": "system", "content": "You are a home agent."},
            {"role": "user", "content": "Schalte das Licht aus"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "ha_control",
                            "arguments": '{"entity_id":"light.x","action":"turn_off"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": '{"success": true}'},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "ha_control",
                    "description": "control",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        return to_anthropic_request(
            messages, tools, model="claude-opus-4-8", max_tokens=256, temperature=0.7, top_p=1.0
        )

    def test_system_hoisted_to_top_level(self):
        body = self._request()
        assert body["system"] == "You are a home agent."
        assert all(m["role"] != "system" for m in body["messages"])

    def test_max_tokens_required_field(self):
        assert self._request()["max_tokens"] == 256

    def test_tool_use_input_is_object_not_string(self):
        """The core regression: arguments JSON-string -> input object."""
        body = self._request()
        tool_use = [
            b
            for m in body["messages"]
            if m["role"] == "assistant"
            for b in m["content"]
            if b["type"] == "tool_use"
        ]
        assert len(tool_use) == 1
        assert isinstance(tool_use[0]["input"], dict)
        assert tool_use[0]["input"] == {"entity_id": "light.x", "action": "turn_off"}
        assert tool_use[0]["id"] == "toolu_1"

    def test_tool_result_block(self):
        body = self._request()
        results = [
            b
            for m in body["messages"]
            if m["role"] == "user"
            for b in m["content"]
            if b.get("type") == "tool_result"
        ]
        assert len(results) == 1
        assert results[0]["tool_use_id"] == "toolu_1"

    def test_tools_use_input_schema(self):
        body = self._request()
        assert body["tools"][0]["name"] == "ha_control"
        assert "input_schema" in body["tools"][0]
        assert "function" not in body["tools"][0]

    def test_malformed_arguments_default_to_empty_object(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "t", "type": "function", "function": {"name": "x", "arguments": "not json"}}
                ],
            }
        ]
        body = to_anthropic_request(messages, None, model="m", max_tokens=8, temperature=0.0, top_p=1.0)
        tu = body["messages"][0]["content"][0]
        assert tu["type"] == "tool_use" and tu["input"] == {}


class TestFromAnthropicResponse:
    """Anthropic response -> OpenAI response translation."""

    def test_text_and_tool_use(self):
        resp = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Mach ich."},
                {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "ha_control",
                    "input": {"entity_id": "light.x", "action": "turn_off"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
        oai = from_anthropic_response(resp)
        msg = oai["choices"][0]["message"]
        assert msg["content"] == "Mach ich."
        assert oai["choices"][0]["finish_reason"] == "tool_calls"
        assert len(msg["tool_calls"]) == 1
        # arguments must be a JSON string for the OpenAI consumer (core json.loads it)
        args = msg["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"entity_id": "light.x", "action": "turn_off"}

    def test_usage_mapped(self):
        resp = {"content": [], "stop_reason": "end_turn", "usage": {"input_tokens": 100, "output_tokens": 20}}
        assert from_anthropic_response(resp)["usage"] == {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }

    def test_plain_text_has_no_tool_calls(self):
        resp = {"content": [{"type": "text", "text": "Hallo"}], "stop_reason": "end_turn", "usage": {}}
        msg = from_anthropic_response(resp)["choices"][0]["message"]
        assert "tool_calls" not in msg
        assert msg["content"] == "Hallo"

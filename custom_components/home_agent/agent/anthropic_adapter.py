"""Native Anthropic Messages API adapter for HomeAgent.

The rest of HomeAgent speaks the OpenAI chat-completions shape everywhere
(``messages``/``tools`` in, ``choices[0].message`` out). Some gateways translate
that to Anthropic for Claude models, but the OpenAI→Anthropic tool-call rewrite
is a common source of breakage: OpenAI carries ``tool_calls[].function.arguments``
as a JSON **string**, while Anthropic requires ``tool_use.input`` to be an
**object**. A gateway that forwards the string verbatim makes Claude reject every
follow-up turn with ``messages.N.content.0.tool_use.input: Input should be an
object``.

This adapter sidesteps the gateway translation entirely by talking the native
Anthropic Messages API. It translates ONLY at the wire boundary:

    OpenAI messages/tools  --to_anthropic_request-->  Anthropic /v1/messages
    Anthropic response     --from_anthropic_response->  OpenAI choices[0].message

so ``core.py`` keeps consuming the OpenAI shape unchanged. Activated when the
configured base URL points at an Anthropic-style endpoint (see
``helpers.is_anthropic_backend``); streaming is disabled in that mode so only the
synchronous ``_call_llm`` path runs through here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from ..exceptions import AuthenticationError, HomeAgentError

_LOGGER = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"

# Anthropic requires max_tokens; use a sane default if the caller omits it.
DEFAULT_MAX_TOKENS = 1024

_STOP_REASON_TO_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def messages_endpoint(base_url: str) -> str:
    """Return the Anthropic messages URL for a configured base URL.

    Accepts both ``…/anthropic`` and ``…/anthropic/v1`` style bases and
    normalises to a single ``/v1/messages`` suffix.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _coalesce_text(blocks: list[dict[str, Any]]) -> str:
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def to_anthropic_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Translate an OpenAI-format request into an Anthropic Messages request.

    - ``system`` messages are hoisted to the top-level ``system`` field.
    - assistant ``tool_calls`` become ``tool_use`` content blocks with the
      JSON-string ``arguments`` parsed back into an ``input`` object.
    - ``role: tool`` messages become ``tool_result`` blocks, coalesced into the
      preceding user message (Anthropic groups tool results in one user turn).
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def _append_user_block(block: dict[str, Any]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if content:
                system_parts.append(content if isinstance(content, str) else str(content))
            continue

        if role == "user":
            text = content if isinstance(content, str) else json.dumps(content)
            # Plain user turns start a fresh message (don't merge into a tool-result turn).
            out.append({"role": "user", "content": [{"type": "text", "text": text}]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (ValueError, TypeError):
                    parsed = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": parsed,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            out.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            result = content if isinstance(content, str) else json.dumps(content)
            _append_user_block(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": result,
                }
            )
            continue

    body: dict[str, Any] = {
        "model": model,
        "messages": out,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "temperature": temperature,
        "top_p": top_p,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    if tools:
        body["tools"] = [
            {
                "name": t.get("function", {}).get("name", ""),
                "description": t.get("function", {}).get("description", ""),
                "input_schema": t.get("function", {}).get("parameters", {"type": "object"}),
            }
            for t in tools
            if t.get("type") == "function"
        ]
    return body


def from_anthropic_response(resp: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages response into the OpenAI shape.

    ``text`` blocks join into ``message.content``; ``tool_use`` blocks become
    ``message.tool_calls`` with ``input`` re-serialised to the JSON-string
    ``arguments`` that the OpenAI path expects. Token usage is mapped to
    ``prompt_tokens``/``completion_tokens``/``total_tokens``.
    """
    content_blocks = resp.get("content", []) or []
    text = _coalesce_text(content_blocks)

    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp.get("usage", {}) or {}
    prompt = usage.get("input_tokens", 0)
    completion = usage.get("output_tokens", 0)

    return {
        "choices": [
            {
                "message": message,
                "finish_reason": _STOP_REASON_TO_FINISH.get(resp.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


async def call_anthropic(
    session: aiohttp.ClientSession,
    *,
    url: str,
    api_key: str,
    proxy_headers: dict[str, str] | None,
    body: dict[str, Any],
) -> dict[str, Any]:
    """POST a translated request to the Anthropic Messages API and return the
    raw Anthropic response dict. Errors map to HomeAgent's exception types so the
    retry/handling in ``_call_llm`` behaves identically to the OpenAI path."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
    }
    if api_key:
        # Native Anthropic uses x-api-key; keep Bearer too for gateways that
        # authenticate that way (harmless to Anthropic, which ignores it).
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    if proxy_headers:
        headers.update(proxy_headers)

    try:
        async with session.post(url, headers=headers, json=body, allow_redirects=False) as response:
            if response.status == 401:
                raise AuthenticationError(
                    "Anthropic API authentication failed. Check your API key configuration."
                )
            if response.status != 200:
                error_text = await response.text()
                raise HomeAgentError(
                    f"Anthropic API returned status {response.status}: {error_text}"
                )
            return await response.json()
    except aiohttp.ClientError as err:
        raise HomeAgentError(f"Failed to connect to Anthropic API: {err}") from err

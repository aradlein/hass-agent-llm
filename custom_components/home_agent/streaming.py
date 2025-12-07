"""Streaming response handler for Home Agent (OpenAI/Ollama compatible)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncGenerator

from homeassistant.components import conversation
from homeassistant.helpers import llm

from .helpers import strip_thinking_blocks

_LOGGER = logging.getLogger(__name__)

# Pattern for detecting incomplete thinking blocks in streaming content
_THINK_START_PATTERN = re.compile(r"<think>")
_THINK_END_PATTERN = re.compile(r"</think>")


class OpenAIStreamingHandler:
    """Handles streaming responses from OpenAI-compatible APIs (Ollama).

    This class processes Server-Sent Events (SSE) from OpenAI-compatible streaming
    APIs and converts them into Home Assistant's conversation delta format. It handles:
    - Text content streaming
    - Tool call detection and accumulation
    - Incremental JSON argument parsing
    - Multiple indexed tool calls
    """

    def __init__(self) -> None:
        """Initialize the streaming handler."""
        self._current_tool_calls: dict[int, dict[str, Any]] = {}
        # OpenAI uses indexed tool calls that can be streamed incrementally
        self._usage: dict[str, int] | None = None
        # Track token usage from the stream
        self._in_thinking_block: bool = False
        # Track if we're currently inside a <think>...</think> block
        self._thinking_buffer: str = ""
        # Buffer for potential partial thinking block tags

    def _filter_thinking_content(self, content: str) -> str | None:
        """Filter out thinking blocks from streaming content.

        Handles thinking blocks that may span multiple chunks by tracking
        state across calls. Returns only content that should be displayed
        to the user.

        Args:
            content: Raw content from the stream chunk

        Returns:
            Filtered content to yield, or None if all content was filtered
        """
        # Combine with any buffered content
        full_content = self._thinking_buffer + content
        self._thinking_buffer = ""

        result_parts = []
        i = 0

        while i < len(full_content):
            if self._in_thinking_block:
                # Look for closing tag
                end_match = _THINK_END_PATTERN.search(full_content, i)
                if end_match:
                    # Found end of thinking block
                    self._in_thinking_block = False
                    i = end_match.end()
                else:
                    # Still in thinking block, check for partial closing tag
                    # Buffer the last few characters in case </think> spans chunks
                    if len(full_content) >= 8 and full_content[-8:].startswith("</"):
                        self._thinking_buffer = full_content[-8:]
                    break
            else:
                # Look for opening tag
                start_match = _THINK_START_PATTERN.search(full_content, i)
                if start_match:
                    # Yield content before the thinking block
                    if start_match.start() > i:
                        result_parts.append(full_content[i:start_match.start()])
                    self._in_thinking_block = True
                    i = start_match.end()
                else:
                    # Check for partial opening tag at end
                    # Buffer content that might be start of <think>
                    for j in range(min(7, len(full_content)), 0, -1):
                        potential_start = full_content[-j:]
                        if "<think>"[:j] == potential_start:
                            result_parts.append(full_content[i:-j])
                            self._thinking_buffer = potential_start
                            break
                    else:
                        # No partial tag, yield all remaining content
                        result_parts.append(full_content[i:])
                    break

        result = "".join(result_parts)
        return result if result else None

    def _parse_sse_line(self, line: str) -> dict[str, Any] | None:
        """Parse an SSE line.

        Args:
            line: SSE line (e.g., "data: {...}")

        Returns:
            Parsed JSON dict or None if [DONE] or empty
        """
        line = line.strip()
        if not line or not line.startswith("data: "):
            return None

        data = line[6:]  # Remove "data: " prefix

        if data == "[DONE]":
            return None

        try:
            result: dict[str, Any] = json.loads(data)
            return result
        except json.JSONDecodeError:
            _LOGGER.error("Failed to parse SSE data: %s", data)
            return None

    async def transform_openai_stream(
        self,
        stream: AsyncGenerator[str, None],  # SSE lines from aiohttp
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
        """Transform OpenAI streaming events to HA delta format.

        This method processes Server-Sent Events from OpenAI-compatible APIs
        (like Ollama) and yields Home Assistant conversation delta dictionaries.
        It handles:
        - Message initialization
        - Text content streaming
        - Tool call detection and accumulation
        - Tool call JSON argument parsing
        - Multiple indexed tool calls
        - Error handling and logging

        Args:
            stream: OpenAI SSE streaming response (text lines)

        Yields:
            AssistantContentDeltaDict objects for HA conversation API

        Example:
            async for delta in handler.transform_openai_stream(stream):
                # Process delta (role, content, or tool_calls)
                pass
        """
        try:
            # Yield initial role to begin the message
            yield {"role": "assistant"}

            async for line in stream:
                # Parse SSE line
                chunk = self._parse_sse_line(line)
                if chunk is None:
                    continue

                _LOGGER.debug("Processing streaming chunk: %s", chunk)

                # Log if we see any usage-related fields (for debugging Ollama compatibility)
                if any(key in chunk for key in ["usage", "prompt_eval_count", "eval_count"]):
                    _LOGGER.info(
                        "Found usage/token data in chunk: %s",
                        {
                            k: v
                            for k, v in chunk.items()
                            if k
                            in [
                                "usage",
                                "prompt_eval_count",
                                "eval_count",
                                "prompt_eval_duration",
                                "eval_duration",
                            ]
                        },
                    )

                # Capture usage data if present
                # (check BEFORE choices, as usage chunks may have empty choices)
                # OpenAI format: chunk["usage"] = {prompt_tokens, completion_tokens, total_tokens}
                if "usage" in chunk:
                    self._usage = chunk["usage"]
                    _LOGGER.info("Token usage received (OpenAI format): %s", self._usage)

                # Ollama format: chunk has prompt_eval_count, eval_count at root level
                elif "prompt_eval_count" in chunk or "eval_count" in chunk:
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)
                    self._usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                    _LOGGER.info("Token usage received (Ollama format): %s", self._usage)

                # Extract delta from choices array
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # Handle role (first chunk)
                if "role" in delta:
                    _LOGGER.debug("Stream started with role: %s", delta["role"])

                # Handle text content - filter out thinking blocks from reasoning models
                if "content" in delta and delta["content"]:
                    filtered_content = self._filter_thinking_content(delta["content"])
                    if filtered_content:
                        yield {"content": filtered_content}

                # Handle tool calls
                if "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        index = tool_call_delta.get("index", 0)

                        # Initialize tool call if this is the first chunk for this index
                        if index not in self._current_tool_calls:
                            self._current_tool_calls[index] = {
                                "id": tool_call_delta.get("id", ""),
                                "name": "",
                                "arguments": "",
                            }
                            _LOGGER.debug("Tool call started at index %d", index)

                        # Update tool call ID if present
                        if "id" in tool_call_delta:
                            self._current_tool_calls[index]["id"] = tool_call_delta["id"]

                        # Update function info if present
                        if "function" in tool_call_delta:
                            function = tool_call_delta["function"]

                            # Update name if present
                            if "name" in function:
                                self._current_tool_calls[index]["name"] = function["name"]
                                _LOGGER.debug(
                                    "Tool call %d name: %s",
                                    index,
                                    function["name"],
                                )

                            # Accumulate arguments
                            if "arguments" in function:
                                self._current_tool_calls[index]["arguments"] += function[
                                    "arguments"
                                ]
                                _LOGGER.debug(
                                    "Accumulated tool args for %d (length: %d)",
                                    index,
                                    len(self._current_tool_calls[index]["arguments"]),
                                )

                # Handle finish_reason - finalize tool calls if present
                if finish_reason and self._current_tool_calls:
                    _LOGGER.debug(
                        "Stream finished with reason: %s, finalizing %d tool calls",
                        finish_reason,
                        len(self._current_tool_calls),
                    )

                    # Yield all accumulated tool calls
                    tool_inputs = []
                    for index in sorted(self._current_tool_calls.keys()):
                        tool_call = self._current_tool_calls[index]

                        try:
                            # Parse accumulated JSON arguments
                            tool_args = (
                                json.loads(tool_call["arguments"]) if tool_call["arguments"] else {}
                            )

                            tool_inputs.append(
                                llm.ToolInput(
                                    id=tool_call["id"],
                                    tool_name=tool_call["name"],
                                    tool_args=tool_args,
                                )
                            )

                            _LOGGER.debug(
                                "Tool call %d completed: %s with %d arguments",
                                index,
                                tool_call["name"],
                                len(tool_args),
                            )

                        except json.JSONDecodeError as err:
                            _LOGGER.error(
                                "Failed to parse tool arguments JSON for tool %d: %s. "
                                "Raw arguments: %s",
                                index,
                                err,
                                tool_call["arguments"][:100],
                            )
                            # Yield empty tool call to maintain conversation flow
                            tool_inputs.append(
                                llm.ToolInput(
                                    id=tool_call["id"],
                                    tool_name=tool_call["name"],
                                    tool_args={},
                                )
                            )

                    if tool_inputs:
                        yield {"tool_calls": tool_inputs}

                    # Reset tool call tracking
                    self._current_tool_calls.clear()

            # Handle case where stream ends without finish_reason
            # (Some APIs may not send finish_reason in every chunk)
            if self._current_tool_calls:
                _LOGGER.debug(
                    "Stream ended with pending tool calls, finalizing %d tool calls",
                    len(self._current_tool_calls),
                )

                tool_inputs = []
                for index in sorted(self._current_tool_calls.keys()):
                    tool_call = self._current_tool_calls[index]

                    try:
                        tool_args = (
                            json.loads(tool_call["arguments"]) if tool_call["arguments"] else {}
                        )

                        tool_inputs.append(
                            llm.ToolInput(
                                id=tool_call["id"],
                                tool_name=tool_call["name"],
                                tool_args=tool_args,
                            )
                        )

                        _LOGGER.debug(
                            "Tool call %d completed (stream end): %s",
                            index,
                            tool_call["name"],
                        )

                    except json.JSONDecodeError as err:
                        _LOGGER.error(
                            "Failed to parse tool arguments JSON at stream end: %s",
                            err,
                        )
                        tool_inputs.append(
                            llm.ToolInput(
                                id=tool_call["id"],
                                tool_name=tool_call["name"],
                                tool_args={},
                            )
                        )

                if tool_inputs:
                    yield {"tool_calls": tool_inputs}

                self._current_tool_calls.clear()

            # Flush any remaining content in the thinking buffer
            # This handles the case where stream ends with a partial opening tag
            # (e.g., "<th" that might have become "<think>" but never will now)
            if self._thinking_buffer and not self._in_thinking_block:
                _LOGGER.debug(
                    "Flushing thinking buffer at stream end: %s",
                    self._thinking_buffer,
                )
                yield {"content": self._thinking_buffer}
                self._thinking_buffer = ""

            _LOGGER.debug("Stream transformation completed")

        except Exception as err:
            _LOGGER.error(
                "Error during stream transformation: %s",
                err,
                exc_info=True,
            )
            raise

    def get_usage(self) -> dict[str, int] | None:
        """Get token usage statistics from the stream.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens or None
        """
        return self._usage

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Uses a simple heuristic: ~4 characters per token (based on GPT tokenization).
        This is approximate but works reasonably well for most languages.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Simple estimation: 4 chars per token on average
        return max(1, len(text) // 4)

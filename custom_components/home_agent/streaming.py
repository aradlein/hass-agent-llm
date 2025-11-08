"""Streaming response handler for Home Agent (OpenAI/Ollama compatible)."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from homeassistant.components import conversation
from homeassistant.helpers import llm

_LOGGER = logging.getLogger(__name__)


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
            return json.loads(data)
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

                # Extract delta from choices array
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # Capture usage data if present (OpenAI sends this at the end with stream_options)
                if "usage" in chunk:
                    self._usage = chunk["usage"]
                    _LOGGER.debug("Token usage received: %s", self._usage)

                # Handle role (first chunk)
                if "role" in delta:
                    _LOGGER.debug("Stream started with role: %s", delta["role"])

                # Handle text content
                if "content" in delta and delta["content"]:
                    yield {"content": delta["content"]}

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

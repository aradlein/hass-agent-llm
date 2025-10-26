"""Tools for Home Agent integration.

This module provides the tool system for executing Home Assistant operations
and external queries through the LLM conversation interface.
"""

from __future__ import annotations

from .ha_control import HomeAssistantControlTool
from .ha_query import HomeAssistantQueryTool
from .registry import ToolRegistry

__all__ = [
    "ToolRegistry",
    "HomeAssistantControlTool",
    "HomeAssistantQueryTool",
]

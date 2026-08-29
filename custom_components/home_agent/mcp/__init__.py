"""MCP (Model Context Protocol) server support for Home Agent.

This package provides an isolated, self-contained client for remote MCP
servers exposed over HTTP or Server-Sent Events (SSE). It is intentionally
independent from the custom_tools framework and has no runtime dependency on
``custom_components.home_agent.tools.custom``.
"""

from __future__ import annotations

from .manager import McpServerManager
from .tool import MCPTool

__all__ = ["MCPTool", "McpServerManager"]

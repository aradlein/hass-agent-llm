"""Context providers for home_agent.

This package provides different strategies for gathering and formatting
entity context to be injected into LLM prompts.

Available Providers:
    - DirectContextProvider: Directly fetches configured entities
    - (VectorDBContextProvider: Coming in Phase 2)
"""

from .base import ContextProvider
from .direct import DirectContextProvider

__all__ = [
    "ContextProvider",
    "DirectContextProvider",
]

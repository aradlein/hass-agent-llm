"""Context providers for home_agent.

This package provides different strategies for gathering and formatting
entity context to be injected into LLM prompts.

Available Providers:
    - DirectContextProvider: Directly fetches configured entities
    - VectorDBContextProvider: Semantic search using ChromaDB
"""

from .base import ContextProvider
from .direct import DirectContextProvider
from .vector_db import VectorDBContextProvider

__all__ = [
    "ContextProvider",
    "DirectContextProvider",
    "VectorDBContextProvider",
]

"""Mock utilities for Home Agent integration tests.

This package provides reusable mock implementations for external services:
- LLM backends (OpenAI, Ollama, vLLM)
- Embedding services (OpenAI, Ollama)
- ChromaDB vector database

These mocks allow integration tests to run quickly without requiring
actual external services, while still testing the full integration flow.

## What Mocks Test vs What They Don't

Mocks test:
- Agent initialization and configuration
- Tool registration and dispatch
- Message routing and history
- Error handling paths
- Integration between components

Mocks do NOT test:
- Actual LLM intelligence (tool selection, reasoning)
- Real semantic similarity (embedding quality)
- Network reliability
- Real-world latency characteristics

For testing actual LLM/embedding behavior, run with real services
by setting the appropriate environment variables (see conftest.py).
"""

from .embedding_mocks import (
    MockEmbeddingServer,
    create_embedding_response,
    create_similar_embeddings,
    generate_deterministic_embedding,
)
from .fixtures import (
    MockChromaDBClient,
    MockChromaDBCollection,
    create_mock_llm_for_home_agent,
)
from .llm_mocks import (
    RESPONSES,
    MockLLMServer,
    create_chat_completion_response,
    create_multiple_tool_calls_response,
    create_streaming_chunks,
    create_streaming_response,
    create_streaming_tool_call_chunks,
    create_tool_call_response,
    mock_aiohttp_session,
)

__all__ = [
    # LLM mocks
    "MockLLMServer",
    "RESPONSES",
    "create_chat_completion_response",
    "create_multiple_tool_calls_response",
    "create_streaming_chunks",
    "create_streaming_response",
    "create_streaming_tool_call_chunks",
    "create_tool_call_response",
    "mock_aiohttp_session",
    # Embedding mocks
    "MockEmbeddingServer",
    "create_embedding_response",
    "create_similar_embeddings",
    "generate_deterministic_embedding",
    # ChromaDB mocks
    "MockChromaDBClient",
    "MockChromaDBCollection",
    # Pre-configured mocks
    "create_mock_llm_for_home_agent",
]

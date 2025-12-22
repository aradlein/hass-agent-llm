"""Mock embedding server implementations for testing.

This module provides mock implementations for embedding APIs,
supporting both OpenAI and Ollama-compatible embedding endpoints.

The mocks generate deterministic embeddings based on input text,
allowing for consistent testing without requiring actual embedding models.
"""

from __future__ import annotations

import functools
import hashlib
from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch


@functools.lru_cache(maxsize=1000)
def generate_deterministic_embedding(
    text: str,
    dimensions: int = 384,
    seed: int = 42,
) -> tuple[float, ...]:
    """Generate a deterministic embedding vector from text.

    This creates consistent embeddings for the same input text,
    which is useful for testing semantic search behavior.

    Args:
        text: Input text to embed
        dimensions: Number of embedding dimensions (default: 384 for mxbai-embed-large)
        seed: Random seed for reproducibility

    Returns:
        List of floats representing the embedding vector
    """
    # Create a hash of the text for deterministic output
    # Use multiple hashes to ensure we have enough data for all dimensions
    base_hash = hashlib.sha256(f"{text}{seed}".encode()).hexdigest()

    # Generate embedding values from hash
    embedding = []
    for i in range(dimensions):
        # Create a unique hash for each dimension to avoid collisions
        dim_hash = hashlib.md5(f"{base_hash}{i}".encode()).hexdigest()
        # Use first 8 chars of hash as hex value
        value = int(dim_hash[:8], 16) / 0xFFFFFFFF - 0.5  # Normalize to [-0.5, 0.5]
        embedding.append(value)

    # Normalize the vector (L2 normalization)
    magnitude = sum(v * v for v in embedding) ** 0.5
    if magnitude > 0:
        embedding = [v / magnitude for v in embedding]

    return tuple(embedding)


def create_embedding_response(
    texts: list[str] | str,
    model: str = "mock-embedding-model",
    dimensions: int = 384,
) -> dict[str, Any]:
    """Create a mock OpenAI-compatible embedding response.

    Args:
        texts: Single text or list of texts to embed
        model: Model name
        dimensions: Embedding dimensions

    Returns:
        Dict matching OpenAI embedding response format
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings_data = []
    for i, text in enumerate(texts):
        embedding = generate_deterministic_embedding(text, dimensions)
        embeddings_data.append({
            "object": "embedding",
            "index": i,
            "embedding": embedding,
        })

    return {
        "object": "list",
        "data": embeddings_data,
        "model": model,
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    }


def create_ollama_embedding_response(
    texts: list[str] | str,
    model: str = "mxbai-embed-large",
    dimensions: int = 1024,
) -> dict[str, Any]:
    """Create a mock Ollama embedding response.

    Args:
        texts: Single text or list of texts to embed
        model: Model name
        dimensions: Embedding dimensions

    Returns:
        Dict matching Ollama embedding response format
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = [generate_deterministic_embedding(text, dimensions) for text in texts]

    # Ollama returns embeddings directly
    if len(texts) == 1:
        return {
            "model": model,
            "embeddings": embeddings,
        }

    return {
        "model": model,
        "embeddings": embeddings,
    }


class MockEmbeddingServer:
    """Mock embedding server for testing.

    This class provides a flexible way to mock embedding API calls
    with configurable responses and behavior.

    Example:
        server = MockEmbeddingServer(dimensions=384)

        with server.patch_aiohttp():
            # Run your test
            embeddings = await get_embeddings(["hello", "world"])
    """

    def __init__(
        self,
        dimensions: int = 384,
        model: str = "mock-embedding-model",
        provider: str = "openai",
    ):
        """Initialize the mock embedding server.

        Args:
            dimensions: Embedding vector dimensions
            model: Model name to report
            provider: API provider style ("openai" or "ollama")
        """
        self.dimensions = dimensions
        self.model = model
        self.provider = provider
        self.call_history: list[dict[str, Any]] = []
        self._custom_embeddings: dict[str, list[float]] = {}

    def set_embedding(self, text: str, embedding: list[float]) -> "MockEmbeddingServer":
        """Set a custom embedding for specific text.

        Args:
            text: Text that should return this embedding
            embedding: Custom embedding vector

        Returns:
            Self for chaining
        """
        self._custom_embeddings[text] = embedding
        return self

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text (custom or generated).

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if text in self._custom_embeddings:
            return self._custom_embeddings[text]
        return generate_deterministic_embedding(text, self.dimensions)

    def get_response(self, texts: list[str]) -> dict[str, Any]:
        """Get embedding response for texts.

        Args:
            texts: List of texts to embed

        Returns:
            Embedding response dict
        """
        self.call_history.append({"texts": texts})

        if self.provider == "ollama":
            # Ollama returns singular 'embedding' for single text
            if len(texts) == 1:
                return {
                    "model": self.model,
                    "embedding": self.get_embedding(texts[0]),
                }
            else:
                embeddings = [self.get_embedding(text) for text in texts]
                return {
                    "model": self.model,
                    "embeddings": embeddings,
                }
        else:
            # OpenAI format
            embeddings_data = []
            for i, text in enumerate(texts):
                embedding = self.get_embedding(text)
                embeddings_data.append({
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding,
                })

            return {
                "object": "list",
                "data": embeddings_data,
                "model": self.model,
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in texts),
                    "total_tokens": sum(len(t.split()) for t in texts),
                },
            }

    def reset(self) -> None:
        """Reset call history."""
        self.call_history.clear()

    @contextmanager
    def patch_aiohttp(self):
        """Context manager to patch aiohttp with this mock server.

        Yields:
            The mock session for additional assertions
        """
        def response_handler(url: str, **kwargs) -> MagicMock:
            payload = kwargs.get("json", {})

            # Extract texts from different API formats
            texts = []
            if "input" in payload:
                # OpenAI format
                input_data = payload["input"]
                if isinstance(input_data, str):
                    texts = [input_data]
                else:
                    texts = input_data
            elif "prompt" in payload:
                # Ollama format
                prompt = payload["prompt"]
                if isinstance(prompt, str):
                    texts = [prompt]
                else:
                    texts = prompt

            response_data = self.get_response(texts)

            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_data)
            mock_response.raise_for_status = MagicMock()
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            return mock_response

        mock_session = MagicMock()
        # session.post should return the mock_response directly (which is an async context manager)
        mock_session.post = MagicMock(side_effect=response_handler)
        mock_session.closed = False
        # The session itself also needs to be an async context manager
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            yield mock_session


def create_similar_embeddings(
    base_text: str,
    similar_texts: list[str],
    similarity: float = 0.9,
    dimensions: int = 384,
) -> dict[str, list[float]]:
    """Create a set of embeddings where similar_texts are close to base_text.

    This is useful for testing semantic search where you want to control
    which texts are "similar" to each other.

    Args:
        base_text: The reference text
        similar_texts: Texts that should be similar to base_text
        similarity: Target cosine similarity (0-1)
        dimensions: Embedding dimensions

    Returns:
        Dict mapping text to embedding
    """
    import math

    base_embedding = generate_deterministic_embedding(base_text, dimensions)

    embeddings = {base_text: base_embedding}

    for i, text in enumerate(similar_texts):
        # Generate a base random embedding
        random_embedding = generate_deterministic_embedding(text, dimensions)

        # Blend with base embedding to achieve target similarity
        blended = []
        for j in range(dimensions):
            # Linear interpolation towards base embedding
            value = similarity * base_embedding[j] + (1 - similarity) * random_embedding[j]
            blended.append(value)

        # Normalize
        magnitude = sum(v * v for v in blended) ** 0.5
        if magnitude > 0:
            blended = [v / magnitude for v in blended]

        embeddings[text] = blended

    return embeddings


# Pre-configured embedding servers for common use cases
def create_openai_embedding_server(dimensions: int = 1536) -> MockEmbeddingServer:
    """Create a mock server configured for OpenAI embeddings."""
    return MockEmbeddingServer(
        dimensions=dimensions,
        model="text-embedding-ada-002",
        provider="openai",
    )


def create_ollama_embedding_server(dimensions: int = 1024) -> MockEmbeddingServer:
    """Create a mock server configured for Ollama embeddings."""
    return MockEmbeddingServer(
        dimensions=dimensions,
        model="mxbai-embed-large",
        provider="ollama",
    )

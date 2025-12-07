"""Pytest fixtures for mock services.

These fixtures provide mock implementations of external services
(LLM, embedding, ChromaDB) for integration testing when real
services are not available.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .embedding_mocks import MockEmbeddingServer, generate_deterministic_embedding
from .llm_mocks import (
    MockLLMServer,
    create_chat_completion_response,
    create_tool_call_response,
)


class MockChromaDBCollection:
    """Mock ChromaDB collection for testing.

    Provides an in-memory implementation of ChromaDB collection
    operations including add, get, query, update, and delete.
    """

    def __init__(self, name: str, embedding_function=None):
        self.name = name
        self._data: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._embedding_function = embedding_function

    def add(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add documents to the collection."""
        for i, id_ in enumerate(ids):
            self._data[id_] = {
                "id": id_,
                "document": documents[i] if documents else None,
                "metadata": metadatas[i] if metadatas else {},
            }
            if embeddings:
                self._embeddings[id_] = embeddings[i]
            elif documents:
                # Generate embedding from document
                self._embeddings[id_] = generate_deterministic_embedding(
                    documents[i], dimensions=1024
                )

    def upsert(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents (add or update)."""
        self.add(ids, documents, embeddings, metadatas)

    def get(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get documents by ID or filter."""
        if ids:
            result_ids = [id_ for id_ in ids if id_ in self._data]
        else:
            result_ids = list(self._data.keys())

        # Apply where filter if provided
        if where:
            filtered_ids = []
            for id_ in result_ids:
                metadata = self._data[id_].get("metadata", {})
                match = True
                for key, value in where.items():
                    if key.startswith("$"):
                        # Skip operators for now
                        continue
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_ids.append(id_)
            result_ids = filtered_ids

        return {
            "ids": result_ids,
            "documents": [self._data[id_]["document"] for id_ in result_ids],
            "metadatas": [self._data[id_]["metadata"] for id_ in result_ids],
            "embeddings": [self._embeddings.get(id_, []) for id_ in result_ids],
        }

    def query(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """Query the collection by embedding similarity."""
        if query_texts and not query_embeddings:
            query_embeddings = [
                generate_deterministic_embedding(text, dimensions=1024)
                for text in query_texts
            ]

        results: dict[str, list] = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

        for query_emb in query_embeddings or [[]]:
            # Calculate distances to all documents
            distances = []
            for id_, emb in self._embeddings.items():
                # Simple L2 distance calculation
                if len(emb) == len(query_emb):
                    dist = sum((a - b) ** 2 for a, b in zip(query_emb, emb)) ** 0.5
                else:
                    dist = float("inf")
                distances.append((id_, dist))

            # Sort by distance and take top n
            distances.sort(key=lambda x: x[1])
            top_results = distances[:n_results]

            results["ids"].append([id_ for id_, _ in top_results])
            results["documents"].append([self._data[id_]["document"] for id_, _ in top_results])
            results["metadatas"].append([self._data[id_]["metadata"] for id_, _ in top_results])
            results["distances"].append([dist for _, dist in top_results])

        return results

    def update(
        self,
        ids: list[str],
        documents: list[str] | None = None,
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update existing documents."""
        for i, id_ in enumerate(ids):
            if id_ in self._data:
                if documents:
                    self._data[id_]["document"] = documents[i]
                if metadatas:
                    self._data[id_]["metadata"].update(metadatas[i])
                if embeddings:
                    self._embeddings[id_] = embeddings[i]

    def delete(self, ids: list[str] | None = None, where: dict | None = None) -> None:
        """Delete documents from the collection."""
        if ids:
            for id_ in ids:
                self._data.pop(id_, None)
                self._embeddings.pop(id_, None)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return len(self._data)


class MockChromaDBClient:
    """Mock ChromaDB client for testing."""

    def __init__(self):
        self._collections: dict[str, MockChromaDBCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        embedding_function=None,
        metadata: dict | None = None,
    ) -> MockChromaDBCollection:
        """Get or create a collection."""
        if name not in self._collections:
            self._collections[name] = MockChromaDBCollection(name, embedding_function)
        return self._collections[name]

    def get_collection(self, name: str) -> MockChromaDBCollection:
        """Get an existing collection."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def create_collection(
        self,
        name: str,
        embedding_function=None,
        metadata: dict | None = None,
    ) -> MockChromaDBCollection:
        """Create a new collection."""
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")
        self._collections[name] = MockChromaDBCollection(name, embedding_function)
        return self._collections[name]

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        del self._collections[name]

    def list_collections(self) -> list[MockChromaDBCollection]:
        """List all collections."""
        return list(self._collections.values())

    def heartbeat(self) -> int:
        """Return a heartbeat timestamp."""
        return 1234567890


def create_mock_llm_for_home_agent() -> MockLLMServer:
    """Create a MockLLMServer pre-configured for Home Agent testing.

    This server is configured to:
    - Respond to greetings appropriately
    - Trigger ha_control tool for light/switch commands
    - Trigger ha_query tool for status questions
    - Remember context in multi-turn conversations
    - Handle streaming requests

    Returns:
        Configured MockLLMServer
    """
    server = MockLLMServer(
        default_response="I'm your home assistant. How can I help you control your smart home?"
    )

    # Greeting responses
    server.add_response(
        "hello",
        "Hello! I'm your home assistant. I can help you control lights, "
        "check temperatures, and manage your smart home devices. "
        "What would you like me to do?"
    )

    server.add_response(
        "how are you",
        "I'm doing well, thank you for asking! I'm here to help you "
        "with your smart home. Is there anything you'd like me to help with?"
    )

    # Light control - trigger tool calls
    server.add_tool_call_response(
        "turn on the living room light",
        "ha_control",
        {"action": "turn_on", "entity_id": "light.living_room"},
    )

    # Also match variations without "the"
    server.add_tool_call_response(
        lambda text: "turn on" in text.lower() and "living room" in text.lower() and "light" in text.lower(),
        "ha_control",
        {"action": "turn_on", "entity_id": "light.living_room"},
    )

    server.add_tool_call_response(
        "turn off the living room light",
        "ha_control",
        {"action": "turn_off", "entity_id": "light.living_room"},
    )

    server.add_tool_call_response(
        "turn on the bedroom light",
        "ha_control",
        {"action": "turn_on", "entity_id": "light.bedroom"},
    )

    server.add_tool_call_response(
        "turn on the kitchen light",
        "ha_control",
        {"action": "turn_on", "entity_id": "light.kitchen"},
    )

    # Switch control
    server.add_tool_call_response(
        "turn on the coffee maker",
        "ha_control",
        {"action": "turn_on", "entity_id": "switch.coffee_maker"},
    )

    server.add_tool_call_response(
        "turn on the fan",
        "ha_control",
        {"action": "turn_on", "entity_id": "switch.bedroom_fan"},
    )

    # Temperature queries - these should use ha_query
    server.add_response(
        "temperature",
        "The current temperature in the living room is 72.5°F. "
        "The thermostat is set to 72°F in heat mode."
    )

    server.add_response(
        "what is the temp",
        "The living room temperature sensor shows 72.5°F."
    )

    # Multi-turn context - name/color memory
    server.add_response(
        "my name is alice",
        "Nice to meet you, Alice! I'll remember that. "
        "Is there anything you'd like me to help with in your home?"
    )

    server.add_response(
        "what is my name",
        "Your name is Alice, as you mentioned earlier."
    )

    server.add_response(
        "color blue",
        "I've noted that you like the color blue. "
        "I can help adjust lighting to match your preferences if you'd like."
    )

    server.add_response(
        "what color do i like",
        "You mentioned that you like the color blue."
    )

    # Streaming test
    server.add_streaming_response(
        "count from 1 to 5",
        "1, 2, 3, 4, 5. There you go! Is there anything else I can help with?",
        chunk_size=5,
    )

    return server


@pytest.fixture
def mock_llm_server():
    """Provide a pre-configured mock LLM server for testing."""
    return create_mock_llm_for_home_agent()


@pytest.fixture
def mock_embedding_server():
    """Provide a mock embedding server for testing."""
    return MockEmbeddingServer(
        dimensions=1024,  # Match mxbai-embed-large
        model="mxbai-embed-large",
        provider="ollama",
    )


@pytest.fixture
def mock_chromadb_client():
    """Provide a mock ChromaDB client for testing."""
    return MockChromaDBClient()


@pytest.fixture
def mock_chromadb_collection(mock_chromadb_client):
    """Provide a mock ChromaDB collection for testing."""
    name = f"test_collection_{uuid.uuid4().hex[:8]}"
    return mock_chromadb_client.get_or_create_collection(name)

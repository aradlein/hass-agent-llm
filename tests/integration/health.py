"""Health check utilities for integration tests.

This module provides utilities to check if external services (ChromaDB, LLM, embeddings)
are available before running integration tests.
"""

import logging
from typing import Any

import aiohttp
import pytest

_LOGGER = logging.getLogger(__name__)


async def check_chromadb_health(host: str, port: int, timeout: int = 5) -> bool:
    """Check if ChromaDB is reachable and healthy.

    Args:
        host: ChromaDB host
        port: ChromaDB port
        timeout: Timeout in seconds

    Returns:
        True if ChromaDB is healthy, False otherwise
    """
    # Try both v2 (newer) and v1 (legacy) API endpoints
    endpoints = [
        f"http://{host}:{port}/api/v2/heartbeat",  # ChromaDB v2 API
        f"http://{host}:{port}/api/v1/heartbeat",  # ChromaDB v1 API (legacy)
    ]

    try:
        async with aiohttp.ClientSession() as session:
            for url in endpoints:
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            # ChromaDB heartbeat returns a timestamp in nanoseconds
                            is_healthy = "nanosecond heartbeat" in data or isinstance(
                                data.get("nanosecond heartbeat"), int
                            )
                            if is_healthy:
                                _LOGGER.info("ChromaDB health check passed: %s:%s (endpoint: %s)", host, port, url)
                                return True
                except aiohttp.ClientError:
                    continue

            _LOGGER.warning(
                "ChromaDB health check failed: %s:%s (tried all endpoints)", host, port
            )
            return False
    except aiohttp.ClientError as err:
        _LOGGER.warning("ChromaDB health check failed: %s:%s (%s)", host, port, err)
        return False
    except Exception as err:
        _LOGGER.warning(
            "ChromaDB health check failed: %s:%s (unexpected error: %s)", host, port, err
        )
        return False


async def check_llm_health(base_url: str, timeout: int = 5) -> bool:
    """Check if LLM endpoint is reachable and healthy.

    Args:
        base_url: LLM API base URL (e.g., http://localhost:11434)
        timeout: Timeout in seconds

    Returns:
        True if LLM endpoint is healthy, False otherwise
    """
    # Try multiple common health check endpoints
    endpoints = [
        "/api/tags",  # Ollama
        "/v1/models",  # OpenAI-compatible
        "/models",  # Alternative
        "/health",  # Generic health endpoint
        "",  # Root endpoint
    ]

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = f"{base_url.rstrip('/')}{endpoint}"
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status in (200, 404):  # 404 is ok for root endpoint
                        _LOGGER.info("LLM health check passed: %s (endpoint: %s)", base_url, endpoint)
                        return True
            except aiohttp.ClientError:
                continue
            except Exception:
                continue

    _LOGGER.warning("LLM health check failed: %s (tried all endpoints)", base_url)
    return False


async def check_embedding_health(base_url: str, timeout: int = 5) -> bool:
    """Check if embedding endpoint is reachable and healthy.

    Args:
        base_url: Embedding API base URL (e.g., http://localhost:11434)
        timeout: Timeout in seconds

    Returns:
        True if embedding endpoint is healthy, False otherwise
    """
    # For Ollama-style embeddings, check the /api/tags endpoint
    # For OpenAI-style, check /v1/models
    endpoints = [
        "/api/tags",  # Ollama
        "/v1/models",  # OpenAI-compatible
        "/models",  # Alternative
        "",  # Root endpoint
    ]

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = f"{base_url.rstrip('/')}{endpoint}"
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status in (200, 404):
                        _LOGGER.info(
                            "Embedding health check passed: %s (endpoint: %s)", base_url, endpoint
                        )
                        return True
            except aiohttp.ClientError:
                continue
            except Exception:
                continue

    _LOGGER.warning("Embedding health check failed: %s (tried all endpoints)", base_url)
    return False


def skip_if_chromadb_unavailable(host: str, port: int) -> Any:
    """Pytest marker to skip test if ChromaDB is unavailable.

    Usage:
        @skip_if_chromadb_unavailable("localhost", 8000)
        async def test_my_chromadb_feature():
            ...

    Args:
        host: ChromaDB host
        port: ChromaDB port

    Returns:
        Pytest skip marker
    """

    async def check() -> bool:
        """Check if ChromaDB is available."""
        return await check_chromadb_health(host, port)

    import asyncio

    is_available = asyncio.run(check())
    return pytest.mark.skipif(
        not is_available, reason=f"ChromaDB not available at {host}:{port}"
    )


def skip_if_llm_unavailable(base_url: str) -> Any:
    """Pytest marker to skip test if LLM endpoint is unavailable.

    Usage:
        @skip_if_llm_unavailable("http://localhost:11434")
        async def test_my_llm_feature():
            ...

    Args:
        base_url: LLM API base URL

    Returns:
        Pytest skip marker
    """

    async def check() -> bool:
        """Check if LLM is available."""
        return await check_llm_health(base_url)

    import asyncio

    is_available = asyncio.run(check())
    return pytest.mark.skipif(not is_available, reason=f"LLM not available at {base_url}")


def skip_if_embedding_unavailable(base_url: str) -> Any:
    """Pytest marker to skip test if embedding endpoint is unavailable.

    Usage:
        @skip_if_embedding_unavailable("http://localhost:11434")
        async def test_my_embedding_feature():
            ...

    Args:
        base_url: Embedding API base URL

    Returns:
        Pytest skip marker
    """

    async def check() -> bool:
        """Check if embedding endpoint is available."""
        return await check_embedding_health(base_url)

    import asyncio

    is_available = asyncio.run(check())
    return pytest.mark.skipif(not is_available, reason=f"Embedding not available at {base_url}")


async def check_all_services(
    chromadb_host: str,
    chromadb_port: int,
    llm_base_url: str,
    embedding_base_url: str,
) -> dict[str, bool]:
    """Check health of all services.

    Args:
        chromadb_host: ChromaDB host
        chromadb_port: ChromaDB port
        llm_base_url: LLM API base URL
        embedding_base_url: Embedding API base URL

    Returns:
        Dictionary mapping service names to health status
    """
    return {
        "chromadb": await check_chromadb_health(chromadb_host, chromadb_port),
        "llm": await check_llm_health(llm_base_url),
        "embedding": await check_embedding_health(embedding_base_url),
    }

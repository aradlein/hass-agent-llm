"""Integration test fixtures for Home Agent.

This module provides fixtures for integration tests that interact with real services
(ChromaDB, LLM endpoints, etc.) configured via environment variables.
"""

import os
import uuid
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.core import HomeAssistant, State

from .health import (
    check_chromadb_health,
    check_embedding_health,
    check_llm_health,
)

# Default test service endpoints (can be overridden with environment variables)
DEFAULT_TEST_CHROMADB_HOST = "localhost"
DEFAULT_TEST_CHROMADB_PORT = 8000
DEFAULT_TEST_LLM_BASE_URL = "http://localhost:11434"
DEFAULT_TEST_LLM_MODEL = "qwen2.5:3b"
DEFAULT_TEST_EMBEDDING_BASE_URL = "http://localhost:11434"
DEFAULT_TEST_EMBEDDING_MODEL = "mxbai-embed-large"


@pytest.fixture(scope="session")
def chromadb_config() -> dict[str, Any]:
    """Provide ChromaDB connection settings from environment.

    Environment variables:
        TEST_CHROMADB_HOST: ChromaDB host (default: localhost)
        TEST_CHROMADB_PORT: ChromaDB port (default: 8000)

    Returns:
        Dictionary with ChromaDB configuration
    """
    return {
        "host": os.getenv("TEST_CHROMADB_HOST", DEFAULT_TEST_CHROMADB_HOST),
        "port": int(os.getenv("TEST_CHROMADB_PORT", str(DEFAULT_TEST_CHROMADB_PORT))),
    }


@pytest.fixture(scope="session")
def llm_config() -> dict[str, Any]:
    """Provide LLM endpoint settings from environment.

    Environment variables:
        TEST_LLM_BASE_URL: LLM API base URL (default: http://localhost:11434)
        TEST_LLM_API_KEY: LLM API key (optional)
        TEST_LLM_MODEL: LLM model name (default: qwen2.5:3b)

    Returns:
        Dictionary with LLM configuration
    """
    return {
        "base_url": os.getenv("TEST_LLM_BASE_URL", DEFAULT_TEST_LLM_BASE_URL),
        "api_key": os.getenv("TEST_LLM_API_KEY", ""),
        "model": os.getenv("TEST_LLM_MODEL", DEFAULT_TEST_LLM_MODEL),
    }


@pytest.fixture(scope="session")
def embedding_config() -> dict[str, Any]:
    """Provide embedding endpoint settings from environment.

    Environment variables:
        TEST_EMBEDDING_BASE_URL: Embedding API base URL (default: http://localhost:11434)
        TEST_EMBEDDING_API_KEY: Embedding API key (optional)
        TEST_EMBEDDING_MODEL: Embedding model name (default: mxbai-embed-large)

    Returns:
        Dictionary with embedding configuration
    """
    return {
        "base_url": os.getenv("TEST_EMBEDDING_BASE_URL", DEFAULT_TEST_EMBEDDING_BASE_URL),
        "api_key": os.getenv("TEST_EMBEDDING_API_KEY", ""),
        "model": os.getenv("TEST_EMBEDDING_MODEL", DEFAULT_TEST_EMBEDDING_MODEL),
        # Additional keys for VectorDBManager compatibility
        "host": os.getenv("TEST_CHROMADB_HOST", DEFAULT_TEST_CHROMADB_HOST),
        "port": int(os.getenv("TEST_CHROMADB_PORT", str(DEFAULT_TEST_CHROMADB_PORT))),
        "provider": "ollama",  # Default provider for embeddings
    }


@pytest.fixture(scope="session")
async def chromadb_client(chromadb_config: dict[str, Any]) -> AsyncGenerator[Any, None]:
    """Provide real ChromaDB HttpClient with health check.

    Skips tests if ChromaDB is unavailable.

    Args:
        chromadb_config: ChromaDB configuration from chromadb_config fixture

    Yields:
        ChromaDB HttpClient instance

    Raises:
        pytest.skip: If ChromaDB is not available
    """
    host = chromadb_config["host"]
    port = chromadb_config["port"]

    # Check if ChromaDB is available
    is_healthy = await check_chromadb_health(host, port)
    if not is_healthy:
        pytest.skip(f"ChromaDB not available at {host}:{port}")

    # Import ChromaDB (skip if not installed)
    try:
        import chromadb
    except ImportError:
        pytest.skip("ChromaDB not installed")

    # Create client
    client = chromadb.HttpClient(host=host, port=port)

    yield client

    # Cleanup is handled by cleanup_test_collections in individual tests


@pytest.fixture
def test_collection_name() -> str:
    """Generate unique collection name with test_ prefix.

    Returns:
        Unique collection name for testing
    """
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def memory_collection_name() -> str:
    """Generate unique memory collection name for test isolation.

    Returns:
        Unique memory collection name for testing
    """
    return f"test_memories_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def test_collection(chromadb_client: Any, test_collection_name: str) -> AsyncGenerator[Any, None]:
    """Create a test collection and clean it up after test.

    Args:
        chromadb_client: ChromaDB client instance
        test_collection_name: Unique test collection name

    Yields:
        ChromaDB collection instance
    """
    collection = chromadb_client.get_or_create_collection(name=test_collection_name)

    yield collection

    # Cleanup: delete collection after test
    try:
        chromadb_client.delete_collection(name=test_collection_name)
    except Exception:
        pass  # Collection might not exist if test failed early


@pytest.fixture
def mock_hass_integration() -> HomeAssistant:
    """Alias for test_hass for backward compatibility."""
    return _create_test_hass()


@pytest.fixture(autouse=True)
async def cleanup_background_tasks(request):
    """Automatically wait for background tasks after each test to avoid warnings."""
    import asyncio

    # Let the test run
    yield

    # After test completes, wait for any background tasks
    # This prevents "Task was destroyed but it is pending" warnings
    # Get all pending tasks (excluding the current task)
    try:
        current_task = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if not task.done() and task != current_task]
        if pending:
            # Wait up to 0.5 seconds for tasks to complete
            done, still_pending = await asyncio.wait(pending, timeout=0.5)
            # Cancel any remaining tasks
            for task in still_pending:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        # Silently ignore errors during cleanup
                        pass
    except Exception:
        # Ignore cleanup errors
        pass


def _create_test_hass() -> HomeAssistant:
    """Create a test Home Assistant instance."""
    import asyncio
    import tempfile

    hass = MagicMock(spec=HomeAssistant)

    # Mock exposed entities data structure (required by VectorDBManager)
    # All test entities are exposed by default
    mock_exposed_entities = MagicMock()
    mock_exposed_entities.async_should_expose.return_value = True

    hass.data = {
        "homeassistant.exposed_entites": mock_exposed_entities,
    }

    # Add event loop reference (required by some HA helpers)
    try:
        hass.loop = asyncio.get_running_loop()
    except RuntimeError:
        hass.loop = asyncio.new_event_loop()

    # Add state attribute for HA lifecycle checks
    hass.state = MagicMock()

    # Mock states with some realistic entities
    hass.states = MagicMock()
    hass.states.async_entity_ids = MagicMock(
        return_value=[
            "light.living_room",
            "light.bedroom",
            "sensor.temperature",
            "switch.coffee_maker",
            "climate.thermostat",
        ]
    )
    hass.states.async_all = MagicMock(return_value=[])

    # Mock services
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.services.async_services = MagicMock(
        return_value={
            "light": {"turn_on": {}, "turn_off": {}, "toggle": {}},
            "switch": {"turn_on": {}, "turn_off": {}, "toggle": {}},
            "climate": {"set_temperature": {}, "set_hvac_mode": {}},
            "homeassistant": {"turn_on": {}, "turn_off": {}, "toggle": {}},
        }
    )

    # Mock config with proper path method for storage
    temp_dir = tempfile.mkdtemp(prefix="ha_test_")
    hass.config = MagicMock()
    hass.config.config_dir = temp_dir
    hass.config.path = MagicMock(side_effect=lambda *args: "/".join([temp_dir] + list(args)))
    hass.config.location_name = "Test Home"

    # Mock bus for events
    hass.bus = MagicMock()
    # async_fire is sync in HA, not actually async
    hass.bus.async_fire = MagicMock(return_value=None)
    hass.bus.async_listen = MagicMock(return_value=lambda: None)

    # Mock async_add_executor_job - executes the callable immediately
    async def mock_executor_job(func, *args):
        """Execute job immediately in test context."""
        return func(*args) if args else func()

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)

    # Track created tasks for cleanup
    created_tasks = []

    # Mock async_create_task - schedule the coroutine properly
    def mock_create_task(coro, *args, **kwargs):
        """Create a task that properly awaits the coroutine."""
        task = hass.loop.create_task(coro)
        created_tasks.append(task)
        return task

    hass.async_create_task = MagicMock(side_effect=mock_create_task)

    # Store tasks list for cleanup
    hass._test_tasks = created_tasks

    # Mock entity registry to prevent AttributeError
    # This is needed for get_exposed_entities() in agent.py
    mock_entity_registry = MagicMock()
    mock_entity_registry.async_get = MagicMock(return_value=None)

    # Store the mock in hass.data for er.async_get(hass) to find
    from homeassistant.helpers import entity_registry as er
    hass.data[er.DATA_REGISTRY] = mock_entity_registry

    return hass


@pytest.fixture
def test_hass() -> HomeAssistant:
    """Create a test Home Assistant instance with real-ish states.

    This provides a more realistic mock than the basic mock_hass fixture,
    with actual entity states and service registrations for integration testing.

    Returns:
        Mock Home Assistant instance
    """
    import asyncio

    hass = MagicMock(spec=HomeAssistant)

    # Mock exposed entities data structure (required by VectorDBManager)
    mock_exposed_entities = MagicMock()
    mock_exposed_entities.async_should_expose.return_value = True

    hass.data = {
        "homeassistant.exposed_entites": mock_exposed_entities,
    }

    # Add event loop reference (required by HA helpers like Store)
    try:
        hass.loop = asyncio.get_running_loop()
    except RuntimeError:
        hass.loop = asyncio.new_event_loop()

    # Add state attribute for HA lifecycle checks
    hass.state = MagicMock()

    # Mock states with some realistic entities
    hass.states = MagicMock()
    hass.states.async_entity_ids = MagicMock(
        return_value=[
            "light.living_room",
            "light.bedroom",
            "sensor.temperature",
            "switch.coffee_maker",
            "climate.thermostat",
        ]
    )

    # Mock services
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock()
    hass.services.async_services = MagicMock(
        return_value={
            "light": {"turn_on": {}, "turn_off": {}, "toggle": {}},
            "switch": {"turn_on": {}, "turn_off": {}, "toggle": {}},
            "climate": {"set_temperature": {}, "set_hvac_mode": {}},
            "homeassistant": {"turn_on": {}, "turn_off": {}, "toggle": {}},
        }
    )

    # Mock config with proper path method for storage
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="ha_test_")
    hass.config = MagicMock()
    hass.config.config_dir = temp_dir
    hass.config.path = MagicMock(side_effect=lambda *args: "/".join([temp_dir] + list(args)))
    hass.config.location_name = "Test Home"

    # Mock bus for events
    hass.bus = MagicMock()
    # async_fire is sync in HA, not actually async
    hass.bus.async_fire = MagicMock(return_value=None)
    hass.bus.async_listen = MagicMock(return_value=lambda: None)

    # Mock async_add_executor_job - executes the callable immediately
    async def mock_executor_job(func, *args):
        """Execute job immediately in test context."""
        return func(*args) if args else func()

    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor_job)

    # Track created tasks for cleanup
    created_tasks = []

    # Mock async_create_task - schedule the coroutine properly
    def mock_create_task(coro, *args, **kwargs):
        """Create a task that properly awaits the coroutine."""
        task = hass.loop.create_task(coro)
        created_tasks.append(task)
        return task

    hass.async_create_task = MagicMock(side_effect=mock_create_task)

    # Store tasks list for cleanup
    hass._test_tasks = created_tasks

    # Mock entity registry to prevent AttributeError
    # This is needed for get_exposed_entities() in agent.py
    mock_entity_registry = MagicMock()
    mock_entity_registry.async_get = MagicMock(return_value=None)

    # Store the mock in hass.data for er.async_get(hass) to find
    from homeassistant.helpers import entity_registry as er
    hass.data[er.DATA_REGISTRY] = mock_entity_registry

    return hass


@pytest.fixture
def sample_entity_states() -> list[State]:
    """Create sample entity states for testing.

    Returns:
        List of mock Home Assistant State objects
    """
    return [
        State(
            "light.living_room",
            "on",
            {"brightness": 255, "friendly_name": "Living Room Light"},
        ),
        State(
            "light.bedroom",
            "off",
            {"friendly_name": "Bedroom Light"},
        ),
        State(
            "sensor.temperature",
            "72.5",
            {
                "unit_of_measurement": "°F",
                "device_class": "temperature",
                "friendly_name": "Temperature",
            },
        ),
        State(
            "climate.thermostat",
            "heat",
            {
                "temperature": 72,
                "current_temperature": 70,
                "hvac_mode": "heat",
                "friendly_name": "Thermostat",
            },
        ),
        State(
            "switch.coffee_maker",
            "off",
            {"friendly_name": "Coffee Maker"},
        ),
    ]


@pytest.fixture
async def session_manager(test_hass: HomeAssistant):
    """Create a ConversationSessionManager for testing.

    Args:
        test_hass: Test Home Assistant instance

    Returns:
        ConversationSessionManager instance
    """
    from custom_components.home_agent.conversation_session import ConversationSessionManager

    manager = ConversationSessionManager(test_hass)
    await manager.async_load()
    return manager


@pytest.fixture(scope="session")
def integration_test_marker() -> str:
    """Marker to identify integration test runs.

    This can be used to conditionally enable/disable certain behaviors
    during integration testing.

    Returns:
        Marker string
    """
    return "INTEGRATION_TEST_RUN"


@pytest.fixture
async def check_services_health(
    chromadb_config: dict[str, Any],
    llm_config: dict[str, Any],
    embedding_config: dict[str, Any],
) -> dict[str, bool]:
    """Check health of all test services.

    This fixture can be used at the start of integration tests to verify
    that all required services are available.

    Args:
        chromadb_config: ChromaDB configuration
        llm_config: LLM configuration
        embedding_config: Embedding configuration

    Returns:
        Dictionary mapping service names to health status
    """
    health_status = {
        "chromadb": await check_chromadb_health(
            chromadb_config["host"], chromadb_config["port"]
        ),
        "llm": await check_llm_health(llm_config["base_url"]),
        "embedding": await check_embedding_health(embedding_config["base_url"]),
    }
    return health_status


@pytest.fixture(autouse=True)
async def skip_if_services_unavailable(
    request: pytest.FixtureRequest,
    check_services_health: dict[str, bool],
) -> None:
    """Automatically skip tests if required services are unavailable.

    This fixture is autouse, meaning it runs for every test. Tests can specify
    which services they require using markers:

        @pytest.mark.requires_chromadb
        @pytest.mark.requires_llm
        @pytest.mark.requires_embedding

    Args:
        request: Pytest request object
        check_services_health: Health status of all services
    """
    # Check for service requirement markers
    if request.node.get_closest_marker("requires_chromadb"):
        if not check_services_health["chromadb"]:
            pytest.skip("ChromaDB service not available")

    if request.node.get_closest_marker("requires_llm"):
        if not check_services_health["llm"]:
            pytest.skip("LLM service not available")

    if request.node.get_closest_marker("requires_embedding"):
        if not check_services_health["embedding"]:
            pytest.skip("Embedding service not available")


# Register custom pytest markers
def pytest_configure(config: Any) -> None:
    """Register custom markers for integration tests.

    Args:
        config: Pytest config object
    """
    config.addinivalue_line(
        "markers", "requires_chromadb: mark test as requiring ChromaDB service"
    )
    config.addinivalue_line("markers", "requires_llm: mark test as requiring LLM service")
    config.addinivalue_line(
        "markers", "requires_embedding: mark test as requiring embedding service"
    )


def pytest_sessionstart(session: Any) -> None:
    """Enable sockets at session start, after all plugins have configured.

    This runs after pytest_configure from all plugins, so it can override
    the socket blocking set up by pytest-homeassistant-custom-component.

    Args:
        session: Pytest session object
    """
    import pytest_socket

    pytest_socket.enable_socket()


@pytest.fixture(autouse=True)
def enable_socket_for_integration_tests(request: pytest.FixtureRequest):
    """Enable real socket connections for all integration tests.

    This fixture disables socket blocking for integration tests to allow
    real network calls to external services.

    Args:
        request: Pytest request object
    """
    import pytest_socket

    # Re-enable sockets for real network calls
    pytest_socket.enable_socket()

    yield

    # Note: We don't re-disable here since each test gets a fresh fixture state

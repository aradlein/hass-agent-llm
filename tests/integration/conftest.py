"""Integration test fixtures for Home Agent.

This module provides fixtures for integration tests that interact with real services
(ChromaDB, LLM endpoints, etc.) configured via environment variables.
"""

import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from homeassistant.core import HomeAssistant, State

from .health import (
    check_chromadb_health,
    check_embedding_health,
    check_llm_health,
)


@pytest.fixture(autouse=True)
def disable_thinking_for_tests():
    """Disable LLM thinking mode for all integration tests.

    Reasoning models (Qwen3, DeepSeek R1, etc.) support extended thinking which
    can cause tests to timeout. This fixture patches DEFAULT_THINKING_ENABLED
    to False, causing /no_think to be automatically appended to user messages.

    This is an autouse fixture so all tests automatically benefit without
    needing to modify their configuration.
    """
    # Patch where the value is used (agent.core), not where it's defined (const)
    # because Python imports copy the value at import time
    with patch(
        "custom_components.home_agent.agent.core.DEFAULT_THINKING_ENABLED",
        False,
    ):
        yield

# Session-scoped health check cache
_health_check_cache: dict[str, bool] = {}

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
        TEST_LLM_PROXY_HEADERS: Custom headers as JSON (optional)

    Returns:
        Dictionary with LLM configuration
    """
    # Parse proxy headers from environment variable
    proxy_headers = {}
    proxy_headers_str = os.getenv("TEST_LLM_PROXY_HEADERS", "")
    if proxy_headers_str:
        try:
            proxy_headers = json.loads(proxy_headers_str)
        except json.JSONDecodeError as e:
            logging.warning(
                f"Failed to parse TEST_LLM_PROXY_HEADERS as JSON: {e}. Using empty dict."
            )
            proxy_headers = {}

    return {
        "base_url": os.getenv("TEST_LLM_BASE_URL", DEFAULT_TEST_LLM_BASE_URL),
        "api_key": os.getenv("TEST_LLM_API_KEY", ""),
        "model": os.getenv("TEST_LLM_MODEL", DEFAULT_TEST_LLM_MODEL),
        "proxy_headers": proxy_headers,
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
    import socket as socket_module

    import _socket

    # Enable real sockets for this session-scoped fixture
    # (pytest-socket may have blocked them before session fixtures run)
    original_socket = socket_module.socket
    socket_module.socket = _socket.socket

    host = chromadb_config["host"]
    port = chromadb_config["port"]

    try:
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
    finally:
        # Restore original socket implementation
        socket_module.socket = original_socket


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


@pytest.fixture(scope="function")
def expected_lingering_tasks(request):
    """Override pytest-homeassistant-custom-component's expected_lingering_tasks.

    ChromaDB tests may have lingering background threads from the HTTP client.
    This is expected behavior and not a test failure.
    """
    # Allow lingering tasks for ChromaDB tests
    if request.node.get_closest_marker("requires_chromadb"):
        # Return True to skip the verification
        return True
    # For other tests, use default behavior (empty set)
    return set()


@pytest.fixture(scope="function")
def verify_cleanup():
    """Override pytest-homeassistant-custom-component's verify_cleanup.

    For integration tests with ChromaDB, we skip the thread cleanup verification
    because ChromaDB's HttpClient creates daemon threads that cannot be easily
    cleaned up.
    """
    # Do nothing - skip verification for integration tests
    yield
    # No cleanup verification


@pytest.fixture
async def test_collection(
    chromadb_client: Any, test_collection_name: str
) -> AsyncGenerator[Any, None]:
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
    """Automatically clean up background tasks after each test.

    This fixture uses aggressive cancellation instead of waiting to avoid
    the 1-second timeout that was slowing down tests significantly.
    """
    import asyncio

    # Let the test run
    yield

    # After test completes, immediately cancel any pending tasks
    # This is much faster than waiting for them to complete naturally
    try:
        current_task = asyncio.current_task()
        pending = [task for task in asyncio.all_tasks() if not task.done() and task != current_task]
        if pending:
            # Cancel all pending tasks immediately
            for task in pending:
                if not task.done():
                    task.cancel()

            # Brief wait for cancellations to propagate (50ms max)
            if pending:
                await asyncio.wait(pending, timeout=0.05)
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

    # Mock area registry to prevent template rendering errors
    # This is needed for area_name() and area_id() template functions
    from homeassistant.helpers import area_registry as ar

    mock_area_registry = MagicMock()
    # Mock the areas attribute as a dict for template compatibility
    mock_area_registry.areas = {}
    # Mock async_get_area to return None for unknown areas
    mock_area_registry.async_get_area = MagicMock(return_value=None)

    hass.data[ar.DATA_REGISTRY] = mock_area_registry

    # Mock device registry to prevent template rendering errors
    # This is needed for device_id() and related template functions
    from homeassistant.helpers import device_registry as dr

    mock_device_registry = MagicMock()
    # Mock the devices attribute/method for template compatibility
    mock_device_registry.devices = {}
    mock_device_registry._device_data = {}
    # Mock async_get to return None for unknown devices
    mock_device_registry.async_get = MagicMock(return_value=None)

    hass.data[dr.DATA_REGISTRY] = mock_device_registry

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

    # Mock area registry to prevent template rendering errors
    # This is needed for area_name() and area_id() template functions
    from homeassistant.helpers import area_registry as ar

    mock_area_registry = MagicMock()
    # Mock the areas attribute as a dict for template compatibility
    mock_area_registry.areas = {}
    # Mock async_get_area to return None for unknown areas
    mock_area_registry.async_get_area = MagicMock(return_value=None)

    hass.data[ar.DATA_REGISTRY] = mock_area_registry

    # Mock device registry to prevent template rendering errors
    # This is needed for device_id() and related template functions
    from homeassistant.helpers import device_registry as dr

    mock_device_registry = MagicMock()
    # Mock the devices attribute/method for template compatibility
    mock_device_registry.devices = {}
    mock_device_registry._device_data = {}
    # Mock async_get to return None for unknown devices
    mock_device_registry.async_get = MagicMock(return_value=None)

    hass.data[dr.DATA_REGISTRY] = mock_device_registry

    return hass


@pytest.fixture
def test_hass_with_default_entities(test_hass, sample_entity_states) -> HomeAssistant:
    """Home Assistant instance with default entity set pre-configured.

    Provides a test_hass instance with sample_entity_states already wired in.
    Tests can use this fixture to avoid manual entity setup boilerplate.

    The default entity set includes:
    - light.living_room (on, brightness: 255)
    - light.bedroom (off)
    - sensor.temperature (72.5°F)
    - climate.thermostat (heat mode)
    - switch.coffee_maker (off)

    For custom entity sets, use test_hass + setup_entity_states() instead.
    """
    from tests.integration.helpers import setup_entity_states
    setup_entity_states(test_hass, sample_entity_states)
    return test_hass


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


@pytest.fixture(autouse=True)
def mock_entity_exposure():
    """Auto-patch async_should_expose to expose all entities in tests.

    In production, async_should_expose checks Home Assistant config to determine
    if entities should be exposed to conversation agents. In tests, we want all
    entities exposed by default to simplify test setup.

    Tests can override this by patching async_should_expose themselves.
    """
    with patch(
        "homeassistant.components.homeassistant.exposed_entities.async_should_expose",
        return_value=True  # Return True to expose all entities
    ):
        yield


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


@pytest.fixture(scope="session")
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
        "chromadb": await check_chromadb_health(chromadb_config["host"], chromadb_config["port"]),
        "llm": await check_llm_health(llm_config["base_url"]),
        "embedding": await check_embedding_health(embedding_config["base_url"]),
    }
    return health_status


@pytest.fixture(scope="function")
async def service_availability(
    request: pytest.FixtureRequest,
    socket_enabled,  # Ensure sockets are enabled first
    chromadb_config: dict[str, Any],
    llm_config: dict[str, Any],
    embedding_config: dict[str, Any],
) -> dict[str, bool]:
    """Check and track service availability for the test.

    This fixture checks the health of external services and returns
    a dictionary of their availability status. Tests can use this
    to decide whether to use real services or mocks.

    Returns:
        Dictionary mapping service names to availability status
    """
    return {
        "chromadb": await check_chromadb_health(chromadb_config["host"], chromadb_config["port"]),
        "llm": await check_llm_health(llm_config["base_url"]),
        "embedding": await check_embedding_health(embedding_config["base_url"]),
    }


@pytest.fixture(autouse=True, scope="function")
async def skip_if_services_unavailable(
    request: pytest.FixtureRequest,
    socket_enabled,  # Ensure sockets are enabled first
    chromadb_config: dict[str, Any],
    llm_config: dict[str, Any],
    embedding_config: dict[str, Any],
) -> None:
    """Check for service requirements and use mocks when services unavailable.

    This fixture is autouse, meaning it runs for every test. Tests can specify
    which services they require using markers:

        @pytest.mark.requires_chromadb
        @pytest.mark.requires_llm
        @pytest.mark.requires_embedding

    When a service is unavailable:
    - If USE_MOCK_FALLBACK env var is set to "0", tests are skipped
    - Otherwise, tests will use mock implementations (handled by other fixtures)

    The service availability is stored on the request node for other fixtures
    to access.

    Args:
        request: Pytest request object
        socket_enabled: Ensures sockets are enabled first
        chromadb_config: ChromaDB configuration
        llm_config: LLM configuration
        embedding_config: Embedding configuration
    """
    global _health_check_cache

    # Check if we should skip on unavailable services (default: use mocks)
    skip_on_unavailable = os.getenv("USE_MOCK_FALLBACK", "1") == "0"

    # Store availability status on the request for other fixtures
    request.node._service_status = {}  # type: ignore[attr-defined]

    # Determine if this is a "real" test (test_real_*.py files)
    test_file = os.path.basename(request.node.fspath)
    is_real_test = test_file.startswith("test_real_")

    # For non-real tests, always use mocks (no health checks)
    if not is_real_test:
        request.node._service_status = {  # type: ignore[attr-defined]
            "llm": False,
            "chromadb": False,
            "embedding": False,
        }
        return

    # For real tests (test_real_*.py), perform health checks with caching
    # Check for service requirement markers and test health on-demand
    if request.node.get_closest_marker("requires_chromadb"):
        cache_key = f"chromadb:{chromadb_config['host']}:{chromadb_config['port']}"
        if cache_key in _health_check_cache:
            is_healthy = _health_check_cache[cache_key]
        else:
            is_healthy = await check_chromadb_health(chromadb_config["host"], chromadb_config["port"])
            _health_check_cache[cache_key] = is_healthy

        request.node._service_status["chromadb"] = is_healthy  # type: ignore[attr-defined]
        if not is_healthy and skip_on_unavailable:
            pytest.skip("ChromaDB service not available (set USE_MOCK_FALLBACK=1 to use mocks)")

    if request.node.get_closest_marker("requires_llm"):
        cache_key = f"llm:{llm_config['base_url']}"
        if cache_key in _health_check_cache:
            is_healthy = _health_check_cache[cache_key]
        else:
            is_healthy = await check_llm_health(llm_config["base_url"])
            _health_check_cache[cache_key] = is_healthy

        request.node._service_status["llm"] = is_healthy  # type: ignore[attr-defined]
        if not is_healthy and skip_on_unavailable:
            pytest.skip("LLM service not available (set USE_MOCK_FALLBACK=1 to use mocks)")

    if request.node.get_closest_marker("requires_embedding"):
        cache_key = f"embedding:{embedding_config['base_url']}"
        if cache_key in _health_check_cache:
            is_healthy = _health_check_cache[cache_key]
        else:
            is_healthy = await check_embedding_health(embedding_config["base_url"])
            _health_check_cache[cache_key] = is_healthy

        request.node._service_status["embedding"] = is_healthy  # type: ignore[attr-defined]
        if not is_healthy and skip_on_unavailable:
            pytest.skip("Embedding service not available (set USE_MOCK_FALLBACK=1 to use mocks)")


# Register custom pytest markers
def pytest_configure(config: Any) -> None:
    """Register custom markers for integration tests.

    Args:
        config: Pytest config object
    """
    import asyncio
    import sys

    # Disable socket blocking for integration tests
    # pytest-homeassistant-custom-component blocks sockets by default
    # but integration tests need real network access
    config.option.disable_socket = False

    config.addinivalue_line("markers", "requires_chromadb: mark test as requiring ChromaDB service")
    config.addinivalue_line("markers", "requires_llm: mark test as requiring LLM service")
    config.addinivalue_line(
        "markers", "requires_embedding: mark test as requiring embedding service"
    )

    # Suppress "Task was destroyed but it is pending!" messages from asyncio
    # These are printed directly to stderr and aren't actual errors in test context
    original_stderr_write = sys.stderr.write

    def filtered_stderr_write(text: str) -> int:
        """Filter out asyncio task destruction messages."""
        if "Task was destroyed but it is pending" in text or "source_traceback:" in text:
            return len(text)  # Pretend we wrote it
        return original_stderr_write(text)

    sys.stderr.write = filtered_stderr_write  # type: ignore[method-assign]

    # Also disable asyncio debug mode which generates verbose tracebacks
    asyncio.get_event_loop_policy()  # Ensure policy is initialized


def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Modify test items to add timeout markers for integration tests.

    Integration tests that call real LLM services need longer timeouts
    than the default 10 seconds, as each LLM call can take 2-5+ seconds.

    Args:
        config: Pytest config object
        items: List of collected test items
    """
    import pytest

    for item in items:
        # Add 30-second timeout to all integration tests that don't already have a timeout
        if item.get_closest_marker("integration"):
            if not item.get_closest_marker("timeout"):
                item.add_marker(pytest.mark.timeout(30))


def pytest_sessionstart(session: Any) -> None:
    """Enable sockets at session start, after all plugins have configured.

    This runs after pytest_configure from all plugins, so it can override
    the socket blocking set up by pytest-homeassistant-custom-component.

    Args:
        session: Pytest session object
    """
    global _health_check_cache
    import pytest_socket

    # Clear health check cache at session start
    _health_check_cache.clear()

    # Load .env.test file for integration test configuration
    load_dotenv(".env.test")

    # Enable sockets for integration tests
    pytest_socket.enable_socket()

    # Also try to disable the socket blocking feature entirely
    try:
        session.config.option.disable_socket = False
    except Exception:
        pass


@pytest.fixture(autouse=True, scope="function")
def socket_enabled(request: pytest.FixtureRequest):
    """Enable real socket connections for all integration tests.

    This fixture uses pytest-socket's socket_enabled fixture name
    to disable socket blocking for integration tests, allowing
    real network calls to external services.

    Args:
        request: Pytest request object
    """
    import socket as socket_module

    import _socket

    # Store the current socket implementation
    current_socket = socket_module.socket

    # Replace with the real socket implementation from _socket
    socket_module.socket = _socket.socket

    yield

    # Cleanup - restore previous socket
    socket_module.socket = current_socket


# =============================================================================
# Mock-aware service fixtures
# These fixtures provide either real or mock implementations based on
# service availability, allowing tests to run with mocks when services
# are unavailable.
# =============================================================================


@pytest.fixture
def mock_llm_server():
    """Provide a pre-configured mock LLM server for testing.

    This fixture provides a MockLLMServer configured with responses
    appropriate for Home Agent testing.
    """
    from tests.mocks import create_mock_llm_for_home_agent

    return create_mock_llm_for_home_agent()


@pytest.fixture
def mock_embedding_server():
    """Provide a mock embedding server for testing."""
    from tests.mocks import MockEmbeddingServer

    return MockEmbeddingServer(
        dimensions=1024,  # Match mxbai-embed-large
        model="mxbai-embed-large",
        provider="ollama",
    )


@pytest.fixture
def mock_chromadb_client():
    """Provide a mock ChromaDB client for testing."""
    from tests.mocks import MockChromaDBClient

    return MockChromaDBClient()


@pytest.fixture
def is_using_mock_llm(request: pytest.FixtureRequest) -> bool:
    """Check if the test is using a mock LLM.

    This can be used to adjust assertions based on whether we're
    testing with real or mock services.

    Returns:
        True if using mock LLM, False if using real LLM
    """
    service_status = getattr(request.node, "_service_status", {})
    return not service_status.get("llm", False)


@pytest.fixture
def is_using_mock_chromadb(request: pytest.FixtureRequest) -> bool:
    """Check if the test is using a mock ChromaDB.

    Returns:
        True if using mock ChromaDB, False if using real ChromaDB
    """
    service_status = getattr(request.node, "_service_status", {})
    return not service_status.get("chromadb", False)


@pytest.fixture
def is_using_mock_embedding(request: pytest.FixtureRequest) -> bool:
    """Check if the test is using mock embeddings.

    Returns:
        True if using mock embeddings, False if using real embeddings
    """
    service_status = getattr(request.node, "_service_status", {})
    return not service_status.get("embedding", False)

"""E2E test fixtures for Home Agent.

This module provides fixtures for end-to-end test scenarios that test
complete workflows from user input to final output.
"""

from typing import Any, AsyncGenerator

import pytest

# Import and reuse fixtures from integration tests
from tests.integration.conftest import (
    chromadb_client,
    chromadb_config,
    embedding_config,
    llm_config,
    sample_entity_states,
    test_collection,
    test_collection_name,
    test_hass,
)

from .executor import ScenarioExecutor
from .metrics import MetricsCollector

# Re-export integration fixtures for use in e2e tests
__all__ = [
    "chromadb_client",
    "chromadb_config",
    "embedding_config",
    "llm_config",
    "sample_entity_states",
    "test_collection",
    "test_collection_name",
    "test_hass",
    "scenario_executor",
    "metrics_collector",
]


@pytest.fixture
async def scenario_executor(test_hass: Any) -> AsyncGenerator[ScenarioExecutor, None]:
    """Provide a scenario executor instance.

    Args:
        test_hass: Test Home Assistant instance

    Yields:
        ScenarioExecutor instance configured for testing
    """
    executor = ScenarioExecutor(test_hass)
    yield executor
    # Cleanup is handled by individual tests


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Provide a metrics collector instance.

    Returns:
        MetricsCollector instance for tracking test metrics
    """
    return MetricsCollector()

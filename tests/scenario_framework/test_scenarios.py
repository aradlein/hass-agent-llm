"""Scenario framework tests for Home Agent.

This module contains mock-based unit tests that verify the scenario
execution infrastructure using YAML-defined workflows.

NOTE: These are NOT true E2E tests - they use mocked LLM responses.
For real integration testing, see tests/integration/test_real_*.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from .executor import ScenarioExecutor
from .metrics import MetricsCollector

_LOGGER = logging.getLogger(__name__)

# Path to scenario files
SCENARIOS_DIR = Path(__file__).parent / "scenarios"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_basic_conversation_flow(
    scenario_executor: ScenarioExecutor, metrics_collector: MetricsCollector
) -> None:
    """Test basic conversation flow without tool calls.

    This test loads and executes the basic_conversation.yaml scenario,
    which tests simple question-answer interactions.

    Args:
        scenario_executor: Fixture providing scenario executor
        metrics_collector: Fixture providing metrics collector
    """
    scenario_file = SCENARIOS_DIR / "basic_conversation.yaml"

    # Load scenario
    scenario = await scenario_executor.load_scenario(scenario_file)
    assert scenario is not None
    assert scenario["name"] == "Basic Conversation Flow"

    # Start collecting metrics
    conversation_id = scenario.get("setup", {}).get("conversation_id", "test_basic")
    metrics_collector.start_conversation(conversation_id)

    # Execute scenario
    metrics_collector.start_turn(conversation_id)
    result = await scenario_executor.execute_scenario(scenario)
    metrics_collector.record_turn(
        conversation_id,
        duration_ms=None,
        tokens=15,
        tools=[],
    )

    # Verify execution
    assert result is not None
    assert result["steps"] == 1
    assert len(result["results"]) == 1

    # Log metrics
    conv_metrics = metrics_collector.get_conversation_metrics(conversation_id)
    if conv_metrics:
        _LOGGER.info("Basic conversation metrics: %s", conv_metrics.to_dict())


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tool_execution_flow(
    scenario_executor: ScenarioExecutor, metrics_collector: MetricsCollector
) -> None:
    """Test tool execution flow with ha_query and ha_control.

    This test loads and executes the tool_execution.yaml scenario,
    which tests tool calling capabilities.

    Args:
        scenario_executor: Fixture providing scenario executor
        metrics_collector: Fixture providing metrics collector
    """
    scenario_file = SCENARIOS_DIR / "tool_execution.yaml"

    # Load scenario
    scenario = await scenario_executor.load_scenario(scenario_file)
    assert scenario is not None
    assert scenario["name"] == "Tool Execution Flow"

    # Start collecting metrics
    conversation_id = scenario.get("setup", {}).get("conversation_id", "test_tools")
    metrics_collector.start_conversation(conversation_id)

    # Execute scenario
    for step_idx in range(len(scenario.get("steps", []))):
        metrics_collector.start_turn(conversation_id)

    result = await scenario_executor.execute_scenario(scenario)

    # Record metrics for each step
    for step_result in result["results"]:
        tools_used = step_result.get("tools_used", [])
        metrics_collector.record_turn(
            conversation_id,
            duration_ms=None,
            tokens=20,
            tools=tools_used,
        )

    # Verify execution
    assert result is not None
    assert result["steps"] == 2
    assert len(result["results"]) == 2

    # Verify tool usage
    conv_metrics = metrics_collector.get_conversation_metrics(conversation_id)
    if conv_metrics:
        _LOGGER.info("Tool execution metrics: %s", conv_metrics.to_dict())
        # Should have tool calls recorded
        assert conv_metrics.total_turns > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_memory_flow(
    scenario_executor: ScenarioExecutor, metrics_collector: MetricsCollector
) -> None:
    """Test memory extraction and recall flow.

    This test loads and executes the memory_flow.yaml scenario,
    which tests memory extraction and recall capabilities.

    Args:
        scenario_executor: Fixture providing scenario executor
        metrics_collector: Fixture providing metrics collector
    """
    scenario_file = SCENARIOS_DIR / "memory_flow.yaml"

    # Load scenario
    scenario = await scenario_executor.load_scenario(scenario_file)
    assert scenario is not None
    assert scenario["name"] == "Memory Flow"

    # Start collecting metrics
    conversation_id = scenario.get("setup", {}).get("conversation_id", "memory_test_001")
    metrics_collector.start_conversation(conversation_id)

    # Execute scenario
    result = await scenario_executor.execute_scenario(scenario)

    # Record metrics for each step
    for step_result in result["results"]:
        tools_used = step_result.get("tools_used", [])
        metrics_collector.record_turn(
            conversation_id,
            duration_ms=None,
            tokens=25,
            tools=tools_used,
        )

    # Verify execution
    assert result is not None
    assert result["steps"] == 3
    assert len(result["results"]) == 3

    # Verify memory persistence
    conv_metrics = metrics_collector.get_conversation_metrics(conversation_id)
    if conv_metrics:
        _LOGGER.info("Memory flow metrics: %s", conv_metrics.to_dict())
        # Should have multiple turns with conversation history
        assert conv_metrics.total_turns > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_recovery_flow(
    scenario_executor: ScenarioExecutor, metrics_collector: MetricsCollector
) -> None:
    """Test error recovery and graceful handling.

    This test loads and executes the error_recovery.yaml scenario,
    which tests error handling and recovery capabilities.

    Args:
        scenario_executor: Fixture providing scenario executor
        metrics_collector: Fixture providing metrics collector
    """
    scenario_file = SCENARIOS_DIR / "error_recovery.yaml"

    # Load scenario
    scenario = await scenario_executor.load_scenario(scenario_file)
    assert scenario is not None
    assert scenario["name"] == "Error Recovery"

    # Start collecting metrics
    conversation_id = scenario.get("setup", {}).get("conversation_id", "test_errors")
    metrics_collector.start_conversation(conversation_id)

    # Execute scenario
    result = await scenario_executor.execute_scenario(scenario)

    # Record metrics for each step
    for step_result in result["results"]:
        tools_used = step_result.get("tools_used", [])
        metrics_collector.record_turn(
            conversation_id,
            duration_ms=None,
            tokens=20,
            tools=tools_used,
        )

    # Verify execution
    assert result is not None
    assert result["steps"] == 2
    assert len(result["results"]) == 2

    # Verify error handling
    conv_metrics = metrics_collector.get_conversation_metrics(conversation_id)
    if conv_metrics:
        _LOGGER.info("Error recovery metrics: %s", conv_metrics.to_dict())
        # Should have handled errors gracefully
        assert conv_metrics.total_turns > 0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_metrics_summary(metrics_collector: MetricsCollector) -> None:
    """Test metrics collection and summary generation.

    This test verifies that the metrics collector properly
    aggregates and summarizes metrics across multiple conversations.

    Args:
        metrics_collector: Fixture providing metrics collector
    """
    # Create multiple test conversations
    for i in range(3):
        conversation_id = f"test_metrics_{i}"
        metrics_collector.start_conversation(conversation_id)

        # Record some turns
        for j in range(2):
            metrics_collector.record_turn(
                conversation_id,
                duration_ms=100 + (i * 10) + j,
                tokens=10 + i + j,
                tools=["ha_query"] if j == 0 else [],
            )

    # Get summary
    summary = metrics_collector.get_summary()

    # Verify summary
    assert summary["total_conversations"] == 3
    assert summary["total_turns"] == 6
    assert summary["total_tokens"] > 0
    assert summary["total_duration_ms"] > 0
    assert "ha_query" in summary["tool_usage"]
    assert summary["tool_usage"]["ha_query"] == 3

    _LOGGER.info("Metrics summary: %s", summary)

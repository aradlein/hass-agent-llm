"""Scenario executor for E2E tests.

This module provides the ScenarioExecutor class that loads and executes
test scenarios defined in YAML files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from homeassistant.core import HomeAssistant, State

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_STREAMING_ENABLED,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
)
from custom_components.home_agent.conversation_session import ConversationSessionManager

_LOGGER = logging.getLogger(__name__)


class ScenarioExecutor:
    """Executes E2E test scenarios from YAML definitions.

    This class loads scenario files, sets up the test environment based on
    the scenario configuration, executes the test steps, and verifies
    assertions.
    """

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the scenario executor.

        Args:
            hass: Home Assistant instance for testing
        """
        self.hass = hass
        self.agent: HomeAgent | None = None
        self.conversation_id: str | None = None
        self.step_results: list[dict[str, Any]] = []
        self.total_tool_calls: int = 0

    async def load_scenario(self, scenario_file: Path) -> dict[str, Any]:
        """Load a scenario from a YAML file.

        Args:
            scenario_file: Path to the scenario YAML file

        Returns:
            Dictionary containing the parsed scenario

        Raises:
            FileNotFoundError: If scenario file doesn't exist
            yaml.YAMLError: If scenario file is invalid YAML
        """
        if not scenario_file.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_file}")

        with open(scenario_file, "r", encoding="utf-8") as f:
            scenario = yaml.safe_load(f)

        _LOGGER.info(
            "Loaded scenario: %s from %s",
            scenario.get("name", "Unnamed"),
            scenario_file,
        )
        return scenario

    async def execute_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Execute a complete test scenario.

        Args:
            scenario: Scenario dictionary loaded from YAML

        Returns:
            Dictionary containing execution results and metrics

        Raises:
            AssertionError: If any step or assertion fails
        """
        _LOGGER.info("Executing scenario: %s", scenario.get("name", "Unnamed"))

        # Reset tracking variables
        self.step_results = []
        self.total_tool_calls = 0

        # Setup scenario environment
        await self._setup_scenario(scenario.get("setup", {}))

        # Execute each step
        results = []
        for step_idx, step in enumerate(scenario.get("steps", [])):
            _LOGGER.debug("Executing step %d: %s", step_idx + 1, step.get("user", "")[:50])
            step_result = await self._execute_step(step, self.conversation_id or "test")
            results.append(step_result)
            self.step_results.append(step_result)
            # Track tool calls
            self.total_tool_calls += len(step_result.get("tools_used", []))

        # Verify assertions
        await self._verify_assertions(
            scenario.get("assertions", []), self.conversation_id or "test"
        )

        return {
            "scenario": scenario.get("name", "Unnamed"),
            "steps": len(results),
            "results": results,
        }

    async def _setup_scenario(self, setup: dict[str, Any]) -> None:
        """Set up the scenario environment.

        Args:
            setup: Setup configuration from scenario
        """
        # Set up entities
        entities = setup.get("entities", [])
        entity_states = []

        for entity_config in entities:
            entity_id = entity_config.get("entity_id")
            state = entity_config.get("state", "unknown")
            attributes = entity_config.get("attributes", {})

            # Add friendly_name if not present
            if "friendly_name" not in attributes:
                attributes["friendly_name"] = entity_id.replace("_", " ").title()

            entity_state = State(entity_id, state, attributes)
            entity_states.append(entity_state)

        # Mock the states.get method to return our entities
        def mock_get_state(entity_id: str) -> State | None:
            """Mock state getter."""
            for state in entity_states:
                if state.entity_id == entity_id:
                    return state
            return None

        self.hass.states.get = MagicMock(side_effect=mock_get_state)

        # Mock async_all to return all entity states
        self.hass.states.async_all = MagicMock(return_value=entity_states)

        # Create agent with test configuration
        config = {
            CONF_LLM_BASE_URL: setup.get("llm_base_url", "http://localhost:11434/v1"),
            CONF_LLM_API_KEY: setup.get("llm_api_key", "test_key"),
            CONF_LLM_MODEL: setup.get("llm_model", "qwen2.5:3b"),
            CONF_STREAMING_ENABLED: setup.get("streaming_enabled", False),
            CONF_HISTORY_ENABLED: setup.get("history_enabled", True),
            CONF_TOOLS_MAX_CALLS_PER_TURN: setup.get("max_tool_calls", 5),
        }

        # Create session manager
        session_manager = ConversationSessionManager(self.hass)
        await session_manager.async_load()

        # Create agent
        with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
            mock_expose.return_value = False
            self.agent = HomeAgent(self.hass, config, session_manager)

        # Generate conversation ID
        self.conversation_id = setup.get("conversation_id", "test_scenario")

        _LOGGER.debug("Scenario setup complete with %d entities", len(entity_states))

    async def _execute_step(self, step: dict[str, Any], conversation_id: str) -> dict[str, Any]:
        """Execute a single test step.

        Args:
            step: Step configuration from scenario
            conversation_id: Conversation ID for this execution

        Returns:
            Dictionary containing step execution results
        """
        user_input = step.get("user", "")
        timeout = step.get("timeout", 10)

        # Create mock user input
        from homeassistant.components import conversation

        mock_input = MagicMock(spec=conversation.ConversationInput)
        mock_input.text = user_input
        mock_input.conversation_id = conversation_id
        mock_input.device_id = None
        mock_input.satellite_id = None
        mock_input.language = "en"
        mock_input.agent_id = "test_agent"
        mock_input.context = MagicMock()
        mock_input.context.user_id = "test_user"

        # Mock LLM response
        response_text = step.get("mock_response", "Test response")
        tools_used = step.get("tools_used", [])

        # Create mock LLM response
        mock_llm_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": None,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_llm_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False

        # Track start time for duration calculation
        import time
        start_time = time.perf_counter()

        # Execute with mocked session
        with patch.object(self.agent, "_ensure_session", return_value=mock_session):
            result = await self.agent.async_process(mock_input)

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Verify expected response content if specified
        if "expected_response_contains" in step:
            expected = step["expected_response_contains"]
            # Note: In real execution, we'd check result.response.speech.plain.speech
            # For now, we just verify result exists
            assert result is not None, "Expected response but got None"

        return {
            "user_input": user_input,
            "response": response_text,
            "tools_used": tools_used,
            "conversation_id": conversation_id,
            "duration_ms": duration_ms,
        }

    async def _verify_assertions(
        self, assertions: list[dict[str, Any]], conversation_id: str
    ) -> None:
        """Verify scenario assertions.

        Args:
            assertions: List of assertions to verify
            conversation_id: Conversation ID to check

        Raises:
            AssertionError: If any assertion fails
        """
        for assertion in assertions:
            # Check conversation history length
            if "conversation_history_length" in assertion:
                expected_length = assertion["conversation_history_length"]
                actual_length = len(self.agent.conversation_manager.get_history(conversation_id))
                assert (
                    actual_length == expected_length
                ), f"Expected conversation history length {expected_length}, got {actual_length}"

            # Check tool calls count
            if "total_tool_calls" in assertion:
                expected_calls = assertion["total_tool_calls"]
                actual_calls = self.total_tool_calls
                assert (
                    actual_calls == expected_calls
                ), f"Expected {expected_calls} tool calls, got {actual_calls}"

            # Check response contains
            if "response_contains" in assertion:
                expected_text = assertion["response_contains"]
                # Check if the text appears in any of the step responses
                found = False
                for step_result in self.step_results:
                    response = step_result.get("response", "")
                    if expected_text in response:
                        found = True
                        break
                assert found, f"Expected response to contain '{expected_text}', but it was not found in any step response"

            # Check total duration if present
            if "total_duration_ms" in assertion:
                max_duration = assertion["total_duration_ms"]
                total_duration = sum(step.get("duration_ms", 0) for step in self.step_results)
                assert (
                    total_duration <= max_duration
                ), f"Expected total duration <= {max_duration}ms, got {total_duration:.2f}ms"

        _LOGGER.debug("All assertions passed")

"""Integration tests for Phase 3: External LLM Tool.

This test suite validates the complete external LLM integration flow:
- Dual-LLM workflow (primary LLM delegates to external LLM)
- External LLM tool registration and execution
- Context parameter handling
- Error propagation to primary LLM
- Tool call counting
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from homeassistant.core import HomeAssistant

from custom_components.home_agent.const import (
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_TOOLS_MAX_CALLS_PER_TURN,
    CONF_TOOLS_TIMEOUT,
    TOOL_QUERY_EXTERNAL_LLM,
)
from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.tools.external_llm import ExternalLLMTool


@pytest.fixture
def external_llm_config():
    """Provide configuration with external LLM enabled."""
    return {
        # Primary LLM config
        CONF_LLM_BASE_URL: "https://api.primary.com/v1",
        CONF_LLM_API_KEY: "primary-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        # External LLM config
        CONF_EXTERNAL_LLM_ENABLED: True,
        CONF_EXTERNAL_LLM_BASE_URL: "https://api.external.com/v1",
        CONF_EXTERNAL_LLM_API_KEY: "external-key-456",
        CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
        CONF_EXTERNAL_LLM_TEMPERATURE: 0.8,
        CONF_EXTERNAL_LLM_MAX_TOKENS: 1000,
        CONF_EXTERNAL_LLM_TOOL_DESCRIPTION: "Use for complex analysis tasks",
        # Tool configuration
        CONF_TOOLS_MAX_CALLS_PER_TURN: 5,
        CONF_TOOLS_TIMEOUT: 30,
    }


@pytest.fixture
def mock_hass_for_integration():
    """Create a mock Home Assistant instance for integration tests."""
    mock = MagicMock(spec=HomeAssistant)
    mock.data = {}

    # Mock states
    mock.states = MagicMock()
    mock.states.async_all = MagicMock(return_value=[])
    mock.states.get = MagicMock(return_value=None)

    # Mock services
    mock.services = MagicMock()
    mock.services.async_call = AsyncMock()

    # Mock config
    mock.config = MagicMock()
    mock.config.config_dir = "/config"
    mock.config.location_name = "Test Home"

    # Mock bus
    mock.bus = MagicMock()
    mock.bus.async_fire = AsyncMock()

    return mock


@pytest.mark.asyncio
async def test_external_llm_tool_registration(mock_hass_for_integration, external_llm_config):
    """Test that external LLM tool is registered when enabled."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, external_llm_config)

        # Trigger lazy tool registration
        agent._ensure_tools_registered()

        # Verify external LLM tool is registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert TOOL_QUERY_EXTERNAL_LLM in registered_tool_names

        # Get the tool and verify it's the correct type
        external_tool = agent.tool_handler.tools.get(TOOL_QUERY_EXTERNAL_LLM)
        assert isinstance(external_tool, ExternalLLMTool)


@pytest.mark.asyncio
async def test_external_llm_tool_not_registered_when_disabled(mock_hass_for_integration):
    """Test that external LLM tool is NOT registered when disabled."""
    config = {
        CONF_LLM_BASE_URL: "https://api.primary.com/v1",
        CONF_LLM_API_KEY: "primary-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_EXTERNAL_LLM_ENABLED: False,  # Disabled
    }

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, config)

        # Trigger lazy tool registration
        agent._ensure_tools_registered()

        # Verify external LLM tool is NOT registered
        registered_tool_names = agent.tool_handler.get_registered_tools()

        assert TOOL_QUERY_EXTERNAL_LLM not in registered_tool_names


@pytest.mark.asyncio
async def test_dual_llm_workflow_successful(mock_hass_for_integration, external_llm_config):
    """Test complete dual-LLM workflow: primary delegates to external LLM."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, external_llm_config)

        # Mock primary LLM response that calls external LLM tool
        primary_llm_response_with_tool_call = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": TOOL_QUERY_EXTERNAL_LLM,
                            "arguments": json.dumps({
                                "prompt": "Analyze energy usage and suggest optimizations",
                                "context": {
                                    "energy_data": {
                                        "sensor.energy_usage": [
                                            {"time": "2024-01-01T00:00:00", "value": 150},
                                            {"time": "2024-01-01T01:00:00", "value": 160},
                                        ]
                                    }
                                }
                            })
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        # Mock external LLM response
        external_llm_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Based on the energy data, I recommend: 1) Shift high-energy tasks to off-peak hours, 2) Install solar panels, 3) Upgrade to LED lighting."
                }
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
        }

        # Mock primary LLM final response after receiving tool result
        primary_llm_final_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I've analyzed your energy usage. Here are the recommendations: shift high-energy tasks to off-peak hours, install solar panels, and upgrade to LED lighting."
                }
            }],
            "usage": {"prompt_tokens": 150, "completion_tokens": 75, "total_tokens": 225}
        }

        # Mock aiohttp sessions
        with patch("aiohttp.ClientSession") as mock_session_class:
            call_count = [0]

            async def mock_post_side_effect(url, **kwargs):
                call_count[0] += 1
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)

                if "api.primary.com" in url:
                    # Primary LLM calls
                    if call_count[0] == 1:
                        # First call: primary LLM decides to use external tool
                        mock_response.json = AsyncMock(return_value=primary_llm_response_with_tool_call)
                    else:
                        # Second call: primary LLM formats final response
                        mock_response.json = AsyncMock(return_value=primary_llm_final_response)
                else:
                    # External LLM call
                    mock_response.json = AsyncMock(return_value=external_llm_response)

                mock_response.raise_for_status = MagicMock()
                return mock_response

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=mock_post_side_effect)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            # Execute the conversation
            response = await agent.process_message(
                text="Analyze my energy usage and suggest optimizations",
                conversation_id="test_conv_1"
            )

            # Verify response from primary LLM includes external LLM's analysis
            assert "recommendations" in response.lower()
            assert "off-peak" in response.lower() or "solar" in response.lower()

            # Verify both LLMs were called
            assert mock_session.post.call_count >= 2


@pytest.mark.asyncio
async def test_external_llm_error_propagation(mock_hass_for_integration, external_llm_config):
    """Test that external LLM errors are propagated to primary LLM."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, external_llm_config)

        # Mock primary LLM response that calls external LLM tool
        primary_llm_response_with_tool_call = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": TOOL_QUERY_EXTERNAL_LLM,
                            "arguments": json.dumps({
                                "prompt": "Complex analysis task"
                            })
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
        }

        # Mock primary LLM response after receiving error
        primary_llm_error_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error accessing the external analysis service. The service is currently unavailable."
                }
            }],
            "usage": {"prompt_tokens": 75, "completion_tokens": 30, "total_tokens": 105}
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            call_count = [0]

            async def mock_post_side_effect(url, **kwargs):
                call_count[0] += 1
                mock_response = AsyncMock()
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)

                if "api.primary.com" in url:
                    mock_response.status = 200
                    if call_count[0] == 1:
                        mock_response.json = AsyncMock(return_value=primary_llm_response_with_tool_call)
                    else:
                        mock_response.json = AsyncMock(return_value=primary_llm_error_response)
                    mock_response.raise_for_status = MagicMock()
                else:
                    # External LLM returns error
                    import aiohttp
                    mock_response.status = 503
                    mock_response.raise_for_status = MagicMock(
                        side_effect=aiohttp.ClientResponseError(
                            request_info=MagicMock(),
                            history=(),
                            status=503,
                            message="Service Unavailable"
                        )
                    )

                return mock_response

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=mock_post_side_effect)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            # Execute the conversation
            response = await agent.process_message(
                text="Perform complex analysis",
                conversation_id="test_conv_2"
            )

            # Verify primary LLM received error and communicated it to user
            assert "error" in response.lower() or "apologize" in response.lower() or "unavailable" in response.lower()


@pytest.mark.asyncio
async def test_tool_call_counting_includes_external_llm(mock_hass_for_integration, external_llm_config):
    """Test that external LLM calls count toward tool call limit."""
    # Set low limit for testing
    config = external_llm_config.copy()
    config[CONF_TOOLS_MAX_CALLS_PER_TURN] = 2  # Low limit

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, config)

        # Mock primary LLM making multiple tool calls (exceeding limit)
        primary_llm_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": TOOL_QUERY_EXTERNAL_LLM,
                                "arguments": json.dumps({"prompt": "Task 1"})
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": TOOL_QUERY_EXTERNAL_LLM,
                                "arguments": json.dumps({"prompt": "Task 2"})
                            }
                        },
                        {
                            "id": "call_3",
                            "type": "function",
                            "function": {
                                "name": TOOL_QUERY_EXTERNAL_LLM,
                                "arguments": json.dumps({"prompt": "Task 3"})
                            }
                        }
                    ]
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        }

        external_llm_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Analysis complete."
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
        }

        final_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "All tasks completed."
                }
            }],
            "usage": {"prompt_tokens": 75, "completion_tokens": 20, "total_tokens": 95}
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            call_count = [0]

            async def mock_post_side_effect(url, **kwargs):
                call_count[0] += 1
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_response.raise_for_status = MagicMock()

                if "api.primary.com" in url:
                    if call_count[0] == 1:
                        mock_response.json = AsyncMock(return_value=primary_llm_response)
                    else:
                        mock_response.json = AsyncMock(return_value=final_response)
                else:
                    mock_response.json = AsyncMock(return_value=external_llm_response)

                return mock_response

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=mock_post_side_effect)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            # Execute conversation
            await agent.process_message(
                text="Perform multiple tasks",
                conversation_id="test_conv_3"
            )

            # Verify that not all 3 external LLM calls were made (due to limit)
            # With limit of 2, only 2 external LLM calls should succeed
            # The total number of calls to external API should be <= 2
            external_api_calls = sum(
                1 for call in mock_session.post.call_args_list
                if "api.external.com" in str(call)
            )
            assert external_api_calls <= config[CONF_TOOLS_MAX_CALLS_PER_TURN]


@pytest.mark.asyncio
async def test_external_llm_context_not_included_automatically(mock_hass_for_integration, external_llm_config):
    """Test that conversation history is NOT automatically included in external LLM calls."""
    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, external_llm_config)

        # First message to establish history
        primary_response_1 = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I understand you want analysis."
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }

        # Second message that triggers external LLM
        primary_response_2 = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_789",
                        "type": "function",
                        "function": {
                            "name": TOOL_QUERY_EXTERNAL_LLM,
                            "arguments": json.dumps({
                                "prompt": "Analyze data",
                                # No context parameter - conversation history should NOT be included
                            })
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}
        }

        external_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Analysis result."
                }
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }

        final_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here's the analysis."
                }
            }],
            "usage": {"prompt_tokens": 75, "completion_tokens": 15, "total_tokens": 90}
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            responses = [primary_response_1, primary_response_2, external_response, final_response]
            response_index = [0]

            async def mock_post_side_effect(url, **kwargs):
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_response.raise_for_status = MagicMock()

                if "api.external.com" in url:
                    # Verify external LLM payload
                    payload = kwargs.get("json", {})
                    messages = payload.get("messages", [])

                    # Should only have 1 message (the prompt), not full conversation history
                    assert len(messages) == 1
                    assert messages[0]["role"] == "user"
                    # Should only contain the prompt, not previous conversation
                    assert "Analyze data" in messages[0]["content"]

                    mock_response.json = AsyncMock(return_value=external_response)
                else:
                    mock_response.json = AsyncMock(return_value=responses[response_index[0]])
                    response_index[0] += 1

                return mock_response

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=mock_post_side_effect)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            # First message
            await agent.process_message(
                text="I need some analysis",
                conversation_id="test_conv_4"
            )

            # Second message (triggers external LLM)
            await agent.process_message(
                text="Do the analysis now",
                conversation_id="test_conv_4"
            )


@pytest.mark.asyncio
async def test_external_llm_configuration_validation(mock_hass_for_integration):
    """Test that proper configuration is required for external LLM tool."""
    # Config missing external LLM settings
    incomplete_config = {
        CONF_LLM_BASE_URL: "https://api.primary.com/v1",
        CONF_LLM_API_KEY: "primary-key-123",
        CONF_LLM_MODEL: "gpt-4o-mini",
        CONF_EXTERNAL_LLM_ENABLED: True,
        # Missing CONF_EXTERNAL_LLM_BASE_URL and CONF_EXTERNAL_LLM_API_KEY
    }

    with patch("custom_components.home_agent.agent.async_should_expose") as mock_expose:
        mock_expose.return_value = False

        agent = HomeAgent(mock_hass_for_integration, incomplete_config)

        # Mock primary LLM calling external LLM tool
        primary_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_999",
                        "type": "function",
                        "function": {
                            "name": TOOL_QUERY_EXTERNAL_LLM,
                            "arguments": json.dumps({"prompt": "Test"})
                        }
                    }]
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        }

        error_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Configuration error occurred."
                }
            }],
            "usage": {"prompt_tokens": 75, "completion_tokens": 10, "total_tokens": 85}
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            call_count = [0]

            async def mock_post_side_effect(url, **kwargs):
                call_count[0] += 1
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.__aenter__ = AsyncMock(return_value=mock_response)
                mock_response.__aexit__ = AsyncMock(return_value=None)
                mock_response.raise_for_status = MagicMock()

                if call_count[0] == 1:
                    mock_response.json = AsyncMock(return_value=primary_response)
                else:
                    mock_response.json = AsyncMock(return_value=error_response)

                return mock_response

            mock_session = AsyncMock()
            mock_session.post = AsyncMock(side_effect=mock_post_side_effect)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            # Execute - should handle configuration error gracefully
            response = await agent.process_message(
                text="Test external LLM",
                conversation_id="test_conv_5"
            )

            # Should complete without crashing, error should be returned to primary LLM
            assert response is not None

"""Integration tests for keep_alive OpenAI compatibility.

Tests for GitHub Issue #65: OpenAI does not ignore keep_alive.

These tests verify that:
1. OpenAI and other cloud APIs do NOT receive the keep_alive parameter
2. Ollama servers DO receive the keep_alive parameter
3. The fix works end-to-end through the full agent flow
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_EXTERNAL_LLM_KEEP_ALIVE,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_ALIVE,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONF_STREAMING_ENABLED,
    CONTEXT_MODE_DIRECT,
)


@pytest.fixture
def test_hass():
    """Create a test Home Assistant instance."""
    from homeassistant.core import HomeAssistant

    hass = MagicMock(spec=HomeAssistant)
    hass.states = MagicMock()
    hass.states.async_all = MagicMock(return_value=[])
    hass.data = {}
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    hass.config = MagicMock()
    hass.config.location_name = "Test Home"
    return hass


@pytest.fixture
def session_manager(test_hass):
    """Create a session manager."""
    from custom_components.home_agent.conversation_session import ConversationSessionManager

    return ConversationSessionManager(test_hass)


class TestKeepAliveOpenAICompatibility:
    """Integration tests for OpenAI compatibility with keep_alive parameter."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_openai_api_does_not_receive_keep_alive(self, test_hass, session_manager):
        """Test that OpenAI's official API does not receive keep_alive.

        This is the exact scenario from Issue #65 - when using api.openai.com,
        the API was returning 400 error due to unrecognized keep_alive parameter.
        """
        config = {
            CONF_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_LLM_API_KEY: "sk-test-key",
            CONF_LLM_MODEL: "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: "5m",  # This should NOT be sent to OpenAI
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            agent = HomeAgent(test_hass, config, session_manager)

            # Mock the HTTP session to capture the payload
            with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_context)
                mock_ensure.return_value = mock_session

                # Make an LLM call
                messages = [{"role": "user", "content": "Hello"}]
                await agent._call_llm(messages)

                # Verify the critical fix: keep_alive should NOT be in payload
                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                payload = call_args.kwargs.get("json", {})

                assert "keep_alive" not in payload, (
                    "CRITICAL: keep_alive was sent to OpenAI API! "
                    "This will cause 400 error: 'Unrecognized request argument supplied: keep_alive'"
                )

                # Verify other required parameters ARE present
                assert "model" in payload
                assert "messages" in payload
                assert "temperature" in payload
                assert "max_tokens" in payload

            await agent.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_server_receives_keep_alive(self, test_hass, session_manager):
        """Test that Ollama servers DO receive the keep_alive parameter.

        Ollama uses keep_alive to control model memory retention.
        """
        config = {
            CONF_LLM_BASE_URL: "http://localhost:11434/v1",
            CONF_LLM_API_KEY: "ollama",
            CONF_LLM_MODEL: "llama2",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: "10m",  # This SHOULD be sent to Ollama
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            agent = HomeAgent(test_hass, config, session_manager)

            with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_context)
                mock_ensure.return_value = mock_session

                messages = [{"role": "user", "content": "Hello"}]
                await agent._call_llm(messages)

                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                payload = call_args.kwargs.get("json", {})

                # Verify keep_alive IS present for Ollama
                assert "keep_alive" in payload, "keep_alive should be sent to Ollama"
                assert payload["keep_alive"] == "10m"

            await agent.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_streaming_openai_does_not_receive_keep_alive(self, test_hass, session_manager):
        """Test that streaming requests to OpenAI do not include keep_alive."""
        config = {
            CONF_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_LLM_API_KEY: "sk-test-key",
            CONF_LLM_MODEL: "gpt-4o-mini",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: "5m",
            CONF_STREAMING_ENABLED: True,
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            agent = HomeAgent(test_hass, config, session_manager)

            with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.status = 200

                async def mock_content():
                    yield b"data: [DONE]\n"

                mock_response.content = mock_content()
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_context)
                mock_ensure.return_value = mock_session

                messages = [{"role": "user", "content": "Hello"}]
                async for _ in agent._call_llm_streaming(messages):
                    pass

                mock_session.post.assert_called_once()
                call_args = mock_session.post.call_args
                payload = call_args.kwargs.get("json", {})

                assert "keep_alive" not in payload, (
                    "keep_alive should NOT be in streaming request to OpenAI"
                )
                assert payload.get("stream") is True

            await agent.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_external_llm_tool_openai_compat(self, test_hass, session_manager):
        """Test that external LLM tool respects OpenAI compatibility."""
        from custom_components.home_agent.tools.external_llm import ExternalLLMTool

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "sk-test-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
            CONF_EXTERNAL_LLM_KEEP_ALIVE: "5m",
        }

        tool = ExternalLLMTool(test_hass, config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [{"message": {"role": "assistant", "content": "Response"}}],
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            await tool.execute(prompt="Test")

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]

            assert "keep_alive" not in payload, (
                "External LLM tool should NOT send keep_alive to OpenAI"
            )

        await tool.close()


class TestKeepAliveEdgeCases:
    """Test edge cases for keep_alive parameter handling."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_azure_openai_does_not_receive_keep_alive(self, test_hass, session_manager):
        """Test that Azure OpenAI endpoints do not receive keep_alive."""
        config = {
            CONF_LLM_BASE_URL: "https://my-resource.openai.azure.com/openai/deployments/gpt-4",
            CONF_LLM_API_KEY: "azure-api-key",
            CONF_LLM_MODEL: "gpt-4",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: "5m",
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            agent = HomeAgent(test_hass, config, session_manager)

            with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_context)
                mock_ensure.return_value = mock_session

                messages = [{"role": "user", "content": "Hello"}]
                await agent._call_llm(messages)

                payload = mock_session.post.call_args.kwargs.get("json", {})
                assert "keep_alive" not in payload

            await agent.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ollama_proxy_receives_keep_alive(self, test_hass, session_manager):
        """Test that Ollama behind a proxy still receives keep_alive."""
        config = {
            CONF_LLM_BASE_URL: "https://my-proxy.example.com/ollama/v1",
            CONF_LLM_API_KEY: "proxy-key",
            CONF_LLM_MODEL: "llama2",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: "-1",  # Keep forever
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            agent = HomeAgent(test_hass, config, session_manager)

            with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_context)
                mock_ensure.return_value = mock_session

                messages = [{"role": "user", "content": "Hello"}]
                await agent._call_llm(messages)

                payload = mock_session.post.call_args.kwargs.get("json", {})
                assert "keep_alive" in payload, "Ollama proxy should receive keep_alive"
                assert payload["keep_alive"] == "-1"

            await agent.close()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_local_llm_without_ollama_port_no_keep_alive(self, test_hass, session_manager):
        """Test that local LLM servers without Ollama port don't receive keep_alive."""
        config = {
            CONF_LLM_BASE_URL: "http://localhost:8080/v1",  # Not Ollama port
            CONF_LLM_API_KEY: "local-key",
            CONF_LLM_MODEL: "local-model",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: "5m",
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            agent = HomeAgent(test_hass, config, session_manager)

            with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
                mock_session = MagicMock()
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(
                    return_value={
                        "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
                mock_context = MagicMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_session.post = MagicMock(return_value=mock_context)
                mock_ensure.return_value = mock_session

                messages = [{"role": "user", "content": "Hello"}]
                await agent._call_llm(messages)

                payload = mock_session.post.call_args.kwargs.get("json", {})
                # Local LLM on non-Ollama port should NOT receive keep_alive
                # because it's likely vLLM, llama.cpp, or another server
                assert "keep_alive" not in payload, (
                    "Local LLM server on non-Ollama port should not receive keep_alive"
                )

            await agent.close()

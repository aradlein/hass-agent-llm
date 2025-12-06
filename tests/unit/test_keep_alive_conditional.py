"""Unit tests for keep_alive conditional logic.

Tests for GitHub Issue #65: OpenAI does not ignore keep_alive.

The keep_alive parameter is Ollama-specific and should NOT be sent to
OpenAI or other non-Ollama LLM providers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.home_agent.const import (
    CONF_CONTEXT_MODE,
    CONF_EMIT_EVENTS,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_ALIVE,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONTEXT_MODE_DIRECT,
    DEFAULT_LLM_KEEP_ALIVE,
)


class TestKeepAliveConditionalLLM:
    """Test keep_alive conditional logic for primary LLM calls."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.async_all = MagicMock(return_value=[])
        hass.data = {}
        hass.bus = MagicMock()
        hass.bus.async_fire = MagicMock()
        hass.config = MagicMock()
        hass.config.location_name = "Test Home"
        return hass

    @pytest.fixture
    def session_manager(self, mock_hass):
        """Create a session manager."""
        from custom_components.home_agent.conversation_session import (
            ConversationSessionManager,
        )

        return ConversationSessionManager(mock_hass)

    def _create_agent(self, mock_hass, session_manager, base_url: str, keep_alive: str = "5m"):
        """Create a HomeAgent with the given base URL."""
        from custom_components.home_agent.agent import HomeAgent

        config = {
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "test-model",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: keep_alive,
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            return HomeAgent(mock_hass, config, session_manager)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://api.openai.com",
            "https://openai.azure.com/v1",
            "https://api.anthropic.com/v1",
            "https://generativelanguage.googleapis.com/v1",
            "https://api.together.xyz/v1",
            "https://api.groq.com/openai/v1",
        ],
    )
    async def test_keep_alive_not_sent_to_openai_compatible_apis(
        self, mock_hass, session_manager, base_url
    ):
        """Test that keep_alive is NOT sent to OpenAI-compatible cloud APIs.

        These providers do not support the Ollama-specific keep_alive parameter
        and will return a 400 error if it's included.
        """
        agent = self._create_agent(mock_hass, session_manager, base_url)

        # Mock the HTTP session
        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "choices": [{"message": {"role": "assistant", "content": "Test"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            # Make an LLM call
            messages = [{"role": "user", "content": "test"}]
            await agent._call_llm(messages)

            # Verify payload does NOT contain keep_alive
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get("json", {})

            assert "keep_alive" not in payload, (
                f"keep_alive should NOT be sent to {base_url} "
                "(Ollama-specific parameter not supported by OpenAI-compatible APIs)"
            )

        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url,keep_alive_value",
        [
            ("http://localhost:11434/v1", "5m"),
            ("http://localhost:11434/v1", "10m"),
            ("http://localhost:11434/v1", "-1"),
            ("http://127.0.0.1:11434/v1", "5m"),
            ("http://ollama:11434/v1", "5m"),
            ("http://my-ollama-server:11434/v1", "5m"),
            ("http://192.168.1.100:11434/v1", "5m"),
        ],
    )
    async def test_keep_alive_sent_to_ollama(
        self, mock_hass, session_manager, base_url, keep_alive_value
    ):
        """Test that keep_alive IS sent to Ollama servers.

        Ollama uses keep_alive to control how long models stay loaded in memory.
        """
        agent = self._create_agent(mock_hass, session_manager, base_url, keep_alive_value)

        # Mock the HTTP session
        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "choices": [{"message": {"role": "assistant", "content": "Test"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            # Make an LLM call
            messages = [{"role": "user", "content": "test"}]
            await agent._call_llm(messages)

            # Verify payload DOES contain keep_alive with correct value
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get("json", {})

            assert "keep_alive" in payload, f"keep_alive should be sent to Ollama at {base_url}"
            assert payload["keep_alive"] == keep_alive_value, (
                f"Expected keep_alive={keep_alive_value}, got {payload.get('keep_alive')}"
            )

        await agent.close()


class TestKeepAliveConditionalStreaming:
    """Test keep_alive conditional logic for streaming LLM calls."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.async_all = MagicMock(return_value=[])
        hass.data = {}
        hass.bus = MagicMock()
        hass.bus.async_fire = MagicMock()
        hass.config = MagicMock()
        hass.config.location_name = "Test Home"
        return hass

    @pytest.fixture
    def session_manager(self, mock_hass):
        """Create a session manager."""
        from custom_components.home_agent.conversation_session import (
            ConversationSessionManager,
        )

        return ConversationSessionManager(mock_hass)

    def _create_agent(self, mock_hass, session_manager, base_url: str, keep_alive: str = "5m"):
        """Create a HomeAgent with the given base URL."""
        from custom_components.home_agent.agent import HomeAgent

        config = {
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "test-model",
            CONF_LLM_TEMPERATURE: 0.7,
            CONF_LLM_MAX_TOKENS: 500,
            CONF_LLM_KEEP_ALIVE: keep_alive,
            CONF_CONTEXT_MODE: CONTEXT_MODE_DIRECT,
            CONF_HISTORY_ENABLED: False,
            CONF_EMIT_EVENTS: False,
        }

        with patch(
            "custom_components.home_agent.agent.core.async_should_expose",
            return_value=False,
        ):
            return HomeAgent(mock_hass, config, session_manager)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com/v1",
        ],
    )
    async def test_streaming_keep_alive_not_sent_to_openai(
        self, mock_hass, session_manager, base_url
    ):
        """Test that keep_alive is NOT sent in streaming requests to OpenAI."""
        agent = self._create_agent(mock_hass, session_manager, base_url)

        # Mock the HTTP session for streaming
        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200

            # Mock async iterator for streaming response
            async def mock_content_generator():
                yield b"data: [DONE]\n"

            mock_response.content = mock_content_generator()
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            # Make a streaming LLM call
            messages = [{"role": "user", "content": "test"}]
            async for _ in agent._call_llm_streaming(messages):
                pass

            # Verify payload does NOT contain keep_alive
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get("json", {})

            assert "keep_alive" not in payload, (
                f"keep_alive should NOT be sent to {base_url} in streaming mode"
            )

        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:11434/v1",
            "http://ollama:11434/v1",
        ],
    )
    async def test_streaming_keep_alive_sent_to_ollama(
        self, mock_hass, session_manager, base_url
    ):
        """Test that keep_alive IS sent in streaming requests to Ollama."""
        agent = self._create_agent(mock_hass, session_manager, base_url, keep_alive="10m")

        # Mock the HTTP session for streaming
        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200

            # Mock async iterator for streaming response
            async def mock_content_generator():
                yield b"data: [DONE]\n"

            mock_response.content = mock_content_generator()
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            # Make a streaming LLM call
            messages = [{"role": "user", "content": "test"}]
            async for _ in agent._call_llm_streaming(messages):
                pass

            # Verify payload DOES contain keep_alive
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args.kwargs.get("json", {})

            assert "keep_alive" in payload, f"keep_alive should be sent to Ollama at {base_url}"
            assert payload["keep_alive"] == "10m"

        await agent.close()


class TestKeepAliveConditionalExternalLLM:
    """Test keep_alive conditional logic for external LLM tool."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.data = {}
        return hass

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com/v1",
            "https://api.together.xyz/v1",
        ],
    )
    async def test_external_llm_keep_alive_not_sent_to_openai(self, mock_hass, base_url):
        """Test that keep_alive is NOT sent by external LLM tool to OpenAI-compatible APIs."""
        from custom_components.home_agent.const import (
            CONF_EXTERNAL_LLM_API_KEY,
            CONF_EXTERNAL_LLM_BASE_URL,
            CONF_EXTERNAL_LLM_KEEP_ALIVE,
            CONF_EXTERNAL_LLM_MODEL,
        )
        from custom_components.home_agent.tools.external_llm import ExternalLLMTool

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: base_url,
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
            CONF_EXTERNAL_LLM_KEEP_ALIVE: "5m",
        }

        tool = ExternalLLMTool(mock_hass, config)

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
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

            await tool.execute(prompt="Test prompt")

            # Verify payload does NOT contain keep_alive
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]

            assert "keep_alive" not in payload, (
                f"External LLM tool should NOT send keep_alive to {base_url}"
            )

        await tool.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url,keep_alive_value",
        [
            ("http://localhost:11434/v1", "5m"),
            ("http://ollama:11434/v1", "10m"),
            ("http://192.168.1.100:11434/v1", "-1"),
        ],
    )
    async def test_external_llm_keep_alive_sent_to_ollama(
        self, mock_hass, base_url, keep_alive_value
    ):
        """Test that keep_alive IS sent by external LLM tool to Ollama."""
        from custom_components.home_agent.const import (
            CONF_EXTERNAL_LLM_API_KEY,
            CONF_EXTERNAL_LLM_BASE_URL,
            CONF_EXTERNAL_LLM_KEEP_ALIVE,
            CONF_EXTERNAL_LLM_MODEL,
        )
        from custom_components.home_agent.tools.external_llm import ExternalLLMTool

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: base_url,
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
            CONF_EXTERNAL_LLM_MODEL: "llama2",
            CONF_EXTERNAL_LLM_KEEP_ALIVE: keep_alive_value,
        }

        tool = ExternalLLMTool(mock_hass, config)

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
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

            await tool.execute(prompt="Test prompt")

            # Verify payload DOES contain keep_alive
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            payload = call_args[1]["json"]

            assert "keep_alive" in payload, (
                f"External LLM tool should send keep_alive to Ollama at {base_url}"
            )
            assert payload["keep_alive"] == keep_alive_value

        await tool.close()


class TestIsOllamaBackendHelper:
    """Test the helper function that detects Ollama backends."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            # Should be detected as Ollama (port 11434)
            ("http://localhost:11434/v1", True),
            ("http://127.0.0.1:11434/v1", True),
            ("http://ollama:11434/v1", True),
            ("http://my-server:11434/v1", True),
            ("http://192.168.1.100:11434/v1", True),
            ("https://ollama.example.com:11434/v1", True),
            # Should be detected as Ollama (contains 'ollama' in path/host)
            ("http://localhost:8080/ollama/v1", True),
            ("https://my-proxy.com/ollama/api/v1", True),
            # Should NOT be detected as Ollama (cloud APIs)
            ("https://api.openai.com/v1", False),
            ("https://api.anthropic.com/v1", False),
            ("https://openai.azure.com/v1", False),
            ("https://api.together.xyz/v1", False),
            ("https://api.groq.com/openai/v1", False),
            ("https://generativelanguage.googleapis.com/v1", False),
            # Should NOT be detected as Ollama (other local servers)
            ("http://localhost:8080/v1", False),
            ("http://localhost:5000/v1", False),
            ("http://my-llm-server:3000/v1", False),
        ],
    )
    def test_is_ollama_backend_detection(self, base_url, expected):
        """Test that Ollama backends are correctly detected by URL."""
        from custom_components.home_agent.helpers import is_ollama_backend

        result = is_ollama_backend(base_url)
        assert result == expected, (
            f"Expected is_ollama_backend('{base_url}') to be {expected}, got {result}"
        )

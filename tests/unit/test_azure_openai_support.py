"""Unit tests for Azure OpenAI support.

Tests for Azure OpenAI backend detection and URL/header construction.

Azure OpenAI uses a different URL structure and auth mechanism than standard
OpenAI-compatible endpoints:
- URL: {base_url}/openai/deployments/{model}/chat/completions?api-version={version}
  instead of {base_url}/chat/completions
- Auth: api-key: {value} header instead of Authorization: Bearer {value}
- Detection: Check for openai.azure.com in the base URL

These tests validate:
1. Detection of Azure OpenAI backends via is_azure_openai_backend()
2. URL construction via build_api_url() for Azure vs standard endpoints
3. Auth header construction via build_auth_headers() for Azure vs standard
4. Primary LLM (_call_llm) uses correct Azure URL and headers
5. Streaming LLM (_call_llm_streaming) uses correct Azure URL and headers
6. External LLM tool uses correct Azure URL and headers
7. Non-Azure backends remain unchanged (no regression)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientSession

from custom_components.home_agent.agent.llm import LLMMixin
from custom_components.home_agent.agent.streaming import StreamingMixin
from custom_components.home_agent.const import (
    CONF_AZURE_API_VERSION,
    CONF_CONTEXT_MODE,
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_HISTORY_ENABLED,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_KEEP_ALIVE,
    CONF_LLM_MAX_TOKENS,
    CONF_LLM_MODEL,
    CONF_LLM_TEMPERATURE,
    CONTEXT_MODE_DIRECT,
    DEFAULT_AZURE_API_VERSION,
)


class TestIsAzureOpenAIBackendHelper:
    """Test the helper function that detects Azure OpenAI backends."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            # Azure URLs that SHOULD be detected
            ("https://myresource.openai.azure.com", True),
            ("https://myresource.openai.azure.com/v1", True),
            ("https://MYRESOURCE.OPENAI.AZURE.COM/v1", True),  # Case insensitive
            ("https://my-resource.openai.azure.com", True),
            ("https://my-resource.openai.azure.com/openai/deployments/gpt-4/chat/completions", True),
            # Non-Azure URLs that should NOT be detected
            ("https://api.openai.com/v1", False),
            ("http://localhost:11434/v1", False),
            ("https://api.anthropic.com/v1", False),
            ("https://api.together.xyz/v1", False),
            ("", False),
            ("https://azure.com/something", False),  # azure without openai.azure.com
            ("http://localhost:8080/v1", False),
            ("https://api.groq.com/openai/v1", False),
        ],
    )
    def test_is_azure_openai_backend_detection(self, base_url, expected):
        """Test that Azure OpenAI backends are correctly detected by URL."""
        from custom_components.home_agent.helpers import is_azure_openai_backend

        result = is_azure_openai_backend(base_url)
        assert result == expected, (
            f"Expected is_azure_openai_backend('{base_url}') to be {expected}, got {result}"
        )


class TestBuildApiUrl:
    """Test the URL construction helper for Azure vs standard endpoints."""

    def test_standard_openai_url(self):
        """Test URL construction for standard OpenAI endpoint."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url("https://api.openai.com/v1", "gpt-4")
        assert result == "https://api.openai.com/v1/chat/completions"

    def test_ollama_url(self):
        """Test URL construction for Ollama endpoint."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url("http://localhost:11434/v1", "llama3")
        assert result == "http://localhost:11434/v1/chat/completions"

    def test_azure_url_default_api_version(self):
        """Test URL construction for Azure OpenAI with default API version."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url("https://myresource.openai.azure.com", "gpt-4o-mini")
        expected = (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o-mini"
            f"/chat/completions?api-version={DEFAULT_AZURE_API_VERSION}"
        )
        assert result == expected

    def test_azure_url_trailing_slash_stripped(self):
        """Test that trailing slash on Azure URL is stripped before constructing path."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url("https://myresource.openai.azure.com/", "gpt-4o-mini")
        expected = (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o-mini"
            f"/chat/completions?api-version={DEFAULT_AZURE_API_VERSION}"
        )
        assert result == expected

    def test_azure_url_custom_api_version(self):
        """Test URL construction for Azure OpenAI with custom API version."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url(
            "https://myresource.openai.azure.com",
            "gpt-4",
            azure_api_version="2024-06-01",
        )
        expected = (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4"
            "/chat/completions?api-version=2024-06-01"
        )
        assert result == expected

    def test_standard_url_ignores_azure_api_version(self):
        """Test that azure_api_version is ignored for non-Azure URLs."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url(
            "https://api.openai.com/v1",
            "gpt-4",
            azure_api_version="2024-06-01",
        )
        assert result == "https://api.openai.com/v1/chat/completions"

    def test_azure_url_case_insensitive(self):
        """Test that Azure detection in build_api_url is case-insensitive."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url("https://MYRESOURCE.OPENAI.AZURE.COM", "gpt-4o")
        assert "/openai/deployments/gpt-4o/chat/completions" in result
        assert f"api-version={DEFAULT_AZURE_API_VERSION}" in result

    def test_together_url(self):
        """Test URL construction for Together AI endpoint."""
        from custom_components.home_agent.helpers import build_api_url

        result = build_api_url("https://api.together.xyz/v1", "meta-llama/Llama-3-70b")
        assert result == "https://api.together.xyz/v1/chat/completions"


class TestBuildAuthHeaders:
    """Test the auth header construction for Azure vs standard endpoints."""

    def test_standard_with_key(self):
        """Test auth headers for standard OpenAI with API key."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://api.openai.com/v1", "test-key")
        assert result == {"Authorization": "Bearer test-key"}

    def test_standard_without_key(self):
        """Test auth headers for standard OpenAI without API key (local LLM)."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://api.openai.com/v1", "")
        assert result == {}

    def test_standard_with_none_key(self):
        """Test auth headers for standard OpenAI with None API key."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("http://localhost:11434/v1", None)
        assert result == {}

    def test_azure_with_key(self):
        """Test auth headers for Azure OpenAI with API key."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://myresource.openai.azure.com", "test-key")
        assert result == {"api-key": "test-key"}

    def test_azure_without_key(self):
        """Test auth headers for Azure OpenAI without API key."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://myresource.openai.azure.com", "")
        assert result == {}

    def test_azure_with_none_key(self):
        """Test auth headers for Azure OpenAI with None API key."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://myresource.openai.azure.com", None)
        assert result == {}

    def test_azure_case_insensitive(self):
        """Test that Azure detection for auth headers is case-insensitive."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://MYRESOURCE.OPENAI.AZURE.COM", "test-key")
        assert result == {"api-key": "test-key"}

    def test_ollama_with_key(self):
        """Test auth headers for Ollama with API key (uses standard Bearer)."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("http://localhost:11434/v1", "test-key")
        assert result == {"Authorization": "Bearer test-key"}

    def test_azure_no_bearer_prefix(self):
        """Test that Azure auth headers do NOT use Bearer prefix."""
        from custom_components.home_agent.helpers import build_auth_headers

        result = build_auth_headers("https://myresource.openai.azure.com", "test-key")
        assert "Authorization" not in result
        assert "Bearer" not in str(result)
        assert result.get("api-key") == "test-key"


class TestAzureOpenAIPrimaryLLM:
    """Test that the HomeAgent _call_llm correctly uses Azure URL and headers."""

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

    def _create_agent(self, mock_hass, session_manager, base_url: str, api_key: str = "test-key"):
        """Create a HomeAgent with the given base URL."""
        from custom_components.home_agent.agent import HomeAgent

        config = {
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: api_key,
            CONF_LLM_MODEL: "gpt-4o-mini",
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
            return HomeAgent(mock_hass, config, session_manager)

    @pytest.mark.asyncio
    async def test_azure_llm_uses_deployment_url(self, mock_hass, session_manager):
        """Test that Azure OpenAI LLM calls use the Azure deployment URL format."""
        agent = self._create_agent(
            mock_hass, session_manager, "https://myresource.openai.azure.com"
        )

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

            messages = [{"role": "user", "content": "test"}]
            await agent._call_llm(messages)

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args

            # Verify the URL contains Azure deployment path
            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
            assert "/openai/deployments/" in url, (
                f"Azure URL should contain /openai/deployments/, got: {url}"
            )
            assert "api-version=" in url, (
                f"Azure URL should contain api-version query parameter, got: {url}"
            )

        await agent.close()

    @pytest.mark.asyncio
    async def test_azure_llm_uses_api_key_header(self, mock_hass, session_manager):
        """Test that Azure OpenAI LLM calls use api-key header instead of Authorization."""
        agent = self._create_agent(
            mock_hass, session_manager, "https://myresource.openai.azure.com"
        )

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

            messages = [{"role": "user", "content": "test"}]
            await agent._call_llm(messages)

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            headers = call_args.kwargs.get("headers", {})

            # Azure should use api-key header
            assert "api-key" in headers, (
                f"Azure should use 'api-key' header, got headers: {headers}"
            )
            assert headers["api-key"] == "test-key"

            # Azure should NOT use Authorization: Bearer header
            assert "Authorization" not in headers, (
                f"Azure should NOT use 'Authorization' header, got headers: {headers}"
            )

        await agent.close()


class TestAzureOpenAIStreaming:
    """Test that streaming _call_llm_streaming uses Azure URL and headers."""

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

    def _create_agent(self, mock_hass, session_manager, base_url: str, api_key: str = "test-key"):
        """Create a HomeAgent with the given base URL."""
        from custom_components.home_agent.agent import HomeAgent

        config = {
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: api_key,
            CONF_LLM_MODEL: "gpt-4o-mini",
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
            return HomeAgent(mock_hass, config, session_manager)

    @pytest.mark.asyncio
    async def test_azure_streaming_uses_deployment_url(self, mock_hass, session_manager):
        """Test that Azure OpenAI streaming calls use the Azure deployment URL format."""
        agent = self._create_agent(
            mock_hass, session_manager, "https://myresource.openai.azure.com"
        )

        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200

            async def mock_content_generator():
                yield b"data: [DONE]\n"

            mock_response.content = mock_content_generator()
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            messages = [{"role": "user", "content": "test"}]
            async for _ in agent._call_llm_streaming(messages):
                pass

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args

            # Verify the URL contains Azure deployment path
            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
            assert "/openai/deployments/" in url, (
                f"Azure streaming URL should contain /openai/deployments/, got: {url}"
            )
            assert "api-version=" in url, (
                f"Azure streaming URL should contain api-version parameter, got: {url}"
            )

        await agent.close()

    @pytest.mark.asyncio
    async def test_azure_streaming_uses_api_key_header(self, mock_hass, session_manager):
        """Test that Azure OpenAI streaming calls use api-key header."""
        agent = self._create_agent(
            mock_hass, session_manager, "https://myresource.openai.azure.com"
        )

        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200

            async def mock_content_generator():
                yield b"data: [DONE]\n"

            mock_response.content = mock_content_generator()
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            messages = [{"role": "user", "content": "test"}]
            async for _ in agent._call_llm_streaming(messages):
                pass

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            headers = call_args.kwargs.get("headers", {})

            # Azure should use api-key header
            assert "api-key" in headers, (
                f"Azure streaming should use 'api-key' header, got: {headers}"
            )
            assert headers["api-key"] == "test-key"

            # Azure should NOT use Authorization: Bearer header
            assert "Authorization" not in headers, (
                f"Azure streaming should NOT use 'Authorization' header, got: {headers}"
            )

        await agent.close()


class TestAzureOpenAIExternalLLM:
    """Test that the ExternalLLMTool uses Azure URL and headers."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.data = {}
        return hass

    @pytest.mark.asyncio
    async def test_external_llm_azure_uses_deployment_url(self, mock_hass):
        """Test that external LLM tool uses Azure deployment URL format."""
        from custom_components.home_agent.tools.external_llm import ExternalLLMTool

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://myresource.openai.azure.com",
            CONF_EXTERNAL_LLM_API_KEY: "test-azure-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
        }

        tool = ExternalLLMTool(mock_hass, config)

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

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args

            # Verify the URL contains Azure deployment path
            url = call_args.args[0] if call_args.args else call_args[0][0]
            assert "/openai/deployments/" in url, (
                f"External LLM Azure URL should contain /openai/deployments/, got: {url}"
            )
            assert "api-version=" in url, (
                f"External LLM Azure URL should contain api-version, got: {url}"
            )

        await tool.close()

    @pytest.mark.asyncio
    async def test_external_llm_azure_uses_api_key_header(self, mock_hass):
        """Test that external LLM tool uses api-key header for Azure."""
        from custom_components.home_agent.tools.external_llm import ExternalLLMTool

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://myresource.openai.azure.com",
            CONF_EXTERNAL_LLM_API_KEY: "test-azure-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
        }

        tool = ExternalLLMTool(mock_hass, config)

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

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            headers = call_args[1]["headers"]

            # Azure should use api-key header
            assert "api-key" in headers, (
                f"External LLM Azure should use 'api-key' header, got: {headers}"
            )
            assert headers["api-key"] == "test-azure-key"

            # Azure should NOT use Authorization: Bearer header
            assert "Authorization" not in headers, (
                f"External LLM Azure should NOT use 'Authorization' header, got: {headers}"
            )

        await tool.close()

    @pytest.mark.asyncio
    async def test_external_llm_standard_still_uses_bearer(self, mock_hass):
        """Test that external LLM tool still uses Bearer auth for standard endpoints."""
        from custom_components.home_agent.tools.external_llm import ExternalLLMTool

        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-openai-key",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
        }

        tool = ExternalLLMTool(mock_hass, config)

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

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            headers = call_args[1]["headers"]

            # Standard endpoint should use Authorization: Bearer header
            assert "Authorization" in headers, (
                f"Standard endpoint should use 'Authorization' header, got: {headers}"
            )
            assert headers["Authorization"] == "Bearer test-openai-key"

            # Standard endpoint should NOT use api-key header
            assert "api-key" not in headers, (
                f"Standard endpoint should NOT use 'api-key' header, got: {headers}"
            )

        await tool.close()


class TestNonAzureBackendsUnchanged:
    """Parametrized tests confirming standard OpenAI/Ollama/Anthropic URLs still work exactly as before.

    These tests serve as regression tests to ensure Azure OpenAI support does not
    break existing functionality for non-Azure backends.
    """

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

    def _create_agent(self, mock_hass, session_manager, base_url: str, api_key: str = "test-key"):
        """Create a HomeAgent with the given base URL."""
        from custom_components.home_agent.agent import HomeAgent

        config = {
            CONF_LLM_BASE_URL: base_url,
            CONF_LLM_API_KEY: api_key,
            CONF_LLM_MODEL: "test-model",
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
            return HomeAgent(mock_hass, config, session_manager)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "http://localhost:11434/v1",
            "https://api.anthropic.com/v1",
            "https://api.together.xyz/v1",
            "https://api.groq.com/openai/v1",
        ],
    )
    async def test_non_azure_llm_uses_standard_url(
        self, mock_hass, session_manager, base_url
    ):
        """Test that non-Azure backends use standard /chat/completions URL."""
        agent = self._create_agent(mock_hass, session_manager, base_url)

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

            messages = [{"role": "user", "content": "test"}]
            await agent._call_llm(messages)

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args

            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")

            # Non-Azure should use standard /chat/completions
            assert url.endswith("/chat/completions"), (
                f"Non-Azure URL should end with /chat/completions, got: {url}"
            )
            # Non-Azure should NOT contain Azure deployment path
            assert "/openai/deployments/" not in url, (
                f"Non-Azure URL should NOT contain /openai/deployments/, got: {url}"
            )
            # Non-Azure should NOT contain api-version parameter
            assert "api-version=" not in url, (
                f"Non-Azure URL should NOT contain api-version, got: {url}"
            )

        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com/v1",
            "https://api.together.xyz/v1",
        ],
    )
    async def test_non_azure_llm_uses_bearer_auth(
        self, mock_hass, session_manager, base_url
    ):
        """Test that non-Azure backends use Authorization: Bearer header."""
        agent = self._create_agent(mock_hass, session_manager, base_url)

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

            messages = [{"role": "user", "content": "test"}]
            await agent._call_llm(messages)

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            headers = call_args.kwargs.get("headers", {})

            # Non-Azure should use Authorization: Bearer
            assert "Authorization" in headers, (
                f"Non-Azure should use 'Authorization' header for {base_url}, got: {headers}"
            )
            assert headers["Authorization"] == "Bearer test-key", (
                f"Non-Azure should use 'Bearer test-key' for {base_url}, got: {headers['Authorization']}"
            )

            # Non-Azure should NOT use api-key header
            assert "api-key" not in headers, (
                f"Non-Azure should NOT use 'api-key' header for {base_url}, got: {headers}"
            )

        await agent.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.openai.com/v1",
            "https://api.anthropic.com/v1",
        ],
    )
    async def test_non_azure_streaming_uses_standard_url(
        self, mock_hass, session_manager, base_url
    ):
        """Test that non-Azure streaming uses standard /chat/completions URL."""
        agent = self._create_agent(mock_hass, session_manager, base_url)

        with patch.object(agent, "_ensure_session", new_callable=AsyncMock) as mock_ensure:
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.status = 200

            async def mock_content_generator():
                yield b"data: [DONE]\n"

            mock_response.content = mock_content_generator()
            mock_context = MagicMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=mock_context)
            mock_ensure.return_value = mock_session

            messages = [{"role": "user", "content": "test"}]
            async for _ in agent._call_llm_streaming(messages):
                pass

            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args

            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")

            # Non-Azure should use standard /chat/completions
            assert url.endswith("/chat/completions"), (
                f"Non-Azure streaming URL should end with /chat/completions, got: {url}"
            )
            assert "/openai/deployments/" not in url, (
                f"Non-Azure streaming URL should NOT contain /openai/deployments/, got: {url}"
            )

        await agent.close()


class TestAzureApiVersionConstant:
    """Test that the Azure API version constant exists and has a valid value."""

    def test_default_azure_api_version_format(self):
        """Test that DEFAULT_AZURE_API_VERSION is a valid date-based version string."""
        assert isinstance(DEFAULT_AZURE_API_VERSION, str)
        assert len(DEFAULT_AZURE_API_VERSION) > 0
        # Azure API versions follow a date-based format like "2024-12-01-preview" or "2024-06-01"
        assert DEFAULT_AZURE_API_VERSION[0:4].isdigit(), (
            f"Azure API version should start with a year, got: {DEFAULT_AZURE_API_VERSION}"
        )

    def test_conf_azure_api_version_is_string(self):
        """Test that CONF_AZURE_API_VERSION config key is defined."""
        assert isinstance(CONF_AZURE_API_VERSION, str)
        assert len(CONF_AZURE_API_VERSION) > 0

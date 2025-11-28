"""Unit tests for proxy headers configuration feature."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientSession

from custom_components.home_agent.agent.llm import LLMMixin
from custom_components.home_agent.agent.streaming import StreamingMixin
from custom_components.home_agent.config_flow import (
    _migrate_legacy_backend,
    _validate_proxy_headers,
)
from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BACKEND,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROXY_HEADERS,
    LLM_BACKEND_DEFAULT,
    LLM_BACKEND_LLAMA_CPP,
)

# Note: CONF_LLM_BACKEND and LLM_BACKEND_* are still needed for migration tests
from custom_components.home_agent.exceptions import ValidationError


class TestProxyHeadersValidation:
    """Tests for proxy headers validation function."""

    def test_validate_empty_headers(self):
        """Test that empty headers return empty dict."""
        assert _validate_proxy_headers(None) == {}
        assert _validate_proxy_headers("") == {}
        assert _validate_proxy_headers("   ") == {}
        assert _validate_proxy_headers({}) == {}

    def test_validate_valid_headers_dict(self):
        """Test validation of valid headers as dict."""
        headers = {"X-Custom-Header": "value", "X-Another-Header": "value2"}
        result = _validate_proxy_headers(headers)
        assert result == headers

    def test_validate_valid_headers_json_string(self):
        """Test validation of valid headers as JSON string."""
        headers = {"X-Custom-Header": "value", "X-Another-Header": "value2"}
        json_str = json.dumps(headers)
        result = _validate_proxy_headers(json_str)
        assert result == headers

    def test_validate_headers_with_hyphens_underscores(self):
        """Test that headers with hyphens and underscores are valid."""
        headers = {
            "X-Ollama-Backend": "llama-cpp",
            "X_Custom_Header": "value",
            "X-Mixed_Header-Style": "value",
        }
        result = _validate_proxy_headers(headers)
        assert result == headers

    def test_reject_invalid_header_name_special_chars(self):
        """Test that header names with special characters are rejected."""
        headers = {"X-Header@Invalid": "value"}
        with pytest.raises(ValidationError, match="Invalid header name"):
            _validate_proxy_headers(headers)

    def test_reject_invalid_header_name_spaces(self):
        """Test that header names with spaces are rejected."""
        headers = {"X Header": "value"}
        with pytest.raises(ValidationError, match="Invalid header name"):
            _validate_proxy_headers(headers)

    def test_reject_invalid_header_name_colon(self):
        """Test that header names with colons are rejected."""
        headers = {"X-Header:": "value"}
        with pytest.raises(ValidationError, match="Invalid header name"):
            _validate_proxy_headers(headers)

    def test_reject_non_string_header_value(self):
        """Test that non-string header values are rejected."""
        headers = {"X-Header": 123}
        with pytest.raises(ValidationError, match="must be a string"):
            _validate_proxy_headers(headers)

    def test_reject_non_dict_headers(self):
        """Test that non-dictionary headers are rejected."""
        with pytest.raises(ValidationError, match="must be a JSON object"):
            _validate_proxy_headers("[1, 2, 3]")

    def test_reject_invalid_json(self):
        """Test that invalid JSON is rejected."""
        with pytest.raises(ValidationError, match="Invalid JSON format"):
            _validate_proxy_headers("{invalid json}")


class TestLegacyBackendMigration:
    """Tests for legacy backend migration."""

    def test_migrate_default_backend_no_migration(self):
        """Test that default backend is not migrated."""
        config = {CONF_LLM_BACKEND: LLM_BACKEND_DEFAULT}
        result = _migrate_legacy_backend(config)
        assert CONF_LLM_PROXY_HEADERS not in result

    def test_migrate_custom_backend(self):
        """Test that custom backend is migrated to proxy headers."""
        config = {CONF_LLM_BACKEND: LLM_BACKEND_LLAMA_CPP}
        result = _migrate_legacy_backend(config)
        assert result[CONF_LLM_PROXY_HEADERS] == {"X-Ollama-Backend": LLM_BACKEND_LLAMA_CPP}

    def test_migrate_no_backend_no_migration(self):
        """Test that missing backend does not trigger migration."""
        config = {}
        result = _migrate_legacy_backend(config)
        assert CONF_LLM_PROXY_HEADERS not in result

    def test_no_migration_when_proxy_headers_exist(self):
        """Test that migration does not overwrite existing proxy headers."""
        config = {
            CONF_LLM_BACKEND: LLM_BACKEND_LLAMA_CPP,
            CONF_LLM_PROXY_HEADERS: {"X-Custom": "value"},
        }
        result = _migrate_legacy_backend(config)
        # Should keep original proxy headers
        assert result[CONF_LLM_PROXY_HEADERS] == {"X-Custom": "value"}


class TestProxyHeadersInRequests:
    """Tests for proxy headers in LLM API requests."""

    @pytest.fixture
    def mock_llm_mixin(self):
        """Create a mock LLM mixin instance."""

        class MockAgent(LLMMixin):
            def __init__(self, config):
                self.config = config
                self._session = None

        return MockAgent

    @pytest.mark.asyncio
    async def test_proxy_headers_added_to_request(self, mock_llm_mixin):
        """Test that proxy headers are added to API requests."""
        config = {
            CONF_LLM_BASE_URL: "http://localhost:11434/v1",
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "llama3",
            CONF_LLM_PROXY_HEADERS: {
                "X-Ollama-Backend": "llama-cpp",
                "X-Custom-Router": "gpu-1",
            },
        }

        agent = mock_llm_mixin(config)

        # Mock the session and response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        )

        mock_session = MagicMock(spec=ClientSession)
        mock_session.closed = False
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock()

        agent._session = mock_session

        # Make the API call
        await agent._call_llm([{"role": "user", "content": "test"}])

        # Verify headers were passed
        call_args = mock_session.post.call_args
        headers = call_args.kwargs["headers"]

        assert "X-Ollama-Backend" in headers
        assert headers["X-Ollama-Backend"] == "llama-cpp"
        assert "X-Custom-Router" in headers
        assert headers["X-Custom-Router"] == "gpu-1"

    @pytest.mark.asyncio
    async def test_empty_proxy_headers_work(self, mock_llm_mixin):
        """Test that empty proxy headers don't cause issues."""
        config = {
            CONF_LLM_BASE_URL: "http://localhost:11434/v1",
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "llama3",
            CONF_LLM_PROXY_HEADERS: {},
        }

        agent = mock_llm_mixin(config)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        )

        mock_session = MagicMock(spec=ClientSession)
        mock_session.closed = False
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock()

        agent._session = mock_session

        # Should work without errors
        await agent._call_llm([{"role": "user", "content": "test"}])

        # Verify standard headers are present
        call_args = mock_session.post.call_args
        headers = call_args.kwargs["headers"]

        assert "Content-Type" in headers
        assert "Authorization" in headers

    @pytest.mark.asyncio
    async def test_multiple_headers_configured(self, mock_llm_mixin):
        """Test that multiple custom headers are all applied."""
        config = {
            CONF_LLM_BASE_URL: "http://localhost:11434/v1",
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "llama3",
            CONF_LLM_PROXY_HEADERS: {
                "X-Header-1": "value1",
                "X-Header-2": "value2",
                "X-Header-3": "value3",
            },
        }

        agent = mock_llm_mixin(config)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": "test"}}]}
        )

        mock_session = MagicMock(spec=ClientSession)
        mock_session.closed = False
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock()

        agent._session = mock_session

        await agent._call_llm([{"role": "user", "content": "test"}])

        call_args = mock_session.post.call_args
        headers = call_args.kwargs["headers"]

        assert headers["X-Header-1"] == "value1"
        assert headers["X-Header-2"] == "value2"
        assert headers["X-Header-3"] == "value3"

class TestProxyHeadersInStreaming:
    """Tests for proxy headers in streaming requests."""

    @pytest.fixture
    def mock_streaming_mixin(self):
        """Create a mock streaming mixin instance."""

        class MockAgent(StreamingMixin):
            def __init__(self, config):
                self.config = config
                self._session = None
                self.tool_handler = MagicMock()
                self.tool_handler.get_tool_definitions.return_value = []

            async def _ensure_session(self):
                """Override to return mock session."""
                return self._session

        return MockAgent

    @pytest.mark.asyncio
    async def test_proxy_headers_in_streaming(self, mock_streaming_mixin):
        """Test that proxy headers are added to streaming requests."""
        config = {
            CONF_LLM_BASE_URL: "http://localhost:11434/v1",
            CONF_LLM_API_KEY: "test-key",
            CONF_LLM_MODEL: "llama3",
            CONF_LLM_PROXY_HEADERS: {
                "X-Ollama-Backend": "llama-cpp",
                "X-Stream-Priority": "high",
            },
        }

        agent = mock_streaming_mixin(config)

        # Mock the streaming response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MagicMock()

        # Create async iterator for streaming
        async def async_iter():
            yield b'data: {"choices":[{"delta":{"content":"test"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        mock_response.content.__aiter__ = lambda self: async_iter()

        mock_session = MagicMock(spec=ClientSession)
        mock_session.closed = False
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock()

        agent._session = mock_session

        # Consume the stream
        chunks = []
        async for chunk in agent._call_llm_streaming([{"role": "user", "content": "test"}]):
            chunks.append(chunk)

        # Verify headers were passed
        call_args = mock_session.post.call_args
        headers = call_args.kwargs["headers"]

        assert "X-Ollama-Backend" in headers
        assert headers["X-Ollama-Backend"] == "llama-cpp"
        assert "X-Stream-Priority" in headers
        assert headers["X-Stream-Priority"] == "high"

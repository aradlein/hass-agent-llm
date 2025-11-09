"""Unit tests for the ExternalLLMTool."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.home_agent.const import (
    CONF_EXTERNAL_LLM_API_KEY,
    CONF_EXTERNAL_LLM_BASE_URL,
    CONF_EXTERNAL_LLM_MAX_TOKENS,
    CONF_EXTERNAL_LLM_MODEL,
    CONF_EXTERNAL_LLM_TEMPERATURE,
    CONF_EXTERNAL_LLM_TOOL_DESCRIPTION,
    CONF_TOOLS_TIMEOUT,
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_MODEL,
    DEFAULT_EXTERNAL_LLM_TEMPERATURE,
    DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION,
    TOOL_QUERY_EXTERNAL_LLM,
)
from custom_components.home_agent.exceptions import ValidationError
from custom_components.home_agent.tools.external_llm import ExternalLLMTool


class TestExternalLLMTool:
    """Test the ExternalLLMTool class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for external LLM."""
        return {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.example.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-api-key-123",
            CONF_EXTERNAL_LLM_MODEL: "gpt-4o",
            CONF_EXTERNAL_LLM_TEMPERATURE: 0.8,
            CONF_EXTERNAL_LLM_MAX_TOKENS: 1000,
            CONF_EXTERNAL_LLM_TOOL_DESCRIPTION: "Custom description for external LLM",
            CONF_TOOLS_TIMEOUT: 30,
        }

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM API response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a detailed analysis from the external LLM.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150,
            },
        }

    def test_tool_initialization(self, mock_hass, mock_config):
        """Test that tool initializes correctly."""
        tool = ExternalLLMTool(mock_hass, mock_config)
        assert tool.hass == mock_hass
        assert tool._config == mock_config
        assert tool._session is None

    def test_tool_name(self, mock_hass, mock_config):
        """Test that tool name is correct."""
        tool = ExternalLLMTool(mock_hass, mock_config)
        assert tool.name == TOOL_QUERY_EXTERNAL_LLM

    def test_tool_description_custom(self, mock_hass, mock_config):
        """Test that tool returns custom description from config."""
        tool = ExternalLLMTool(mock_hass, mock_config)
        assert tool.description == "Custom description for external LLM"

    def test_tool_description_default(self, mock_hass):
        """Test that tool returns default description when not in config."""
        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.example.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
        }
        tool = ExternalLLMTool(mock_hass, config)
        assert tool.description == DEFAULT_EXTERNAL_LLM_TOOL_DESCRIPTION

    def test_tool_parameters_schema(self, mock_hass, mock_config):
        """Test that parameter schema is valid."""
        tool = ExternalLLMTool(mock_hass, mock_config)
        params = tool.parameters

        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        # Check required parameters
        assert "prompt" in params["required"]

        # Check properties
        assert "prompt" in params["properties"]
        assert "context" in params["properties"]

        # Verify prompt is string type
        assert params["properties"]["prompt"]["type"] == "string"

        # Verify context is object type
        assert params["properties"]["context"]["type"] == "object"

    def test_to_openai_format(self, mock_hass, mock_config):
        """Test conversion to OpenAI format."""
        tool = ExternalLLMTool(mock_hass, mock_config)
        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == TOOL_QUERY_EXTERNAL_LLM
        assert "description" in openai_format["function"]
        assert "parameters" in openai_format["function"]

    @pytest.mark.asyncio
    async def test_execute_successful_query(self, mock_hass, mock_config, mock_llm_response):
        """Test successful external LLM query."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_llm_response)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Analyze this energy usage data")

            assert result["success"] is True
            assert result["result"] == "This is a detailed analysis from the external LLM."
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_hass, mock_config, mock_llm_response):
        """Test external LLM query with context parameter."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        context_data = {
            "energy_data": {
                "sensor.energy_usage": [
                    {"time": "2024-01-01T00:00:00", "value": 150},
                    {"time": "2024-01-01T01:00:00", "value": 160},
                ]
            }
        }

        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_llm_response)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Analyze this energy usage", context=context_data)

            assert result["success"] is True
            assert result["result"] == "This is a detailed analysis from the external LLM."

            # Verify context was included in the request
            mock_session.post.assert_called_once()
            call_kwargs = mock_session.post.call_args[1]
            payload = call_kwargs["json"]

            # Check that context is formatted and included in message
            assert "messages" in payload
            assert len(payload["messages"]) == 1
            assert "energy_data" in payload["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_execute_missing_prompt_raises_validation_error(self, mock_hass, mock_config):
        """Test that missing prompt raises ValidationError."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute()

        assert "prompt" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_empty_prompt_raises_validation_error(self, mock_hass, mock_config):
        """Test that empty prompt raises ValidationError."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(prompt="")

        assert "prompt" in str(exc_info.value).lower()
        assert "required" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_prompt_type_raises_validation_error(
        self, mock_hass, mock_config
    ):
        """Test that non-string prompt raises ValidationError."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(prompt=12345)  # Not a string

        assert "prompt" in str(exc_info.value).lower()
        assert "string" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_context_type_raises_validation_error(
        self, mock_hass, mock_config
    ):
        """Test that non-dict context raises ValidationError."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        with pytest.raises(ValidationError) as exc_info:
            await tool.execute(prompt="Test prompt", context="invalid context")  # Not a dict

        assert "context" in str(exc_info.value).lower()
        assert "dict" in str(exc_info.value).lower() or "object" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_execute_timeout_returns_error(self, mock_hass, mock_config):
        """Test that timeout returns error response."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Mock aiohttp session to simulate timeout
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await tool.execute(prompt="Test prompt")

                assert result["success"] is False
                assert result["result"] is None
                assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_auth_error_returns_error(self, mock_hass, mock_config):
        """Test that 401 auth error returns error response."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Mock aiohttp session to simulate 401 error
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=401,
                message="Unauthorized",
            )
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Test prompt")

            assert result["success"] is False
            assert result["result"] is None
            assert "authentication failed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_rate_limit_error_returns_error(self, mock_hass, mock_config):
        """Test that 429 rate limit error returns error response."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Mock aiohttp session to simulate 429 error
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=429,
                message="Too Many Requests",
            )
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Test prompt")

            assert result["success"] is False
            assert result["result"] is None
            assert "rate limit" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_connection_error_returns_error(self, mock_hass, mock_config):
        """Test that connection error returns error response."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Mock aiohttp session to simulate connection error
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Test prompt")

            assert result["success"] is False
            assert result["result"] is None
            assert "failed to connect" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_empty_response_returns_error(self, mock_hass, mock_config):
        """Test that empty response returns error."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Mock response with no choices
        empty_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [],  # Empty choices
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=empty_response)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Test prompt")

            assert result["success"] is False
            assert result["result"] is None
            assert "empty response" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_base_url_returns_error(self, mock_hass):
        """Test that missing base URL returns error response."""
        config = {
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
        }
        tool = ExternalLLMTool(mock_hass, config)

        result = await tool.execute(prompt="Test prompt")

        assert result["success"] is False
        assert result["result"] is None
        assert "base url" in result["error"].lower()
        assert "not configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_missing_api_key_returns_error(self, mock_hass):
        """Test that missing API key returns error response."""
        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.example.com/v1",
        }
        tool = ExternalLLMTool(mock_hass, config)

        result = await tool.execute(prompt="Test prompt")

        assert result["success"] is False
        assert result["result"] is None
        assert "api key" in result["error"].lower()
        assert "not configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_uses_config_values(self, mock_hass, mock_config, mock_llm_response):
        """Test that execute uses configuration values correctly."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_llm_response)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            await tool.execute(prompt="Test prompt")

            # Verify correct URL and headers
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args

            # Check URL
            assert call_args[0][0] == "https://api.example.com/v1/chat/completions"

            # Check headers
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test-api-key-123"
            assert headers["Content-Type"] == "application/json"

            # Check payload
            payload = call_args[1]["json"]
            assert payload["model"] == "gpt-4o"
            assert payload["temperature"] == 0.8
            assert payload["max_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_execute_uses_default_values(self, mock_hass, mock_llm_response):
        """Test that execute uses default values when not in config."""
        config = {
            CONF_EXTERNAL_LLM_BASE_URL: "https://api.example.com/v1",
            CONF_EXTERNAL_LLM_API_KEY: "test-key",
        }
        tool = ExternalLLMTool(mock_hass, config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_llm_response)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            await tool.execute(prompt="Test prompt")

            # Check payload uses defaults
            call_kwargs = mock_session.post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["model"] == DEFAULT_EXTERNAL_LLM_MODEL
            assert payload["temperature"] == DEFAULT_EXTERNAL_LLM_TEMPERATURE
            assert payload["max_tokens"] == DEFAULT_EXTERNAL_LLM_MAX_TOKENS

    @pytest.mark.asyncio
    async def test_format_context_as_json(self, mock_hass, mock_config):
        """Test that context is formatted as JSON."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        context = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "object"}}

        formatted = tool._format_context(context)

        # Should be valid JSON
        import json

        parsed = json.loads(formatted)
        assert parsed == context

    def test_format_context_handles_non_serializable(self, mock_hass, mock_config):
        """Test that format_context handles non-JSON-serializable objects."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Create context with non-serializable object
        class CustomObject:
            def __str__(self):
                return "CustomObject"

        context = {
            "key1": "value1",
            "key2": CustomObject(),
        }

        formatted = tool._format_context(context)

        # Should still return a string representation
        assert isinstance(formatted, str)
        assert "CustomObject" in formatted

    @pytest.mark.asyncio
    async def test_close_session(self, mock_hass, mock_config):
        """Test that close() properly closes the session."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Create a mock session
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()
        tool._session = mock_session

        await tool.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session_already_closed(self, mock_hass, mock_config):
        """Test that close() handles already closed session."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        # Create a mock session that's already closed
        mock_session = AsyncMock()
        mock_session.closed = True
        mock_session.close = AsyncMock()
        tool._session = mock_session

        await tool.close()

        # Should not call close on already closed session
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_no_session(self, mock_hass, mock_config):
        """Test that close() handles None session gracefully."""
        tool = ExternalLLMTool(mock_hass, mock_config)
        tool._session = None

        # Should not raise an error
        await tool.close()

    @pytest.mark.asyncio
    async def test_standardized_response_format(self, mock_hass, mock_config, mock_llm_response):
        """Test that all responses follow standardized format."""
        tool = ExternalLLMTool(mock_hass, mock_config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_llm_response)
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post = MagicMock(return_value=mock_response)
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await tool.execute(prompt="Test prompt")

            # Verify standardized format
            assert "success" in result
            assert "result" in result
            assert "error" in result
            assert isinstance(result["success"], bool)

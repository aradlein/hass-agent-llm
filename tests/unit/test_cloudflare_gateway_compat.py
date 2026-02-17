"""Unit tests for Cloudflare AI Gateway compatibility.

This module tests three bugs related to proxied LLM API requests through
gateways like Cloudflare AI Gateway:

Bug 1: Streaming path doesn't render Jinja2 API key templates.
    The streaming mixin reads the raw config value instead of calling
    render_template_value(), so template strings like
    {{ states('input_text.api_key') }} are sent as literal text.

Bug 2: aiohttp strips Authorization header on cross-origin redirects.
    The default aiohttp session follows redirects automatically. Per
    RFC 7235 section 2.2, aiohttp strips the Authorization header when
    a redirect changes the origin. Gateway proxies (e.g., Cloudflare)
    commonly return cross-origin redirects, causing "Missing Authorization
    header" errors.

Bug 3: Poor error messaging for auth-related failures.
    When the API returns HTTP 400 with "Missing Authorization header" in
    the body, the error is raised as a generic HomeAgentError with the raw
    text. A helpful hint about checking API key configuration and proxy
    redirect behavior would make debugging much easier.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import ClientSession

from custom_components.home_agent.agent.llm import LLMMixin
from custom_components.home_agent.agent.streaming import StreamingMixin
from custom_components.home_agent.const import (
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    CONF_LLM_PROXY_HEADERS,
)
from custom_components.home_agent.exceptions import HomeAgentError

# Patch path for the lazy Template import inside render_template_value
_TEMPLATE_PATCH = "homeassistant.helpers.template.Template"

# Patch path for render_template_value inside the streaming module
_RENDER_IN_STREAMING = "custom_components.home_agent.agent.streaming.render_template_value"

# Patch path for render_template_value inside the llm module
_RENDER_IN_LLM = "custom_components.home_agent.helpers.render_template_value"


# ---------------------------------------------------------------------------
# Helpers: reusable mock agent factories (matching existing test conventions)
# ---------------------------------------------------------------------------


class _MockLLMAgent(LLMMixin):
    """Concrete class using LLMMixin for testing."""

    def __init__(self, config, hass=None):
        self.config = config
        self.hass = hass or MagicMock()
        self._session = None


class _MockStreamingAgent(StreamingMixin):
    """Concrete class using StreamingMixin for testing."""

    def __init__(self, config, hass=None):
        self.config = config
        self.hass = hass or MagicMock()
        self._session = None
        self.tool_handler = MagicMock()
        self.tool_handler.get_tool_definitions.return_value = []

    async def _ensure_session(self):
        """Return the pre-assigned mock session."""
        return self._session


def _make_mock_session(response):
    """Build a MagicMock aiohttp session that returns *response* on POST.

    NOTE: __aexit__ must return a falsy value (False) so that exceptions
    raised inside ``async with session.post(...) as resp:`` blocks propagate
    correctly.  The default AsyncMock() returns a truthy AsyncMock object,
    which tells ``async with`` to suppress the exception.
    """
    mock_session = MagicMock(spec=ClientSession)
    mock_session.closed = False
    mock_session.post = MagicMock()
    mock_session.post.return_value.__aenter__ = AsyncMock(return_value=response)
    mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_session


def _make_ok_response():
    """Build a mock 200 JSON response for non-streaming calls."""
    resp = MagicMock()
    resp.status = 200
    resp.json = AsyncMock(
        return_value={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        }
    )
    return resp


def _make_streaming_response():
    """Build a mock 200 streaming response."""
    resp = MagicMock()
    resp.status = 200
    resp.content = MagicMock()

    async def _aiter():
        yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    resp.content.__aiter__ = lambda self: _aiter()
    return resp


def _make_error_response(status, body_text):
    """Build a mock error response with a given status and body."""
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body_text)
    resp.json = AsyncMock(side_effect=Exception("not json"))
    return resp


# ===========================================================================
# Bug 1: Streaming path doesn't render Jinja2 API key templates
# ===========================================================================


class TestStreamingTemplateRendering:
    """Tests proving that the streaming path does NOT render API key templates.

    The LLM (non-streaming) path correctly calls render_template_value() on
    the API key.  The streaming path should do the same, but currently uses
    the raw config string instead.
    """

    @pytest.mark.asyncio
    async def test_streaming_renders_api_key_template(self):
        """Streaming should call render_template_value on the API key.

        This test FAILS with the current code because streaming.py line 220
        does:
            api_key = self.config.get(CONF_LLM_API_KEY, "")
        instead of:
            api_key = render_template_value(
                self.hass, self.config.get(CONF_LLM_API_KEY, ""))

        When the API key is a Jinja template like
        {{ states('input_text.api_key') }}, the streaming path sends the
        literal template string as the Bearer token.
        """
        template_key = "{{ states('input_text.api_key') }}"
        rendered_key = "sk-rendered-secret-key"

        config = {
            CONF_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_LLM_API_KEY: template_key,
            CONF_LLM_MODEL: "gpt-4",
        }

        mock_hass = MagicMock()
        agent = _MockStreamingAgent(config, hass=mock_hass)

        mock_response = _make_streaming_response()
        agent._session = _make_mock_session(mock_response)

        # Patch render_template_value in the streaming module to return
        # the rendered key.  If streaming actually called it, the header
        # would contain the rendered value.
        with patch(_TEMPLATE_PATCH) as mock_template_cls:
            mock_tmpl_inst = MagicMock()
            mock_tmpl_inst.async_render = MagicMock(return_value=rendered_key)
            mock_template_cls.return_value = mock_tmpl_inst

            # Consume the stream
            chunks = []
            async for chunk in agent._call_llm_streaming(
                [{"role": "user", "content": "hi"}]
            ):
                chunks.append(chunk)

        # Verify the Authorization header contains the RENDERED key,
        # not the raw template string.
        call_args = agent._session.post.call_args
        headers = call_args.kwargs["headers"]

        assert "Authorization" in headers, (
            "Authorization header should be present"
        )

        # This assertion will FAIL: current code sends the literal template
        # string because it never calls render_template_value.
        assert headers["Authorization"] == f"Bearer {rendered_key}", (
            f"Expected rendered key '{rendered_key}' in Authorization header, "
            f"but got '{headers['Authorization']}'. "
            "The streaming path does not render Jinja2 API key templates."
        )

    @pytest.mark.asyncio
    async def test_streaming_sends_auth_header_with_plain_key(self):
        """Plain string API keys should work in streaming (baseline).

        This is a sanity check: when the API key is a plain string (not a
        template), the streaming path should include it in the Authorization
        header without any rendering.  This test should PASS with current code.
        """
        plain_key = "sk-plain-test-key-12345"

        config = {
            CONF_LLM_BASE_URL: "https://api.openai.com/v1",
            CONF_LLM_API_KEY: plain_key,
            CONF_LLM_MODEL: "gpt-4",
        }

        agent = _MockStreamingAgent(config)
        mock_response = _make_streaming_response()
        agent._session = _make_mock_session(mock_response)

        chunks = []
        async for chunk in agent._call_llm_streaming(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        call_args = agent._session.post.call_args
        headers = call_args.kwargs["headers"]

        assert headers["Authorization"] == f"Bearer {plain_key}"


# ===========================================================================
# Bug 2: aiohttp strips Authorization header on cross-origin redirects
# ===========================================================================


class TestRedirectAuthPreservation:
    """Tests proving that sessions/requests don't auto-follow redirects.

    When an LLM API request is routed through a gateway (e.g., Cloudflare
    AI Gateway), the gateway may issue a cross-origin redirect (302).
    aiohttp's default behaviour is to follow redirects but strip the
    Authorization header when the redirect target has a different origin
    (per RFC 7235 section 2.2).

    The fix is to either:
    (a) Create the session with auto_redirect=False, or
    (b) Pass allow_redirects=False to session.post().

    These tests verify that the code takes one of these approaches.
    """

    @pytest.mark.asyncio
    async def test_llm_preserves_auth_header_no_auto_redirect(self):
        """LLMMixin must not let aiohttp auto-follow redirects.

        This test FAILS with the current code because _ensure_session()
        creates a plain aiohttp.ClientSession() without setting
        auto_redirect=False, and _call_llm() calls session.post()
        without allow_redirects=False.
        """
        config = {
            CONF_LLM_BASE_URL: "https://gateway.ai.cloudflare.com/v1/acct/gw/openai/v1",
            CONF_LLM_API_KEY: "sk-test-key",
            CONF_LLM_MODEL: "gpt-4",
        }

        agent = _MockLLMAgent(config)

        # Capture how aiohttp.ClientSession is constructed
        with patch("custom_components.home_agent.agent.llm.aiohttp.ClientSession") as mock_cls:
            mock_session_inst = MagicMock(spec=ClientSession)
            mock_session_inst.closed = False

            # Set up the session.post to return a successful response
            ok_resp = _make_ok_response()
            mock_session_inst.post = MagicMock()
            mock_session_inst.post.return_value.__aenter__ = AsyncMock(return_value=ok_resp)
            mock_session_inst.post.return_value.__aexit__ = AsyncMock()
            mock_cls.return_value = mock_session_inst

            # Force creation of a new session
            agent._session = None
            await agent._call_llm([{"role": "user", "content": "test"}])

            # Check option 1: session created with auto_redirect=False
            session_kwargs = mock_cls.call_args
            session_has_no_redirect = False
            if session_kwargs:
                # Check keyword args
                kw = session_kwargs.kwargs if session_kwargs.kwargs else {}
                if kw.get("auto_decompress") is not None:
                    pass  # unrelated kwarg
                if "auto_redirect" in kw and kw["auto_redirect"] is False:
                    session_has_no_redirect = True

            # Check option 2: post called with allow_redirects=False
            post_kwargs = mock_session_inst.post.call_args
            post_kw = post_kwargs.kwargs if post_kwargs.kwargs else {}
            post_has_no_redirect = post_kw.get("allow_redirects") is False

            assert session_has_no_redirect or post_has_no_redirect, (
                "Neither the session was created with auto_redirect=False "
                "nor was session.post() called with allow_redirects=False. "
                "aiohttp will follow cross-origin redirects and strip the "
                "Authorization header (RFC 7235), causing 'Missing Authorization "
                "header' errors when using gateway proxies like Cloudflare."
            )

    @pytest.mark.asyncio
    async def test_streaming_preserves_auth_header_no_auto_redirect(self):
        """StreamingMixin must not let aiohttp auto-follow redirects.

        This test FAILS with the current code because the streaming path
        uses the same session (from _ensure_session) that does not disable
        auto-redirect, and session.post() is called without
        allow_redirects=False.
        """
        config = {
            CONF_LLM_BASE_URL: "https://gateway.ai.cloudflare.com/v1/acct/gw/openai/v1",
            CONF_LLM_API_KEY: "sk-test-key",
            CONF_LLM_MODEL: "gpt-4",
        }

        agent = _MockStreamingAgent(config)

        # Create a mock session and track how .post() is called
        mock_response = _make_streaming_response()
        mock_session = _make_mock_session(mock_response)
        agent._session = mock_session

        # We also need to check how the session was created. Since
        # _MockStreamingAgent overrides _ensure_session, we test the post
        # call's allow_redirects parameter directly.
        chunks = []
        async for chunk in agent._call_llm_streaming(
            [{"role": "user", "content": "test"}]
        ):
            chunks.append(chunk)

        post_call = mock_session.post.call_args
        post_kw = post_call.kwargs if post_call.kwargs else {}
        post_has_no_redirect = post_kw.get("allow_redirects") is False

        # For the streaming mixin, we can only check the post call since
        # the session is provided externally. If the session itself was
        # created by LLMMixin._ensure_session, the session-level check
        # (from the test above) covers it. But the post call should also
        # explicitly disable redirects for defense in depth.
        assert post_has_no_redirect, (
            "session.post() in the streaming path was not called with "
            "allow_redirects=False. Cross-origin redirects from gateway "
            "proxies (e.g., Cloudflare AI Gateway) will cause aiohttp to "
            "strip the Authorization header, resulting in auth failures."
        )


# ===========================================================================
# Bug 3: Poor error messaging for auth-related failures
# ===========================================================================


class TestAuthErrorMessaging:
    """Tests proving that auth-related errors lack helpful diagnostics.

    When a gateway proxy strips the Authorization header (due to a
    cross-origin redirect) and the upstream API returns an error, the
    current code raises a generic HomeAgentError with the raw error text.
    A better error message would hint at common causes like:
    - API key not configured or template not rendering
    - Proxy/gateway stripping Authorization header on redirect
    """

    @pytest.mark.asyncio
    async def test_error_message_includes_auth_hint_on_missing_auth(self):
        """A 400 response with 'Missing Authorization header' should include a hint.

        This test FAILS with the current code because line 262 of llm.py
        raises:
            HomeAgentError(f"LLM API returned status {response.status}: {error_text}")
        which is a raw error dump with no actionable guidance.

        The error message should include a hint about checking:
        - API key configuration (possibly a template that isn't rendering)
        - Proxy/gateway headers (redirect stripping auth)
        """
        config = {
            CONF_LLM_BASE_URL: "https://gateway.ai.cloudflare.com/v1/acct/gw/openai/v1",
            CONF_LLM_API_KEY: "sk-test-key",
            CONF_LLM_MODEL: "gpt-4",
        }

        agent = _MockLLMAgent(config)

        error_body = '{"error": {"message": "Missing Authorization header", "type": "invalid_request_error"}}'
        mock_response = _make_error_response(400, error_body)
        agent._session = _make_mock_session(mock_response)

        with pytest.raises(HomeAgentError) as exc_info:
            await agent._call_llm([{"role": "user", "content": "test"}])

        error_msg = str(exc_info.value).lower()

        # The error message should contain a helpful hint — not just the
        # raw API response text. We check for keywords that indicate the
        # error message provides guidance about possible causes.
        has_api_key_hint = (
            "api key" in error_msg
            or "api_key" in error_msg
            or "authentication" in error_msg
            or "authorization" in error_msg
        )
        has_config_hint = (
            "check" in error_msg
            or "verify" in error_msg
            or "configuration" in error_msg
            or "config" in error_msg
        )

        assert has_api_key_hint and has_config_hint, (
            f"Error message should include a hint about checking API key "
            f"configuration when the response body contains "
            f"'Missing Authorization header'. Got: '{exc_info.value}'"
        )

    @pytest.mark.asyncio
    async def test_error_message_includes_auth_hint_on_401(self):
        """A 401 response should raise AuthenticationError with helpful text.

        This test verifies that a 401 raises AuthenticationError (not just
        HomeAgentError) and includes guidance about checking the API key
        and configuration.

        Note: The current code does raise AuthenticationError for 401, so
        this test may partially pass. The key assertion is that the message
        includes actionable hints beyond just "authentication failed".
        """
        config = {
            CONF_LLM_BASE_URL: "https://gateway.ai.cloudflare.com/v1/acct/gw/openai/v1",
            CONF_LLM_API_KEY: "sk-test-key",
            CONF_LLM_MODEL: "gpt-4",
        }

        agent = _MockLLMAgent(config)

        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(
            return_value='{"error": {"message": "Incorrect API key provided"}}'
        )
        agent._session = _make_mock_session(mock_response)

        with pytest.raises(HomeAgentError) as exc_info:
            await agent._call_llm([{"role": "user", "content": "test"}])

        error_msg = str(exc_info.value).lower()

        # The error should mention checking configuration — not just say
        # "authentication failed" without guidance.
        has_config_guidance = (
            "check" in error_msg
            or "verify" in error_msg
            or "ensure" in error_msg
        )
        has_key_reference = (
            "api key" in error_msg
            or "api_key" in error_msg
            or "key" in error_msg
        )
        has_template_or_proxy_hint = (
            "template" in error_msg
            or "proxy" in error_msg
            or "redirect" in error_msg
            or "gateway" in error_msg
            or "header" in error_msg
            or "configuration" in error_msg
            or "config" in error_msg
        )

        assert has_config_guidance and has_key_reference, (
            f"401 error message should include guidance about checking API key "
            f"configuration. Got: '{exc_info.value}'"
        )

        # Stronger assertion: the message should mention potential causes
        # specific to proxy/gateway setups.
        assert has_template_or_proxy_hint, (
            f"401 error message should hint at common causes like template "
            f"rendering, proxy headers, or gateway configuration. "
            f"Got: '{exc_info.value}'"
        )

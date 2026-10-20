"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _content_has_visible_payload,
    _normalize_multimodal_content,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Pure-function tests for _normalize_multimodal_content
# ---------------------------------------------------------------------------


class TestNormalizeMultimodalContent:
    def test_string_passthrough(self):
        assert _normalize_multimodal_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _normalize_multimodal_content(None) == ""

    def test_text_only_list_collapses_to_string(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _normalize_multimodal_content(content) == "hi\nthere"

    def test_responses_input_text_canonicalized(self):
        content = [{"type": "input_text", "text": "hello"}]
        assert _normalize_multimodal_content(content) == "hello"

    def test_inline_data_image_passes_through_native_guard(self):
        original = "data:image/jpeg;base64,b3JpZ2luYWw="
        guarded = "data:image/jpeg;base64,Z3VhcmRlZA=="
        content = [{"type": "image_url", "image_url": {"url": original}}]

        guarded_parts = [{"type": "image_url", "image_url": {"url": guarded}}]
        with patch(
            "gateway.platforms.api_server.prepare_native_image_parts", return_value=guarded_parts
        ) as guard:
            normalized = _normalize_multimodal_content(content)

        guard.assert_called_once_with(original, detail=None, allow_tiling=False)
        assert normalized == [{"type": "image_url", "image_url": {"url": guarded}}]

    def test_live_single_image_can_expand_to_tiled_parts(self):
        original = "data:image/png;base64,b3JpZ2luYWw="
        expanded = [
            {"type": "text", "text": "[High-detail image split into four overlapping tiles]"},
            *[
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{idx}"}}
                for idx in range(4)
            ],
        ]
        content = [{"type": "image_url", "image_url": {"url": original}}]
        with patch(
            "gateway.platforms.api_server.prepare_native_image_parts", return_value=expanded
        ) as prepare:
            normalized = _normalize_multimodal_content(content, allow_tiling=True)
        prepare.assert_called_once_with(original, detail=None, allow_tiling=True)
        assert normalized == expanded

    def test_multiple_images_disable_tiling(self):
        content = [
            {"type": "image_url", "image_url": {"url": "https://example.com/one.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/two.png"}},
        ]
        with patch(
            "gateway.platforms.api_server.prepare_native_image_parts",
            side_effect=lambda url, **_: [{"type": "image_url", "image_url": {"url": url}}],
        ) as prepare:
            _normalize_multimodal_content(content, allow_tiling=True)
        assert all(call.kwargs["allow_tiling"] is False for call in prepare.call_args_list)


class TestContentHasVisiblePayload:


    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestChatCompletionsMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_inline_image_preserved_to_run_agent(self, adapter):
        """Multimodal user content reaches _run_agent as a list of parts."""
        image_payload = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new=MagicMock(),
            ) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": image_payload}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == image_payload


class TestResponsesMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_input_image_canonicalized_and_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.com/cat.png",
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            expected = [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]
            assert mock_run.captured["user_message"] == expected

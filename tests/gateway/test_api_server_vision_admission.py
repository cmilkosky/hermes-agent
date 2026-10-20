"""Behavior contracts for single-flight admission of four-tile vision turns."""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.image_routing import TILED_NATIVE_MARKER
from gateway.platforms import api_server_vision_admission as admission
from tests.waiting import async_wait_until


def _message(image_count: int = 4, *, marker: bool = True) -> list[dict]:
    text = TILED_NATIVE_MARKER if marker else "ordinary image request"
    return [
        {"type": "text", "text": text},
        *[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{index}"}}
            for index in range(image_count)
        ],
    ]


def _submit(loop, fn):
    return loop.run_in_executor(None, fn)


def test_only_canonical_four_tile_payload_needs_admission():
    assert admission.is_tiled_native_message(_message()) is True
    assert admission.is_tiled_native_message(_message(3)) is False
    assert admission.is_tiled_native_message(_message(marker=False)) is False
    assert admission.is_tiled_native_message(TILED_NATIVE_MARKER) is False


@pytest.mark.asyncio
async def test_tiled_workers_are_serialized_while_text_workers_bypass_gate():
    loop = asyncio.get_running_loop()
    release_first = threading.Event()
    first_started = threading.Event()
    starts: list[str] = []

    def first_worker():
        starts.append("first")
        first_started.set()
        release_first.wait(timeout=5)
        return "first"

    def second_worker():
        starts.append("second")
        return "second"

    first = asyncio.create_task(admission.submit_api_worker(
        loop, first_worker, _message(), _submit))
    try:
        await async_wait_until(first_started.is_set, what="first tiled worker to start")
        second = asyncio.create_task(admission.submit_api_worker(
            loop, second_worker, _message(), _submit))
        await async_wait_until(
            lambda: admission.admission_snapshot(loop) == (1, 1),
            what="second tiled worker to queue",
        )
        assert starts == ["first"]

        text_result = await admission.submit_api_worker(
            loop, lambda: "text", "ordinary text turn", _submit)
        assert text_result == "text"
        assert starts == ["first"]

        release_first.set()
        assert await first == "first"
        assert await second == "second"
        await async_wait_until(
            lambda: admission.admission_snapshot(loop) == (0, 0),
            what="tiled admission to drain",
        )
        assert starts == ["first", "second"]
    finally:
        release_first.set()


@pytest.mark.asyncio
async def test_cancelled_request_keeps_slot_until_its_worker_exits():
    loop = asyncio.get_running_loop()
    release_worker = threading.Event()
    worker_started = threading.Event()

    def worker():
        worker_started.set()
        release_worker.wait(timeout=5)

    task = asyncio.create_task(admission.submit_api_worker(loop, worker, _message(), _submit))
    try:
        await async_wait_until(worker_started.is_set, what="tiled worker to start")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admission.admission_snapshot(loop) == (1, 0)
    finally:
        release_worker.set()
    await async_wait_until(
        lambda: admission.admission_snapshot(loop) == (0, 0),
        what="cancelled tiled worker to release its slot",
    )

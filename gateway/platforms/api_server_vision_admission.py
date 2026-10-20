"""Single-flight admission for memory-heavy tiled native-vision turns."""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from agent.image_routing import TILED_NATIVE_MARKER


logger = logging.getLogger("gateway.platforms.api_server")
_T = TypeVar("_T")


@dataclass
class _AdmissionState:
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))
    active: int = 0
    queued: int = 0


_LOOP_STATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _AdmissionState] = (
    weakref.WeakKeyDictionary()
)


def _state_for(loop: asyncio.AbstractEventLoop) -> _AdmissionState:
    state = _LOOP_STATES.get(loop)
    if state is None:
        state = _AdmissionState()
        _LOOP_STATES[loop] = state
    return state


def is_tiled_native_message(user_message: Any) -> bool:
    """True only for the canonical marker plus exactly four native image parts."""
    if not isinstance(user_message, list):
        return False
    marker = any(
        isinstance(part, dict)
        and part.get("type") == "text"
        and TILED_NATIVE_MARKER in str(part.get("text") or "")
        for part in user_message
    )
    image_count = sum(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in user_message
    )
    return marker and image_count == 4


def admission_snapshot(
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[int, int]:
    """Return ``(active, queued)`` for diagnostics and deterministic tests."""
    state = _state_for(loop or asyncio.get_running_loop())
    return state.active, state.queued


async def submit_api_worker(
    loop: asyncio.AbstractEventLoop,
    fn: Callable[[], _T],
    user_message: Any,
    submit_worker: Callable[[asyncio.AbstractEventLoop, Callable[[], _T]], Awaitable[_T]],
) -> _T:
    """Submit a worker, serializing only turns carrying a real four-tile payload.

    Admission happens on the event loop, while release happens when the executor
    worker itself exits.  A disconnected/cancelled HTTP task therefore cannot free
    the slot while its model turn is still consuming TensorFold memory.
    """
    if not is_tiled_native_message(user_message):
        return await submit_worker(loop, fn)

    state = _state_for(loop)
    wait_started = time.perf_counter()
    state.queued += 1
    logger.info("tiled vision request queued (active=%d queued=%d)", state.active, state.queued)
    try:
        await state.semaphore.acquire()
    except BaseException:
        state.queued -= 1
        raise
    state.queued -= 1
    state.active += 1
    logger.info(
        "tiled vision request admitted after %.3fs (queued=%d)",
        time.perf_counter() - wait_started,
        state.queued,
    )

    released = False

    def _release() -> None:
        nonlocal released
        if released:
            return
        released = True
        state.active -= 1
        state.semaphore.release()
        logger.info("tiled vision request released (queued=%d)", state.queued)

    def _guarded_worker() -> _T:
        try:
            return fn()
        finally:
            try:
                loop.call_soon_threadsafe(_release)
            except RuntimeError:
                pass  # Event loop already closed during process shutdown.

    try:
        worker = submit_worker(loop, _guarded_worker)
    except BaseException:
        _release()
        raise

    def _release_cancelled_worker(done: asyncio.Future) -> None:
        if done.cancelled():
            _release()

    worker.add_done_callback(_release_cancelled_worker)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # The shield keeps the executor worker alive; consume a later exception
        # because this request task no longer awaits it.
        worker.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        raise


__all__ = ["admission_snapshot", "is_tiled_native_message", "submit_api_worker"]

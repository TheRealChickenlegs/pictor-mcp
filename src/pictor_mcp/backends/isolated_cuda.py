"""Lazy CUDA backend hosted in a short-lived, isolated process.

The MCP server itself never imports a CUDA runtime. The worker is spawned by
Python's ``spawn`` context on first use, then terminated after the configured
idle interval. Process exit releases the CUDA context and allocator completely.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from contextlib import suppress
from typing import Any

from PIL import Image

from ..errors import BackendUnavailableError
from .base import BackendStatus

logger = logging.getLogger(__name__)


class IsolatedCudaBackend:
    """Proxy for CUDA resize and ONNX-backed background removal."""

    name = "torch-cuda"
    device = "cuda:0"

    def __init__(
        self,
        *,
        idle_seconds: float = 300.0,
        startup_timeout: float = 120.0,
        resize_enabled: bool = True,
    ) -> None:
        self.idle_seconds = max(0.0, idle_seconds)
        self.resize_enabled = resize_enabled
        self.startup_timeout = startup_timeout
        self._lock = threading.RLock()
        # Activity bookkeeping has its own short-held lock so a new MCP request
        # never blocks the event loop behind a long CUDA IPC operation.
        self._activity_lock = threading.Lock()
        self._process: Any = None
        self._connection: Any = None
        self._timer: threading.Timer | None = None
        self._active_tool_calls = 0
        self._worker_available: bool | None = None
        self._detail = (
            "configured; worker starts on first GPU operation (no VRAM allocated at startup)"
            if resize_enabled
            else "disabled by PICTOR_GPU=off; ML GPU work remains isolated"
        )
        self._disabled_reason = ""
        self._device_info: list[dict[str, object]] = []

    @property
    def initialized(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    def available(self) -> bool:
        """Whether GPU work may be attempted, without starting CUDA."""
        return self.resize_enabled and self._worker_available is not False and not self._disabled_reason

    def status(self) -> BackendStatus:
        # These are immutable snapshots/references. Do not take the IPC lock here:
        # image_capabilities runs on the event loop and must not wait for a GPU job.
        process = self._process
        if self._worker_available is True and process is not None and process.is_alive():
            return BackendStatus(
                self.name,
                "resize",
                True,
                device=self.device,
                detail=self._detail,
            )
        return BackendStatus(
            self.name,
            "resize",
            False,
            device=self.device,
            detail=self._detail,
            disabled_reason=self._disabled_reason,
        )

    @property
    def device_info(self) -> list[dict[str, object]]:
        info = self._device_info
        return [dict(row) for row in info]

    def begin_tool_call(self) -> None:
        """Keep a live worker while any MCP tool call is in flight."""
        with self._activity_lock:
            self._active_tool_calls += 1
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def end_tool_call(self) -> None:
        with self._activity_lock:
            self._active_tool_calls = max(0, self._active_tool_calls - 1)
        self._schedule_idle_shutdown()

    def resize(self, image: Image.Image, size: tuple[int, int], pil_filter: Image.Resampling) -> Image.Image | None:
        """Resize in the worker, or return None for the guaranteed Pillow fallback."""
        if image.mode not in {"L", "LA", "RGB", "RGBA"} or min(size) < 1:
            return None
        with self._lock:
            reply = self._request_locked(
                ("resize", image.mode, image.size, image.tobytes(), size, int(pil_filter)),
            )
            if reply is None or reply[0] == "fallback":
                self._schedule_idle_shutdown()
                return None
            if reply[0] != "image":
                logger.warning("isolated CUDA resize failed; using Pillow: %s", reply[1:])
                self._schedule_idle_shutdown()
                return None
            _, mode, result_size, raw = reply
            self._schedule_idle_shutdown()
            return Image.frombytes(mode, result_size, raw)

    def remove_background_ml(
        self,
        image: Image.Image,
        *,
        model: str,
        background: tuple[int, int, int, int] | None = None,
    ) -> tuple[Image.Image, list[str]]:
        """Run rembg/ONNX in the isolated worker so its CUDA provider is also released."""
        with self._lock:
            reply = self._request_locked(
                ("remove_background", image.mode, image.size, image.tobytes(), model, background)
            )
            self._schedule_idle_shutdown()
        if reply is None or reply[0] != "image":
            raise BackendUnavailableError(
                "ML background removal is unavailable in the isolated GPU worker",
                hint="check the server log for worker startup details, or use method='color'",
            )
        _, mode, size, raw, notes = reply
        return Image.frombytes(mode, size, raw), list(notes)

    def gpu_device_info(self) -> list[dict[str, object]]:
        """Return cached device details without probing CUDA from the server process."""
        return self.device_info

    def close(self) -> None:
        """Terminate the worker, releasing its CUDA context and VRAM."""
        with self._lock:
            self._cancel_timer()
            self._terminate_locked()

    def _request_locked(self, request: tuple[Any, ...]) -> tuple[Any, ...] | None:
        self._cancel_timer()
        if not self._ensure_worker_locked():
            self._schedule_idle_shutdown()
            return None
        try:
            assert self._connection is not None
            self._connection.send(request)
            if not self._connection.poll(self.startup_timeout):
                logger.warning("isolated CUDA worker timed out; falling back to CPU")
                self._terminate_locked()
                self._disabled_reason = "GPU worker timed out"
                self._detail = self._disabled_reason
                return None
            reply = self._connection.recv()
            if reply and reply[0] == "error":
                error_type = reply[1] if len(reply) > 1 else "unknown error"
                logger.warning("isolated CUDA worker operation failed (%s)", error_type)
                if len(reply) > 1 and reply[1] == "OutOfMemoryError":
                    return None
            return reply
        except (EOFError, BrokenPipeError, OSError) as exc:
            logger.warning("isolated CUDA worker exited unexpectedly: %s", type(exc).__name__)
            self._terminate_locked()
            self._disabled_reason = "GPU worker exited unexpectedly"
            self._detail = self._disabled_reason
            return None

    def _ensure_worker_locked(self) -> bool:
        if self._disabled_reason:
            return False
        if self._process is not None and self._process.is_alive() and self._connection is not None:
            return True
        self._terminate_locked()
        try:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe(duplex=True)
            from ._cuda_worker import run

            process = context.Process(
                target=run,
                args=(child, self.resize_enabled),
                name="pictor-cuda-worker",
                daemon=True,
            )
            process.start()
            child.close()
            self._process = process
            self._connection = parent
            if not parent.poll(self.startup_timeout):
                raise TimeoutError("worker startup timed out")
            ready = parent.recv()
            if not ready or ready[0] != "ready":
                details = ready[1:] if ready else ("empty startup response",)
                raise RuntimeError("; ".join(str(item) for item in details[:2]))
            _, status, devices = ready
            self._detail = str(status.get("detail") or "CUDA worker initialized")
            self._device_info = list(devices)
            self._worker_available = bool(status.get("available"))
            if not self._worker_available:
                self._detail = str(status.get("disabledReason") or self._detail)
            logger.info("isolated GPU worker ready: %s", self._detail)
            # Keep the process alive even when CUDA resize is unavailable: the
            # optional ONNX background-removal provider may still be usable.
            return True
        except Exception as exc:
            self._terminate_locked()
            self._worker_available = False
            self._disabled_reason = f"GPU worker could not start ({type(exc).__name__})"
            self._detail = self._disabled_reason
            logger.info("GPU acceleration unavailable: %s", self._disabled_reason)
            return False

    def _schedule_idle_shutdown(self) -> None:
        with self._activity_lock:
            previous, self._timer = self._timer, None
            should_stop_now = self._process is not None and self._active_tool_calls == 0 and self.idle_seconds <= 0
            if self._process is not None and self._active_tool_calls == 0 and self.idle_seconds > 0:
                timer = threading.Timer(self.idle_seconds, self._on_idle_timeout)
                timer.daemon = True
                self._timer = timer
            else:
                timer = None
        if previous is not None:
            previous.cancel()
        if timer is not None:
            timer.start()
        elif should_stop_now:
            self.close()

    def _on_idle_timeout(self) -> None:
        with self._lock, self._activity_lock:
            self._timer = None
            if self._active_tool_calls == 0:
                logger.info("GPU idle for %.1f seconds; stopping isolated CUDA worker", self.idle_seconds)
                self._terminate_locked()

    def _cancel_timer(self) -> None:
        with self._activity_lock:
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _terminate_locked(self) -> None:
        process, self._process = self._process, None
        connection, self._connection = self._connection, None
        self._worker_available = None
        if connection is not None:
            with suppress(OSError):
                connection.close()
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        self._device_info = []
        if process is not None and not self._disabled_reason:
            self._detail = "worker stopped; CUDA context released and no VRAM retained"


__all__ = ["IsolatedCudaBackend"]

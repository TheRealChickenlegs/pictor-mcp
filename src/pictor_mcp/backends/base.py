"""Resampling backends.

The interface is deliberately tiny - one method - because the only operation
where a GPU wins decisively for this workload is resampling. Encoding stays on
the CPU (libjpeg-turbo and libwebp are already highly optimised and have no
CUDA path in Pillow), and so does every geometric operation that is memory-bound
rather than compute-bound.

A backend is an *accelerator*, never a requirement. Every method returns
``None`` to mean "I could not do this", and the caller falls back to Pillow.
That contract is what makes the GPU genuinely optional: a broken driver, an
out-of-memory card, or an unsupported pixel format degrades throughput, never
correctness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """What a backend is and whether it is usable."""

    name: str
    kind: str
    available: bool
    device: str = "cpu"
    detail: str = ""
    #: Set when a self-test rejected the backend.
    disabled_reason: str = ""

    def to_public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "available": self.available,
            "device": self.device,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.disabled_reason:
            payload["disabledReason"] = self.disabled_reason
        return payload


@runtime_checkable
class ResizeBackend(Protocol):
    """Accelerator for :func:`PIL.Image.Image.resize`."""

    name: str
    device: str

    def available(self) -> bool: ...

    def resize(self, image: Image.Image, size: tuple[int, int], pil_filter: Image.Resampling) -> Image.Image | None:
        """Resize, or return ``None`` to defer to Pillow."""
        ...


@dataclass(slots=True)
class BackendRegistry:
    """Chooses a backend per operation, with CPU as the guaranteed floor."""

    backends: list[ResizeBackend] = field(default_factory=list)
    statuses: list[BackendStatus] = field(default_factory=list)
    #: Images smaller than this are handled on the CPU; device transfer and
    #: kernel-launch overhead dominate below roughly this size.
    min_pixels_for_gpu: int = 4_000_000
    #: Upper bound on what we are willing to move to the device at once.
    max_pixels_for_gpu: int = 64_000_000
    gpu_worker: object | None = None

    def resampler_for(self, image: Image.Image):
        """Return a backend able to resize ``image`` now, or ``None`` for Pillow."""
        pixels = image.width * image.height
        if pixels < self.min_pixels_for_gpu or pixels > self.max_pixels_for_gpu:
            return None
        for backend in self.backends:
            try:
                if backend.available():
                    return backend
            except Exception:  # pragma: no cover - defensive
                logger.warning("backend %s failed its availability check", backend.name, exc_info=True)
        return None

    @property
    def active(self) -> str:
        for backend in self.backends:
            try:
                if backend.available():
                    if getattr(backend, "initialized", True):
                        return f"{backend.name} ({backend.device})"
                    return "pillow (cpu; CUDA worker is lazy and not using VRAM)"
            except Exception:  # pragma: no cover - defensive
                logger.warning("backend %s failed its availability check", backend.name, exc_info=True)
                continue
        return "pillow (cpu)"

    def _lifecycle_backends(self) -> list[object]:
        candidates = [*self.backends, *([self.gpu_worker] if self.gpu_worker else [])]
        unique: list[object] = []
        for backend in candidates:
            if not any(existing is backend for existing in unique):
                unique.append(backend)
        return unique

    def begin_tool_call(self) -> None:
        for backend in self._lifecycle_backends():
            callback = getattr(backend, "begin_tool_call", None)
            if callable(callback):
                callback()

    def end_tool_call(self) -> None:
        for backend in self._lifecycle_backends():
            callback = getattr(backend, "end_tool_call", None)
            if callable(callback):
                callback()

    def gpu_device_info(self) -> list[dict[str, object]]:
        for backend in self._lifecycle_backends():
            callback = getattr(backend, "gpu_device_info", None)
            if callable(callback):
                return cast(list[dict[str, object]], callback())
        return []

    def remove_background_ml(self, image: Image.Image, *, model: str, background=None):
        callback = getattr(self.gpu_worker, "remove_background_ml", None)
        if callable(callback):
            return callback(image, model=model, background=background)
        from ..imaging.background import remove_background_ml

        return remove_background_ml(image, model=model, background=background)

    def to_public_dict(self) -> dict[str, object]:
        statuses = []
        for backend in self._lifecycle_backends():
            callback = getattr(backend, "status", None)
            if callable(callback):
                statuses.append(callback())
        statuses.extend(status for status in self.statuses if status.name == "pillow")
        report: dict[str, object] = {
            "active": self.active,
            "minPixelsForAcceleration": self.min_pixels_for_gpu,
            "maxPixelsForAcceleration": self.max_pixels_for_gpu,
            "backends": [status.to_public_dict() for status in statuses],
        }
        if self.gpu_worker is not None:
            report["gpuWorkerIsolation"] = "separate process; terminated after idle timeout to release CUDA context"
            report["gpuWorkerIdleSeconds"] = getattr(self.gpu_worker, "idle_seconds", None)
        return report


__all__ = ["BackendRegistry", "BackendStatus", "ResizeBackend"]

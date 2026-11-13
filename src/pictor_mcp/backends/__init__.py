"""Backend discovery without initializing CUDA in the server process.

GPU discovery and self-test are deferred to the isolated worker's first use and
reported through capabilities after that operation. Startup remains VRAM-free.
"""

from __future__ import annotations

import logging
from typing import cast

from PIL import __version__ as _pillow_version

from ..config import Config
from .base import BackendRegistry, BackendStatus, ResizeBackend
from .isolated_cuda import IsolatedCudaBackend

logger = logging.getLogger(__name__)


def build_registry(config: Config) -> BackendRegistry:
    """Build an accelerator registry with a CPU floor, without probing CUDA."""
    backends: list[ResizeBackend] = []
    statuses: list[BackendStatus] = []

    # This proxy imports no CUDA libraries and starts no worker. Device discovery
    # and the CUDA self-test happen in the child on first eligible resize, so
    # building the server consumes no VRAM. It also owns isolated ONNX ML calls.
    cuda = IsolatedCudaBackend(
        idle_seconds=config.gpu_idle_seconds,
        resize_enabled=config.gpu != "off",
    )
    if config.gpu != "off":
        backends.append(cuda)
        logger.info("GPU acceleration configured for lazy isolated startup")
    statuses.append(cuda.status())

    statuses.append(
        BackendStatus(
            name="pillow",
            kind="resize",
            available=True,
            device="cpu",
            detail=f"Pillow {_pillow_version} (always available fallback)",
        )
    )

    return BackendRegistry(
        backends=backends,
        statuses=statuses,
        min_pixels_for_gpu=config.gpu_min_pixels,
        gpu_worker=cuda,
    )


def gpu_device_info(registry: BackendRegistry | None = None) -> list[dict[str, object]]:
    """Return cached GPU inventory without importing or probing CUDA in-process."""
    if registry is None:
        return []
    for backend in registry.backends:
        info = getattr(backend, "gpu_device_info", None)
        if callable(info):
            return cast(list[dict[str, object]], info())
    return []


__all__ = ["BackendRegistry", "BackendStatus", "ResizeBackend", "build_registry", "gpu_device_info"]

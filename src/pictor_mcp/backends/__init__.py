"""Backend discovery.

Detection happens once at startup and is reported through the capabilities
tool, so an operator can confirm from their agent whether the GPU is actually
in use rather than guessing from logs.
"""

from __future__ import annotations

import logging

from PIL import __version__ as _pillow_version

from ..config import Config
from .base import BackendRegistry, BackendStatus, ResizeBackend
from .torch_cuda import TorchCudaBackend

logger = logging.getLogger(__name__)


def build_registry(config: Config) -> BackendRegistry:
    """Probe for accelerators and return a registry with a CPU floor."""
    backends: list[ResizeBackend] = []
    statuses: list[BackendStatus] = []

    if config.gpu == "off":
        statuses.append(
            BackendStatus(
                name="torch-cuda",
                kind="resize",
                available=False,
                detail="disabled by PICTOR_GPU=off",
            )
        )
    else:
        cuda = TorchCudaBackend()
        status = cuda.status()
        statuses.append(status)
        if status.available:
            backends.append(cuda)
            logger.info("GPU acceleration enabled: %s", status.detail)
        elif status.disabled_reason:
            # A device was found and then rejected. That is a real problem the
            # operator wants to fix, not a fact about the machine, so it is a
            # warning rather than a note - the server runs correctly without it,
            # but it is not running the way it was configured to.
            logger.warning("GPU acceleration unavailable: %s", status.disabled_reason)
        else:
            # No GPU here, or torch is not installed. Informational: this is the
            # expected state for the CPU image.
            logger.info("GPU acceleration unavailable: %s", status.detail or "unavailable")

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
    )


def gpu_device_info() -> list[dict[str, object]]:
    """Best-effort GPU inventory for the capabilities report."""
    try:
        import torch
    except ImportError:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "computeCapability": f"{properties.major}.{properties.minor}",
                    "totalMemoryBytes": total,
                    "freeMemoryBytes": free,
                }
            )
        return devices
    except Exception:  # pragma: no cover - driver-dependent
        logger.debug("could not enumerate CUDA devices", exc_info=True)
        return []


__all__ = ["BackendRegistry", "BackendStatus", "ResizeBackend", "build_registry", "gpu_device_info"]

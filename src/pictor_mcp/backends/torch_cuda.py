"""CUDA resampling via PyTorch.

Why PyTorch rather than DALI or cuCIM: it is the only one of the three that is
pip-installable for every current CUDA runtime, is already present in most
ML-enabled homelabs, and exposes a stable ``F.interpolate`` that maps cleanly
onto Pillow's resampling semantics. DALI is faster for bulk decode but pins
hard to specific CUDA versions and owns its own thread pool; cuCIM is a smaller
dependency but adds a RAPIDS channel for one function.

Honesty about limits, because a silently-wrong resize is worse than a slow one:

* **Lanczos is approximated.** Torch has no Lanczos kernel. Downscaling with
  ``bicubic`` + ``antialias=True`` is close but not bit-identical, and the
  caller is told so in the result notes.
* **Palette images stay on the CPU.** Quantised images have no meaningful
  float representation to interpolate.
* **The device is only used when it wins.** Below a pixel threshold the
  host-to-device copy costs more than the resize.

A self-test runs once at startup: a small image is resized on both paths and
compared. If the results diverge beyond tolerance the backend disables itself,
so a broken kernel or a bad driver cannot quietly corrupt output.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

from .base import BackendStatus

logger = logging.getLogger(__name__)

#: Maximum mean absolute channel difference tolerated by the self-test.
_SELF_TEST_TOLERANCE = 6.0

#: Pillow resampling -> torch interpolation mode.
_TORCH_MODES = {
    Image.Resampling.NEAREST: "nearest",
    Image.Resampling.BOX: "bilinear",
    Image.Resampling.BILINEAR: "bilinear",
    Image.Resampling.HAMMING: "bilinear",
    Image.Resampling.BICUBIC: "bicubic",
    Image.Resampling.LANCZOS: "bicubic",
}

#: Pillow modes with a straight channel mapping.
_SUPPORTED_MODES = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4}


class TorchCudaBackend:
    """Resizes images on a CUDA device using ``torch.nn.functional.interpolate``."""

    name = "torch-cuda"

    def __init__(self, *, device_index: int = 0) -> None:
        self._device_index = device_index
        self._torch = None
        self._device = "cuda"
        self._lock = threading.Lock()
        self._available = False
        self._detail = ""
        self._disabled_reason = ""
        self._load()

    # ------------------------------------------------------------- lifecycle
    def _load(self) -> None:
        try:
            import torch
        except ImportError:
            self._detail = "torch is not installed (install the 'gpu' extra)"
            return
        try:
            if not torch.cuda.is_available():
                self._detail = "torch is installed but CUDA is not available to this process"
                return
            if torch.cuda.device_count() <= self._device_index:
                self._detail = f"device index {self._device_index} does not exist"
                return
            self._torch = torch
            self._device = f"cuda:{self._device_index}"
            name = torch.cuda.get_device_name(self._device_index)
            self._detail = f"{name} (torch {torch.__version__}, cuda {torch.version.cuda})"
            self._run_self_test()
            self._available = True
        except Exception as exc:  # pragma: no cover - driver-dependent
            self._disabled_reason = f"initialisation failed: {type(exc).__name__}"
            logger.warning("CUDA backend disabled: %s", exc, exc_info=True)

    def _run_self_test(self) -> None:
        """Verify GPU and CPU resampling agree before trusting the GPU.

        A driver or kernel problem shows up as a numerically wrong image, not an
        exception, so a tolerance check is the only way to catch it.
        """
        rng = np.random.default_rng(1234)
        sample = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        reference_source = Image.fromarray(sample, "RGB")
        target = (37, 53)  # deliberately not a clean 2x ratio

        cpu_result = np.asarray(reference_source.resize(target, Image.Resampling.BILINEAR), dtype=np.float32)
        gpu_image = self.resize(reference_source, target, Image.Resampling.BILINEAR)
        if gpu_image is None:
            self._disabled_reason = "self-test could not run a GPU resize"
            return
        gpu_result = np.asarray(gpu_image, dtype=np.float32)

        if gpu_result.shape != cpu_result.shape:
            self._disabled_reason = "self-test produced a differently shaped result"
            return
        difference = float(np.mean(np.abs(gpu_result - cpu_result)))
        if difference > _SELF_TEST_TOLERANCE:
            self._disabled_reason = f"self-test mismatch against Pillow (mean abs diff {difference:.1f})"
            logger.error("CUDA backend disabled: %s", self._disabled_reason)

    def available(self) -> bool:
        return self._available and not self._disabled_reason

    @property
    def device(self) -> str:
        return self._device

    def status(self) -> BackendStatus:
        return BackendStatus(
            name=self.name,
            kind="resize",
            available=self.available(),
            device=self._device,
            detail=self._detail,
            disabled_reason=self._disabled_reason,
        )

    # --------------------------------------------------------------- resizing
    def resize(self, image: Image.Image, size: tuple[int, int], pil_filter: Image.Resampling) -> Image.Image | None:
        if not self.available() or self._torch is None:
            return None
        if image.mode not in _SUPPORTED_MODES:
            return None
        if size[0] < 1 or size[1] < 1:
            return None

        mode = _TORCH_MODES.get(pil_filter, "bilinear")
        torch = self._torch
        try:
            with self._lock:  # one device transfer at a time keeps peak memory bounded
                array = np.asarray(image)
                # (H, W, C) uint8 -> (1, C, H, W) float32
                tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device=self._device, dtype=torch.float32)
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) if array.ndim == 3 else (tensor.unsqueeze(0).unsqueeze(0))
                with torch.inference_mode():
                    resized = torch.nn.functional.interpolate(
                        tensor,
                        size=(size[1], size[0]),
                        mode=mode,
                        # Antialiasing is what keeps downscaling from aliasing;
                        # without it bicubic on a 4x reduction looks worse than
                        # Pillow's Lanczos by a wide margin.
                        antialias=mode != "nearest",
                    )
                resized = resized.clamp_(0, 255).to(torch.uint8)
                result = resized.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        except Exception as exc:
            # Never let an accelerator failure surface as a tool error: the CPU
            # path can always produce the answer.
            if type(exc).__name__ == "OutOfMemoryError":
                logger.warning("CUDA out of memory during resize; falling back to CPU")
                self._empty_cache()
            else:
                logger.warning("CUDA resize failed (%s); falling back to CPU", exc)
            return None

        if result.ndim == 2:
            return Image.fromarray(result, "L")
        channels = result.shape[2]
        mode_name = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}.get(channels)
        if mode_name is None:  # pragma: no cover - guarded by _SUPPORTED_MODES
            return None
        return Image.fromarray(result, mode_name)

    def _empty_cache(self) -> None:
        try:
            if self._torch is not None:
                self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            logger.debug("cuda cache flush failed", exc_info=True)


__all__ = ["TorchCudaBackend"]

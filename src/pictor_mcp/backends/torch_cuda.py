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
import re
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


def _capability(tag: str) -> tuple[int, int] | None:
    """``(major, minor)`` for an arch tag, or None if it is not one.

    Handles every form PyTorch emits: ``sm_90``, ``sm_120``, the arch-specific
    ``sm_120a``, and the PTX ``compute_120``. Tuples compare in the right order,
    so ``(12, 0) > (9, 0)`` without any packing arithmetic.
    """
    match = re.match(r"(?:sm|compute)_(\d+)", tag.strip().lower())
    if not match:
        return None
    digits = match.group(1)
    if len(digits) >= 3:
        return int(digits[:-1]), int(digits[-1])
    if len(digits) == 2:
        return int(digits[0]), int(digits[1])
    return int(digits), 0


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
        #: Why the last resize attempt failed, so the self-test can report a
        #: cause rather than only that nothing ran.
        self._last_error = ""
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

            # torch is importable and a CUDA device is present, so the backend is
            # usable *in principle*. This is set before the checks below rather
            # than after, because resize() refuses to touch the device unless
            # available() is true - and the self-test resizes. Setting it last
            # meant the self-test always found the backend unavailable, gave up
            # with "could not run a GPU resize", and disabled itself: the GPU path
            # was dead code on every machine, working card or not.
            self._available = True

            # Check the architecture before the self-test, because the self-test
            # can only say "the resize did not run" - it cannot say why, and the
            # usual reason is the most confusing one: torch is installed
            # correctly, sees the card, and is simply not built for it.
            unsupported = self._unsupported_architecture()
            if unsupported:
                self._disabled_reason = unsupported
                logger.warning("GPU acceleration unavailable: %s", unsupported)
                return

            self._run_self_test()
        except Exception as exc:  # pragma: no cover - driver-dependent
            self._disabled_reason = f"initialisation failed: {type(exc).__name__}"
            logger.warning("CUDA backend disabled: %s", exc, exc_info=True)

    def _unsupported_architecture(self) -> str:
        """Explain an architecture mismatch, or return an empty string.

        A PyTorch build only contains kernels for the compute capabilities it was
        compiled for. When the installed card is newer than the build - an RTX 50
        series (sm_120) against a pre-CUDA-12.8 wheel, say - torch imports, sees
        the device, and *then* fails at the first kernel launch. Torch's own
        warning says the capabilities it supports but not what to do about it,
        and the failure surfaces here as an opaque "resize did not run".

        Comparing the two lists turns that into a message naming the card, the
        capability, what the build does support, and the fix.
        """
        torch = self._torch
        try:
            major, minor = torch.cuda.get_device_capability(self._device_index)
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:  # pragma: no cover - driver-dependent
            return ""

        if not arch_list:
            return ""

        wanted = f"sm_{major}{minor}"
        tags = {entry.strip().lower() for entry in arch_list}
        supported = {cap for tag in tags if (cap := _capability(tag)) is not None}
        if (major, minor) in supported:
            return ""

        name = self._detail or "the installed GPU"
        ordered = sorted(supported)

        # Direction matters. A card newer than everything in the list means the
        # build predates the hardware; a card older than everything in the list
        # means the build dropped that architecture. Advising the wrong direction
        # sends the operator to change the one thing that will not help.
        if ordered and (major, minor) > ordered[-1]:
            remedy = (
                "This build predates that architecture. Rebuild with a CUDA 12.8 or "
                "newer wheel index - TORCH_INDEX_URL="
                "https://download.pytorch.org/whl/cu128 is the image default - and "
                "leave TORCH_VERSION empty so it resolves for the base image's Python."
            )
        elif ordered and (major, minor) < ordered[0]:
            remedy = (
                "This build no longer includes that architecture. Rebuild with an "
                "older CUDA line, e.g. TORCH_INDEX_URL="
                "https://download.pytorch.org/whl/cu126, and expect to pair it with "
                "an older base image if no wheel exists for the current Python."
            )
        else:
            remedy = "Rebuild with a wheel index whose kernels cover this card; TORCH_INDEX_URL selects it."
        return (
            f"{name} is {wanted}, which this PyTorch build has no kernels for; "
            f"it supports {', '.join(sorted(tags))}. {remedy}"
        )

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
            cause = f": {self._last_error}" if self._last_error else ""
            self._disabled_reason = f"self-test could not run a GPU resize{cause}"
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

                # torch.from_numpy requires a writable buffer, and a PIL image
                # exposes a read-only one. np.ascontiguousarray returns the same
                # object when the array is already contiguous - which it is,
                # coming straight from Pillow - so it does not copy and the
                # read-only flag survives into torch, which warns and reserves
                # the right to produce undefined behaviour on write.
                if not array.flags.c_contiguous or not array.flags.writeable:
                    array = np.array(array, copy=True, order="C")

                # (H, W, C) uint8 -> (1, C, H, W) float32
                tensor = torch.from_numpy(array).to(device=self._device, dtype=torch.float32)
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) if array.ndim == 3 else (tensor.unsqueeze(0).unsqueeze(0))

                # The whole conversion lives inside inference_mode. Tensors made
                # there are "inference tensors", and torch forbids in-place
                # updates to them once the mode has exited:
                #
                #   RuntimeError: Inplace update to inference tensor outside
                #   InferenceMode is not allowed.
                #
                # `clamp` is therefore the out-of-place form, so the code is
                # correct even if a later edit moves a line back out of the block.
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
                    resized = torch.clamp(resized, 0, 255).to(torch.uint8)
                    result = resized.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        except Exception as exc:
            # Never let an accelerator failure surface as a tool error: the CPU
            # path can always produce the answer.
            if type(exc).__name__ == "OutOfMemoryError":
                self._last_error = "the device ran out of memory"
                logger.warning("CUDA out of memory during resize; falling back to CPU")
                self._empty_cache()
            else:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("CUDA resize failed (%s); falling back to CPU", exc)
            return None

        if result.ndim == 2:
            return Image.fromarray(result, "L")

        channels = result.shape[2]
        if channels == 1:
            # The permute above always yields (H, W, C), so a grayscale result
            # arrives as (H, W, 1) - which Pillow rejects for mode "L". Without
            # this the GPU path raised on every single-channel image and fell
            # back to the CPU, so grayscale was never actually accelerated.
            return Image.fromarray(result[:, :, 0], "L")

        mode_name = {2: "LA", 3: "RGB", 4: "RGBA"}.get(channels)
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

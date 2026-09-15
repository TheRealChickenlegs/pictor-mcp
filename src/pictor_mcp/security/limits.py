"""Decode-time resource guards.

Image formats are attacker-controlled input even on an internal network, and
the classic failure modes are cheap to trigger and expensive to survive:

* **Decompression bombs** - a 40 KiB PNG that expands to 64 GiB of pixels
  (the "billion laughs" of raster formats). Caught by a pixel ceiling that is
  enforced *before* pixel data is materialised, plus Pillow's own
  ``MAX_IMAGE_PIXELS`` as a second line of defence.
* **Dimension overflows** - a huge single axis, which some C decoders handle
  less gracefully than a large area.
* **Animation amplification** - a GIF declaring thousands of frames, each
  fully decoded.
* **Truncated / malformed streams** - handled by the decoder, but the errors
  are normalised so they cannot leak library internals.

Nothing here trusts a header. The header is used to *reject* work, never to
authorise it; anything that passes is still decoded inside a bounded budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFile

from ..config import Limits
from ..errors import LimitExceededError


def configure_pillow(limits: Limits) -> None:
    """Apply process-wide Pillow safety limits.

    Pillow warns above ``MAX_IMAGE_PIXELS`` and raises ``DecompressionBombError``
    above twice that. We set the ceiling to our own limit so the library's guard
    sits just behind ours as a backstop; :func:`check_decoded_geometry` is the
    primary check and runs before any pixel data is materialised.
    """
    Image.MAX_IMAGE_PIXELS = limits.max_pixels
    # A truncated file should fail loudly rather than decode into a
    # partially-populated image, which would silently corrupt output.
    ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass(frozen=True, slots=True)
class Geometry:
    width: int
    height: int
    frames: int = 1

    @property
    def pixels(self) -> int:
        return self.width * self.height


def check_decoded_geometry(geo: Geometry, limits: Limits, *, context: str = "input") -> None:
    """Reject an image whose declared geometry is too large to decode."""
    if geo.width <= 0 or geo.height <= 0:
        raise LimitExceededError(f"{context} image has invalid dimensions {geo.width}x{geo.height}")
    if geo.width > limits.max_dimension or geo.height > limits.max_dimension:
        raise LimitExceededError(
            f"{context} image is {geo.width}x{geo.height}, exceeding the {limits.max_dimension}px per-axis limit",
            width=geo.width,
            height=geo.height,
            limit=limits.max_dimension,
        )
    if geo.pixels > limits.max_pixels:
        raise LimitExceededError(
            f"{context} image has {geo.pixels} pixels, exceeding the {limits.max_pixels} pixel limit "
            "(possible decompression bomb)",
            pixels=geo.pixels,
            limit=limits.max_pixels,
        )
    if geo.frames > limits.max_frames:
        raise LimitExceededError(
            f"{context} image has {geo.frames} frames, exceeding the {limits.max_frames} frame limit",
            frames=geo.frames,
            limit=limits.max_frames,
        )
    # The per-frame and per-frame-count limits together still permit an
    # unbounded *product*, and a multi-frame image is decoded into one full
    # bitmap per frame before it is re-encoded. Bound the total.
    if geo.frames > 1:
        total_pixels = geo.pixels * geo.frames
        if total_pixels > limits.max_animation_pixels:
            raise LimitExceededError(
                f"{context} animation is {geo.width}x{geo.height} across {geo.frames} frames, "
                f"a total of {total_pixels} pixels, exceeding the "
                f"{limits.max_animation_pixels} pixel animation budget",
                pixels=total_pixels,
                limit=limits.max_animation_pixels,
            )


def check_output_geometry(width: int, height: int, limits: Limits) -> None:
    """Reject a *requested* output size that would exhaust memory.

    Checked before the operation runs so that a request to upscale a small
    image to 200000x200000 fails immediately instead of after allocating.
    """
    if width <= 0 or height <= 0:
        raise LimitExceededError(f"requested output size {width}x{height} is not positive")
    if width > limits.max_dimension or height > limits.max_dimension:
        raise LimitExceededError(
            f"requested output size {width}x{height} exceeds the {limits.max_dimension}px per-axis limit",
            limit=limits.max_dimension,
        )
    pixels = width * height
    if pixels > limits.max_output_pixels:
        raise LimitExceededError(
            f"requested output size {width}x{height} is {pixels} pixels, exceeding the "
            f"{limits.max_output_pixels} pixel output limit",
            pixels=pixels,
            limit=limits.max_output_pixels,
        )


def check_base64_size(data: bytes, limits: Limits) -> None:
    """Bound inline base64 payloads before decoding them."""
    if len(data) > limits.max_input_base64_bytes:
        raise LimitExceededError(
            f"inline image is {len(data)} bytes, exceeding the "
            f"{limits.max_input_base64_bytes} byte limit for inline data",
            size_bytes=len(data),
            limit_bytes=limits.max_input_base64_bytes,
        )


__all__ = [
    "Geometry",
    "check_base64_size",
    "check_decoded_geometry",
    "check_output_geometry",
    "configure_pillow",
]

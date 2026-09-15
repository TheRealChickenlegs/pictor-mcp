"""Background removal.

Two implementations, because "remove the background" means two different things
and only one of them needs a neural network:

``color``
    Chroma-key style. The dominant border colour is sampled and everything
    within a tolerance of it is made transparent. For the overwhelmingly common
    case - a product, logo or screenshot on a flat white or flat colour
    backdrop - this is exact, instant, deterministic, and works offline. It is
    also *edge-connected* by default: only background-coloured regions touching
    the border are removed, so a white shirt on a white background keeps its
    shirt.

``ml``
    ``rembg`` with a U²-Net-family model. Handles arbitrary backgrounds and
    hair, at the cost of a multi-hundred-megabyte model download and a GPU or
    a few seconds of CPU per image. Optional dependency; when absent the
    server says so precisely rather than failing obscurely.

``auto`` picks ML when the model is installed and falls back to colour, which
means the same tool call works in a minimal container and a full GPU image.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..errors import BackendUnavailableError, InvalidArgumentError

logger = logging.getLogger(__name__)

Method = Literal["auto", "color", "ml"]

#: Border sampling thickness, in pixels.
_BORDER_SAMPLE = 2

#: Long edge of the mask used for the connectivity flood fill.
_FLOOD_MAX_EDGE = 512


def _border_colour(rgb: np.ndarray) -> np.ndarray:
    """Median colour of a thin frame around the image.

    Median rather than mean so a few foreground pixels touching the border (a
    subject that bleeds off-frame) cannot drag the estimate.
    """
    frame = np.concatenate(
        [
            rgb[:_BORDER_SAMPLE, :, :].reshape(-1, 3),
            rgb[-_BORDER_SAMPLE:, :, :].reshape(-1, 3),
            rgb[:, :_BORDER_SAMPLE, :].reshape(-1, 3),
            rgb[:, -_BORDER_SAMPLE:, :].reshape(-1, 3),
        ]
    )
    return np.median(frame, axis=0)


def _soft_alpha(distance: np.ndarray, tolerance: float, softness: float) -> np.ndarray:
    """Map colour distance to alpha with a linear ramp at the edge.

    A hard threshold produces jagged edges on anti-aliased boundaries; a ramp
    over ``softness`` levels gives a clean cutout without a separate feather
    pass.
    """
    if softness <= 0:
        alpha = np.where(distance <= tolerance, 0.0, 255.0)
    else:
        low = tolerance - softness / 2.0
        high = tolerance + softness / 2.0
        ramp = (distance - low) / max(high - low, 1e-6)
        alpha = np.clip(ramp, 0.0, 1.0) * 255.0
    return alpha


def _edge_connected_mask(mask: Image.Image) -> Image.Image:
    """Return a *binary* mask of candidate regions reachable from the border.

    The result is strictly 255 (reached) or 0 (not reached). Returning a
    three-valued mask instead - candidate, reached-candidate, foreground - is a
    trap: the "candidate but unreached" value must end up *outside* the removal
    mask, and any linear ramp mapping the reached value to 1 tends to map the
    unreached candidate value to 1 as well. That silently deletes exactly the
    interior regions this function exists to protect.
    """
    flooded = mask.copy()
    width, height = flooded.size
    # 128 marks "reached from the border". Starting from every border pixel is
    # cheap because ImageDraw.floodfill returns immediately on an already
    # filled pixel, so each region is traversed exactly once.
    for x in range(width):
        for y in (0, height - 1):
            if flooded.getpixel((x, y)) == 255:
                ImageDraw.floodfill(flooded, (x, y), 128)
    for y in range(height):
        for x in (0, width - 1):
            if flooded.getpixel((x, y)) == 255:
                ImageDraw.floodfill(flooded, (x, y), 128)
    return flooded.point(lambda value: 255 if value == 128 else 0)


def remove_background_color(
    image: Image.Image,
    *,
    tolerance: float = 32.0,
    softness: float = 24.0,
    edge_connected: bool = True,
    background: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, list[str]]:
    """Make the border-connected dominant colour transparent."""
    if tolerance < 0:
        raise InvalidArgumentError("tolerance must not be negative")
    if softness < 0:
        raise InvalidArgumentError("softness must not be negative")

    notes: list[str] = []
    rgba = image.convert("RGBA")
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32)
    key = _border_colour(rgb)

    distance = np.sqrt(np.sum((rgb - key) ** 2, axis=2))
    alpha = _soft_alpha(distance, tolerance, softness)

    if edge_connected:
        # Connectivity is evaluated on a small binary mask and then upscaled:
        # the flood fill is the only O(pixels) Python loop in this module, and
        # running it at 512px instead of 6000px is the difference between
        # milliseconds and minutes.
        #
        # The candidate mask uses the *outer* edge of the soft ramp, so the
        # anti-aliased boundary is inside the connected region and keeps its
        # smooth alpha instead of being clipped to a hard edge.
        gate_threshold = tolerance + max(softness, 0.0) / 2.0
        candidate = Image.fromarray(np.where(distance <= gate_threshold, 255, 0).astype(np.uint8), "L")
        scale = min(1.0, _FLOOD_MAX_EDGE / max(candidate.size))
        small = (
            candidate.resize(
                (max(1, int(candidate.width * scale)), max(1, int(candidate.height * scale))),
                Image.Resampling.NEAREST,
            )
            if scale < 1.0
            else candidate
        )
        reached = _edge_connected_mask(small)
        if scale < 1.0:
            # Bilinear on a binary mask yields a soft 0..255 edge, which is what
            # gives the upscaled cutout a clean boundary instead of a staircase.
            reached = reached.resize(candidate.size, Image.Resampling.BILINEAR)
        connected = np.asarray(reached, dtype=np.float32) / 255.0

        # Gate, do not scale: alpha is *replaced* with opaque outside the
        # connected region. Multiplying instead would leave an interior region
        # that already scored as background at alpha 0 - the exact opposite of
        # the protection this option promises.
        alpha = np.where(connected > 0.5, alpha, 255.0)
        notes.append("removed border-connected background only")

    result = rgba.copy()
    result.putalpha(Image.fromarray(alpha.astype(np.uint8), "L"))
    notes.append(f"keyed out rgb({int(key[0])},{int(key[1])},{int(key[2])}) with tolerance {tolerance:g}")

    if background is not None:
        result = _composite_over(result, background)
        notes.append("composited onto the requested background")

    return result, notes


def _composite_over(rgba: Image.Image, background: tuple[int, int, int, int]) -> Image.Image:
    if len(background) == 3:
        background = (*background, 255)
    if background[3] == 0:
        return rgba
    backdrop = Image.new("RGBA", rgba.size, background)
    backdrop.alpha_composite(rgba)
    return backdrop


def ml_available() -> tuple[bool, str]:
    """Whether ``rembg`` and an ONNX runtime are importable."""
    try:
        import rembg  # noqa: F401
    except ImportError:
        return False, "rembg is not installed (install the 'bg' extra)"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False, "onnxruntime is not installed (install the 'bg' extra)"
    return True, "rembg + onnxruntime available"


def remove_background_ml(
    image: Image.Image,
    *,
    model: str = "u2net",
    background: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, list[str]]:
    """Remove the background with a U²-Net-family model via ``rembg``."""
    available, detail = ml_available()
    if not available:
        raise BackendUnavailableError(
            f"ML background removal is unavailable: {detail}",
            hint="use method='color' for flat backgrounds, or install the 'bg' extra",
        )

    from rembg import new_session, remove

    notes: list[str] = []
    try:
        session = new_session(model)
    except Exception as exc:
        raise BackendUnavailableError(
            f"background-removal model {model!r} could not be loaded: {exc}",
            hint="the model is downloaded on first use; pre-download it at build time for offline hosts",
        ) from exc

    try:
        result = remove(image.convert("RGB"), session=session)
    except Exception as exc:  # pragma: no cover - runtime/ONNX failure
        raise BackendUnavailableError(f"background removal failed: {exc}") from exc

    if not isinstance(result, Image.Image):  # pragma: no cover - rembg contract
        result = Image.open(result)
    result = result.convert("RGBA")
    notes.append(f"removed background with the {model} model")

    if background is not None:
        result = _composite_over(result, background)
        notes.append("composited onto the requested background")

    return result, notes


def remove_background(
    image: Image.Image,
    *,
    method: str = "auto",
    tolerance: float = 32.0,
    softness: float = 24.0,
    edge_connected: bool = True,
    model: str = "u2net",
    background: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, list[str]]:
    """Dispatch to the requested background-removal implementation."""
    key = (method or "auto").strip().lower()
    if key == "color":
        return remove_background_color(
            image,
            tolerance=tolerance,
            softness=softness,
            edge_connected=edge_connected,
            background=background,
        )
    if key == "ml":
        return remove_background_ml(image, model=model, background=background)
    if key == "auto":
        available, _ = ml_available()
        if available:
            try:
                return remove_background_ml(image, model=model, background=background)
            except BackendUnavailableError as exc:
                logger.info("ML background removal unavailable, using colour keying: %s", exc)
        result, notes = remove_background_color(
            image,
            tolerance=tolerance,
            softness=softness,
            edge_connected=edge_connected,
            background=background,
        )
        notes.append("method 'auto' selected the colour keyer")
        return result, notes
    raise InvalidArgumentError(
        f"unknown background-removal method {method!r}",
        supported=["auto", "color", "ml"],
    )


def feather_alpha(image: Image.Image, radius: float) -> Image.Image:
    """Soften a cutout's alpha channel by ``radius`` pixels."""
    if radius <= 0:
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius))
    rgba.putalpha(alpha)
    return rgba


__all__ = [
    "Method",
    "feather_alpha",
    "ml_available",
    "remove_background",
    "remove_background_color",
    "remove_background_ml",
]

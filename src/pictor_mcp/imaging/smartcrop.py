"""Content-aware cropping.

A centre crop on a photograph usually decapitates the subject. This module
finds the most *informative* window instead, using two cheap saliency proxies
that need no model and no downloads:

``attention``
    Gradient magnitude. Edges and texture are where the detail is, so the
    window with the most edge energy tends to hold the subject.
``entropy``
    Local standard deviation, which tracks busy regions rather than strong
    single edges - better for text, foliage and crowd scenes.

Both are computed on a downscaled copy and scored with an integral image, so
choosing the window costs one pass over a few tens of thousands of cells
regardless of the original size. A mild centre bias keeps the result sane when
the energy is uniform (a flat wall has no best window, and drifting to a corner
would be arbitrary).
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from ..errors import InvalidArgumentError
from .ops import Size

logger = logging.getLogger(__name__)

Method = str

#: Long edge of the working copy used for saliency analysis.
_ANALYSIS_EDGE = 256

#: Weight given to the centre when scoring windows. Keeps ties resolving
#: predictably instead of jumping to whichever corner has one noisy pixel.
_CENTRE_BIAS = 0.15


def _working_copy(image: Image.Image, max_edge: int = _ANALYSIS_EDGE) -> Image.Image:
    scale = min(1.0, max_edge / max(image.width, image.height))
    if scale >= 1.0:
        return image.convert("L")
    return image.convert("L").resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.BILINEAR,
    )


def _gradient_energy(gray: np.ndarray) -> np.ndarray:
    """Sobel-ish gradient magnitude."""
    array = gray.astype(np.float32)
    # np.gradient handles the interior; edges get a one-sided difference.
    grad_y, grad_x = np.gradient(array)
    magnitude = np.hypot(grad_x, grad_y)
    # A gentle blur prevents a single high-contrast pixel from dominating.
    return _box_blur(magnitude, radius=2)


def _entropy_energy(gray: np.ndarray) -> np.ndarray:
    """Local standard deviation as a stand-in for local entropy."""
    array = gray.astype(np.float32)
    radius = 3
    mean = _box_blur(array, radius)
    mean_square = _box_blur(array * array, radius)
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return np.sqrt(variance)


def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via a summed-area table (no scipy needed)."""
    if radius < 1:
        return array
    padded = np.pad(array.astype(np.float64), radius, mode="edge")
    window = 2 * radius + 1
    return _window_sums(_integral(padded), window, window) / (window * window)


def _integral(array: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero border, shape ``(h+1, w+1)``.

    ``integral[a, b]`` is the sum of ``array[0:a, 0:b]``, which makes every
    rectangular window sum a four-term expression.
    """
    table = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = array.cumsum(0).cumsum(1)
    return table


def _window_sums(integral: np.ndarray, width: int, height: int) -> np.ndarray:
    """Sum over every ``height x width`` window, returned as a 2D score map."""
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _centre_weight(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, cols, dtype=np.float32)[None, :]
    distance = np.sqrt(x * x + y * y) / np.sqrt(2.0)
    return (1.0 - _CENTRE_BIAS * distance).astype(np.float32)


def saliency_map(image: Image.Image, method: str = "attention") -> np.ndarray:
    """Return a normalised 2D saliency map for ``image``."""
    gray = _working_copy(image)
    array = np.asarray(gray)
    key = (method or "attention").strip().lower()
    if key in {"attention", "edges", "gradient", "detail"}:
        energy = _gradient_energy(array)
    elif key in {"entropy", "variance", "busy"}:
        energy = _entropy_energy(array)
    elif key in {"center", "centre", "none"}:
        energy = np.ones_like(array, dtype=np.float32)
    else:
        raise InvalidArgumentError(
            f"unknown smart-crop method {method!r}",
            supported=["attention", "entropy", "center"],
        )
    total = float(energy.sum())
    if total <= 0:
        return np.ones_like(energy, dtype=np.float32)
    return energy / total


def best_window(
    image: Image.Image,
    target: Size,
    *,
    method: str = "attention",
) -> tuple[int, int, int, int]:
    """Return the pixel box of the best ``target``-sized window."""
    if target.width > image.width or target.height > image.height:
        # Cannot crop a window larger than the source; clamp to the whole image.
        return (0, 0, image.width, image.height)

    energy = saliency_map(image, method)
    rows, cols = energy.shape
    scale_x = cols / image.width
    scale_y = rows / image.height
    win_w = max(1, min(cols, round(target.width * scale_x)))
    win_h = max(1, min(rows, round(target.height * scale_y)))

    scores = _window_sums(_integral(energy), win_w, win_h)
    scores = scores * _centre_weight(scores.shape)
    best = int(np.argmax(scores))
    row, col = divmod(best, scores.shape[1])

    # Map the analysis-space window back to source pixels.
    left = round(col / scale_x)
    top = round(row / scale_y)
    left = max(0, min(left, image.width - target.width))
    top = max(0, min(top, image.height - target.height))
    return (left, top, left + target.width, top + target.height)


def smart_crop(image: Image.Image, target: Size, *, method: str = "attention") -> Image.Image:
    """Crop to ``target`` at the most salient window."""
    box = best_window(image, target, method=method)
    return image.crop(box)


def smart_thumbnail(
    image: Image.Image,
    target: Size,
    *,
    method: str = "attention",
    resampler=None,
    limits=None,
) -> Image.Image:
    """Scale-and-crop to exactly ``target``, choosing the crop by saliency.

    Differs from ``fit="cover"`` only in *where* the crop is taken from.

    ``limits`` bounds the scaled intermediate, which is what gets allocated:
    a very wide, very short target scales the source up enormously before the
    crop brings it back down.
    """
    from .ops import scale_exact

    ratio = max(target.width / image.width, target.height / image.height)
    scaled_size = Size(
        max(target.width, round(image.width * ratio)),
        max(target.height, round(image.height * ratio)),
    )
    if limits is not None:
        from ..security.limits import check_output_geometry

        check_output_geometry(scaled_size.width, scaled_size.height, limits)
    scaled = scale_exact(image, scaled_size, resampler=resampler)
    return smart_crop(scaled, target, method=method)


def focal_point(image: Image.Image, *, method: str = "attention") -> tuple[float, float]:
    """Return the saliency centroid as ``(x, y)`` ratios in ``0..1``."""
    energy = saliency_map(image, method)
    rows, cols = energy.shape
    y_axis = np.linspace(0.0, 1.0, rows, dtype=np.float64)[:, None]
    x_axis = np.linspace(0.0, 1.0, cols, dtype=np.float64)[None, :]
    total = float(energy.sum())
    if total <= 0:
        return (0.5, 0.5)
    return (float((energy * x_axis).sum() / total), float((energy * y_axis).sum() / total))


__all__ = [
    "best_window",
    "focal_point",
    "saliency_map",
    "smart_crop",
    "smart_thumbnail",
]

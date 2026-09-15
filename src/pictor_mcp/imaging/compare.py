"""Image comparison and perceptual hashing.

Two jobs, both of which an agent needs to close the loop on an image edit:

**Did my transform do what I meant?** RMSE/PSNR answer "how different", SSIM
answers "does it look different", and the percentage of pixels past a threshold
answers "how much of it changed". SSIM is the useful one - a shift of one pixel
produces a huge RMSE and a tiny SSIM change, which matches human judgement.

**Are these the same image?** Perceptual hashes (aHash, dHash, pHash) survive
resizing and re-encoding, so they detect duplicates that a byte comparison
misses. pHash uses a DCT, implemented here with a plain matrix multiply rather
than pulling in SciPy for one function.

Two implementation notes that matter more than they look:

*Memory.* Everything is accumulated one horizontal strip at a time, so peak
memory is O(rows_per_strip x width) rather than O(pixels). The naive
formulation materialises float64 copies of the whole image - and half a dozen
more inside SSIM - which measured about 2 GB for a 16 MP pair and extrapolated
past 8 GB at the 64 MP input ceiling. A flat 64 MP PNG is only ~77 KiB, so that
was reachable from a tiny upload.

*Numerical care.* The variance terms are computed as a difference of means,
which cancels catastrophically on high-contrast images and can come out slightly
negative. Left alone, a negative variance makes the SSIM denominator negative,
and dividing by it yields a large negative score - reporting "maximally
different" for two nearly identical images. Variances are therefore clamped at
zero and the covariance is clamped to the range the variances permit, which is
the mathematically correct treatment and keeps SSIM inside ``[-1, 1]``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops

from ..errors import InvalidArgumentError

logger = logging.getLogger(__name__)

#: SSIM stabilisation constants from Wang et al. (2004), for 8-bit data.
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_L = 255.0
_SSIM_WINDOW = 7

#: Rows processed per strip. Comfortably larger than the SSIM window halo, and
#: what makes peak memory independent of image size.
_STRIP_ROWS = 256


@dataclass(slots=True)
class ComparisonResult:
    identical: bool
    width_a: int
    height_a: int
    width_b: int
    height_b: int
    same_size: bool
    rmse: float
    psnr_db: float | None
    mean_absolute_error: float
    max_difference: int
    changed_pixel_ratio: float
    ssim: float
    hash_hamming_distance: int
    hash_similarity: float

    def to_public_dict(self) -> dict[str, object]:
        return {
            "identical": self.identical,
            "sameSize": self.same_size,
            "sizeA": {"width": self.width_a, "height": self.height_a},
            "sizeB": {"width": self.width_b, "height": self.height_b},
            "rmse": round(self.rmse, 4),
            "psnrDb": round(self.psnr_db, 2) if self.psnr_db is not None else None,
            "meanAbsoluteError": round(self.mean_absolute_error, 4),
            "maxDifference": self.max_difference,
            "changedPixelRatio": round(self.changed_pixel_ratio, 6),
            "ssim": round(self.ssim, 6),
            "perceptualHashDistance": self.hash_hamming_distance,
            "perceptualHashSimilarity": round(self.hash_similarity, 4),
        }


def _gray_array(image: Image.Image) -> np.ndarray:
    """Grayscale pixels as uint8.

    Deliberately not converted to float here: a float64 copy of a 64 MP image is
    512 MB, and each metric below would need several such copies live at once.
    Metrics convert one strip at a time instead.
    """
    return np.asarray(image.convert("L"))


def _box_filter(array: np.ndarray, radius: int) -> np.ndarray:
    """Uniform mean filter via a summed-area table."""
    if radius < 1:
        return array
    padded = np.pad(array, radius, mode="reflect")
    height, width = array.shape
    window = 2 * radius + 1
    # The running sum is accumulated in float64 (a float32 cumsum over a large
    # strip loses precision badly) and narrowed to float32 for the pointwise
    # arithmetic, which is where memory would otherwise multiply.
    table = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    table[1:, 1:] = padded.cumsum(0).cumsum(1)
    total = table[window:, window:] - table[:-window, window:] - table[window:, :-window] + table[:-window, :-window]
    return (total[:height, :width] / (window * window)).astype(np.float32)


def _ssim_map(a: np.ndarray, b: np.ndarray, radius: int) -> np.ndarray:
    """Per-pixel SSIM for one block of float32 data, over a uniform window.

    A uniform window is used instead of an 11-tap Gaussian: it is what the
    scikit-image default approximates, it is cheaper, and for a "did anything
    change" signal the difference is immaterial.
    """
    mu_a = _box_filter(a, radius)
    mu_b = _box_filter(b, radius)

    # E[x^2] - E[x]^2 cancels badly on high-contrast regions and can come out
    # slightly negative. Clamping is not cosmetic here; see the module docstring.
    var_a = np.maximum(_box_filter(a * a, radius) - mu_a * mu_a, 0.0)
    var_b = np.maximum(_box_filter(b * b, radius) - mu_b * mu_b, 0.0)
    cov_ab = _box_filter(a * b, radius) - mu_a * mu_b

    # |cov| <= sqrt(var_a * var_b) by Cauchy-Schwarz; enforce it so the
    # numerator cannot exceed what the variances allow.
    cov_ab = np.clip(cov_ab, -np.sqrt(var_a * var_b), np.sqrt(var_a * var_b))

    c1 = (_SSIM_K1 * _SSIM_L) ** 2
    c2 = (_SSIM_K2 * _SSIM_L) ** 2

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    # Both denominator factors are now strictly positive, so this is safe; the
    # maximum only guards a degenerate all-zero block.
    return numerator / np.maximum(denominator, 1e-12)


def _strips(height: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` row ranges covering ``height``."""
    for start in range(0, height, _STRIP_ROWS):
        yield start, min(height, start + _STRIP_ROWS)


def _ssim_block(a: np.ndarray, b: np.ndarray, start: int, stop: int, height: int, radius: int) -> np.ndarray:
    """SSIM map for one strip, expanded by the window halo on each side."""
    top = max(0, start - radius)
    bottom = min(height, stop + radius)
    return _ssim_map(
        np.asarray(a[top:bottom], dtype=np.float32),
        np.asarray(b[top:bottom], dtype=np.float32),
        radius,
    )


def _ssim_accumulate(a: np.ndarray, b: np.ndarray, *, radius: int) -> tuple[float, int]:
    """Sum and count of the SSIM map, accumulated strip-wise."""
    height = a.shape[0]
    total = 0.0
    count = 0
    for start, stop in _strips(height):
        block = _ssim_block(a, b, start, stop, height, radius)
        lead = start - max(0, start - radius)
        usable = block[lead : lead + (stop - start)]
        if usable.size:
            total += float(usable.sum())
            count += int(usable.size)
    return total, count


def ssim(a: np.ndarray, b: np.ndarray, *, window: int = _SSIM_WINDOW) -> float:
    """Mean structural similarity, computed strip-wise to bound memory."""
    if a.shape != b.shape:
        raise InvalidArgumentError("SSIM requires identically shaped arrays")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise InvalidArgumentError("SSIM requires a non-empty image")

    total, count = _ssim_accumulate(a, b, radius=max(1, window // 2))
    if not count:  # pragma: no cover - guarded above
        raise InvalidArgumentError("SSIM requires a non-empty image")
    return float(np.clip(total / count, -1.0, 1.0))


def compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
    *,
    change_threshold: int = 8,
) -> ComparisonResult:
    """Compute every metric for two equally-shaped grayscale arrays."""
    if a.shape != b.shape:
        raise InvalidArgumentError("comparison requires identically shaped images")

    height, width = a.shape
    if height == 0 or width == 0:
        raise InvalidArgumentError("comparison requires a non-empty image")

    radius = max(1, _SSIM_WINDOW // 2)
    sum_sq = 0.0
    sum_abs = 0.0
    max_difference = 0
    changed_count = 0
    ssim_total = 0.0
    ssim_count = 0
    identical = True

    for start, stop in _strips(height):
        strip_a = np.asarray(a[start:stop], dtype=np.float32)
        strip_b = np.asarray(b[start:stop], dtype=np.float32)

        difference = np.abs(strip_a - strip_b)
        sum_sq += float(np.einsum("ij,ij->", difference, difference))
        sum_abs += float(difference.sum())
        max_difference = max(max_difference, int(difference.max()))
        changed_count += int(np.count_nonzero(difference > change_threshold))
        if identical and not np.array_equal(strip_a, strip_b):
            identical = False
        del difference, strip_a, strip_b

        block = _ssim_block(a, b, start, stop, height, radius)
        lead = start - max(0, start - radius)
        usable = block[lead : lead + (stop - start)]
        if usable.size:
            ssim_total += float(usable.sum())
            ssim_count += int(usable.size)
        del block

    pixels = float(height * width)
    mse = sum_sq / pixels
    rmse = float(np.sqrt(mse))
    mae = sum_abs / pixels
    changed = changed_count / pixels

    psnr: float | None = None
    if mse > 0:
        psnr = float(10.0 * np.log10((255.0**2) / mse))

    mean_ssim = float(np.clip(ssim_total / ssim_count, -1.0, 1.0)) if ssim_count else 0.0

    hash_a = phash_bits(a)
    hash_b = phash_bits(b)
    distance = int(np.count_nonzero(hash_a != hash_b))
    similarity = 1.0 - distance / hash_a.size

    return ComparisonResult(
        identical=identical,
        width_a=width,
        height_a=height,
        width_b=width,
        height_b=height,
        same_size=True,
        rmse=rmse,
        psnr_db=psnr,
        mean_absolute_error=mae,
        max_difference=max_difference,
        changed_pixel_ratio=changed,
        ssim=mean_ssim,
        hash_hamming_distance=distance,
        hash_similarity=similarity,
    )


def compare_images(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    align: bool = True,
    change_threshold: int = 8,
) -> ComparisonResult:
    """Compare two images, optionally resizing ``b`` to match ``a``.

    ``align`` is on by default because "same picture, wrong size" is the common
    case when verifying a resize; it is reported via ``same_size`` so the caller
    knows a rescale happened.
    """
    same_size = image_a.size == image_b.size
    prepared_b = image_b
    if not same_size:
        if not align:
            raise InvalidArgumentError(
                f"images differ in size ({image_a.width}x{image_a.height} vs "
                f"{image_b.width}x{image_b.height}); pass align=true to compare anyway"
            )
        prepared_b = image_b.resize(image_a.size, Image.Resampling.LANCZOS)

    result = compare_arrays(_gray_array(image_a), _gray_array(prepared_b), change_threshold=change_threshold)
    result.width_b = image_b.width
    result.height_b = image_b.height
    result.same_size = same_size
    return result


def difference_image(image_a: Image.Image, image_b: Image.Image, *, amplify: float = 1.0) -> Image.Image:
    """Return a visual difference image (white where identical)."""
    if image_a.size != image_b.size:
        image_b = image_b.resize(image_a.size, Image.Resampling.LANCZOS)
    difference = ImageChops.difference(image_a.convert("RGB"), image_b.convert("RGB"))
    if amplify and amplify != 1.0:
        difference = difference.point(lambda value: min(255, int(value * amplify)))
    return difference


# ------------------------------------------------------------------ hashing


def _dct_matrix(size: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape ``(size, size)``.

    Built explicitly rather than imported: a 32x32 matrix costs nothing and it
    keeps SciPy out of the dependency list for a single transform.
    """
    index = np.arange(size, dtype=np.float64)
    factor = np.pi / size * (index + 0.5)
    matrix = np.cos(np.outer(index, factor))
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix


def _downscale(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downscale through Pillow, narrowing to uint8 first.

    Narrowing before the resize matters: resizing a float array would allocate a
    float buffer the size of the *source* image for a 32x32 result.
    """
    clipped = np.clip(gray, 0, 255)
    if clipped.dtype != np.uint8:
        clipped = clipped.astype(np.uint8)
    array = np.asarray(Image.fromarray(clipped, "L").resize((width, height), Image.Resampling.LANCZOS))
    return array.astype(np.float64)


def phash_bits(gray: np.ndarray, *, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Perceptual hash bits (DCT-based), returned as a boolean array."""
    size = hash_size * highfreq_factor
    small = _downscale(gray, size, size)
    matrix = _dct_matrix(size)
    coefficients = matrix @ small @ matrix.T
    low = coefficients[:hash_size, :hash_size]
    # The DC term carries only overall brightness, so it is excluded from the
    # median and from the hash itself.
    values = low.flatten()[1:]
    threshold = np.median(values)
    return values > threshold


def dhash_bits(gray: np.ndarray, *, hash_size: int = 8) -> np.ndarray:
    """Difference hash: sign of horizontal gradients on a downscaled image."""
    small = _downscale(gray, hash_size + 1, hash_size).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def ahash_bits(gray: np.ndarray, *, hash_size: int = 8) -> np.ndarray:
    """Average hash: pixels brighter than the mean."""
    small = _downscale(gray, hash_size, hash_size)
    return (small > small.mean()).flatten()


def hash_hex(bits: np.ndarray) -> str:
    """Render hash bits as lowercase hex, most significant bit first."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    width = (bits.size + 3) // 4
    return f"{value:0{width}x}"


def perceptual_hashes(image: Image.Image) -> dict[str, str]:
    """All three hashes for an image, as hex strings."""
    gray = _gray_array(image)
    return {
        "phash": hash_hex(phash_bits(gray)),
        "dhash": hash_hex(dhash_bits(gray)),
        "ahash": hash_hex(ahash_bits(gray)),
    }


__all__ = [
    "ComparisonResult",
    "ahash_bits",
    "compare_arrays",
    "compare_images",
    "dhash_bits",
    "difference_image",
    "hash_hex",
    "perceptual_hashes",
    "phash_bits",
    "ssim",
]

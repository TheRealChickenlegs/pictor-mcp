"""Geometry and pixel operations.

Each function takes and returns a ``PIL.Image`` and does exactly one thing, so
the pipeline executor can compose them and the tool layer can expose them
individually. Sizing arithmetic is separated from resampling so the "what size
should this be" decision is unit-testable without touching pixels.

Resampling is delegated to a pluggable backend (see
:mod:`pictor_mcp.backends`) so the same call can run on the CPU or on a CUDA
device without any caller-visible difference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageChops, ImageFilter, ImageOps

from ..errors import InvalidArgumentError
from ..security.limits import check_output_geometry

logger = logging.getLogger(__name__)

FitMode = Literal["contain", "cover", "fill", "inside", "outside", "pad"]
Gravity = Literal[
    "center",
    "top",
    "bottom",
    "left",
    "right",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
]

#: Friendly resampling filter names mapped onto Pillow's enum.
_FILTERS: dict[str, Image.Resampling] = {
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
    "bilinear": Image.Resampling.BILINEAR,
    "hamming": Image.Resampling.HAMMING,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

#: Default filter per direction. Lanczos for downscaling (avoids aliasing),
#: bicubic for upscaling (Lanczos ringing is visible when enlarging).
_DOWNSCALE_FILTER = "lanczos"
_UPSCALE_FILTER = "bicubic"

_RGBA = tuple[int, int, int, int]


def resolve_filter(name: str | None, *, upscaling: bool) -> Image.Resampling:
    """Map a filter name to a Pillow enum, defaulting sensibly by direction."""
    if name in (None, "", "auto"):
        return _FILTERS[_UPSCALE_FILTER if upscaling else _DOWNSCALE_FILTER]
    key = name.strip().lower()
    if key not in _FILTERS:
        raise InvalidArgumentError(
            f"unknown resampling filter {name!r}",
            supported=sorted(_FILTERS),
        )
    return _FILTERS[key]


def parse_gravity(gravity: str | None) -> tuple[float, float]:
    """Return ``(x_ratio, y_ratio)`` in ``0..1`` for a named position."""
    if not gravity:
        return (0.5, 0.5)
    key = gravity.strip().lower().replace("_", "-")
    mapping = {
        "center": (0.5, 0.5),
        "centre": (0.5, 0.5),
        "top": (0.5, 0.0),
        "bottom": (0.5, 1.0),
        "left": (0.0, 0.5),
        "right": (1.0, 0.5),
        "top-left": (0.0, 0.0),
        "top-right": (1.0, 0.0),
        "bottom-left": (0.0, 1.0),
        "bottom-right": (1.0, 1.0),
    }
    if key not in mapping:
        raise InvalidArgumentError(
            f"unknown gravity {gravity!r}",
            supported=sorted(mapping),
        )
    return mapping[key]


def _offset(container: int, inner: int, ratio: float) -> int:
    return max(0, round((container - inner) * ratio))


# --------------------------------------------------------------------- sizing


@dataclass(frozen=True, slots=True)
class Size:
    width: int
    height: int


def compute_target_size(
    source: Size,
    *,
    width: int | None = None,
    height: int | None = None,
    percent: float | None = None,
    fit: FitMode = "contain",
    only_shrink: bool = False,
) -> Size:
    """Work out the output size for a resize request.

    ``width`` and ``height`` may be given independently; the missing axis is
    derived from the aspect ratio for ``contain``/``inside``, while ``fill``
    treats a missing axis as "keep the other one unchanged".
    """
    if source.width < 1 or source.height < 1:
        raise InvalidArgumentError("source image has no area")

    if percent is not None:
        if percent <= 0:
            raise InvalidArgumentError("percent must be greater than zero")
        width = max(1, round(source.width * percent / 100.0))
        height = max(1, round(source.height * percent / 100.0))
    elif width is None and height is None:
        raise InvalidArgumentError("provide width, height or percent")
    elif width is None:
        assert height is not None
        width = max(1, round(source.width * (height / source.height)))
    elif height is None:
        assert width is not None
        height = max(1, round(source.height * (width / source.width)))

    assert width is not None and height is not None
    if width < 1 or height < 1:
        raise InvalidArgumentError("computed output size has no area")

    if only_shrink and (width > source.width or height > source.height):
        return Size(source.width, source.height)

    return Size(width, height)


# --------------------------------------------------------------------- resize


def resize(
    image: Image.Image,
    target: Size,
    *,
    fit: FitMode = "contain",
    filter_name: str | None = None,
    background: _RGBA = (255, 255, 255, 0),
    gravity: str | None = None,
    resampler=None,
    limits=None,
) -> Image.Image:
    """Resize ``image`` to ``target`` honouring ``fit``.

    ``resampler`` lets a GPU backend take over the expensive scaling step; when
    it is ``None`` Pillow does the work.

    ``limits`` bounds the *intermediate* allocation, not just the result.
    ``cover`` and ``outside`` scale until the box is covered, so a target like
    ``24000x1`` on a square source produces a 24000x24000 intermediate (~2.3 GB
    as RGBA) before cropping back down to something tiny. Checking only the
    final size would let that through.
    """
    if fit == "fill":
        return _scale(image, target, filter_name, resampler)
    if fit in {"contain", "inside", "pad"}:
        scaled = _fit_inside(image, target, strict=fit != "inside", filter_name=filter_name, resampler=resampler)
        if fit == "pad" and (scaled.width, scaled.height) != (target.width, target.height):
            return _pad_to(scaled, target, background, gravity)
        return scaled
    if fit == "cover":
        return _cover(image, target, filter_name=filter_name, gravity=gravity, resampler=resampler, limits=limits)
    if fit == "outside":
        return _fit_outside(image, target, filter_name=filter_name, resampler=resampler, limits=limits)
    raise InvalidArgumentError(
        f"unknown fit mode {fit!r}",
        supported=["contain", "cover", "fill", "inside", "outside", "pad"],
    )


def _scale(image: Image.Image, target: Size, filter_name: str | None, resampler) -> Image.Image:
    upscaling = target.width > image.width or target.height > image.height
    pil_filter = resolve_filter(filter_name, upscaling=upscaling)
    if resampler is not None:
        converted = resampler.resize(image, (target.width, target.height), pil_filter)
        if converted is not None:
            return converted
    return image.resize((target.width, target.height), pil_filter)


def scale_exact(
    image: Image.Image,
    target: Size,
    *,
    filter_name: str | None = None,
    resampler=None,
) -> Image.Image:
    """Scale to exactly ``target``, ignoring aspect ratio.

    Public wrapper so other modules can reuse the acceleration path without
    reaching for a private helper.
    """
    if target.width == image.width and target.height == image.height:
        return image
    return _scale(image, target, filter_name, resampler)


def _fit_inside(
    image: Image.Image,
    target: Size,
    *,
    strict: bool,
    filter_name: str | None,
    resampler,
) -> Image.Image:
    ratio = min(target.width / image.width, target.height / image.height)
    if not strict:
        ratio = min(ratio, 1.0)
    width = max(1, round(image.width * ratio))
    height = max(1, round(image.height * ratio))
    if (width, height) == (image.width, image.height):
        return image
    return _scale(image, Size(width, height), filter_name, resampler)


def _fit_outside(image: Image.Image, target: Size, *, filter_name: str | None, resampler, limits=None) -> Image.Image:
    ratio = max(target.width / image.width, target.height / image.height)
    width = max(1, round(image.width * ratio))
    height = max(1, round(image.height * ratio))
    if limits is not None:
        # Bound the scaled size before it is allocated, because that is the
        # buffer that actually consumes memory - not the cropped result.
        check_output_geometry(width, height, limits)
    return _scale(image, Size(width, height), filter_name, resampler)


def _cover(
    image: Image.Image,
    target: Size,
    *,
    filter_name: str | None,
    gravity: str | None,
    resampler,
    limits=None,
) -> Image.Image:
    scaled = _fit_outside(image, target, filter_name=filter_name, resampler=resampler, limits=limits)
    return crop_to_size(scaled, target, gravity=gravity)


def crop_to_size(image: Image.Image, target: Size, *, gravity: str | None = None) -> Image.Image:
    """Centre (or gravity-align) crop to exactly ``target``."""
    x_ratio, y_ratio = parse_gravity(gravity)
    left = _offset(image.width, target.width, x_ratio)
    top = _offset(image.height, target.height, y_ratio)
    box = (left, top, left + min(target.width, image.width), top + min(target.height, image.height))
    return image.crop(box)


def _pad_to(image: Image.Image, target: Size, background: _RGBA, gravity: str | None) -> Image.Image:
    x_ratio, y_ratio = parse_gravity(gravity)
    canvas = Image.new(canvas_mode_for(image, background), (target.width, target.height), background)
    left = _offset(target.width, image.width, x_ratio)
    top = _offset(target.height, image.height, y_ratio)
    if image.mode == "RGBA" and canvas.mode == "RGBA":
        canvas.alpha_composite(image, (left, top))
    elif image.mode == "LA" and canvas.mode == "LA":
        canvas.paste(image, (left, top))
    else:
        canvas.paste(image, (left, top))
    return canvas


def canvas_mode_for(image: Image.Image, background: tuple[int, ...]) -> str:
    if len(background) == 4 and image.mode in {"RGBA", "LA", "P"}:
        return "RGBA"
    if image.mode == "LA":
        return "LA"
    if image.mode == "L":
        return "L"
    return "RGB"


# ----------------------------------------------------------------------- crop


def crop_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    clamp: bool = True,
) -> Image.Image:
    """Crop to an explicit pixel box.

    ``clamp`` clips the box to the image instead of failing, which is what an
    agent wants when it computed the box from a slightly different scale.
    """
    left, top, right, bottom = box
    if right <= left or bottom <= top:
        raise InvalidArgumentError(f"crop box {box} has no area (expected right > left and bottom > top)")
    if clamp:
        left = max(0, min(left, image.width - 1))
        top = max(0, min(top, image.height - 1))
        right = max(left + 1, min(right, image.width))
        bottom = max(top + 1, min(bottom, image.height))
    elif left < 0 or top < 0 or right > image.width or bottom > image.height:
        raise InvalidArgumentError(
            f"crop box {box} falls outside the {image.width}x{image.height} image",
            image_size=[image.width, image.height],
        )
    return image.crop((left, top, right, bottom))


def crop_to_aspect(
    image: Image.Image,
    aspect: float,
    *,
    gravity: str | None = None,
) -> Image.Image:
    """Crop to a width/height ratio, keeping as much of the image as possible."""
    if aspect <= 0:
        raise InvalidArgumentError("aspect_ratio must be greater than zero")
    current = image.width / image.height
    if current > aspect:
        width = max(1, round(image.height * aspect))
        target = Size(width, image.height)
    else:
        height = max(1, round(image.width / aspect))
        target = Size(image.width, height)
    return crop_to_size(image, target, gravity=gravity)


def trim_border(image: Image.Image, *, tolerance: int = 12, background: tuple[int, ...] | None = None) -> Image.Image:
    """Remove a uniform border (the ``-trim`` operation).

    Uses :func:`PIL.ImageChops.difference` against the detected corner colour so
    it works on photographic scans, not just exact-colour graphics.
    """
    if tolerance < 0:
        raise InvalidArgumentError("trim tolerance must not be negative")
    rgb = image.convert("RGB")
    if background is None:
        corners = [
            rgb.getpixel((0, 0)),
            rgb.getpixel((rgb.width - 1, 0)),
            rgb.getpixel((0, rgb.height - 1)),
            rgb.getpixel((rgb.width - 1, rgb.height - 1)),
        ]
        background = corners[0] if len(set(corners)) == 1 else _average_colour(corners)

    backdrop = Image.new("RGB", rgb.size, background)
    difference = ImageChops.difference(rgb, backdrop).convert("L")
    if tolerance:
        difference = difference.point(lambda value: 255 if value > tolerance else 0)
    bbox = difference.getbbox()
    if bbox is None:
        raise InvalidArgumentError("nothing to trim: the image is a single uniform colour")
    return image.crop(bbox)


def _average_colour(colours: list[tuple[int, ...]]) -> tuple[int, ...]:
    count = len(colours)
    channels = len(colours[0])
    return tuple(  # type: ignore[return-value]
        int(sum(colour[index] for colour in colours) / count) for index in range(channels)
    )


# --------------------------------------------------------------------- rotate


def rotate(
    image: Image.Image,
    angle: float,
    *,
    expand: bool = True,
    background: _RGBA = (255, 255, 255, 0),
    filter_name: str | None = "bicubic",
) -> Image.Image:
    """Rotate by an arbitrary angle (counter-clockwise, like Pillow)."""
    if angle % 360 == 0:
        return image
    pil_filter = resolve_filter(filter_name, upscaling=False)
    fill = background if image.mode in {"RGBA", "LA", "P"} else background[:3]
    return image.rotate(
        angle,
        resample=pil_filter,
        expand=expand,
        fillcolor=fill,
    )


def flip(image: Image.Image, direction: str = "horizontal") -> Image.Image:
    """Mirror horizontally or vertically."""
    key = (direction or "horizontal").strip().lower()
    if key in {"horizontal", "h", "left-right", "x"}:
        return ImageOps.mirror(image)
    if key in {"vertical", "v", "top-bottom", "y"}:
        return ImageOps.flip(image)
    if key in {"both", "transpose", "180"}:
        return ImageOps.mirror(ImageOps.flip(image))
    raise InvalidArgumentError(
        f"unknown flip direction {direction!r}",
        supported=["horizontal", "vertical", "both"],
    )


def auto_orient(image: Image.Image) -> Image.Image:
    """Apply the EXIF orientation tag so the pixels match what a viewer shows.

    ``exif_transpose`` removes the tag it applied, which keeps a later
    "rotate 90" from double-applying.
    """
    try:
        return ImageOps.exif_transpose(image) or image
    except Exception:  # pragma: no cover - malformed EXIF
        logger.debug("exif_transpose failed; leaving orientation untouched", exc_info=True)
        return image


def sharpen(image: Image.Image, *, amount: float = 1.0, radius: float = 2.0, threshold: int = 3) -> Image.Image:
    """Unsharp mask, scaled by ``amount`` (0 disables)."""
    if amount <= 0:
        return image
    if not 0 <= threshold <= 255:
        raise InvalidArgumentError("sharpen threshold must be between 0 and 255")
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100), threshold=threshold))


def blur(image: Image.Image, *, radius: float) -> Image.Image:
    if radius <= 0:
        raise InvalidArgumentError("blur radius must be greater than zero")
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def enforce_output_limits(width: int, height: int, limits) -> None:
    """Reject an operation whose *result* would be too large to hold."""
    check_output_geometry(width, height, limits)


__all__ = [
    "FitMode",
    "Gravity",
    "Size",
    "auto_orient",
    "blur",
    "compute_target_size",
    "crop_box",
    "crop_to_aspect",
    "crop_to_size",
    "enforce_output_limits",
    "flip",
    "parse_gravity",
    "resize",
    "resolve_filter",
    "rotate",
    "sharpen",
    "trim_border",
]
